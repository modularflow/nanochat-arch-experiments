"""
RYS-GPT: GPT with Repeated Middle Layers.

Inspired by the RYS (Repeat Your Self) finding that middle transformer layers
form "reasoning circuits" that can be profitably re-traversed. This architecture
trains with weight-sharing in the middle layers: fewer unique parameter blocks
than effective depth, with the middle block traversed multiple times.

Single-block example (d12, rys_block_start=3, rys_block_end=6, rys_num_repeats=2):
  - 9 unique transformer blocks (d9 parameter budget)
  - 12 effective layer traversals (d12 compute budget)
  - Forward order: [0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8]

Multi-block composition (from RYS-II beam search findings):
  - rys_blocks="28,34;43,45" → two disjoint blocks each repeated once
  - This overrides the single-block rys_block_start/end/num_repeats params
  - Blog post found Pareto frontier dominated by contiguous single blocks,
    but multi-block can produce stronger raw scores at higher overhead

Ref: https://dnhkng.github.io/posts/rys-ii/
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
from nanochat.gpt import Block, KVCache, norm


def _parse_rys_blocks(spec):
    """Parse multi-block spec string into sorted list of (start, end) tuples.

    Format: "start1,end1;start2,end2;..." where each (start, end) is a
    half-open range of unique-block indices to repeat once.
    """
    blocks = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        start_s, end_s = part.split(",")
        blocks.append((int(start_s.strip()), int(end_s.strip())))
    blocks.sort(key=lambda b: b[0])
    for i in range(len(blocks) - 1):
        assert blocks[i][1] <= blocks[i + 1][0], \
            f"RYS blocks must not overlap: {blocks[i]} and {blocks[i+1]}"
    return blocks


@dataclass
class RYSGPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12           # effective depth (total layer traversals)
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    # Single-block config (used when rys_blocks is empty)
    rys_block_start: int = 3    # first block of repeated section (inclusive)
    rys_block_end: int = 6      # end of repeated section (exclusive, in unique-block indices)
    rys_num_repeats: int = 2    # times the reasoning block is traversed
    # Multi-block config (overrides single-block when set)
    # Format: "start1,end1;start2,end2;..." - each block repeated once
    rys_blocks: str = ""

    @property
    def n_unique_layers(self):
        if self.rys_blocks:
            blocks = _parse_rys_blocks(self.rys_blocks)
            extra = sum(end - start for start, end in blocks)
            return self.n_layer - extra
        block_size = self.rys_block_end - self.rys_block_start
        return self.n_layer - (self.rys_num_repeats - 1) * block_size


class RYSGPT(nn.Module):
    def __init__(self, config: RYSGPTConfig, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.max_seq_len = config.sequence_len
        n_unique = config.n_unique_layers

        assert n_unique > 0, f"RYS config yields {n_unique} unique layers (need > 0)"

        # effective→physical block mapping
        if config.rys_blocks:
            blocks = _parse_rys_blocks(config.rys_blocks)
            for s, e in blocks:
                assert 0 <= s < e <= n_unique, f"Block ({s},{e}) out of range [0, {n_unique})"
            # Build layer map: traverse unique layers, inserting repeats after each block
            layer_map = list(range(n_unique))
            for start, end in reversed(blocks):
                repeat = list(range(start, end))
                layer_map = layer_map[:end] + repeat + layer_map[end:]
        else:
            assert 0 <= config.rys_block_start < config.rys_block_end <= n_unique
            assert config.rys_num_repeats >= 1
            layer_map = []
            for i in range(config.rys_block_start):
                layer_map.append(i)
            for _ in range(config.rys_num_repeats):
                for i in range(config.rys_block_start, config.rys_block_end):
                    layer_map.append(i)
            for i in range(config.rys_block_end, n_unique):
                layer_map.append(i)
        assert len(layer_map) == config.n_layer, \
            f"Layer map length {len(layer_map)} != n_layer {config.n_layer}"
        self._layer_map = layer_map  # Python list for use in forward/flops (avoids meta tensor issues)
        self.register_buffer('rys_layer_map', torch.tensor(layer_map, dtype=torch.long))

        # Per-effective-layer window sizes
        self.window_sizes = self._compute_window_sizes(config)

        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx=i) for i in range(n_unique)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)

        # n_layer entries (effective depth), not n_unique
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        with torch.no_grad():
            self.resid_lambdas.fill_(1.0)
            self.x0_lambdas.fill_(0.0)

        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}"
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """FLOPs based on effective computation (shared blocks counted per traversal)."""
        block_params = [sum(p.numel() for p in block.parameters()) for block in self.transformer.h]
        effective_matmul_params = sum(block_params[self._layer_map[i]] for i in range(self.config.n_layer))
        effective_matmul_params += self.lm_head.weight.numel()
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * effective_matmul_params + attn_flops

    def num_scaling_params(self):
        """Unique parameter count (memory footprint)."""
        return sum(p.numel() for p in self.parameters())

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(resid_params) + len(x0_params)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=resid_params, lr=scalar_lr * 0.01),
            dict(params=x0_params, lr=scalar_lr),
        ]
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward_to_final_hidden(self, idx):
        """Full effective-depth forward returning the final hidden state (before lm_head).

        Used by JEPA to get a representation that respects the RYS layer map,
        rather than iterating only over the unique physical blocks.
        """
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for eff_i in range(self.config.n_layer):
            phys_i = self._layer_map[eff_i]
            block = self.transformer.h[phys_i]
            x = self.resid_lambdas[eff_i] * x + self.x0_lambdas[eff_i] * x0
            x = block(x, cos_sin, self.window_sizes[eff_i], None)
        return norm(x)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for eff_i in range(self.config.n_layer):
            phys_i = self._layer_map[eff_i]
            block = self.transformer.h[phys_i]
            if kv_cache is not None:
                block.attn.layer_idx = eff_i
            x = self.resid_lambdas[eff_i] * x + self.x0_lambdas[eff_i] * x0
            x = block(x, cos_sin, self.window_sizes[eff_i], kv_cache)
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Autoregressive streaming inference with KV cache for O(n) generation."""
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        cache = KVCache(
            batch_size=1,
            seq_len=len(tokens) + max_tokens,
            n_kv_head=self.config.n_kv_head,
            head_dim=self.config.n_embd // self.config.n_head,
            num_layers=self.config.n_layer,
            device=device,
        )

        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.forward(ids, kv_cache=cache)
        logits = logits[:, -1, :]

        for _ in range(max_tokens):
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            token = next_ids.item()
            yield token
            logits = self.forward(next_ids, kv_cache=cache)
            logits = logits[:, -1, :]
