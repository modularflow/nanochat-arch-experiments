"""
TRM-GPT: Tiny Recursive Model adapted for autoregressive language modeling.

Inspired by "Less is More: Recursive Reasoning with Tiny Networks"
(Jolicoeur-Martineau, 2025). arxiv.org/abs/2510.04871

Core idea: instead of a deep stack of unique layers, use a tiny number of
unique blocks (default: 2) and recurse through them many times. Effective
depth comes from recursion rather than unique parameters.

Key parameters:
  - n_unique_layers: unique transformer blocks (paper finds 2 optimal)
  - n_recur: recursions per cycle (each recursion traverses all unique blocks)
  - T_cycles: total recursion cycles

Effective depth = T_cycles * n_recur * n_unique_layers  (e.g. 3 * 6 * 2 = 36)

Memory optimization (TRM's key insight):
  During training, T-1 cycles run without gradients. Only the last cycle
  backpropagates. Hidden states are detached between cycles. This gives
  effective depth of 36 while only backpropagating through 12 layers.

Usage example:
  python -m scripts.base_train_jepa --architecture trm_gpt --depth 2 \
      --aspect-ratio 256 --trm-n-recur 6 --trm-T-cycles 3
  # → n_unique=2, dim=512, effective_depth=36
"""

from contextlib import nullcontext
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
from nanochat.gpt import Block, KVCache, norm


@dataclass
class TRMGPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_unique_layers: int = 2    # TRM: tiny network (paper finds 2 optimal)
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "L"
    n_recur: int = 6             # recursions per cycle
    T_cycles: int = 3            # recursion cycles (T-1 without grad during training)

    @property
    def effective_depth(self):
        return self.T_cycles * self.n_recur * self.n_unique_layers

    @property
    def n_layer(self):
        """Alias for compatibility with code that expects n_layer."""
        return self.effective_depth


class TRMGPT(nn.Module):
    def __init__(self, config: TRMGPTConfig, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.max_seq_len = config.sequence_len
        n_unique = config.n_unique_layers
        effective = config.effective_depth
        self._layers_per_cycle = config.n_recur * n_unique

        # effective → physical block mapping
        self._layer_map = [i % n_unique for i in range(effective)]

        self.window_sizes = self._compute_window_sizes(config)

        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx=i) for i in range(n_unique)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)

        # Per-(recur, block) scalars shared across T cycles so all get gradients
        # from the last cycle's backprop (TRM's weight-sharing philosophy)
        self.resid_lambdas = nn.Parameter(torch.ones(self._layers_per_cycle))
        self.x0_lambdas = nn.Parameter(torch.zeros(self._layers_per_cycle))

        # Marker buffer for checkpoint architecture detection
        self.register_buffer(
            'trm_marker',
            torch.tensor([config.n_unique_layers, config.n_recur, config.T_cycles], dtype=torch.long),
        )

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
        effective = config.effective_depth
        window_sizes = []
        for layer_idx in range(effective):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def _run_cycle(self, x, x0, cos_sin, eff_offset, kv_cache):
        """Run one recursion cycle: n_recur traversals through n_unique blocks."""
        n_unique = self.config.n_unique_layers
        eff_idx = eff_offset
        for _r in range(self.config.n_recur):
            for i in range(n_unique):
                block = self.transformer.h[i]
                if kv_cache is not None:
                    block.attn.layer_idx = eff_idx
                lambda_idx = eff_idx % self._layers_per_cycle
                x = self.resid_lambdas[lambda_idx] * x + self.x0_lambdas[lambda_idx] * x0
                x = block(x, cos_sin, self.window_sizes[eff_idx], kv_cache)
                eff_idx += 1
        return x

    def estimate_flops(self):
        """FLOPs based on effective computation (shared blocks counted per traversal)."""
        block_params = [sum(p.numel() for p in block.parameters()) for block in self.transformer.h]
        n_unique = self.config.n_unique_layers
        effective = self.config.effective_depth
        effective_matmul_params = sum(block_params[i % n_unique] for i in range(effective))
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

    def _run_cycles(self, x, x0, cos_sin, kv_cache):
        """Run all T recursion cycles with the TRM memory optimisation.

        During training, T-1 cycles run under ``torch.no_grad`` and hidden
        states are detached between cycles.  The autocast weight cache is
        cleared before the final (gradient-tracked) cycle so that shared block
        weights are re-cast with autograd tracking intact (PyTorch's autocast
        caches the bf16 cast; if the first use is under no_grad the cached
        version has no grad_fn, poisoning later uses).
        """
        layers_per_cycle = self._layers_per_cycle
        for t in range(self.config.T_cycles):
            eff_offset = t * layers_per_cycle
            use_no_grad = self.training and t < self.config.T_cycles - 1
            if use_no_grad:
                with torch.no_grad():
                    x = self._run_cycle(x, x0, cos_sin, eff_offset, kv_cache)
                x = x.detach()
            else:
                if self.training and t > 0:
                    torch.clear_autocast_cache()
                x = self._run_cycle(x, x0, cos_sin, eff_offset, kv_cache)
        return x

    def forward_to_final_hidden(self, idx):
        """Full multi-cycle recursive forward returning the final hidden state (before lm_head).

        Used by JEPA to get a representation that respects TRM's recursion
        cycles, rather than iterating only over the unique physical blocks.
        Mirrors training behaviour: T-1 cycles without grad, last cycle with grad.
        """
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        x = self._run_cycles(x, x0, cos_sin, None)
        return norm(x)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x

        x = self._run_cycles(x, x0, cos_sin, kv_cache)

        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Autoregressive streaming inference with KV cache."""
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        effective = self.config.effective_depth
        cache = KVCache(
            batch_size=1,
            seq_len=len(tokens) + max_tokens,
            n_kv_head=self.config.n_kv_head,
            head_dim=self.config.n_embd // self.config.n_head,
            num_layers=effective,
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
