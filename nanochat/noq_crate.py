"""
No-Q CRATE-α: CRATE with the Q projection removed (Q = x).

Based on "The Embedding Geometry Hypothesis" (Rigoni, 2026) applied to
CRATE's Multi-Head Subspace Self-Attention (MSSA).

Original CRATE uses tied Q=K=V via a single 'qkv' projection.
No-Q CRATE uses Q = x (no projection), K = V = kv(x) (one shared projection),
preserving CRATE's weight-tying philosophy for K/V while removing Q's
competition with the embedding geometry.

This is a DROP-IN REPLACEMENT for nanochat's CRATE. All public methods match
CRATE's interface exactly.
"""

from functools import partial
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from nanochat.common import get_dist_info, print0
    from nanochat.muon import Muon, DistMuon
    from nanochat.adamw import DistAdamW
    NANOCHAT_AVAILABLE = True
except ImportError:
    NANOCHAT_AVAILABLE = False
    def get_dist_info():
        return False, 0, 0, 1
    def print0(s="", **kwargs):
        print(s, **kwargs)


@dataclass
class NoQCRATEConfig:
    """Configuration for No-Q CRATE-α (mirrors CRATEConfig interface)."""
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "L"

    odl_expansion: int = 4
    odl_use_residual: bool = True
    odl_use_relu: bool = True

    ista_step_size: float = 0.1
    ista_lambda: float = 0.1
    ista_mode: str = 'relu'
    sparse_block_type: str = "odl"


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def soft_threshold(x: torch.Tensor, lambd: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * F.relu(torch.abs(x) - lambd)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=3)


class NoQMSSA(nn.Module):
    """
    No-Q Multi-Head Subspace Self-Attention.

    Q = x (no projection), K = V = kv(x) (tied, one learned projection).
    Preserves CRATE's weight-tying for K/V while removing Q's distortion.
    """

    def __init__(self, config: NoQCRATEConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert config.n_embd % config.n_head == 0

        self.scale = self.head_dim ** -0.5

        # Single KV projection (tied K=V, as in CRATE philosophy). No Q projection.
        self.kv = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor],
                window_size: Tuple[int, int], kv_cache) -> torch.Tensor:
        B, T, C = x.size()

        q = x.view(B, T, self.n_head, self.head_dim)
        kv_proj = self.kv(x).view(B, T, self.n_head, self.head_dim)

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        kv_proj = apply_rotary_emb(kv_proj, cos, sin)
        q = norm(q)
        kv_proj = norm(kv_proj)

        if kv_cache is not None:
            pos = kv_cache.get_pos()
            k_cache, _ = kv_cache.get_layer_cache(self.layer_idx)
            k_cache[:, pos:pos + T, :, :] = kv_proj
            w_q = q
            w_kv = k_cache[:, :pos + T, :, :]
            T_k = w_kv.size(1)
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

            w_q = w_q.transpose(1, 2)
            w_kv = w_kv.transpose(1, 2)

            dots = torch.matmul(w_q, w_kv.transpose(-1, -2)) * self.scale
            causal_mask = torch.triu(
                torch.ones(T, T_k, dtype=torch.bool, device=x.device),
                diagonal=T_k - T + 1
            )
            dots = dots.masked_fill(causal_mask, float('-inf'))
            window_left, _ = window_size
            if 0 < window_left < T_k:
                positions = torch.arange(T_k, device=x.device)
                query_positions = torch.arange(T, device=x.device) + (T_k - T)
                distance = query_positions.unsqueeze(1) - positions.unsqueeze(0)
                window_mask = distance > window_left
                dots = dots.masked_fill(window_mask, float('-inf'))
            attn = F.softmax(dots, dim=-1)
            out = torch.matmul(attn, w_kv)
        else:
            q_t = q.transpose(1, 2)
            kv_t = kv_proj.transpose(1, 2)
            window_left, _ = window_size
            if 0 < window_left < T:
                mask = torch.zeros(T, T, device=x.device, dtype=q_t.dtype)
                causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
                mask.masked_fill_(causal, float('-inf'))
                positions = torch.arange(T, device=x.device)
                distance = positions.unsqueeze(0) - positions.unsqueeze(1)
                mask.masked_fill_(distance > window_left, float('-inf'))
                out = F.scaled_dot_product_attention(q_t, kv_t, kv_t, attn_mask=mask, scale=self.scale)
            else:
                out = F.scaled_dot_product_attention(q_t, kv_t, kv_t, is_causal=True, scale=self.scale)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.c_proj(out)

        return out


class ODL(nn.Module):
    """Overcomplete Dictionary Learning Block (same as CRATE-α)."""

    def __init__(self, config: NoQCRATEConfig):
        super().__init__()
        self.dim = config.n_embd
        self.expansion = config.odl_expansion
        self.hidden_dim = self.dim * self.expansion
        self.use_relu = config.odl_use_relu
        self.step_size = config.ista_step_size
        self.lambd = config.ista_lambda

        self.D_enc = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.D_dec = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.threshold = nn.Parameter(torch.zeros(self.hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.D_enc(x)
        if self.use_relu:
            h = F.relu(h - self.threshold)
        else:
            h = soft_threshold(h, self.step_size * self.lambd)
        out = self.D_dec(h)
        return out


class ISTA(nn.Module):
    """Original ISTA Block (for legacy checkpoint compatibility)."""

    def __init__(self, config: NoQCRATEConfig):
        super().__init__()
        self.dim = config.n_embd
        self.step_size = config.ista_step_size
        self.lambd = config.ista_lambda
        self.mode = config.ista_mode

        self.weight = nn.Parameter(torch.empty(self.dim, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.linear(x, self.weight, bias=None)
        grad_1 = F.linear(x1, self.weight.t(), bias=None)
        grad_2 = F.linear(x, self.weight.t(), bias=None)
        z = x + self.step_size * (grad_2 - grad_1)
        if self.mode == 'soft_threshold':
            output = soft_threshold(z, self.step_size * self.lambd)
        else:
            output = F.relu(z - self.step_size * self.lambd)
        return output


class Block(nn.Module):
    """One No-Q CRATE-α layer: NoQMSSA + ODL/ISTA with residuals."""

    def __init__(self, config: NoQCRATEConfig, layer_idx: int):
        super().__init__()
        self.use_residual = config.odl_use_residual
        self.sparse_block_type = config.sparse_block_type

        self.mssa = NoQMSSA(config, layer_idx)

        if self.sparse_block_type == "ista":
            self.ista = ISTA(config)
        else:
            self.odl = ODL(config)

    def forward(self, x: torch.Tensor, cos_sin, window_size, kv_cache) -> torch.Tensor:
        x = x + self.mssa(norm(x), cos_sin, window_size, kv_cache)

        sparse_module = self.ista if self.sparse_block_type == "ista" else self.odl
        if self.use_residual:
            x = x + sparse_module(norm(x))
        else:
            x = sparse_module(norm(x))

        return x


class KVCache:
    """KV Cache for No-Q CRATE inference (stores kv projection only)."""

    def __init__(self, batch_size: int, num_heads: int, seq_len: int,
                 head_dim: int, num_layers: int, device, dtype=torch.bfloat16):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        self.w_cache = torch.zeros(
            num_layers, batch_size, seq_len, num_heads, head_dim,
            device=device, dtype=dtype
        )
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    def reset(self):
        self.cache_seqlens.zero_()

    def get_pos(self) -> int:
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx: int):
        return self.w_cache[layer_idx], self.w_cache[layer_idx]

    def advance(self, num_tokens: int):
        self.cache_seqlens += num_tokens


class NoQCRATE(nn.Module):
    """Full No-Q CRATE-α model — drop-in replacement for CRATE."""

    def __init__(self, config: NoQCRATEConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config
        self.max_seq_len = config.sequence_len

        self.window_sizes = self._compute_window_sizes(config)

        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)

        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _compute_window_sizes(self, config: NoQCRATEConfig):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: float = 10000.0, device=None):
        if device is None:
            device = self.transformer.wte.weight.device if hasattr(self, 'transformer') else 'cpu'
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def init_weights(self):
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        for block in self.transformer.h:
            # NoQMSSA: only kv + c_proj (no qkv)
            torch.nn.init.uniform_(block.mssa.kv.weight, -s, s)
            torch.nn.init.zeros_(block.mssa.c_proj.weight)

            if hasattr(block, "odl"):
                torch.nn.init.kaiming_uniform_(block.odl.D_enc.weight, a=5**0.5)
                torch.nn.init.zeros_(block.odl.D_dec.weight)
                torch.nn.init.zeros_(block.odl.threshold)
            else:
                torch.nn.init.kaiming_uniform_(block.ista.weight, a=5**0.5)

        with torch.no_grad():
            self.resid_lambdas.fill_(1.0)
            self.x0_lambdas.fill_(0.0)

        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = self.transformer.wte.weight.numel() + self.resid_lambdas.numel() + self.x0_lambdas.numel()
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = sum(12 * h * q * min(ws[0], t) for ws in self.window_sizes)
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        return sum(p.numel() for p in self.parameters())

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        all_block_params = list(self.transformer.h.parameters())
        matrix_params = [p for p in all_block_params if p.ndim >= 2]
        vector_params = [p for p in all_block_params if p.ndim < 2]

        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]

        if vector_params:
            print0(f"Found {len(vector_params)} 1D parameters in blocks (e.g. ODL thresholds), routing to AdamW")

        assert len(list(self.parameters())) == len(matrix_params) + len(vector_params) + len(embedding_params) + len(lm_head_params) + len(resid_params) + len(x0_params)

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=resid_params, lr=scalar_lr * 0.01),
            dict(params=x0_params, lr=scalar_lr),
        ]
        if vector_params:
            adam_groups.append(dict(params=vector_params, lr=scalar_lr))
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0)

        if NANOCHAT_AVAILABLE:
            AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
            MuonFactory = DistMuon if ddp else Muon
        else:
            AdamWFactory = partial(torch.optim.AdamW, fused=torch.cuda.is_available())
            MuonFactory = partial(torch.optim.AdamW, fused=torch.cuda.is_available())
            print0("Warning: Muon optimizer not available, falling back to AdamW for matrix params")

        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay) if NANOCHAT_AVAILABLE else dict(lr=matrix_lr, weight_decay=weight_decay)
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)

        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

        return optimizers

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                kv_cache=None, loss_reduction: str = 'mean',
                return_hidden_at: Optional[Union[int, List[int]]] = None,
                embedding_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = idx.size()

        assert T <= self.cos.size(1), f"Sequence length grew beyond rotary cache: {T} > {self.cos.size(1)}"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = (self.cos[:, T0:T0+T], self.sin[:, T0:T0+T])

        x = self.transformer.wte(idx)
        x = norm(x)
        if embedding_bias is not None:
            x = x + embedding_bias
        x0 = x

        if return_hidden_at is not None:
            multi_layer = isinstance(return_hidden_at, (list, tuple))
            target_layers = set(return_hidden_at) if multi_layer else {return_hidden_at}
            hidden_snapshots: Dict[int, torch.Tensor] = {}
        else:
            multi_layer = False
            target_layers = set()
            hidden_snapshots = {}

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, cos_sin, self.window_sizes[i], kv_cache)
            if i in target_layers:
                hidden_snapshots[i] = x

        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            output = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction
            )
        else:
            output = logits

        if return_hidden_at is not None:
            if multi_layer:
                return output, hidden_snapshots
            else:
                return output, hidden_snapshots.get(return_hidden_at)
        return output

    @torch.inference_mode()
    def generate(self, tokens: list, max_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None, seed: int = 42):
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        cache = KVCache(
            batch_size=1,
            num_heads=self.config.n_head,
            seq_len=len(tokens) + max_tokens,
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
                logits[logits < v[:, [-1]]] = float('-inf')

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
