"""
TPA-GPT: GPT with Tensor Product Attention.

Replaces standard MHA's three projections (W_Q, W_K, W_V) with contextual
rank-R tensor-product factorizations (Zhang et al., NeurIPS 2025,
"Tensor Product Attention Is All You Need", arXiv:2501.06425).

For each token x_t and each side ∈ {Q, K, V}:

    A_side(x_t) ∈ R^{R_side × n_head}      = reshape(W_a_side @ x_t)
    B_side(x_t) ∈ R^{R_side × head_dim}    = reshape(W_b_side @ x_t)
    side_t      = (1/R_side) · A_side(x_t)^T @ B_side(x_t)   ∈ R^{n_head × head_dim}

The output Q_t, K_t, V_t have the same shape as MHA's, so the rest of the
attention block (RoPE, QK-norm, SDPA / FA3, sliding window, output projection)
is structurally unchanged — we just compute Q/K/V via this rank-R factorization
instead of a single linear projection.

Theorem 3.1 of the TPA paper: applying RoPE to the B_Q and B_K factors (the
head_dim dimension) before the einsum is equivalent to applying RoPE to Q/K
after the einsum, so RoPE composes natively with the factorization.

This file is a DROP-IN sibling of `nanochat/gpt.py` — it reuses GPT's Block,
KV cache, sliding-window machinery, EMA, and PG primitives compatibility
hooks where possible. The only structurally different piece is the attention
module.

KV cache: v1 uses the standard (full K, V) cache from gpt.py. The TPA paper's
"FlashTPA decoding" with factorized KV cache is a v2 inference optimization;
for training and short-context generation it adds complexity without changing
quality.
"""

from functools import partial
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
_USE_FA3 = False
flash_attn = None
if torch.cuda.is_available():
    _major, _minor = torch.cuda.get_device_capability()
    if _major >= 9:
        try:
            from kernels import get_kernel
            flash_attn = get_kernel('varunneal/flash-attention-3').flash_attn_interface
            _USE_FA3 = True
        except Exception:
            pass
if not _USE_FA3:
    print(f"[tpa_gpt.py] Flash Attention 3 not available; using PyTorch SDPA fallback.")


@dataclass
class TPAGPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6                 # number of attention heads (also == n_kv_head; TPA does not use GQA)
    n_kv_head: int = 6              # kept for checkpoint_manager compatibility — must equal n_head for TPA
    n_embd: int = 768
    window_pattern: str = "L"
    # TPA ranks (paper defaults for the "T6" model: R_Q == n_head for full Q expressiveness, R_K == R_V == 2 for KV compression).
    tpa_rank_q: int = 6
    tpa_rank_k: int = 2
    tpa_rank_v: int = 2


class KVCache:
    """Standard K/V cache (same shape as gpt.py). v1 keeps it simple — TPA's
    factorized cache is an inference-time optimization we can wire in later."""

    def __init__(self, batch_size, seq_len, n_kv_head, head_dim, num_layers, device, dtype=torch.bfloat16):
        self.n_layers = num_layers
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, n_kv_head, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, n_kv_head, head_dim, device=device, dtype=dtype)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    def get_pos(self):
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        self.cache_seqlens += num_tokens


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    """RoPE applied to the last dim of a (..., seq, head_or_rank, head_dim) tensor.
    For TPA we call this on B_Q / B_K which have shape (B, T, R, head_dim) — same
    rank as standard (B, T, n_head, head_dim) so the cos/sin broadcast unchanged."""
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class TPACausalSelfAttention(nn.Module):
    """Causal self-attention where Q, K, V are produced by per-token rank-R
    tensor-product factorizations of the hidden state."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        # TPA does not use grouped-query attention — the rank factorization plays
        # the role that K/V head sharing plays in GQA. We assert this loudly so
        # CLI misuse fails fast rather than silently doing the wrong thing.
        assert self.n_kv_head == self.n_head, (
            f"TPA requires n_kv_head == n_head (got n_kv_head={self.n_kv_head}, n_head={self.n_head}). "
            "Use --num-kv-heads == --num-heads with --architecture tpa_gpt; for KV compression "
            "use the tpa_rank_k / tpa_rank_v knobs instead."
        )
        self.R_Q = int(config.tpa_rank_q)
        self.R_K = int(config.tpa_rank_k)
        self.R_V = int(config.tpa_rank_v)
        assert self.R_Q >= 1 and self.R_K >= 1 and self.R_V >= 1, (
            f"TPA ranks must be >= 1 (got R_Q={self.R_Q}, R_K={self.R_K}, R_V={self.R_V})"
        )
        # Per-side projections. We use one Linear per A-side and one per B-side
        # (paper Eq 3.2). Output is reshaped into (R_side, n_head) and (R_side, head_dim)
        # immediately after the projection.
        self.W_aQ = nn.Linear(self.n_embd, self.R_Q * self.n_head, bias=False)
        self.W_bQ = nn.Linear(self.n_embd, self.R_Q * self.head_dim, bias=False)
        self.W_aK = nn.Linear(self.n_embd, self.R_K * self.n_head, bias=False)
        self.W_bK = nn.Linear(self.n_embd, self.R_K * self.head_dim, bias=False)
        self.W_aV = nn.Linear(self.n_embd, self.R_V * self.n_head, bias=False)
        self.W_bV = nn.Linear(self.n_embd, self.R_V * self.head_dim, bias=False)
        # Output projection — same shape as MHA's c_proj.
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def _materialize_qkv(self, x, cos, sin):
        """Return (q, k, v) with shape (B, T, n_head, head_dim) each.

        RoPE is applied to the B_Q and B_K factors (head_dim dimension) before
        contraction — Theorem 3.1 of the TPA paper proves this is equivalent to
        applying RoPE to the materialized Q / K and preserves RoPE's
        translation invariance.
        """
        B, T, _ = x.size()
        # A factors: (B, T, R, n_head)
        a_q = self.W_aQ(x).view(B, T, self.R_Q, self.n_head)
        a_k = self.W_aK(x).view(B, T, self.R_K, self.n_head)
        a_v = self.W_aV(x).view(B, T, self.R_V, self.n_head)
        # B factors: (B, T, R, head_dim)
        b_q = self.W_bQ(x).view(B, T, self.R_Q, self.head_dim)
        b_k = self.W_bK(x).view(B, T, self.R_K, self.head_dim)
        b_v = self.W_bV(x).view(B, T, self.R_V, self.head_dim)
        # RoPE on the B factors of Q and K only. cos/sin have shape (1, T, 1, head_dim/2)
        # and broadcast over the R dim just like they normally broadcast over n_head.
        b_q = apply_rotary_emb(b_q, cos, sin)
        b_k = apply_rotary_emb(b_k, cos, sin)
        # Contract: Q = (A^T @ B) / R_Q  →  (B, T, n_head, head_dim)
        # A.transpose(-1,-2): (B, T, n_head, R)  ;  B: (B, T, R, head_dim)
        q = torch.matmul(a_q.transpose(-1, -2), b_q) / self.R_Q
        k = torch.matmul(a_k.transpose(-1, -2), b_k) / self.R_K
        v = torch.matmul(a_v.transpose(-1, -2), b_v) / self.R_V
        return q, k, v

    def forward(self, x, cos_sin, window_size, kv_cache):
        B, T, _ = x.size()
        cos, sin = cos_sin
        q, k, v = self._materialize_qkv(x, cos, sin)
        # QK-norm (matches gpt.py / noq_gpt.py): RMSNorm on the head_dim of Q and K.
        q, k = norm(q), norm(k)

        if _USE_FA3:
            y = self._forward_fa3(q, k, v, window_size, kv_cache, B, T)
        else:
            y = self._forward_sdpa(q, k, v, window_size, kv_cache, B, T)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

    def _forward_fa3(self, q, k, v, window_size, kv_cache, B, T):
        if kv_cache is None:
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
        y = flash_attn.flash_attn_with_kvcache(
            q, k_cache, v_cache,
            k=k, v=v,
            cache_seqlens=kv_cache.cache_seqlens,
            causal=True,
            window_size=window_size,
        )
        if self.layer_idx == kv_cache.n_layers - 1:
            kv_cache.advance(T)
        return y

    def _forward_sdpa(self, q, k, v, window_size, kv_cache, B, T):
        if kv_cache is not None:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            pos = kv_cache.cache_seqlens[0].item()
            k_cache[:, pos:pos + T] = k
            v_cache[:, pos:pos + T] = v
            k_full = k_cache[:, :pos + T]
            v_full = v_cache[:, :pos + T]
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)
        else:
            k_full = k
            v_full = v

        # SDPA expects (B, H, T, D)
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k_full.transpose(1, 2)
        v_sdpa = v_full.transpose(1, 2)

        is_causal = (kv_cache is None) and (T > 1)

        attn_mask = None
        if window_size is not None and window_size[0] > 0 and kv_cache is None:
            left = window_size[0]
            S = k_sdpa.size(-2)
            row_idx = torch.arange(T, device=q.device).unsqueeze(1)
            col_idx = torch.arange(S, device=q.device).unsqueeze(0)
            causal_mask = col_idx <= row_idx
            window_mask = (row_idx - col_idx) <= left
            attn_mask = causal_mask & window_mask
            is_causal = False

        y = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )
        return y.transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = TPACausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def attn_out(self, x, cos_sin, window_size, kv_cache):
        return self.attn(norm(x), cos_sin, window_size, kv_cache)

    def mlp_out(self, x):
        return self.mlp(norm(x))

    def forward(self, x, cos_sin, window_size, kv_cache):
        x = x + self.attn_out(x, cos_sin, window_size, kv_cache)
        x = x + self.mlp_out(x)
        return x


class TPAGPT(nn.Module):
    """GPT with Tensor Product Attention — drop-in replacement for nanochat's GPT."""

    def __init__(self, config, pad_vocab_size_to=64):
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

    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            # TPA's six projection matrices — initialize all six like nanochat's
            # other Linear weights (uniform, std == 1/sqrt(n_embd)). The 1/R
            # scaling inside the einsum keeps the materialized Q/K/V variance
            # comparable to MHA.
            torch.nn.init.uniform_(block.attn.W_aQ.weight, -s, s)
            torch.nn.init.uniform_(block.attn.W_bQ.weight, -s, s)
            torch.nn.init.uniform_(block.attn.W_aK.weight, -s, s)
            torch.nn.init.uniform_(block.attn.W_bK.weight, -s, s)
            torch.nn.init.uniform_(block.attn.W_aV.weight, -s, s)
            torch.nn.init.uniform_(block.attn.W_bV.weight, -s, s)
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
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
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
        """FLOPs per token (forward + backward).

        We follow nanochat's accounting: 6× per matmul weight parameter for
        forward+backward, plus 12·h·q·effective_seq for the attention dot
        product (which depends on materialized Q/K/V shape, NOT rank, so this
        term is unchanged from MHA).

        The rank factorization affects only the projection params, which are
        already smaller than MHA's W_Q/W_K/W_V — that smaller count is naturally
        captured by `sum(p.numel() for p in self.parameters())`.
        """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = (
            self.transformer.wte.weight.numel()
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
        )
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        return sum(p.numel() for p in self.parameters())

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0,
                         adam_betas=(0.8, 0.95), scalar_lr=0.5, muon_mode="default"):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Same routing as gpt.py: 2D matrix params (Linear weights) → Muon;
        # everything else (embeddings, lm_head, scalars) → AdamW.
        block_params = list(self.transformer.h.parameters())
        matrix_params = [p for p in block_params if p.ndim == 2]
        small_block_params = [p for p in block_params if p.ndim != 2]
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        accounted = (
            len(matrix_params) + len(small_block_params)
            + len(embedding_params) + len(lm_head_params)
            + len(resid_params) + len(x0_params)
        )
        assert accounted == len(list(self.parameters())), \
            f"Param routing accounted={accounted} vs total={len(list(self.parameters()))}"
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=resid_params, lr=scalar_lr * 0.01),
            dict(params=x0_params, lr=scalar_lr),
        ]
        if small_block_params:
            adam_groups.append(dict(params=small_block_params, lr=scalar_lr * 0.01))
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay, mode=muon_mode)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(
        self,
        idx,
        targets=None,
        kv_cache=None,
        loss_reduction="mean",
        return_hidden_at: Optional[Union[int, List[int]]] = None,
        embedding_bias: Optional[torch.Tensor] = None,
    ):
        B, T = idx.size()
        assert T <= self.cos.size(1), f"Sequence length grew beyond rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device
        assert self.cos.dtype == torch.bfloat16
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

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
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            output = loss
        else:
            output = logits

        if return_hidden_at is not None:
            if multi_layer:
                return output, hidden_snapshots
            return output, hidden_snapshots.get(return_hidden_at)
        return output

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
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
