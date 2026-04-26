"""
SVD-GPT: globally-shared *asymmetric* SVD attention transformer.

Distinguishing design (compared to the other nanochat architectures):

  - Single global `Uq`, `Uk`, `Uv` matrices of shape `(rank, n_embd)`, owned
    by the model and SHARED across ALL transformer blocks (asymmetric 3-way
    SVD — Uq ≠ Uk ≠ Uv). Every block's attention reads its query / key /
    value projections from these three global tensors.
  - Single-headed attention (n_head=1, head_dim=rank). With rank=64 this
    is a natural fit for Flash-Attention-3 on SM90+ and for PyTorch SDPA
    on SM89/earlier.
  - SwiGLU FFN with d_ff = round(8 * n_embd / 3 / 8) * 8 (auto-derived).
  - RMSNorm *with* a learnable weight (unlike the other nanochat GPT
    families, which use a parameter-free rmsnorm via `F.rms_norm`). This
    preserves the design described in the SVD spec.
  - RoPE on the rank-dim Q/K projections only; V gets no RoPE.
  - No biases anywhere (attention, ffn, lm_head).

Recipe-parity choices (to fit the existing apples-to-apples sweep):

  - Same optimizer routing as `gpt.py` / `tpa_gpt.py`: 2D matrix params →
    Muon, 1D / small params and embedding / lm_head → AdamW. Crucially,
    the globally-shared `Uq` / `Uk` / `Uv` are ALSO routed to Muon (they
    are 2D `(rank, n_embd)` — shape-compatible with Muon's NS iteration).
  - Same `resid_lambdas` / `x0_lambdas` per-layer scalars the rest of the
    family uses (init 1.0 / 0.0 respectively). This lets the SVD model
    drop into the shared scaler-LR group without bespoke scheduling.
  - Same logit softcap = 15, same `F.cross_entropy(..., ignore_index=-1)`
    loss signature, same bf16 embedding cast, same padded-vocab rule.
  - Same rotary-embedding buffer layout `(1, T, 1, head_dim/2)` and
    `apply_rotary_emb` convention used by `gpt.py`.
  - Same sliding-window attention mechanics as the other arches (FA3
    `window_size` in the FA3 path; additive (causal ∧ window) mask in the
    SDPA fallback).

Checkpoint detection: the model exposes both a unique top-level parameter
name (`Uq`) and a unique buffer (`svd_marker`). The checkpoint_manager
dispatches on the buffer to load the right class.

Default "full-size" config (~375M params at the recipe shown below):
    vocab=50257, sequence_len=2048, n_embd=1024, n_layer=32, rank=64
"""

from functools import partial
from dataclasses import dataclass

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
    print(f"[svd_gpt.py] Flash Attention 3 not available; using PyTorch SDPA fallback.")


@dataclass
class SVDGPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 50257
    n_layer: int = 32
    # n_head / n_kv_head are accepted for CLI-compat with the rest of the family,
    # but SVD attention is single-headed with head_dim == rank. They are not
    # used internally and are not asserted against rank (keeps the CLI uniform).
    n_head: int = 1
    n_kv_head: int = 1
    n_embd: int = 1024
    window_pattern: str = "L"
    # SVD-specific knobs ------------------------------------------------------
    rank: int = 64
    # FFN hidden size. 0 = auto: round(8 * n_embd / 3 / 8) * 8 (SwiGLU parity).
    d_ff: int = 0


class KVCache:
    """KV Cache for efficient autoregressive inference with separate K/V storage.

    Mirrors `gpt.py`'s cache shape `(num_layers, batch, seq, n_kv_head, head_dim)`
    so the same `Engine` code path works. For SVD, n_kv_head == 1 and
    head_dim == rank.
    """

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


def apply_rotary_emb(x, cos, sin):
    """Same rotation convention as `gpt.py` — operates on a 4-D tensor of
    shape (B, T, H, head_dim) with H == 1 (single-headed)."""
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class RMSNorm(nn.Module):
    """RMSNorm with a learnable weight (no bias). Matches the SVD spec."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for numerical stability, then cast back.
        orig_dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(orig_dtype) * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward block: w2( silu(w3 x) * (w1 x) )."""

    def __init__(self, n_embd: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(n_embd, d_ff, bias=False)
        self.w3 = nn.Linear(n_embd, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, n_embd, bias=False)

    def forward(self, x):
        return self.w2(self.w1(x) * F.silu(self.w3(x)))


class SVDBlock(nn.Module):
    """Transformer block with globally-shared asymmetric SVD attention.

    The three SVD bases `Uq`, `Uk`, `Uv` are passed in from the parent model
    at call time (not owned here) so that they remain *shared* across all
    blocks.
    """

    def __init__(self, config, layer_idx: int, d_ff: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_embd = config.n_embd
        self.rank = config.rank

        self.ln_1 = RMSNorm(config.n_embd)
        self.out_proj = nn.Linear(config.rank, config.n_embd, bias=False)
        self.ln_2 = RMSNorm(config.n_embd)
        self.ffn = SwiGLUFFN(config.n_embd, d_ff)

    def forward(self, x, Uq, Uk, Uv, cos_sin, window_size, kv_cache):
        B, T, _ = x.size()
        cos, sin = cos_sin

        normed = self.ln_1(x)

        # Rank-dim projections (shared across layers). Shape (B, T, rank).
        hq = F.linear(normed, Uq)
        hk = F.linear(normed, Uk)
        hv = F.linear(normed, Uv)

        # Promote to (B, T, 1, rank) for FA3 / SDPA. head_dim = rank.
        q = hq.view(B, T, 1, self.rank)
        k = hk.view(B, T, 1, self.rank)
        v = hv.view(B, T, 1, self.rank)

        # RoPE on Q and K only (V gets no positional rotation — SVD spec).
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if _USE_FA3:
            y = self._forward_fa3(q, k, v, window_size, kv_cache, B, T)
        else:
            y = self._forward_sdpa(q, k, v, window_size, kv_cache, B, T)

        y = y.contiguous().view(B, T, self.rank)
        x = x + self.out_proj(y)
        x = x + self.ffn(self.ln_2(x))
        return x

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


class SVDGPT(nn.Module):
    """GPT with globally-shared asymmetric SVD attention.

    Same public interface as `GPT` / `TPAGPT`:
      - `init_weights()` fills all parameters (model is first built on meta
        device, then `to_empty(device=...)` + `init_weights()`).
      - `forward(idx, targets=None, kv_cache=None, loss_reduction='mean')`
        returns either the loss (when targets is given) or the logits.
      - `setup_optimizers(...)` returns `[adamw, muon]`.
      - `estimate_flops()`, `num_scaling_params()`, `get_device()`.
      - `generate(tokens, ...)` for single-batch streaming inference.
      - `forward_to_final_hidden(idx)` for the JEPA aux-loss path.
    """

    def __init__(self, config: SVDGPTConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config
        self.max_seq_len = config.sequence_len

        d = config.n_embd
        r = config.rank
        assert r > 0 and r % 2 == 0, f"rank must be a positive even integer, got {r}"

        # FFN hidden size (SwiGLU parity formula, rounded to a multiple of 8).
        d_ff = config.d_ff if config.d_ff > 0 else (round(8 * d / 3 / 8) * 8)
        self.d_ff = d_ff

        # Sliding-window attention (same mechanism as gpt.py).
        self.window_sizes = self._compute_window_sizes(config)

        # Padded vocab for DDP / tensor-core friendliness.
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, d),
            "h": nn.ModuleList([
                SVDBlock(config, layer_idx=i, d_ff=d_ff) for i in range(config.n_layer)
            ]),
        })
        self.ln_f = RMSNorm(d)
        self.lm_head = nn.Linear(d, padded_vocab_size, bias=False)

        # Globally shared asymmetric SVD bases (rank × n_embd each).
        self.Uq = nn.Parameter(torch.empty(r, d))
        self.Uk = nn.Parameter(torch.empty(r, d))
        self.Uv = nn.Parameter(torch.empty(r, d))

        # Per-layer scalars — recipe-parity with the rest of the family.
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        # Marker buffer used by checkpoint_manager for architecture auto-detection.
        self.register_buffer('svd_marker', torch.tensor([r, config.n_layer, d], dtype=torch.long))

        # Rotary embeddings buffer: same layout as gpt.py — (1, T, 1, head_dim/2).
        # head_dim for SVD is `rank`.
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, r)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    # -----------------------------------------------------------------------
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.config.n_embd
        s = 3 ** 0.5 * n_embd ** -0.5

        # SVD bases: use uniform init with the standard 1/sqrt(n_embd) std so
        # the initial Q/K/V projection statistics match nanochat's linear init.
        torch.nn.init.uniform_(self.Uq, -s, s)
        torch.nn.init.uniform_(self.Uk, -s, s)
        torch.nn.init.uniform_(self.Uv, -s, s)

        for block in self.transformer.h:
            torch.nn.init.ones_(block.ln_1.weight)
            torch.nn.init.ones_(block.ln_2.weight)
            # FFN: SwiGLU with uniform init on the gate / value path and zero
            # init on the output projection so blocks act as an identity at t=0.
            torch.nn.init.uniform_(block.ffn.w1.weight, -s, s)
            torch.nn.init.uniform_(block.ffn.w3.weight, -s, s)
            torch.nn.init.zeros_(block.ffn.w2.weight)
            # Attention output projection: zero-init → residual identity at t=0.
            torch.nn.init.zeros_(block.out_proj.weight)

        torch.nn.init.ones_(self.ln_f.weight)

        with torch.no_grad():
            self.resid_lambdas.fill_(1.0)
            self.x0_lambdas.fill_(0.0)

        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.config.rank)
        self.cos, self.sin = cos, sin

        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    # -----------------------------------------------------------------------
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        """Same (1, T, 1, head_dim/2) layout as gpt.py so apply_rotary_emb
        can broadcast identically over the (single) head dim."""
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
        # Final layer always gets full context (matches gpt.py).
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    # -----------------------------------------------------------------------
    def estimate_flops(self):
        """FLOPs per token (forward + backward), same accounting as gpt.py:
          - 6× per matmul weight parameter (this naturally covers the shared
            `Uq`, `Uk`, `Uv` since they're in `self.parameters()`).
          - Plus 12·h·q·effective_seq for the attention dot product, where
            (h, q) = (1, rank) for single-headed SVD.
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Embeddings and per-layer scalars are not matmul ops; exclude from the
        # 6× matmul accounting (and from the lm_head? the lm_head IS a matmul
        # — we keep it in, same as gpt.py).
        nparams_exclude = (
            self.transformer.wte.weight.numel()
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
        )
        # Also exclude RMSNorm weights (they're pointwise scalings, negligible
        # in matmul terms) for cleanliness; matches the spirit of gpt.py which
        # has no learnable norm params at all.
        for block in self.transformer.h:
            nparams_exclude += block.ln_1.weight.numel() + block.ln_2.weight.numel()
        nparams_exclude += self.ln_f.weight.numel()

        h, q, t = 1, self.config.rank, self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        return sum(p.numel() for p in self.parameters())

    # -----------------------------------------------------------------------
    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                         weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5,
                         muon_mode="default"):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Matrix params: all 2D weights inside each block (out_proj, FFN w1/w2/w3)
        # PLUS the globally-shared `Uq` / `Uk` / `Uv` — these live at the top
        # level but are the 2D projection matrices the whole stack reads from,
        # so Muon is the right optimizer for them.
        block_matrix_params = [p for p in self.transformer.h.parameters() if p.ndim == 2]
        shared_matrix_params = [self.Uq, self.Uk, self.Uv]
        matrix_params = block_matrix_params + shared_matrix_params

        # 1D / small params inside blocks: the RMSNorm `weight` vectors.
        small_block_params = [p for p in self.transformer.h.parameters() if p.ndim != 2]
        # Final-norm weight (top-level, 1D) — goes in the same small group.
        small_block_params.append(self.ln_f.weight)

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
            # RMSNorm gains etc. — same conservative LR as resid_lambdas.
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

    # -----------------------------------------------------------------------
    def forward_to_final_hidden(self, idx):
        """Forward returning the final hidden state (post `ln_f`, pre `lm_head`).

        Used by the JEPA aux-loss path. Kept structurally identical to
        `forward` so the representation used for prediction matches the
        representation used for CE.
        """
        B, T = idx.size()
        assert T <= self.cos.size(1), f"Sequence length grew beyond rotary cache: {T} > {self.cos.size(1)}"
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, self.Uq, self.Uk, self.Uv, cos_sin, self.window_sizes[i], None)
        x = self.ln_f(x)
        return x

    # -----------------------------------------------------------------------
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1), f"Sequence length grew beyond rotary cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device
        assert self.cos.dtype == torch.bfloat16

        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        x = self.transformer.wte(idx)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, self.Uq, self.Uk, self.Uv, cos_sin, self.window_sizes[i], kv_cache)
        x = self.ln_f(x)

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
            return loss
        else:
            return logits

    # -----------------------------------------------------------------------
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
            n_kv_head=1,                  # SVD is single-headed
            head_dim=self.config.rank,    # head_dim == rank
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
