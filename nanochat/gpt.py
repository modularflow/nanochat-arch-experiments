"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
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

# Attention backend selection: FA3 on Hopper (SM 90+), SDPA elsewhere (e.g. 4090 / Ada SM 89)
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
    print(f"[gpt.py] Flash Attention 3 not available; using PyTorch SDPA fallback.")

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "L"
    # Two-lane "parallel residual" forward (GPT-J-style with a learned 2x2 lane mixer).
    # Attention reads lane-A, MLP reads lane-B; outputs are mixed back via a 2x2 matrix.
    parallel_residual: bool = False


class KVCache:
    """KV Cache for efficient autoregressive inference with separate K/V storage."""

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
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
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
        """SDPA fallback for non-Hopper GPUs (e.g. RTX 4090 Ada Lovelace)."""
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

        # GQA: repeat k/v heads to match q heads
        if self.n_kv_head < self.n_head:
            rep = self.n_head // self.n_kv_head
            k_full = k_full.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, k_full.size(1), self.n_head, self.head_dim)
            v_full = v_full.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, v_full.size(1), self.n_head, self.head_dim)

        # SDPA expects (B, H, T, D)
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k_full.transpose(1, 2)
        v_sdpa = v_full.transpose(1, 2)

        is_causal = (kv_cache is None) and (T > 1)

        # Sliding window via additive attention mask
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
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def attn_out(self, x, cos_sin, window_size, kv_cache):
        return self.attn(norm(x), cos_sin, window_size, kv_cache)

    def mlp_out(self, x):
        return self.mlp(norm(x))

    def forward(self, x, cos_sin, window_size, kv_cache):
        x = x + self.attn_out(x, cos_sin, window_size, kv_cache)
        x = x + self.mlp_out(x)
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        self.max_seq_len = config.sequence_len
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Parallel residual: 2-lane forward with a learned 2x2 lane mixer per block.
        # post_lambdas init = identity ⇒ at init each lane carries its own attn/mlp output back.
        # parallel_resid_lambdas init = 1 ⇒ no scaling on the lanes themselves.
        self.parallel_residual = bool(getattr(config, "parallel_residual", False))
        if self.parallel_residual:
            # (n_layer, 2, 2) — mixes (attn_out, mlp_out) into (lane_a, lane_b)
            self.parallel_post_lambdas = nn.Parameter(torch.zeros(config.n_layer, 2, 2))
            # (n_layer, 2) — per-lane scaling on the *input* lane before attn/mlp reads it
            self.parallel_resid_lambdas = nn.Parameter(torch.ones(config.n_layer, 2))
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        with torch.no_grad():
            self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
            self.x0_lambdas.fill_(0.0)      # 0.0 => skip connection to input is disabled at init

        if self.parallel_residual:
            with torch.no_grad():
                # post_lambdas = identity 2x2 ⇒ lane A receives attn_out, lane B receives mlp_out
                self.parallel_post_lambdas.zero_()
                self.parallel_post_lambdas[:, 0, 0] = 1.0
                self.parallel_post_lambdas[:, 1, 1] = 1.0
                self.parallel_resid_lambdas.fill_(1.0)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast token embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars.
        nparams_exclude = self.transformer.wte.weight.numel() + self.resid_lambdas.numel() + self.x0_lambdas.numel()
        if getattr(self, "parallel_residual", False):
            nparams_exclude += self.parallel_post_lambdas.numel() + self.parallel_resid_lambdas.numel()
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return all of the parameters, same as Chinchilla paper.
        Kaplan et al. did not include embedding parameters and said that this led to cleaner scaling laws.
        But Kaplan et al. also had a bug in their results (as pointed out by Chinchilla).
        My own experiments in nanochat confirm the Chinchilla approach gives the much cleaner scaling law.
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper <- good).
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper <- bad)
        """
        nparams = sum(p.numel() for p in self.parameters())
        return nparams

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5, muon_mode="default"):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into groups.
        # Muon is for 2D matrix weights inside transformer blocks (Linear weights).
        # AdamW handles embeddings, lm_head, and all 1D / small scalar params (lambdas, scales, gains).
        block_params = list(self.transformer.h.parameters())
        matrix_params = [p for p in block_params if p.ndim == 2]
        small_block_params = [p for p in block_params if p.ndim != 2]
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        # Parallel-residual trunk-level small params go to AdamW with the small_block group.
        if self.parallel_residual:
            small_block_params.append(self.parallel_post_lambdas)
            small_block_params.append(self.parallel_resid_lambdas)
        accounted = (
            len(matrix_params) + len(small_block_params)
            + len(embedding_params) + len(lm_head_params)
            + len(resid_params) + len(x0_params)
        )
        assert accounted == len(list(self.parameters())), \
            f"Param routing accounted={accounted} vs total={len(list(self.parameters()))}"
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=resid_params, lr=scalar_lr * 0.01), # these are a lot more sensitive because they accumulate in the residual stream
            dict(params=x0_params, lr=scalar_lr),
        ]
        if small_block_params:
            # Per-block scales / gains / mix vectors. Use the same conservative LR as resid_lambdas.
            adam_groups.append(dict(params=small_block_params, lr=scalar_lr * 0.01))
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0) # NOTE: weight decay is hardcoded to 0.0 for AdamW, only used in Muon
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay, mode=muon_mode)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
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
        """
        Args:
            return_hidden_at: Layer index (or list) at which to snapshot pre-head hidden states.
            embedding_bias: Optional [B, T, d] added after embedding RMSNorm (Self-Flow conditioning).
        """
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        if embedding_bias is not None:
            x = x + embedding_bias
        x0 = x  # save initial normalized embedding for x0 residual

        if return_hidden_at is not None:
            multi_layer = isinstance(return_hidden_at, (list, tuple))
            target_layers = set(return_hidden_at) if multi_layer else {return_hidden_at}
            hidden_snapshots: Dict[int, torch.Tensor] = {}
        else:
            multi_layer = False
            target_layers = set()
            hidden_snapshots = {}

        if self.parallel_residual:
            # Two-lane GPT-J-style residual.
            # Lane A feeds attention; lane B feeds MLP. After each block the two outputs are
            # remixed via a learned 2x2 (init=identity ⇒ each lane just gets its own contribution).
            x_a = x
            x_b = x
            for i, block in enumerate(self.transformer.h):
                # Pre-block mix of (lane, x0) via the scalar resid/x0 lambdas applied symmetrically to both lanes.
                x_a = self.resid_lambdas[i] * x_a + self.x0_lambdas[i] * x0
                x_b = self.resid_lambdas[i] * x_b + self.x0_lambdas[i] * x0
                # Per-lane scale on the *input* lane (init 1.0).
                lam = self.parallel_resid_lambdas[i].to(x_a.dtype)
                x_a = x_a * lam[0]
                x_b = x_b * lam[1]
                # Compute attn from lane A and MLP from lane B.
                attn_out = block.attn_out(x_a, cos_sin, self.window_sizes[i], kv_cache)
                mlp_out = block.mlp_out(x_b)
                # Remix outputs into the two lanes via the learned 2x2.
                P = self.parallel_post_lambdas[i].to(x_a.dtype)  # (2, 2)
                x_a_new = x_a + P[0, 0] * attn_out + P[0, 1] * mlp_out
                x_b_new = x_b + P[1, 0] * attn_out + P[1, 1] * mlp_out
                x_a, x_b = x_a_new, x_b_new
                if i in target_layers:
                    # For the JEPA / hidden-snapshot path, we need a single representation.
                    # Average the two lanes — symmetric and parameter-free.
                    hidden_snapshots[i] = 0.5 * (x_a + x_b)
            # Collapse the two lanes back into a single stream for the head.
            x = 0.5 * (x_a + x_b)
        else:
            for i, block in enumerate(self.transformer.h):
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                x = block(x, cos_sin, self.window_sizes[i], kv_cache)
                if i in target_layers:
                    hidden_snapshots[i] = x

        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

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
        """
        Autoregressive streaming inference with KV cache for O(n) generation.
        Batch size is 1; ids and yielded tokens are simple Python lists and ints.
        """
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
