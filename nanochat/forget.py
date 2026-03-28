"""
Adversarial forgetting mechanisms for context efficiency.

Three strategies that force the model to selectively forget unimportant
information, producing compressed recall-focused representations:

1. forget_gate       - xLSTM-inspired exponential per-feature gating
2. context_bottleneck - Dimensionality bottleneck between layers
3. selective_retention - Per-token importance scoring and soft masking

All three share the same adversarial principle: a sparsity/forgetting
penalty encourages maximum information destruction, while LM + rep losses
push back to preserve recall quality.  The equilibrium produces efficient
context utilization.

Integration: these modules sit BETWEEN transformer layers.  The model's
forward pass calls gate.apply(hidden, layer_idx) after each block, and
gate.compute_forget_loss() returns the adversarial sparsity penalty.

Reference:
    xLSTM (Beck et al., NeurIPS 2024) for exponential gating
    Mamba (Gu & Dao, 2024) for selective state forgetting
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Set, Tuple


# =============================================================================
# 1. ForgetGate (xLSTM-inspired exponential gating)
# =============================================================================

class ForgetGate(nn.Module):
    """
    Per-feature forget gate applied between transformer layers.

    For each token position and feature dimension, produces a gate value
    in (0, 1) that multiplicatively scales the hidden state.  Features
    gated toward 0 are "forgotten".

    Uses exponential gating (xLSTM/mLSTM-style) for sharper on/off
    decisions than sigmoid:

        raw  = W @ h + b
        gate = exp(raw) / (1 + exp(raw))     (= sigmoid, but computed via exp)

    A learned per-feature bias initialised to +3 so the gate starts open
    (minimal forgetting), then the sparsity loss gradually pushes it closed.

    Adversarial sparsity loss:  L = -beta * mean(log(gate + eps))
    This encourages gate -> 0 (maximum forgetting).  The model's LM loss
    provides the counter-pressure to keep recall-critical features open.
    """

    def __init__(self, n_embd: int, n_layers: int, active_layers: Optional[Set[int]] = None):
        super().__init__()
        self.n_embd = n_embd
        self.n_layers = n_layers
        self.active_layers = active_layers  # None = every layer

        num_gates = n_layers if active_layers is None else len(active_layers)
        self.gate_projs = nn.ModuleList([
            nn.Linear(n_embd, n_embd) for _ in range(num_gates)
        ])
        # Start with gates wide open (bias = +3 → sigmoid ≈ 0.95)
        for proj in self.gate_projs:
            nn.init.zeros_(proj.weight)
            nn.init.constant_(proj.bias, 3.0)

        self._gate_log: List[torch.Tensor] = []

    def _gate_index(self, layer_idx: int) -> Optional[int]:
        if self.active_layers is None:
            return layer_idx
        if layer_idx not in self.active_layers:
            return None
        return sorted(self.active_layers).index(layer_idx)

    def apply(self, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply forget gate after the given layer.  Returns gated hidden."""
        gi = self._gate_index(layer_idx)
        if gi is None:
            return hidden
        gate = torch.sigmoid(self.gate_projs[gi](hidden))  # [B, T, d]
        if self.training:
            self._gate_log.append(gate)
        return hidden * gate

    def compute_forget_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Sparsity loss encouraging forgetting.  Call once per training step."""
        if not self._gate_log:
            z = torch.tensor(0.0)
            return z, {"forget_loss": 0.0, "gate_mean": 1.0}
        gates = torch.stack([g.mean() for g in self._gate_log])
        # -log(gate): lower gate -> higher loss -> but we WANT low gate
        # So the *adversarial* loss is +log(gate) (we'll negate in training)
        # Equivalently: forget_loss = mean(gate) — push gate toward 0
        forget_loss = gates.mean()
        gate_mean = gates.mean().item()
        self._gate_log.clear()
        return forget_loss, {"forget_loss": forget_loss.item(), "gate_mean": gate_mean}


# =============================================================================
# 2. ContextBottleneck (dimensionality compression)
# =============================================================================

class ContextBottleneck(nn.Module):
    """
    Information bottleneck that compresses representations through a
    lower-dimensional space between transformer layers.

    Architecture:  h -> LayerNorm -> W_down [d -> d//ratio] -> SiLU -> W_up [d//ratio -> d] -> h_out

    The compression forces the model to discard low-importance features.
    Adversarial loss: variance of the bottleneck activations (encourages
    collapse / fewer active dimensions).
    """

    def __init__(self, n_embd: int, n_layers: int,
                 compression_ratio: int = 4,
                 active_layers: Optional[Set[int]] = None):
        super().__init__()
        self.n_embd = n_embd
        self.compression_ratio = compression_ratio
        self.active_layers = active_layers
        bottleneck_dim = max(1, n_embd // compression_ratio)

        num_bottlenecks = n_layers if active_layers is None else len(active_layers)
        self.bottlenecks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(n_embd),
                nn.Linear(n_embd, bottleneck_dim),
                nn.SiLU(),
                nn.Linear(bottleneck_dim, n_embd),
            )
            for _ in range(num_bottlenecks)
        ])
        # Residual scaling — start as near-identity
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.01)) for _ in range(num_bottlenecks)
        ])
        self._bottleneck_activations: List[torch.Tensor] = []

    def _gate_index(self, layer_idx: int) -> Optional[int]:
        if self.active_layers is None:
            return layer_idx
        if layer_idx not in self.active_layers:
            return None
        return sorted(self.active_layers).index(layer_idx)

    def apply(self, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        gi = self._gate_index(layer_idx)
        if gi is None:
            return hidden
        bottleneck_out = self.bottlenecks[gi](hidden)
        # Residual blend: mostly passthrough at start, gradually learns compression
        out = hidden + self.alphas[gi] * (bottleneck_out - hidden)
        if self.training:
            # Log the inner bottleneck activation for the sparsity loss
            inner = self.bottlenecks[gi][1](self.bottlenecks[gi][0](hidden))
            self._bottleneck_activations.append(inner)
        return out

    def compute_forget_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Encourage bottleneck collapse (fewer active dimensions)."""
        if not self._bottleneck_activations:
            z = torch.tensor(0.0)
            return z, {"forget_loss": 0.0, "bottleneck_usage": 1.0}
        # L1 sparsity on bottleneck activations
        acts = torch.cat([a.reshape(-1) for a in self._bottleneck_activations])
        forget_loss = acts.abs().mean()
        usage = (acts.abs() > 0.01).float().mean().item()
        self._bottleneck_activations.clear()
        return forget_loss, {"forget_loss": forget_loss.item(), "bottleneck_usage": usage}


# =============================================================================
# 3. SelectiveRetention (per-token importance scoring)
# =============================================================================

class SelectiveRetention(nn.Module):
    """
    Scores each token's importance and applies soft masking.

    A small network produces a scalar importance score in [0, 1] for each
    token.  Unimportant tokens have their entire representation scaled
    toward zero.

    This directly addresses the "lost in the middle" problem: the model
    must learn which tokens carry recall-critical information, because the
    adversarial pressure will zero out everything else.

    Adversarial sparsity loss: mean(score) — pushes all scores toward 0
    (maximum token dropping).  LM loss pushes back to preserve the tokens
    the model actually needs.
    """

    def __init__(self, n_embd: int, n_layers: int,
                 active_layers: Optional[Set[int]] = None):
        super().__init__()
        self.n_embd = n_embd
        self.active_layers = active_layers

        num_scorers = n_layers if active_layers is None else len(active_layers)
        self.scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, n_embd // 4),
                nn.SiLU(),
                nn.Linear(n_embd // 4, 1),
            )
            for _ in range(num_scorers)
        ])
        # Initialise bias so scores start high (minimal masking)
        for scorer in self.scorers:
            nn.init.constant_(scorer[-1].bias, 3.0)

        self._score_log: List[torch.Tensor] = []

    def _gate_index(self, layer_idx: int) -> Optional[int]:
        if self.active_layers is None:
            return layer_idx
        if layer_idx not in self.active_layers:
            return None
        return sorted(self.active_layers).index(layer_idx)

    def apply(self, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        gi = self._gate_index(layer_idx)
        if gi is None:
            return hidden
        score = torch.sigmoid(self.scorers[gi](hidden))  # [B, T, 1]
        if self.training:
            self._score_log.append(score.squeeze(-1))
        return hidden * score

    def compute_forget_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Sparsity loss: push importance scores toward 0."""
        if not self._score_log:
            z = torch.tensor(0.0)
            return z, {"forget_loss": 0.0, "retention_rate": 1.0}
        scores = torch.cat([s.reshape(-1) for s in self._score_log])
        forget_loss = scores.mean()
        retention_rate = (scores > 0.5).float().mean().item()
        self._score_log.clear()
        return forget_loss, {"forget_loss": forget_loss.item(), "retention_rate": retention_rate}


# =============================================================================
# Factory
# =============================================================================

FORGET_MODES = ["none", "forget_gate", "context_bottleneck", "selective_retention"]


def parse_layer_set(layers_str: str, n_layer: int) -> Optional[Set[int]]:
    """Parse comma-separated layer indices, or None for all layers."""
    if not layers_str.strip() or layers_str.strip().lower() == "all":
        return None
    return {int(x.strip()) for x in layers_str.split(",")}


def build_forget_module(mode: str, n_embd: int, n_layer: int,
                        active_layers: Optional[Set[int]] = None,
                        device=None, dtype=None, **kwargs):
    """Build a forgetting module by mode name."""
    if mode == "none":
        return None
    elif mode == "forget_gate":
        m = ForgetGate(n_embd, n_layer, active_layers)
    elif mode == "context_bottleneck":
        m = ContextBottleneck(
            n_embd, n_layer,
            compression_ratio=kwargs.get("compression_ratio", 4),
            active_layers=active_layers,
        )
    elif mode == "selective_retention":
        m = SelectiveRetention(n_embd, n_layer, active_layers)
    else:
        raise ValueError(f"Unknown forget mode: {mode}. Choose from {FORGET_MODES}")

    if device is not None:
        m = m.to(device)
    if dtype is not None:
        m = m.to(dtype=dtype)
    return m


def build_forget_optimizer(forget_module, lr: float = 0.001):
    """Build optimizer for the forgetting module's parameters."""
    if forget_module is None:
        return None
    params = list(forget_module.parameters())
    if not params:
        return None
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
    return opt
