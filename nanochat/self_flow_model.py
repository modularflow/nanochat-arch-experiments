"""
SelfFlowCRATE: Ground-up Self-Flow architecture for CRATE language models.

Extends CRATE with native support for self-supervised flow matching:
  - Per-token corruption-level conditioning (tells the model how corrupted each token is)
  - Built-in projection heads for multi-scale representation alignment
  - Dedicated forward_selfflow() for dual-path teacher/student training
  - Standard forward() for inference (bypasses conditioning when unused)

The EMA teacher is managed externally by the trainer, keeping this class
clean and serializable.

Reference:
    Chefer et al., "Self-Supervised Flow Matching for Scalable Multi-Modal
    Synthesis" (Black Forest Labs, 2026)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.crate import CRATE, CRATEConfig, norm
from nanochat.corruption import CorruptionStrategy, build_corruption_strategy

try:
    from nanochat.common import get_dist_info, print0
except ImportError:
    def get_dist_info():
        return False, 0, 0, 1
    def print0(s="", **kwargs):
        print(s, **kwargs)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SelfFlowConfig(CRATEConfig):
    """Configuration extending CRATEConfig with Self-Flow parameters."""

    # Layer indices for representation alignment (auto-computed if -1)
    student_layers: str = ""    # comma-separated layer indices, e.g. "4,8"; auto = n_layer//3
    teacher_layers: str = ""    # comma-separated layer indices, e.g. "8,16"; auto = 2*n_layer//3

    # Projection head
    proj_hidden_mult: int = 2   # hidden dim = n_embd * proj_hidden_mult

    # Corruption conditioning
    corruption_conditioning: bool = True
    cond_hidden_mult: int = 4   # hidden dim for conditioning MLP

    # Corruption strategy
    corruption_strategy: str = "embedding_interpolation"

    # Loss
    rep_loss_type: str = "cosine"  # "cosine" | "mse" | "smooth_l1"
    rep_loss_weight: float = 1.0   # gamma in L = L_gen + gamma * L_rep


# =============================================================================
# Sub-modules
# =============================================================================

class CorruptionConditioner(nn.Module):
    """
    Maps per-token corruption level (scalar in [0,1]) to a bias vector in R^d.

    This is the LM analog of the paper's per-token timestep conditioning: it
    tells the model how much each token has been corrupted, allowing it to
    modulate its representations accordingly.

    Architecture: scalar -> sinusoidal embedding -> MLP -> bias [d]
    """

    def __init__(self, n_embd: int, hidden_mult: int = 4, n_freqs: int = 64):
        super().__init__()
        self.n_freqs = n_freqs
        input_dim = n_freqs * 2  # sin + cos
        hidden_dim = n_embd * hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_embd),
        )

    def forward(self, levels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            levels: Per-token corruption levels [B, T] in [0, 1].
        Returns:
            Conditioning bias [B, T, d].
        """
        freqs = torch.arange(self.n_freqs, device=levels.device, dtype=levels.dtype)
        freqs = torch.exp2(freqs) * torch.pi  # geometric spacing
        # [B, T, n_freqs]
        angles = levels.unsqueeze(-1) * freqs
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [B, T, 2*n_freqs]
        return self.mlp(emb)


class ProjectionHead(nn.Module):
    """
    MLP projection head for representation alignment.

    Maps student hidden states to the teacher's representation space.
    Architecture matches the paper's SimpleHead: linear -> SiLU -> linear.
    """

    def __init__(self, n_embd: int, hidden_mult: int = 2):
        super().__init__()
        hidden_dim = n_embd * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Representation Loss
# =============================================================================

def compute_rep_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    proj_head: ProjectionHead,
    loss_type: str = "cosine",
) -> torch.Tensor:
    """
    Compute representation alignment loss between projected student and teacher hidden states.

    Args:
        student_hidden: [B, T, d] from student at layer l.
        teacher_hidden: [B, T, d] from teacher at layer k (detached).
        proj_head: Projection head mapping student space -> teacher space.
        loss_type: "cosine" (paper default), "mse", or "smooth_l1".
    """
    projected = proj_head(student_hidden)  # [B, T, d]

    if loss_type == "cosine":
        # Negative cosine similarity (we minimize, so flip sign)
        proj_norm = F.normalize(projected, dim=-1)
        teacher_norm = F.normalize(teacher_hidden.detach(), dim=-1)
        return -(proj_norm * teacher_norm).sum(dim=-1).mean()
    elif loss_type == "mse":
        return F.mse_loss(projected, teacher_hidden.detach())
    elif loss_type == "smooth_l1":
        return F.smooth_l1_loss(projected, teacher_hidden.detach())
    else:
        raise ValueError(f"Unknown rep_loss_type: {loss_type}")


# =============================================================================
# SelfFlowCRATE Model
# =============================================================================

class SelfFlowCRATE(nn.Module):
    """
    CRATE with native Self-Flow support.

    Wraps a standard CRATE backbone and adds:
      1. CorruptionConditioner: per-token corruption-level embedding
      2. ProjectionHead(s): for multi-scale representation alignment
      3. forward_selfflow(): dedicated dual-path forward for training
      4. forward(): standard LM forward for inference (conditioning bypassed)

    The EMA teacher model is NOT stored here -- it's managed by the trainer.
    """

    def __init__(self, config: SelfFlowConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config

        # Core CRATE backbone
        self.backbone = CRATE(config, pad_vocab_size_to=pad_vocab_size_to)

        # Parse alignment layer indices
        self.student_layer_indices = self._parse_layers(config.student_layers, config.n_layer, "student")
        self.teacher_layer_indices = self._parse_layers(config.teacher_layers, config.n_layer, "teacher")
        assert len(self.student_layer_indices) == len(self.teacher_layer_indices), \
            f"Number of student layers ({len(self.student_layer_indices)}) must match teacher layers ({len(self.teacher_layer_indices)})"

        # Per-alignment-point projection heads
        self.proj_heads = nn.ModuleList([
            ProjectionHead(config.n_embd, config.proj_hidden_mult)
            for _ in self.student_layer_indices
        ])

        # Corruption conditioning
        self.use_conditioning = config.corruption_conditioning
        if self.use_conditioning:
            self.corruption_conditioner = CorruptionConditioner(
                config.n_embd, hidden_mult=config.cond_hidden_mult
            )

        print0(f"SelfFlowCRATE: student_layers={self.student_layer_indices}, "
               f"teacher_layers={self.teacher_layer_indices}, "
               f"conditioning={self.use_conditioning}, "
               f"rep_loss={config.rep_loss_type}")

    @staticmethod
    def _parse_layers(layers_str: str, n_layer: int, role: str) -> List[int]:
        """Parse comma-separated layer indices, or auto-compute defaults."""
        if layers_str.strip():
            indices = [int(x.strip()) for x in layers_str.split(",")]
        else:
            if role == "student":
                indices = [n_layer // 3]
            else:
                indices = [2 * n_layer // 3]
        for idx in indices:
            assert 0 <= idx < n_layer, f"{role} layer {idx} out of range [0, {n_layer})"
        return indices

    # ------------------------------------------------------------------
    # Delegation to backbone for compatibility
    # ------------------------------------------------------------------

    @property
    def transformer(self):
        return self.backbone.transformer

    @property
    def lm_head(self):
        return self.backbone.lm_head

    def get_device(self):
        return self.backbone.get_device()

    def estimate_flops(self):
        return self.backbone.estimate_flops()

    def num_scaling_params(self):
        return sum(p.numel() for p in self.parameters())

    def init_weights(self):
        self.backbone.init_weights()
        for proj in self.proj_heads:
            for m in proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        if self.use_conditioning:
            for m in self.corruption_conditioner.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def generate(self, *args, **kwargs):
        return self.backbone.generate(*args, **kwargs)

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def setup_optimizers(self, proj_lr: float = 0.001, **backbone_kwargs):
        """
        Setup optimizers for both backbone and self-flow heads.

        Returns:
            (backbone_optimizers: list, selfflow_optimizer: AdamW)
        """
        backbone_opts = self.backbone.setup_optimizers(**backbone_kwargs)

        selfflow_params = list(self.proj_heads.parameters())
        if self.use_conditioning:
            selfflow_params += list(self.corruption_conditioner.parameters())

        from functools import partial
        ddp, rank, local_rank, world_size = get_dist_info()
        if ddp:
            from nanochat.adamw import DistAdamW
            AdamWFactory = DistAdamW
        else:
            AdamWFactory = partial(torch.optim.AdamW, fused=torch.cuda.is_available())

        selfflow_opt = AdamWFactory(
            [{"params": selfflow_params, "lr": proj_lr}],
            betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0,
        )
        for group in selfflow_opt.param_groups:
            group["initial_lr"] = group["lr"]

        return backbone_opts, selfflow_opt

    # ------------------------------------------------------------------
    # Forward methods
    # ------------------------------------------------------------------

    def forward(self, idx: torch.Tensor, targets=None, kv_cache=None,
                loss_reduction: str = 'mean', return_hidden_at=None,
                corruption_levels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Standard LM forward, optionally with corruption conditioning.

        When corruption_levels is None, behaves identically to CRATE.forward().
        When provided, injects conditioning bias into embeddings.
        """
        embedding_bias = None
        if corruption_levels is not None and self.use_conditioning:
            embedding_bias = self.corruption_conditioner(corruption_levels)

        return self.backbone(
            idx, targets=targets, kv_cache=kv_cache,
            loss_reduction=loss_reduction, return_hidden_at=return_hidden_at,
            embedding_bias=embedding_bias,
        )

    def forward_selfflow(
        self,
        student_ids: torch.Tensor,
        student_corruption_levels: torch.Tensor,
        targets: torch.Tensor,
        corruption_strategy: CorruptionStrategy,
        forget_module=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Student-side forward pass for Self-Flow training.

        1. Embeds tokens, applies corruption via the strategy, adds conditioning.
        2. Runs through backbone layers, collecting hidden states at student layers.
           If forget_module is provided, applies selective forgetting after each layer.
        3. Computes LM loss on clean targets.
        4. Returns LM loss and student hidden states (projection happens outside).

        Args:
            student_ids: Clean token IDs [B, T].
            student_corruption_levels: Per-token levels [B, T].
            targets: Clean target IDs [B, T].
            corruption_strategy: How to corrupt the embeddings.
            forget_module: Optional forgetting mechanism (ForgetGate, ContextBottleneck,
                or SelectiveRetention) applied between layers to force context compression.

        Returns:
            (lm_loss, student_hiddens) where student_hiddens maps
            layer_idx -> hidden [B, T, d].
        """
        # Get clean embeddings, then corrupt
        clean_emb = self.backbone.transformer.wte(student_ids)
        corrupted_emb = corruption_strategy.corrupt(
            clean_emb, student_ids, student_corruption_levels,
            self.backbone.transformer.wte,
        )
        corrupted_emb = norm(corrupted_emb)

        # Add corruption conditioning bias
        if self.use_conditioning:
            cond_bias = self.corruption_conditioner(student_corruption_levels)
            corrupted_emb = corrupted_emb + cond_bias

        # Run through transformer layers manually (to inject corrupted embeddings)
        B, T = student_ids.size()
        cfg = self.backbone.config
        assert T <= self.backbone.cos.size(1)
        cos_sin = (self.backbone.cos[:, :T], self.backbone.sin[:, :T])

        x = corrupted_emb
        x0 = x
        target_layers = set(self.student_layer_indices)
        hidden_snapshots: Dict[int, torch.Tensor] = {}

        for i, block in enumerate(self.backbone.transformer.h):
            x = self.backbone.resid_lambdas[i] * x + self.backbone.x0_lambdas[i] * x0
            x = block(x, cos_sin, self.backbone.window_sizes[i], None)
            if forget_module is not None:
                x = forget_module.apply(x, i)
            if i in target_layers:
                hidden_snapshots[i] = x

        x = norm(x)

        # LM head
        softcap = 15
        logits = self.backbone.lm_head(x)
        logits = logits[..., :cfg.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        lm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )

        return lm_loss, hidden_snapshots

    def compute_selfflow_loss(
        self,
        student_hiddens: Dict[int, torch.Tensor],
        teacher_hiddens: Dict[int, torch.Tensor],
        lm_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Combine LM loss with representation alignment losses.

        Args:
            student_hiddens: {student_layer: hidden [B,T,d]}
            teacher_hiddens: {teacher_layer: hidden [B,T,d]}
            lm_loss: Scalar LM loss.

        Returns:
            (total_loss, metrics_dict) where metrics_dict contains per-layer
            rep losses and the combined loss for logging.
        """
        rep_losses = []
        metrics = {"lm_loss": lm_loss.item()}

        for i, (s_layer, t_layer) in enumerate(zip(self.student_layer_indices, self.teacher_layer_indices)):
            s_hidden = student_hiddens[s_layer]
            t_hidden = teacher_hiddens[t_layer]
            rep_loss = compute_rep_loss(
                s_hidden, t_hidden, self.proj_heads[i],
                loss_type=self.config.rep_loss_type,
            )
            rep_losses.append(rep_loss)
            metrics[f"rep_loss_s{s_layer}_t{t_layer}"] = rep_loss.item()

        total_rep_loss = sum(rep_losses) / len(rep_losses) if rep_losses else torch.tensor(0.0)
        total_loss = lm_loss + self.config.rep_loss_weight * total_rep_loss

        metrics["rep_loss"] = total_rep_loss.item() if isinstance(total_rep_loss, torch.Tensor) else total_rep_loss
        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics
