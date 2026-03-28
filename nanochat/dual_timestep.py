"""
Dual-Timestep Scheduler for Self-Flow training.

Implements the paper's core scheduling algorithm (Section 3.3):

    1. Sample two timesteps t, s ~ p(t) per batch element
    2. Sample a binary mask M with ratio R_M
    3. Construct per-token corruption levels:
       - Student: tau[i] = s if i in M, else t  (heterogeneous)
       - Teacher: tau_min = min(t, s) for all tokens (uniform, cleaner)

The information asymmetry between the student's mixed view and the teacher's
uniformly-clean view is the key mechanism that drives representation learning.

Reference:
    Chefer et al., "Self-Supervised Flow Matching for Scalable Multi-Modal
    Synthesis" (Black Forest Labs, 2026), Section 3.3
"""

import math
import torch
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DualTimestepSample:
    """Output of the scheduler: per-token corruption levels for student and teacher."""
    student_levels: torch.Tensor   # [B, T] -- heterogeneous per-token levels
    teacher_levels: torch.Tensor   # [B, T] -- uniform (tau_min broadcast)
    mask: torch.Tensor             # [B, T] -- bool, True = token got the secondary timestep


class DualTimestepScheduler:
    """
    Dual-Timestep Scheduling from the Self-Flow paper.

    Supports multiple noise distributions p(t):
        - "uniform":     t ~ U(0, 1)
        - "logit_normal": t ~ sigmoid(N(mean, std))  (paper default for images)
        - "beta":        t ~ Beta(alpha, beta)
        - "cosine":      t ~ 1 - cos(u * pi/2)^2 where u ~ U(0,1)
    """

    def __init__(
        self,
        distribution: str = "uniform",
        mask_ratio: float = 0.5,
        dist_params: dict = None,
    ):
        self.distribution = distribution
        self.mask_ratio = mask_ratio
        self.dist_params = dist_params or {}

    def _sample_timesteps(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample timesteps from the chosen distribution, clamped to (0, 1)."""

        if self.distribution == "uniform":
            t = torch.rand(shape, device=device)

        elif self.distribution == "logit_normal":
            mean = self.dist_params.get("mean", 0.0)
            std = self.dist_params.get("std", 1.0)
            z = torch.randn(shape, device=device) * std + mean
            t = torch.sigmoid(z)

        elif self.distribution == "beta":
            alpha = self.dist_params.get("alpha", 2.0)
            beta_val = self.dist_params.get("beta", 5.0)
            dist = torch.distributions.Beta(alpha, beta_val)
            t = dist.sample(shape).to(device)

        elif self.distribution == "cosine":
            u = torch.rand(shape, device=device)
            t = 1.0 - torch.cos(u * math.pi / 2).pow(2)

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        return t.clamp(1e-4, 1.0 - 1e-4)

    def sample(self, batch_size: int, seq_len: int, device: torch.device) -> DualTimestepSample:
        """
        Sample dual timesteps for a batch.

        Returns a DualTimestepSample with:
            student_levels: [B, T] heterogeneous corruption levels
            teacher_levels: [B, T] uniform (cleaner) corruption levels
            mask:           [B, T] which tokens got the secondary timestep
        """
        t = self._sample_timesteps((batch_size,), device)  # [B]
        s = self._sample_timesteps((batch_size,), device)  # [B]

        # Binary mask: which tokens get the secondary timestep s
        M = torch.rand(batch_size, seq_len, device=device) < self.mask_ratio  # [B, T]

        # Student sees heterogeneous corruption: tau[i] = s if i in M, else t
        t_expanded = t.unsqueeze(1).expand(-1, seq_len)  # [B, T]
        s_expanded = s.unsqueeze(1).expand(-1, seq_len)  # [B, T]
        student_levels = torch.where(M, s_expanded, t_expanded)

        # Teacher sees uniform corruption at tau_min = min(t, s) for all tokens
        tau_min = torch.min(t, s)  # [B]
        teacher_levels = tau_min.unsqueeze(1).expand(-1, seq_len)  # [B, T]

        return DualTimestepSample(
            student_levels=student_levels,
            teacher_levels=teacher_levels,
            mask=M,
        )
