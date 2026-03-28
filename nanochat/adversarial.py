"""
Adversarial components for Self-Flow training.

Four adversarial strategies that can be toggled via --adversarial flag:

1. corrupter   - Learned network that outputs per-token corruption levels
                 to maximally confuse the student's representation alignment.
2. mask        - Learned mask predictor that chooses WHICH tokens to corrupt
                 hardest, while corruption strength stays from the scheduler.
3. gradient    - PGD-style adversarial perturbation in embedding space.
                 No extra parameters -- just inner-loop gradient ascent.
4. discriminator - Discriminative head that distinguishes student vs teacher
                 representations. Student tries to fool it (GAN-style).

All adversarial strategies operate on the representation alignment loss only
(not the LM loss), so the adversary learns to target semantic understanding
rather than simply destroying all information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# =============================================================================
# 1. Adversarial Corrupter Network
# =============================================================================

class AdversarialCorrupter(nn.Module):
    """
    Learned network that produces per-token corruption levels to maximize
    the student's representation alignment loss.

    Takes clean token embeddings and outputs corruption levels in [0, 1] per
    token. Trained adversarially: the corrupter maximizes L_rep while the
    student minimizes it.

    Architecture: 2-layer MLP over embeddings -> sigmoid -> per-token levels.
    Kept small to avoid overpowering the student.
    """

    def __init__(self, n_embd: int, hidden_mult: int = 1):
        super().__init__()
        hidden_dim = n_embd * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: Clean token embeddings [B, T, d].
        Returns:
            Per-token corruption levels [B, T] in [0, 1].
        """
        return self.net(embeddings).squeeze(-1).sigmoid()


# =============================================================================
# 2. Adversarial Mask Predictor
# =============================================================================

class AdversarialMaskPredictor(nn.Module):
    """
    Learns WHICH token positions to corrupt hardest.

    Instead of a random binary mask M (from dual-timestep scheduling), this
    network predicts per-token probabilities of receiving the higher corruption
    level. The corruption levels themselves still come from the scheduler.

    Uses Gumbel-Softmax for differentiable discrete mask sampling during training.
    """

    def __init__(self, n_embd: int, hidden_mult: int = 1):
        super().__init__()
        hidden_dim = n_embd * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.temperature = 1.0

    def forward(
        self,
        embeddings: torch.Tensor,
        t_levels: torch.Tensor,
        s_levels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: Clean token embeddings [B, T, d].
            t_levels: Primary timestep per batch element, broadcast to [B, T].
            s_levels: Secondary timestep per batch element, broadcast to [B, T].
        Returns:
            Per-token corruption levels [B, T] (soft blend of t and s).
        """
        logits = self.net(embeddings).squeeze(-1)  # [B, T]
        # Gumbel-sigmoid for differentiable binary mask
        if self.training:
            u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            gumbel_noise = torch.log(u) - torch.log(1 - u)
            mask_soft = torch.sigmoid((logits + gumbel_noise) / self.temperature)
        else:
            mask_soft = torch.sigmoid(logits)
        # Blend: mask_soft=1 -> s_levels, mask_soft=0 -> t_levels
        return (1 - mask_soft) * t_levels + mask_soft * s_levels


# =============================================================================
# 3. Gradient-Based Adversarial Perturbation (no learned parameters)
# =============================================================================

class GradientAdversarial:
    """
    PGD-style adversarial perturbation in embedding space.

    No extra network -- computes worst-case perturbations via gradient ascent
    on the representation loss w.r.t. the input embeddings.

    At each training step:
      1. Start with corrupted embeddings from the normal pipeline
      2. Run K inner steps of gradient ascent on the rep loss w.r.t. embeddings
      3. Project perturbation onto an epsilon-ball
      4. Use the perturbed embeddings for the student forward pass

    This is similar to FreeLB / SMART from adversarial NLP training.
    """

    def __init__(self, epsilon: float = 0.1, step_size: float = 0.03,
                 num_steps: int = 3):
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps

    def perturb(
        self,
        embeddings: torch.Tensor,
        loss_fn,
    ) -> torch.Tensor:
        """
        Find adversarial perturbation that maximizes loss_fn(embeddings + delta).

        Args:
            embeddings: Input embeddings [B, T, d] (will not be modified).
            loss_fn: Callable that takes embeddings and returns a scalar loss.
                     The perturbation maximizes this loss.
        Returns:
            Perturbed embeddings [B, T, d].
        """
        delta = torch.zeros_like(embeddings, requires_grad=True)

        for _ in range(self.num_steps):
            perturbed = embeddings.detach() + delta
            loss = loss_fn(perturbed)
            loss.backward(retain_graph=False)

            # Gradient ascent step
            grad = delta.grad.detach()
            delta_data = delta.data + self.step_size * grad.sign()
            # Project onto L-inf epsilon ball
            delta_data = delta_data.clamp(-self.epsilon, self.epsilon)
            delta = delta_data.clone().requires_grad_(True)

        return embeddings.detach() + delta.detach()


# =============================================================================
# 4. Discriminative Head
# =============================================================================

class DiscriminativeHead(nn.Module):
    """
    Binary classifier that distinguishes student from teacher representations.

    The student is trained to fool the discriminator (make its projected
    representations indistinguishable from the teacher's). The discriminator
    is trained to tell them apart.

    This adds a GAN-style adversarial loss to the representation alignment:
      L_disc = BCE(D(student_proj), 0) + BCE(D(teacher), 1)  -- train D
      L_gen  = BCE(D(student_proj), 1)                        -- train student

    Uses gradient penalty for stability (WGAN-GP style).
    """

    def __init__(self, n_embd: int, hidden_mult: int = 2):
        super().__init__()
        hidden_dim = n_embd * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits [B, T, 1] (not sigmoid'd)."""
        return self.net(x)

    def compute_discriminator_loss(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        gp_weight: float = 10.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Discriminator loss: classify student as fake (0), teacher as real (1).
        Includes gradient penalty for training stability.
        """
        student_logits = self.forward(student_hidden.detach())
        teacher_logits = self.forward(teacher_hidden.detach())

        loss_fake = F.binary_cross_entropy_with_logits(
            student_logits, torch.zeros_like(student_logits)
        )
        loss_real = F.binary_cross_entropy_with_logits(
            teacher_logits, torch.ones_like(teacher_logits)
        )

        gp = self._gradient_penalty(student_hidden.detach(), teacher_hidden.detach())
        disc_loss = loss_fake + loss_real + gp_weight * gp

        with torch.no_grad():
            acc_fake = (student_logits < 0).float().mean()
            acc_real = (teacher_logits > 0).float().mean()

        return disc_loss, {
            "disc_loss": disc_loss.item(),
            "disc_acc_fake": acc_fake.item(),
            "disc_acc_real": acc_real.item(),
            "disc_gp": gp.item(),
        }

    def compute_generator_loss(
        self,
        student_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Generator (student) loss: fool discriminator into classifying as real."""
        student_logits = self.forward(student_hidden)
        return F.binary_cross_entropy_with_logits(
            student_logits, torch.ones_like(student_logits)
        )

    def _gradient_penalty(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        """WGAN-GP gradient penalty for training stability."""
        alpha = torch.rand(real.size(0), 1, 1, device=real.device, dtype=real.dtype)
        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interpolated = self.forward(interpolated)
        grad = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_norm = grad.view(grad.size(0), -1).norm(2, dim=1)
        return ((grad_norm - 1) ** 2).mean()


# =============================================================================
# Factory + Optimizer Helper
# =============================================================================

ADVERSARIAL_MODES = ["none", "corrupter", "mask", "gradient", "discriminator"]


def build_adversarial(mode: str, n_embd: int, device=None, dtype=None, **kwargs):
    """
    Build an adversarial component by mode name.

    Returns:
        The adversarial module/object, or None if mode is "none".
    """
    if mode == "none":
        return None
    elif mode == "corrupter":
        m = AdversarialCorrupter(n_embd, hidden_mult=kwargs.get("adv_hidden_mult", 1))
    elif mode == "mask":
        m = AdversarialMaskPredictor(n_embd, hidden_mult=kwargs.get("adv_hidden_mult", 1))
    elif mode == "gradient":
        return GradientAdversarial(
            epsilon=kwargs.get("adv_epsilon", 0.1),
            step_size=kwargs.get("adv_step_size", 0.03),
            num_steps=kwargs.get("adv_pgd_steps", 3),
        )
    elif mode == "discriminator":
        m = DiscriminativeHead(n_embd, hidden_mult=kwargs.get("adv_hidden_mult", 2))
    else:
        raise ValueError(f"Unknown adversarial mode: {mode}. Choose from {ADVERSARIAL_MODES}")

    if device is not None:
        m = m.to(device)
    if dtype is not None:
        m = m.to(dtype=dtype)
    return m


def build_adversarial_optimizer(adversarial, lr: float = 0.0003):
    """Build an optimizer for the adversarial module's parameters (if any)."""
    if adversarial is None or isinstance(adversarial, GradientAdversarial):
        return None
    params = list(adversarial.parameters())
    if not params:
        return None
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.5, 0.999), weight_decay=0.0)
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
    return opt
