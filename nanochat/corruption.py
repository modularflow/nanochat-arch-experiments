"""
Pluggable corruption strategies for Self-Flow language model training.

Each strategy implements the same interface: given token IDs and per-token
corruption levels in [0, 1], produce corrupted embeddings. This allows
experimenting with different corruption mechanisms while keeping the rest
of the Self-Flow pipeline unchanged.

Corruption level semantics:
    0.0 = clean (no corruption)
    1.0 = fully corrupted (maximum information destruction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple


class CorruptionStrategy(ABC):
    """Base class for all corruption strategies."""

    @abstractmethod
    def corrupt(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        corruption_levels: torch.Tensor,
        embedding_table: nn.Embedding,
    ) -> torch.Tensor:
        """
        Apply corruption to token embeddings.

        Args:
            embeddings: Clean token embeddings [B, T, d].
            input_ids: Original token IDs [B, T] (for discrete strategies).
            corruption_levels: Per-token corruption level in [0, 1], shape [B, T].
            embedding_table: The model's embedding layer (for strategies that
                need to look up replacement tokens).

        Returns:
            Corrupted embeddings [B, T, d].
        """
        ...


class EmbeddingInterpolationCorruption(CorruptionStrategy):
    """
    Interpolate in embedding space between clean embeddings and Gaussian noise.

    Closest to the paper's continuous flow matching formulation:
        x_corrupted = (1 - t) * embed(token) + t * noise

    where t is the per-token corruption level and noise ~ N(0, I).
    """

    def corrupt(self, embeddings, input_ids, corruption_levels, embedding_table):
        t = corruption_levels.unsqueeze(-1)  # [B, T, 1]
        noise = torch.randn_like(embeddings)
        return (1.0 - t) * embeddings + t * noise


class TokenReplacementCorruption(CorruptionStrategy):
    """
    Replace tokens with random token IDs, then embed.

    For each token, with probability equal to its corruption level, replace it
    with a uniformly random token ID. This is the discrete analog used in the
    bolt-on version.
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def corrupt(self, embeddings, input_ids, corruption_levels, embedding_table):
        mask = torch.rand_like(corruption_levels) < corruption_levels  # [B, T]
        random_ids = torch.randint(
            0, self.vocab_size, input_ids.shape, device=input_ids.device
        )
        replaced_ids = torch.where(mask, random_ids, input_ids)
        return embedding_table(replaced_ids)


class MaskCorruption(CorruptionStrategy):
    """
    Replace corrupted positions with a learned [MASK] embedding.

    The corruption level controls the probability that each token is masked.
    The mask embedding is a learnable parameter passed in at construction.
    """

    def __init__(self, n_embd: int, device=None, dtype=None):
        self.mask_embedding = nn.Parameter(
            torch.randn(n_embd, device=device, dtype=dtype) * 0.02
        )

    def corrupt(self, embeddings, input_ids, corruption_levels, embedding_table):
        mask = (torch.rand_like(corruption_levels) < corruption_levels).unsqueeze(-1)
        mask_emb = self.mask_embedding.to(embeddings.dtype)
        return torch.where(mask, mask_emb.expand_as(embeddings), embeddings)


class SpanCorruption(CorruptionStrategy):
    """
    Corrupt contiguous spans of tokens (inspired by T5/UL2).

    The mean corruption level across the sequence determines the total fraction
    of tokens to corrupt. Spans of length ~mean_span_length are selected and
    replaced with Gaussian noise in embedding space.
    """

    def __init__(self, mean_span_length: int = 3):
        self.mean_span_length = mean_span_length

    def corrupt(self, embeddings, input_ids, corruption_levels, embedding_table):
        B, T, d = embeddings.shape
        device = embeddings.device

        result = embeddings.clone()
        for b in range(B):
            target_count = int(corruption_levels[b].mean().item() * T)
            if target_count == 0:
                continue

            corrupted = 0
            pos = 0
            mask = torch.zeros(T, dtype=torch.bool, device=device)
            while corrupted < target_count and pos < T:
                skip = torch.geometric(torch.tensor([1.0 / self.mean_span_length])).int().item()
                pos = pos + skip
                span_len = max(1, int(torch.poisson(torch.tensor(float(self.mean_span_length))).item()))
                end = min(pos + span_len, T)
                mask[pos:end] = True
                corrupted += end - pos
                pos = end

            noise = torch.randn(T, d, device=device, dtype=embeddings.dtype)
            result[b] = torch.where(mask.unsqueeze(-1), noise, result[b])

        return result


class CompositeCorruption(CorruptionStrategy):
    """
    Blend two corruption strategies with a fixed mixing ratio.

    For each token, with probability `mix_ratio` the first strategy is used,
    otherwise the second. Useful for combining e.g. token replacement with
    embedding noise.
    """

    def __init__(self, strategy_a: CorruptionStrategy, strategy_b: CorruptionStrategy,
                 mix_ratio: float = 0.5):
        self.strategy_a = strategy_a
        self.strategy_b = strategy_b
        self.mix_ratio = mix_ratio

    def corrupt(self, embeddings, input_ids, corruption_levels, embedding_table):
        out_a = self.strategy_a.corrupt(embeddings, input_ids, corruption_levels, embedding_table)
        out_b = self.strategy_b.corrupt(embeddings, input_ids, corruption_levels, embedding_table)
        mask = (torch.rand(embeddings.shape[:2], device=embeddings.device) < self.mix_ratio)
        mask = mask.unsqueeze(-1)
        return torch.where(mask, out_a, out_b)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_corruption_strategy(name: str, vocab_size: int, n_embd: int,
                              device=None, dtype=None, **kwargs) -> CorruptionStrategy:
    """Build a corruption strategy by name."""
    strategies = {
        "embedding_interpolation": lambda: EmbeddingInterpolationCorruption(),
        "token_replacement": lambda: TokenReplacementCorruption(vocab_size),
        "mask": lambda: MaskCorruption(n_embd, device=device, dtype=dtype),
        "span": lambda: SpanCorruption(mean_span_length=kwargs.get("mean_span_length", 3)),
        "composite_interp_replace": lambda: CompositeCorruption(
            EmbeddingInterpolationCorruption(),
            TokenReplacementCorruption(vocab_size),
            mix_ratio=kwargs.get("mix_ratio", 0.5),
        ),
    }
    if name not in strategies:
        raise ValueError(f"Unknown corruption strategy '{name}'. Available: {list(strategies.keys())}")
    return strategies[name]()
