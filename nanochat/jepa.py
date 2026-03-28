"""
Shared JEPA (Joint Embedding Predictive Architecture) utilities.

Provides the auxiliary JEPA loss that can be combined with any primary
training objective (base pretraining, midtraining, SFT, self-training).

The JEPA loss splits a sequence into two views, runs each through the model,
and trains the model to predict the representation of view B from view A
using a special <|pred|> token appended to view A.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import print0

PRED_TOKEN_STR = "<|pred|>"

# Supported JEPA lambda schedules
JEPA_SCHEDULES = ("constant", "linear_decay", "cosine_decay")


def get_jepa_lambda(base_lambda, step, total_steps, schedule="constant"):
    """
    Return the JEPA loss weight at a given training step.

    Schedules:
        constant     - base_lambda throughout training
        linear_decay - base_lambda → 0 linearly over total_steps
        cosine_decay - base_lambda → 0 following a cosine half-period
    """
    if schedule == "constant":
        return base_lambda
    frac = min(step / max(total_steps, 1), 1.0)
    if schedule == "linear_decay":
        return base_lambda * (1.0 - frac)
    if schedule == "cosine_decay":
        return base_lambda * 0.5 * (1.0 + math.cos(math.pi * frac))
    raise ValueError(f"Unknown JEPA schedule '{schedule}'. Choose from: {JEPA_SCHEDULES}")


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def get_backbone(model):
    """Unwrap SelfFlowCRATE to get the underlying CRATE/GPT backbone."""
    return model.backbone if hasattr(model, "backbone") else model


def resize_model_vocab(backbone, new_vocab_size, device):
    """Resize embedding and lm_head to accommodate new tokens."""
    old_wte = backbone.transformer.wte
    old_lm_head = backbone.lm_head
    old_vocab_size, emb_dim = old_wte.weight.shape

    if new_vocab_size <= old_vocab_size:
        return

    new_wte = nn.Embedding(new_vocab_size, emb_dim, device=device, dtype=old_wte.weight.dtype)
    new_lm_head = nn.Linear(emb_dim, new_vocab_size, bias=False, device=device, dtype=old_lm_head.weight.dtype)

    with torch.no_grad():
        new_wte.weight[:old_vocab_size].copy_(old_wte.weight)
        new_lm_head.weight[:old_vocab_size].copy_(old_lm_head.weight)
        torch.nn.init.normal_(new_wte.weight[old_vocab_size:], mean=0.0, std=0.02)
        torch.nn.init.normal_(new_lm_head.weight[old_vocab_size:], mean=0.0, std=0.001)

    backbone.transformer.wte = new_wte
    backbone.lm_head = new_lm_head
    backbone.config.vocab_size = new_vocab_size


def ensure_pred_token_slot(model, tokenizer, device):
    """
    Reserve a model-vocab slot for the <|pred|> token.

    Uses the first unused slot beyond the tokenizer's vocabulary, resizing
    the embedding table if necessary.

    Returns the token ID for <|pred|>.
    """
    backbone = get_backbone(model)
    real_vocab = tokenizer.get_vocab_size()
    current_rows = backbone.transformer.wte.weight.shape[0]
    pred_token_id = real_vocab

    if pred_token_id >= current_rows:
        new_vocab_size = ((pred_token_id + 1 + 63) // 64) * 64
        print0(f"Resizing embedding table from {current_rows} to {new_vocab_size} for {PRED_TOKEN_STR}")
        resize_model_vocab(backbone, new_vocab_size, device)
    else:
        print0(f"Using padded vocab slot {pred_token_id} for {PRED_TOKEN_STR}")

    with torch.no_grad():
        torch.nn.init.normal_(backbone.transformer.wte.weight[pred_token_id:pred_token_id + 1], mean=0.0, std=0.02)
        if pred_token_id < backbone.lm_head.weight.shape[0]:
            torch.nn.init.normal_(backbone.lm_head.weight[pred_token_id:pred_token_id + 1], mean=0.0, std=0.001)

    return pred_token_id


def forward_final_hidden(model, idx):
    """Run model forward and return the final hidden state (before lm_head)."""
    backbone = get_backbone(model)
    _, seq_len = idx.size()
    assert seq_len <= backbone.cos.size(1), (
        f"Sequence length grew beyond rotary embeddings cache: {seq_len} > {backbone.cos.size(1)}"
    )
    cos_sin = backbone.cos[:, :seq_len], backbone.sin[:, :seq_len]
    x = backbone.transformer.wte(idx)
    x = rms_norm(x)
    x0 = x
    for i, block in enumerate(backbone.transformer.h):
        x = backbone.resid_lambdas[i] * x + backbone.x0_lambdas[i] * x0
        x = block(x, cos_sin, backbone.window_sizes[i], None)
    return rms_norm(x)


def compute_jepa_loss(model, view_a, view_b, pred_token_id, device):
    """
    Compute JEPA loss between two views of a sequence.

    Appends <|pred|> to view_a, runs both views through the model, and
    returns 1 - cosine_similarity between the predicted and target embeddings.
    """
    pred_id_tensor = torch.tensor([pred_token_id], dtype=torch.long, device=device)
    input_a = torch.cat([view_a, pred_id_tensor]).unsqueeze(0)
    hidden_a = forward_final_hidden(model, input_a)
    pred_embed = hidden_a[0, -1, :]
    input_b = view_b.unsqueeze(0)
    with torch.no_grad():
        hidden_b = forward_final_hidden(model, input_b)
    target_embed = hidden_b[0, -1, :].detach()
    loss = 1.0 - F.cosine_similarity(
        pred_embed.unsqueeze(0),
        target_embed.unsqueeze(0),
    )
    return loss.squeeze()


def split_views(seq, min_len=32):
    """Split a 1-D token sequence at the midpoint into two views."""
    seq_len = seq.size(0)
    if seq_len < min_len * 2:
        return None, None
    mid = seq_len // 2
    return seq[:mid], seq[mid:]


def extract_last_turn_views(seq, user_start_id, assistant_start_id, min_len=4):
    """
    Extract the final user turn and its matching assistant turn from a packed
    conversational sequence (for conversational JEPA).

    Returns (user_ids, assistant_ids), or (None, None) if no valid pair found.
    """
    assistant_positions = (seq == assistant_start_id).nonzero(as_tuple=True)[0]
    if len(assistant_positions) == 0:
        return None, None
    assistant_start = assistant_positions[-1].item()

    user_positions = (seq[:assistant_start] == user_start_id).nonzero(as_tuple=True)[0]
    if len(user_positions) == 0:
        return None, None
    user_start = user_positions[-1].item()

    user_ids = seq[user_start:assistant_start]
    assistant_ids = seq[assistant_start:]
    if len(user_ids) < min_len or len(assistant_ids) < min_len:
        return None, None
    return user_ids, assistant_ids


def compute_jepa_loss_for_batch(model, x, y, pred_token_id, device,
                                view_min_len=64, max_view_tokens=256):
    """
    Compute the average JEPA loss over a batch of sequences.

    Iterates over sequences in the batch, splits each into two views,
    and computes the JEPA loss. Returns (mean_loss, num_pairs) or
    (None, 0) if no valid pairs were found.
    """
    jepa_loss_accum = torch.tensor(0.0, device=device)
    pair_count = 0

    for b in range(x.shape[0]):
        valid = (y[b] >= 0).nonzero(as_tuple=True)[0]
        if len(valid) == 0:
            continue
        effective_len = valid[-1].item() + 2
        effective_len = min(effective_len, x[b].size(0))
        seq = x[b, :effective_len]

        view_a, view_b = split_views(seq, min_len=view_min_len)
        if view_a is None or view_b is None:
            continue

        view_a = view_a[-max_view_tokens:]
        view_b = view_b[:max_view_tokens]

        j_loss = compute_jepa_loss(model, view_a, view_b, pred_token_id, device)
        jepa_loss_accum = jepa_loss_accum + j_loss
        pair_count += 1

    if pair_count > 0:
        return jepa_loss_accum / pair_count, pair_count
    return None, 0
