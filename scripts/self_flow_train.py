"""
Self-Flow: Self-supervised self-distillation with dual-corruption for CRATE.

Adapts the Self-Flow framework (Black Forest Labs, 2026) to a causal language
model. Both the student (trainable) and teacher (EMA) see the same tokens but
with different corruption levels. The student must match the teacher's deeper-
layer representations via a projection head, forcing it to develop robust
semantic understanding from degraded input.

Usage:
    python -m scripts.self_flow_train \
        --source base --model-tag d12 --model-step 20000 \
        --prompt-task smoltalk \
        --student-corruption 0.75 --teacher-corruption 0.25 \
        --num-train-steps 500

Reference:
    Chefer et al., "Self-Supervised Flow Matching for Scalable Multi-Modal
    Synthesis" (Black Forest Labs, 2026)
"""

import argparse
import copy
import gc
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

from nanochat.common import (
    compute_init, compute_cleanup, get_base_dir, get_dist_info,
    print0, DummyWandb, autodetect_device_type,
)
from nanochat.checkpoint_manager import load_model, save_checkpoint

# ---------------------------------------------------------------------------
# Projection head (matches Self-Flow's SimpleHead)
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 2)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))

# ---------------------------------------------------------------------------
# EMA helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def create_ema_model(model):
    """Deep-copy model weights into a frozen EMA teacher."""
    ema = copy.deepcopy(model)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad = False
    return ema


@torch.no_grad()
def update_ema(student, teacher, decay):
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)

# ---------------------------------------------------------------------------
# Token corruption
# ---------------------------------------------------------------------------

def corrupt_tokens(input_ids, corruption_rate, vocab_size):
    """Replace a fraction of tokens with random token IDs."""
    mask = torch.rand(input_ids.shape, device=input_ids.device) < corruption_rate
    random_tokens = torch.randint(0, vocab_size, input_ids.shape, device=input_ids.device)
    return torch.where(mask, random_tokens, input_ids)

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Self-Flow self-distillation training for CRATE")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument("--source", type=str, default="base", help="checkpoint source")
parser.add_argument("--model-tag", type=str, default=None)
parser.add_argument("--model-step", type=int, default=None)
# Data
parser.add_argument("--prompt-task", type=str, default="smoltalk",
                    help="task name for training data (smoltalk|gsm8k|mmlu|codestack|...)")
parser.add_argument("--max-examples", type=int, default=5000,
                    help="max examples to load from the task")
# Self-Flow architecture
parser.add_argument("--student-layer", type=int, default=None,
                    help="layer to extract student features (default: n_layer // 3)")
parser.add_argument("--teacher-layer", type=int, default=None,
                    help="layer to extract teacher features (default: 2 * n_layer // 3)")
parser.add_argument("--student-corruption", type=float, default=0.75,
                    help="fraction of tokens replaced for student input")
parser.add_argument("--teacher-corruption", type=float, default=0.25,
                    help="fraction of tokens replaced for teacher input")
parser.add_argument("--distill-weight", type=float, default=1.0,
                    help="weight for distillation loss relative to LM loss")
parser.add_argument("--ema-decay", type=float, default=0.999,
                    help="EMA decay rate for teacher model")
# Training
parser.add_argument("--num-train-steps", type=int, default=500)
parser.add_argument("--device-batch-size", type=int, default=4)
parser.add_argument("--train-batch-size", type=int, default=None)
parser.add_argument("--target-examples-per-step", type=int, default=32)
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2)
parser.add_argument("--unembedding-lr", type=float, default=0.004)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--init-lr-frac", type=float, default=0.02)
# Evaluation
parser.add_argument("--eval-every", type=int, default=50)
parser.add_argument("--eval-steps", type=int, default=20)
# Output
parser.add_argument("--save-dir", type=str, default="selfflow_checkpoints")

args = parser.parse_args()
if args.train_batch_size is None:
    args.train_batch_size = args.device_batch_size
user_config = vars(args).copy()

# ---------------------------------------------------------------------------
# Compute init
# ---------------------------------------------------------------------------

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    if device_type == "cuda" else nullcontext()
)

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = (
    DummyWandb() if use_dummy_wandb
    else wandb.init(project="nanochat-selfflow", name=args.run, config=user_config, save_code=True)
)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

print0("Loading student model...")
model, tokenizer, meta = load_model(
    args.source, device, phase="train",
    model_tag=args.model_tag, step=args.model_step,
)
n_layer = model.config.n_layer
n_embd = model.config.n_embd
vocab_size = model.config.vocab_size

student_layer = args.student_layer if args.student_layer is not None else n_layer // 3
teacher_layer = args.teacher_layer if args.teacher_layer is not None else 2 * n_layer // 3
assert 0 <= student_layer < n_layer, f"student_layer {student_layer} out of range [0, {n_layer})"
assert 0 <= teacher_layer < n_layer, f"teacher_layer {teacher_layer} out of range [0, {n_layer})"
print0(f"Self-Flow layers: student={student_layer}, teacher={teacher_layer} (of {n_layer})")

# ---------------------------------------------------------------------------
# Create EMA teacher + projection head
# ---------------------------------------------------------------------------

print0("Creating EMA teacher (deep copy)...")
ema_model = create_ema_model(model)

proj_head = ProjectionHead(n_embd).to(device)
if ptdtype == torch.bfloat16:
    proj_head = proj_head.to(dtype=torch.bfloat16)

if device_type == "cuda":
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Build training data
# ---------------------------------------------------------------------------

def _build_task(name, stop):
    if name == "smoltalk":
        from tasks.smoltalk import SmolTalk
        return SmolTalk(split="train", stop=stop)
    elif name == "gsm8k":
        from tasks.gsm8k import GSM8K
        return GSM8K(subset="main", split="train")
    elif name == "mmlu":
        from tasks.mmlu import MMLU
        return MMLU(subset="auxiliary_train", split="train", stop=stop)
    elif name == "codestack":
        from tasks.codestack import CodeStack
        return CodeStack(split="train", stop=stop)
    else:
        raise ValueError(f"Unknown task: {name}")

train_ds = _build_task(args.prompt_task, args.max_examples)
print0(f"Loaded {len(train_ds)} training examples from {args.prompt_task}")

def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, :n-1] = row_targets
        return inputs.to(device), targets.to(device)
    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

# ---------------------------------------------------------------------------
# Gradient accumulation setup
# ---------------------------------------------------------------------------

train_bs = args.train_batch_size
examples_per_step = train_bs * ddp_world_size
assert args.target_examples_per_step % examples_per_step == 0, \
    "target_examples_per_step must be divisible by train_batch_size * world_size"
grad_accum_steps = args.target_examples_per_step // examples_per_step
print0(f"Batch size: {train_bs}, grad accum: {grad_accum_steps}, effective: {args.target_examples_per_step}")

train_loader = sft_data_generator(train_ds, batch_size=train_bs)

# ---------------------------------------------------------------------------
# Setup optimizers (student model + projection head)
# ---------------------------------------------------------------------------

model_optimizers = model.setup_optimizers(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)
proj_optimizer = torch.optim.AdamW(
    proj_head.parameters(), lr=args.matrix_lr * args.init_lr_frac,
    betas=(0.8, 0.95), eps=1e-10,
)
proj_optimizer.param_groups[0]["initial_lr"] = proj_optimizer.param_groups[0]["lr"]

for opt in model_optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]

all_optimizers = model_optimizers + [proj_optimizer]

def get_lr_multiplier(it):
    return 1.0 - it / args.num_train_steps

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

print0(f"\n{'='*60}")
print0(f"Self-Flow Training")
print0(f"  Student corruption: {args.student_corruption}")
print0(f"  Teacher corruption: {args.teacher_corruption}")
print0(f"  Distillation weight: {args.distill_weight}")
print0(f"  EMA decay: {args.ema_decay}")
print0(f"  Steps: {args.num_train_steps}")
print0(f"{'='*60}\n")

for step in range(args.num_train_steps):
    last_step = step == args.num_train_steps - 1

    # --- Periodic validation (on clean data, LM loss only) ---
    if last_step or step % args.eval_every == 0:
        model.eval()
        val_loader = sft_data_generator(train_ds, batch_size=train_bs)
        losses = []
        for _ in range(args.eval_steps):
            val_inputs, val_targets = next(val_loader)
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
            losses.append(loss)
        val_loss = torch.stack(losses).mean()
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"  Step {step:05d} | Val LM loss: {val_loss.item():.6f}")
        wandb_run.log({"step": step, "val_lm_loss": val_loss.item()})
        model.train()

    if last_step:
        break

    # --- Forward/backward with gradient accumulation ---
    total_lm_loss = 0.0
    total_distill_loss = 0.0
    for micro_step in range(grad_accum_steps):
        inputs, targets = next(train_loader)

        student_inputs = corrupt_tokens(inputs, args.student_corruption, vocab_size)
        teacher_inputs = corrupt_tokens(inputs, args.teacher_corruption, vocab_size)

        with autocast_ctx:
            lm_loss, student_hidden = model(
                student_inputs, targets, return_hidden_at=student_layer,
            )

            with torch.no_grad():
                _, teacher_hidden = ema_model(
                    teacher_inputs, return_hidden_at=teacher_layer,
                )

            projected = proj_head(student_hidden)
            distill_loss = F.mse_loss(projected, teacher_hidden.detach())

            loss = (lm_loss + args.distill_weight * distill_loss) / grad_accum_steps

        loss.backward()
        total_lm_loss += lm_loss.detach().item()
        total_distill_loss += distill_loss.detach().item()

    # --- LR schedule ---
    lrm = get_lr_multiplier(step)
    for opt in all_optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # --- Optimizer step ---
    for opt in all_optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    proj_head.zero_grad(set_to_none=True)

    # --- EMA update ---
    update_ema(model, ema_model, args.ema_decay)

    avg_lm = total_lm_loss / grad_accum_steps
    avg_distill = total_distill_loss / grad_accum_steps
    combined = avg_lm + args.distill_weight * avg_distill
    print0(
        f"  Step {step:05d}/{args.num_train_steps:05d} | "
        f"LM: {avg_lm:.4f} | Distill: {avg_distill:.4f} | "
        f"Combined: {combined:.4f} | lrm: {lrm:.4f}"
    )
    wandb_run.log({
        "step": step,
        "lm_loss": avg_lm,
        "distill_loss": avg_distill,
        "combined_loss": combined,
        "lrm": lrm,
    })

# ---------------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------------

print0("\nSaving Self-Flow checkpoint...")
base_dir = get_base_dir()
model_tag = args.model_tag or "default"
checkpoint_dir = os.path.join(base_dir, args.save_dir, model_tag)

save_meta = {
    "step": args.num_train_steps,
    "model_config": meta.get("model_config", {}),
    "user_config": user_config,
    "selfflow_config": {
        "student_layer": student_layer,
        "teacher_layer": teacher_layer,
        "student_corruption": args.student_corruption,
        "teacher_corruption": args.teacher_corruption,
        "distill_weight": args.distill_weight,
        "ema_decay": args.ema_decay,
    },
}

save_checkpoint(
    checkpoint_dir,
    step=args.num_train_steps,
    model_data=model.state_dict(),
    optimizer_data=None,
    meta_data=save_meta,
    rank=ddp_rank,
)

proj_path = os.path.join(checkpoint_dir, f"proj_head_{args.num_train_steps:06d}.pt")
if master_process:
    torch.save(proj_head.state_dict(), proj_path)
    print0(f"Saved projection head to {proj_path}")

print0(f"Self-Flow training complete. Checkpoint at: {checkpoint_dir}")

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

for opt in all_optimizers:
    del opt
del all_optimizers, ema_model, proj_head
gc.collect()
if device_type == "cuda":
    torch.cuda.empty_cache()

wandb_run.finish()
compute_cleanup()
