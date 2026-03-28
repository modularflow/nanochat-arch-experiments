"""
Self-Flow pretraining from scratch (or from checkpoint).

Ground-up Self-Flow: trains a SelfFlowCRATE model with dual-timestep scheduling,
pluggable corruption, per-token conditioning, and multi-scale representation
alignment -- all as first-class training objectives alongside LM loss.

Usage:
    python -m scripts.self_flow_pretrain --depth 12 --num-iterations 5000

    torchrun --nproc_per_node=8 -m scripts.self_flow_pretrain --depth 20

Reference:
    Chefer et al., "Self-Supervised Flow Matching for Scalable Multi-Modal
    Synthesis" (Black Forest Labs, 2026)
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import copy
import argparse
import gc
import time
from contextlib import nullcontext

import wandb
import torch
import torch.nn.functional as F
import torch.distributed as dist

from nanochat.self_flow_model import SelfFlowCRATE, SelfFlowConfig
from nanochat.dual_timestep import DualTimestepScheduler
from nanochat.corruption import build_corruption_strategy
from nanochat.adversarial import (
    ADVERSARIAL_MODES, build_adversarial, build_adversarial_optimizer,
    GradientAdversarial, AdversarialCorrupter, AdversarialMaskPredictor,
    DiscriminativeHead,
)
from nanochat.forget import (
    FORGET_MODES, build_forget_module, build_forget_optimizer, parse_layer_set,
)
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.common import (
    compute_init, compute_cleanup, print0, DummyWandb, print_banner,
    get_base_dir, autodetect_device_type,
)
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb

print_banner()

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Self-Flow pretraining for CRATE")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps")

# Model architecture
parser.add_argument("--depth", type=int, default=12, help="transformer depth")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension")
parser.add_argument("--max-seq-len", type=int, default=1024, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern")

# Self-Flow architecture
parser.add_argument("--student-layers", type=str, default="", help="comma-separated student layer indices (auto if empty)")
parser.add_argument("--teacher-layers", type=str, default="", help="comma-separated teacher layer indices (auto if empty)")
parser.add_argument("--proj-hidden-mult", type=int, default=2, help="projection head hidden dim multiplier")
parser.add_argument("--corruption-conditioning", action="store_true", default=True, help="use per-token corruption conditioning")
parser.add_argument("--no-corruption-conditioning", dest="corruption_conditioning", action="store_false")
parser.add_argument("--corruption-strategy", type=str, default="embedding_interpolation",
                    choices=["embedding_interpolation", "token_replacement", "mask", "span", "composite_interp_replace"],
                    help="corruption strategy")
parser.add_argument("--rep-loss-type", type=str, default="cosine", choices=["cosine", "mse", "smooth_l1"])
parser.add_argument("--rep-loss-weight", type=float, default=1.0, help="gamma: weight for representation loss")

# Dual-Timestep Scheduler
parser.add_argument("--noise-distribution", type=str, default="uniform",
                    choices=["uniform", "logit_normal", "beta", "cosine"])
parser.add_argument("--mask-ratio", type=float, default=0.5, help="R_M: fraction of tokens getting secondary timestep")
parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay for teacher")

# Adversarial training
parser.add_argument("--adversarial", type=str, default="none", choices=ADVERSARIAL_MODES,
                    help="adversarial mode: none|corrupter|mask|gradient|discriminator")
parser.add_argument("--adv-lr", type=float, default=0.0003, help="adversarial module learning rate")
parser.add_argument("--adv-weight", type=float, default=0.5, help="weight for adversarial loss contribution")
parser.add_argument("--adv-hidden-mult", type=int, default=1, help="hidden dim multiplier for adversarial networks")
parser.add_argument("--adv-epsilon", type=float, default=0.1, help="L-inf epsilon for gradient adversarial (PGD)")
parser.add_argument("--adv-step-size", type=float, default=0.03, help="PGD step size for gradient adversarial")
parser.add_argument("--adv-pgd-steps", type=int, default=3, help="number of PGD inner steps for gradient adversarial")
parser.add_argument("--adv-update-every", type=int, default=1,
                    help="update adversary every N student steps (for stability)")

# Forgetting mechanism (adversarial context compression)
parser.add_argument("--forget", type=str, default="none", choices=FORGET_MODES,
                    help="forgetting mode: none|forget_gate|context_bottleneck|selective_retention")
parser.add_argument("--forget-layers", type=str, default="all",
                    help="comma-separated layer indices for forgetting, or 'all' for every layer")
parser.add_argument("--forget-weight", type=float, default=0.1,
                    help="weight for adversarial forgetting/sparsity loss")
parser.add_argument("--forget-lr", type=float, default=0.001,
                    help="learning rate for forgetting module parameters")
parser.add_argument("--forget-compression-ratio", type=int, default=4,
                    help="compression ratio for context_bottleneck mode")

# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1)
parser.add_argument("--target-param-data-ratio", type=int, default=8, help="data:param ratio (-1 = disable)")

# Optimization
parser.add_argument("--device-batch-size", type=int, default=16, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=65536, help="total batch size in tokens")
parser.add_argument("--embedding-lr", type=float, default=0.3)
parser.add_argument("--unembedding-lr", type=float, default=0.004)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--scalar-lr", type=float, default=0.5)
parser.add_argument("--proj-lr", type=float, default=0.001, help="LR for projection heads + conditioning MLP")
parser.add_argument("--weight-decay", type=float, default=0.2)
parser.add_argument("--warmup-ratio", type=float, default=0.0)
parser.add_argument("--warmdown-ratio", type=float, default=0.4)
parser.add_argument("--final-lr-frac", type=float, default=0.0)
parser.add_argument("--resume-from-step", type=int, default=-1)

# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="eval val bpb every N steps")
parser.add_argument("--eval-tokens", type=int, default=20*524288)
parser.add_argument("--sample-every", type=int, default=2000)
parser.add_argument("--save-every", type=int, default=-1)

# Output
parser.add_argument("--model-tag", type=str, default=None)

args = parser.parse_args()
user_config = vars(args).copy()

# ---------------------------------------------------------------------------
# Compute init
# ---------------------------------------------------------------------------
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanochat-selfflow-pretrain", name=args.run, config=user_config, save_code=True
)

# ---------------------------------------------------------------------------
# Tokenizer + model dimensions
# ---------------------------------------------------------------------------
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

num_layers = args.depth
model_dim = args.depth * args.aspect_ratio

def find_num_heads(model_dim, target_head_dim):
    ideal = max(1, round(model_dim / target_head_dim))
    for offset in range(model_dim):
        for candidate in [ideal + offset, ideal - offset]:
            if candidate > 0 and model_dim % candidate == 0:
                return candidate
    return 1

num_heads = find_num_heads(model_dim, args.head_dim)
print0(f"Model: depth={num_layers}, dim={model_dim}, heads={num_heads}")

# ---------------------------------------------------------------------------
# Batch size / gradient accumulation
# ---------------------------------------------------------------------------
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Batch: {args.device_batch_size}x{args.max_seq_len} * {ddp_world_size} ranks * {grad_accum_steps} accum = {args.total_batch_size:,} tokens/step")

batch_lr_scale = 1.0
reference_batch_size = 2**19
batch_ratio = args.total_batch_size / reference_batch_size
if batch_ratio != 1.0:
    batch_lr_scale = batch_ratio ** 0.5
    print0(f"Scaling LRs by {batch_lr_scale:.4f}")

weight_decay_scaled = args.weight_decay * (12 / args.depth) ** 2

# ---------------------------------------------------------------------------
# Build SelfFlowCRATE model
# ---------------------------------------------------------------------------
model_config_kwargs = dict(
    sequence_len=args.max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_heads,
    n_embd=model_dim,
    window_pattern=args.window_pattern,
    # Self-flow specific
    student_layers=args.student_layers,
    teacher_layers=args.teacher_layers,
    proj_hidden_mult=args.proj_hidden_mult,
    corruption_conditioning=args.corruption_conditioning,
    corruption_strategy=args.corruption_strategy,
    rep_loss_type=args.rep_loss_type,
    rep_loss_weight=args.rep_loss_weight,
)

with torch.device("meta"):
    config = SelfFlowConfig(**model_config_kwargs)
    model = SelfFlowCRATE(config)
model.to_empty(device=device)
model.init_weights()

# Resume from checkpoint if requested
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"
checkpoint_dir = os.path.join(base_dir, "selfflow_pretrain_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank
    )
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data

orig_model = model
model = torch.compile(model, dynamic=False)

num_params = sum(p.numel() for p in model.parameters())
print0(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
num_flops_per_token = orig_model.estimate_flops()

# ---------------------------------------------------------------------------
# EMA teacher (deep copy, frozen)
# ---------------------------------------------------------------------------
print0("Creating EMA teacher (deep copy of student)...")

@torch.no_grad()
def create_ema_model(model):
    ema = copy.deepcopy(model)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad = False
    return ema

@torch.no_grad()
def update_ema(student, teacher, decay):
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)

ema_model = create_ema_model(orig_model)
if device_type == "cuda":
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Corruption strategy + scheduler
# ---------------------------------------------------------------------------
corruption = build_corruption_strategy(
    args.corruption_strategy,
    vocab_size=config.vocab_size,
    n_embd=config.n_embd,
    device=device,
    dtype=torch.bfloat16 if device_type == "cuda" else torch.float32,
)
print0(f"Corruption strategy: {args.corruption_strategy}")

scheduler = DualTimestepScheduler(
    distribution=args.noise_distribution,
    mask_ratio=args.mask_ratio,
)
print0(f"Scheduler: distribution={args.noise_distribution}, mask_ratio={args.mask_ratio}")

# ---------------------------------------------------------------------------
# Adversarial component (optional)
# ---------------------------------------------------------------------------
adversarial = build_adversarial(
    args.adversarial, n_embd=config.n_embd,
    device=device,
    dtype=torch.bfloat16 if device_type == "cuda" else torch.float32,
    adv_hidden_mult=args.adv_hidden_mult,
    adv_epsilon=args.adv_epsilon,
    adv_step_size=args.adv_step_size,
    adv_pgd_steps=args.adv_pgd_steps,
)
adv_optimizer = build_adversarial_optimizer(adversarial, lr=args.adv_lr)
if adversarial is not None:
    print0(f"Adversarial: mode={args.adversarial}, weight={args.adv_weight}, lr={args.adv_lr}")
else:
    print0(f"Adversarial: disabled")

# ---------------------------------------------------------------------------
# Forgetting mechanism (adversarial context compression)
# ---------------------------------------------------------------------------
forget_active_layers = parse_layer_set(args.forget_layers, num_layers)
forget_module = build_forget_module(
    args.forget, n_embd=config.n_embd, n_layer=num_layers,
    active_layers=forget_active_layers,
    device=device,
    dtype=torch.bfloat16 if device_type == "cuda" else torch.float32,
    compression_ratio=args.forget_compression_ratio,
)
forget_optimizer = build_forget_optimizer(forget_module, lr=args.forget_lr)
if forget_module is not None:
    layers_desc = "all" if forget_active_layers is None else sorted(forget_active_layers)
    print0(f"Forget: mode={args.forget}, layers={layers_desc}, weight={args.forget_weight}, lr={args.forget_lr}")
else:
    print0(f"Forget: disabled")

# ---------------------------------------------------------------------------
# Training iterations
# ---------------------------------------------------------------------------
num_scaling_params = orig_model.num_scaling_params()
assert args.num_iterations > 0 or args.target_param_data_ratio > 0
if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print0(f"Training for {num_iterations:,} iterations (user-specified)")
elif args.target_param_data_ratio > 0:
    target_tokens = args.target_param_data_ratio * num_scaling_params
    num_iterations = target_tokens // args.total_batch_size
    print0(f"Training for {num_iterations:,} iterations (ratio={args.target_param_data_ratio})")
else:
    raise ValueError("Specify --num-iterations or --target-param-data-ratio")

total_tokens = args.total_batch_size * num_iterations
print0(f"Total tokens: {total_tokens:,}, Tokens:Params = {total_tokens / num_scaling_params:.1f}")

# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------
adam_betas = (0.8, 0.95)
backbone_opts, selfflow_opt = orig_model.setup_optimizers(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    adam_betas=adam_betas,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    proj_lr=args.proj_lr * batch_lr_scale,
)
all_optimizers = list(backbone_opts) + [selfflow_opt]

if resuming:
    for opt, dat in zip(all_optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data

# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------
dataloader_resume = None if not resuming else meta_data.get("dataloader_state_dict")
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len,
    split="train", device=device, resume_state_dict=dataloader_resume,
)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device,
)
x, y, dataloader_state_dict = next(train_loader)

# ---------------------------------------------------------------------------
# LR / momentum schedulers
# ---------------------------------------------------------------------------

def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# ---------------------------------------------------------------------------
# Loop state
# ---------------------------------------------------------------------------
if not resuming:
    step = 0
    val_bpb = None
    min_val_bpb = float("inf")
    smooth_lm_loss = 0.0
    smooth_rep_loss = 0.0
    total_training_time = 0.0
else:
    step = meta_data["step"]
    loop_state = meta_data.get("loop_state", {})
    val_bpb = meta_data.get("val_bpb")
    min_val_bpb = loop_state.get("min_val_bpb", float("inf"))
    smooth_lm_loss = loop_state.get("smooth_lm_loss", 0.0)
    smooth_rep_loss = loop_state.get("smooth_rep_loss", 0.0)
    total_training_time = loop_state.get("total_training_time", 0.0)

# We need the uncompiled model's teacher_layer_indices for the teacher forward
teacher_layer_indices = orig_model.teacher_layer_indices

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print0(f"\n{'='*60}")
print0(f"Self-Flow Pretraining")
print0(f"  Architecture: SelfFlowCRATE d{num_layers} ({num_params:,} params)")
print0(f"  Student layers: {orig_model.student_layer_indices}")
print0(f"  Teacher layers: {orig_model.teacher_layer_indices}")
print0(f"  Corruption: {args.corruption_strategy}")
print0(f"  Conditioning: {args.corruption_conditioning}")
print0(f"  Rep loss: {args.rep_loss_type} (weight={args.rep_loss_weight})")
print0(f"  Scheduler: {args.noise_distribution} (mask_ratio={args.mask_ratio})")
print0(f"  EMA decay: {args.ema_decay}")
print0(f"  Adversarial: {args.adversarial}" + (f" (weight={args.adv_weight}, lr={args.adv_lr})" if args.adversarial != "none" else ""))
print0(f"  Forget: {args.forget}" + (f" (weight={args.forget_weight}, lr={args.forget_lr}, layers={args.forget_layers})" if args.forget != "none" else ""))
print0(f"  Steps: {num_iterations:,}")
print0(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
while True:
    last_step = step == num_iterations

    # --- Evaluation: val bpb (clean forward, no corruption) ---
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Val bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({"step": step, "val/bpb": val_bpb})
        model.train()

    # --- Sample from model (clean forward) ---
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        from nanochat.engine import Engine
        engine = Engine(orig_model, tokenizer)
        prompts = [
            "The capital of France is",
            "The planets of the solar system are:",
            "If 5*x + 3 = 13, then x is",
        ]
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # --- Save checkpoint ---
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        selfflow_config_dict = {
            "student_layers": args.student_layers,
            "teacher_layers": args.teacher_layers,
            "proj_hidden_mult": args.proj_hidden_mult,
            "corruption_conditioning": args.corruption_conditioning,
            "cond_hidden_mult": config.cond_hidden_mult,
            "corruption_strategy": args.corruption_strategy,
            "rep_loss_type": args.rep_loss_type,
            "rep_loss_weight": args.rep_loss_weight,
        }
        save_checkpoint(
            checkpoint_dir, step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in all_optimizers],
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": model_config_kwargs,
                "selfflow_config": selfflow_config_dict,
                "user_config": user_config,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {
                    "min_val_bpb": min_val_bpb,
                    "smooth_lm_loss": smooth_lm_loss,
                    "smooth_rep_loss": smooth_rep_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )
        # Save EMA teacher separately
        if master_process:
            ema_path = os.path.join(checkpoint_dir, f"ema_{step:06d}.pt")
            torch.save(ema_model.state_dict(), ema_path)
            print0(f"Saved EMA teacher to {ema_path}")

    if last_step:
        break

    # -------------------------------------------------------------------
    # Single training step: Self-Flow dual-path forward
    # -------------------------------------------------------------------
    synchronize()
    t0 = time.time()

    total_lm = 0.0
    total_rep = 0.0
    total_adv = 0.0
    total_forget = 0.0

    for micro_step in range(grad_accum_steps):
        B, T = x.shape

        # 1) Sample dual timesteps
        dts = scheduler.sample(B, T, device)
        student_levels = dts.student_levels

        # 1b) Adversarial corruption level override
        if isinstance(adversarial, AdversarialCorrupter):
            with autocast_ctx:
                clean_emb = orig_model.backbone.transformer.wte(x)
                student_levels = adversarial(clean_emb.detach())
        elif isinstance(adversarial, AdversarialMaskPredictor):
            with autocast_ctx:
                clean_emb = orig_model.backbone.transformer.wte(x)
                # t and s are per-batch scalars broadcast to [B,T]
                t_broadcast = dts.student_levels  # base levels
                s_broadcast = dts.teacher_levels   # secondary levels
                student_levels = adversarial(clean_emb.detach(), t_broadcast, s_broadcast)

        with autocast_ctx:
            # 2) Student forward (corrupted input, with optional forgetting)
            lm_loss, student_hiddens = orig_model.forward_selfflow(
                student_ids=x,
                student_corruption_levels=student_levels,
                targets=y,
                corruption_strategy=corruption,
                forget_module=forget_module,
            )

            # 3) Teacher forward (uniform cleaner input)
            from nanochat.crate import norm as crate_norm
            clean_emb_t = ema_model.backbone.transformer.wte(x)
            corrupted_emb_t = corruption.corrupt(
                clean_emb_t, x, dts.teacher_levels,
                ema_model.backbone.transformer.wte,
            )
            corrupted_emb_t = crate_norm(corrupted_emb_t)
            if orig_model.use_conditioning:
                cond_bias_t = ema_model.corruption_conditioner(dts.teacher_levels)
                corrupted_emb_t = corrupted_emb_t + cond_bias_t

            with torch.no_grad():
                xt = corrupted_emb_t
                x0t = xt
                teacher_hiddens = {}
                target_layers_set = set(teacher_layer_indices)
                for i, block in enumerate(ema_model.backbone.transformer.h):
                    xt = ema_model.backbone.resid_lambdas[i] * xt + ema_model.backbone.x0_lambdas[i] * x0t
                    cos_sin_t = (ema_model.backbone.cos[:, :T], ema_model.backbone.sin[:, :T])
                    xt = block(xt, cos_sin_t, ema_model.backbone.window_sizes[i], None)
                    if i in target_layers_set:
                        teacher_hiddens[i] = xt

            # 4) Compute combined self-flow loss
            total_loss, metrics = orig_model.compute_selfflow_loss(
                student_hiddens, teacher_hiddens, lm_loss,
            )

            # 5) Adversarial loss additions
            adv_loss_val = 0.0

            if isinstance(adversarial, DiscriminativeHead):
                # Get first student/teacher hidden pair for the discriminator
                s_layer = orig_model.student_layer_indices[0]
                t_layer = orig_model.teacher_layer_indices[0]
                s_h = student_hiddens[s_layer]
                t_h = teacher_hiddens[t_layer]
                proj_s = orig_model.proj_heads[0](s_h)

                # Discriminator update (on detached hiddens)
                if step % args.adv_update_every == 0:
                    disc_loss, disc_metrics = adversarial.compute_discriminator_loss(
                        proj_s.detach(), t_h.detach()
                    )
                    disc_loss_scaled = disc_loss / grad_accum_steps
                    disc_loss_scaled.backward()
                    metrics.update(disc_metrics)

                # Generator loss for the student (fool discriminator)
                gen_loss = adversarial.compute_generator_loss(proj_s)
                total_loss = total_loss + args.adv_weight * gen_loss
                adv_loss_val = gen_loss.item()
                metrics["adv_gen_loss"] = adv_loss_val

            elif isinstance(adversarial, GradientAdversarial):
                # PGD perturbation on student embeddings -- re-run student forward
                # with adversarially perturbed embeddings
                clean_emb_pgd = orig_model.backbone.transformer.wte(x)
                base_corrupted = corruption.corrupt(
                    clean_emb_pgd, x, student_levels,
                    orig_model.backbone.transformer.wte,
                )
                base_corrupted = crate_norm(base_corrupted)
                if orig_model.use_conditioning:
                    base_corrupted = base_corrupted + orig_model.corruption_conditioner(student_levels)

                def pgd_loss_fn(perturbed_emb):
                    x_p = perturbed_emb
                    x0_p = x_p
                    pgd_hiddens = {}
                    tgt = set(orig_model.student_layer_indices)
                    for i, block in enumerate(orig_model.backbone.transformer.h):
                        x_p = orig_model.backbone.resid_lambdas[i] * x_p + orig_model.backbone.x0_lambdas[i] * x0_p
                        cos_sin_p = (orig_model.backbone.cos[:, :T], orig_model.backbone.sin[:, :T])
                        x_p = block(x_p, cos_sin_p, orig_model.backbone.window_sizes[i], None)
                        if i in tgt:
                            pgd_hiddens[i] = x_p
                    rep_l = 0.0
                    for idx, (sl, tl) in enumerate(zip(orig_model.student_layer_indices, orig_model.teacher_layer_indices)):
                        from nanochat.self_flow_model import compute_rep_loss
                        rep_l = rep_l + compute_rep_loss(
                            pgd_hiddens[sl], teacher_hiddens[tl],
                            orig_model.proj_heads[idx],
                            loss_type=orig_model.config.rep_loss_type,
                        )
                    return -rep_l  # negate: PGD maximizes, but perturb() calls .backward() assuming maximization

                adv_emb = adversarial.perturb(base_corrupted, pgd_loss_fn)
                # Re-run student with perturbed embeddings to get adversarial rep loss
                x_adv = adv_emb
                x0_adv = x_adv
                adv_hiddens = {}
                for i, block in enumerate(orig_model.backbone.transformer.h):
                    x_adv = orig_model.backbone.resid_lambdas[i] * x_adv + orig_model.backbone.x0_lambdas[i] * x0_adv
                    cos_sin_a = (orig_model.backbone.cos[:, :T], orig_model.backbone.sin[:, :T])
                    x_adv = block(x_adv, cos_sin_a, orig_model.backbone.window_sizes[i], None)
                    if i in set(orig_model.student_layer_indices):
                        adv_hiddens[i] = x_adv
                adv_rep_loss = 0.0
                for idx, (sl, tl) in enumerate(zip(orig_model.student_layer_indices, orig_model.teacher_layer_indices)):
                    from nanochat.self_flow_model import compute_rep_loss
                    adv_rep_loss = adv_rep_loss + compute_rep_loss(
                        adv_hiddens[sl], teacher_hiddens[tl],
                        orig_model.proj_heads[idx],
                        loss_type=orig_model.config.rep_loss_type,
                    )
                total_loss = total_loss + args.adv_weight * adv_rep_loss
                adv_loss_val = adv_rep_loss.item()
                metrics["adv_pgd_loss"] = adv_loss_val

            elif isinstance(adversarial, (AdversarialCorrupter, AdversarialMaskPredictor)):
                # Adversary update: maximize rep loss w.r.t. adversary params
                if step % args.adv_update_every == 0:
                    adv_rep = metrics.get("rep_loss", 0.0)
                    # We need to compute a fresh adversarial loss for the adversary's gradient
                    # (the rep_loss in metrics was already computed; we reuse total_loss's graph)
                    adv_loss_for_update = -metrics["rep_loss"]  # maximize = minimize negative
                    adv_loss_val = metrics["rep_loss"]
                    metrics["adv_corrupter_loss"] = adv_loss_val

            # 6) Forgetting loss (adversarial context compression)
            forget_loss_val = 0.0
            if forget_module is not None:
                forget_loss, forget_metrics = forget_module.compute_forget_loss()
                total_loss = total_loss + args.forget_weight * forget_loss
                forget_loss_val = forget_metrics["forget_loss"]
                metrics.update({f"forget/{k}": v for k, v in forget_metrics.items()})

            scaled_loss = total_loss / grad_accum_steps

        scaled_loss.backward()
        total_lm += metrics["lm_loss"]
        total_rep += metrics["rep_loss"]
        total_adv += adv_loss_val
        total_forget += forget_loss_val

        # Prefetch next batch
        x, y, dataloader_state_dict = next(train_loader)

    # --- LR schedule ---
    lrm = get_lr_multiplier(step)
    for opt in all_optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # Muon-specific scheduling (backbone_opts[1] is muon)
    muon_optimizer = backbone_opts[1]
    muon_momentum = get_muon_momentum(step)
    muon_wd = get_weight_decay(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
        group["weight_decay"] = muon_wd

    # --- Optimizer step ---
    for opt in all_optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    # --- Adversarial optimizer step ---
    if adv_optimizer is not None and step % args.adv_update_every == 0:
        for group in adv_optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        adv_optimizer.step()
        adv_optimizer.zero_grad(set_to_none=True)

    # --- Forget module optimizer step ---
    if forget_optimizer is not None:
        for group in forget_optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        forget_optimizer.step()
        forget_optimizer.zero_grad(set_to_none=True)

    # --- EMA update ---
    update_ema(orig_model, ema_model, args.ema_decay)

    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # --- Logging ---
    avg_lm = total_lm / grad_accum_steps
    avg_rep = total_rep / grad_accum_steps
    avg_adv = total_adv / grad_accum_steps
    avg_forget = total_forget / grad_accum_steps
    ema_beta = 0.9
    smooth_lm_loss = ema_beta * smooth_lm_loss + (1 - ema_beta) * avg_lm
    smooth_rep_loss = ema_beta * smooth_rep_loss + (1 - ema_beta) * avg_rep
    debiased_lm = smooth_lm_loss / (1 - ema_beta ** (step + 1))
    debiased_rep = smooth_rep_loss / (1 - ema_beta ** (step + 1))

    tok_per_sec = int(args.total_batch_size / dt) if dt > 0 else 0
    if step > 10:
        total_training_time += dt

    pct_done = 100 * step / num_iterations
    steps_done = step - 10
    if steps_done > 0:
        eta_seconds = (num_iterations - step) * (total_training_time / steps_done)
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""

    adv_str = f" | adv: {avg_adv:.4f}" if adversarial is not None else ""
    forget_str = f" | fgt: {avg_forget:.4f}" if forget_module is not None else ""
    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.1f}%) | "
        f"lm: {debiased_lm:.4f} | rep: {debiased_rep:.4f}{adv_str}{forget_str} | "
        f"lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/s: {tok_per_sec:,}{eta_str}"
    )

    if step % 100 == 0:
        log_data = {
            "step": step,
            "train/lm_loss": debiased_lm,
            "train/rep_loss": debiased_rep,
            "train/combined_loss": debiased_lm + args.rep_loss_weight * debiased_rep,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
        }
        if adversarial is not None:
            log_data["train/adv_loss"] = avg_adv
        if forget_module is not None:
            log_data["train/forget_loss"] = avg_forget
        wandb_run.log(log_data)

    step += 1

# ---------------------------------------------------------------------------
# Final stats
# ---------------------------------------------------------------------------
print0(f"\nPeak memory: {get_max_memory() / 1024 / 1024:.0f}MiB")
print0(f"Total training time: {total_training_time/60:.1f}m")
if val_bpb is not None:
    print0(f"Min val bpb: {min_val_bpb:.6f}")

# Cleanup
for opt in all_optimizers:
    del opt
if adv_optimizer is not None:
    del adv_optimizer
if adversarial is not None:
    del adversarial
if forget_optimizer is not None:
    del forget_optimizer
if forget_module is not None:
    del forget_module
del all_optimizers, ema_model
gc.collect()
if device_type == "cuda":
    torch.cuda.empty_cache()

wandb_run.finish()
compute_cleanup()
