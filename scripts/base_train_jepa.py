"""
Train model. From root directory of the project, run as:

python -m scripts.base_train_jepa

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train_jepa

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train_jepa --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import time
import warnings
from contextlib import nullcontext

import wandb
import torch
import torch.distributed as dist

from nanochat.crate import CRATE, CRATEConfig
from nanochat.gpt import GPT, GPTConfig
from nanochat.noq_gpt import NoQGPT, NoQGPTConfig
from nanochat.noq_crate import NoQCRATE, NoQCRATEConfig
from nanochat.rys_gpt import RYSGPT, RYSGPTConfig
from nanochat.trm_gpt import TRMGPT, TRMGPTConfig
from nanochat.tpa_gpt import TPAGPT, TPAGPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.jepa import (
    get_backbone, ensure_pred_token_slot, forward_final_hidden,
    compute_jepa_loss, compute_jepa_loss_for_batch, split_views,
    get_jepa_lambda, JEPA_SCHEDULES, resize_model_vocab,
)
from scripts.base_eval import evaluate_model
print_banner()


# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model with an auxiliary JEPA loss")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model architecture
parser.add_argument("--architecture", type=str, default="gpt", choices=["crate", "gpt", "noq_gpt", "noq_crate", "rys_gpt", "trm_gpt", "tpa_gpt"],
    help="Backbone: gpt (default), or crate (deprecated — worse empirical results; kept for legacy checkpoints), "
         "noq_gpt, noq_crate, rys_gpt, trm_gpt, tpa_gpt (Tensor Product Attention, arXiv:2501.06425)")
parser.add_argument("--tpa-rank-q", type=int, default=6,
    help="TPA: rank of the Q tensor-product factorization (paper default: == n_head for full Q expressiveness).")
parser.add_argument("--tpa-rank-k", type=int, default=2,
    help="TPA: rank of the K tensor-product factorization (paper default: 2 — gives the KV-cache compression).")
parser.add_argument("--tpa-rank-v", type=int, default=2,
    help="TPA: rank of the V tensor-product factorization (paper default: 2).")
parser.add_argument("--rys-block-start", type=int, default=3, help="RYS: first unique block in repeated section (inclusive)")
parser.add_argument("--rys-block-end", type=int, default=6, help="RYS: end of repeated section (exclusive)")
parser.add_argument("--rys-num-repeats", type=int, default=2, help="RYS: times the middle block is traversed")
parser.add_argument("--rys-blocks", type=str, default="", help="RYS multi-block: 'start1,end1;start2,end2;...' (overrides single-block params)")
parser.add_argument("--rys-frac-recur-start", type=float, default=0.0,
    help="RYS: fraction of training (0.0-1.0) before recurrence activates. "
         "0 (default) = recurrence always on. e.g. 0.35 disables recurrence for the first 35%% of steps.")
parser.add_argument("--trm-n-recur", type=int, default=6, help="TRM: recursions per cycle (each traverses all unique blocks)")
parser.add_argument("--trm-T-cycles", type=int, default=3, help="TRM: total recursion cycles (T-1 without grad during training)")
parser.add_argument("--parallel-residual", action="store_true",
    help="GPT: GPT-J-style 2-lane residual; attn reads lane A, MLP reads lane B, mixed via learned 2x2.")
# MuonEqR optimizer mode (parameter-golf SOTA recipe).
parser.add_argument("--muon-mode", type=str, default="default", choices=["default", "eqr"],
    help="Muon variant: 'default' (NorMuon variance reduction + cautious WD) or 'eqr' "
         "(parameter-golf MuonEqR: row-normalize gradient before NS, decoupled WD, no var-reduction).")
# EMA shadow weights (saved alongside the regular checkpoint, evaluated on val).
parser.add_argument("--ema-decay", type=float, default=0.0,
    help="Per-step EMA decay for a shadow copy of model weights (0 = disabled). 0.997 ≈ 333-step half-life.")
parser.add_argument("--ema-warmup-steps", type=int, default=100,
    help="When --ema-decay > 0, hard-copy weights for the first N steps before EMA kicks in.")
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument(
    "--num-kv-heads", type=int, default=None,
    help="KV heads for Grouped-Query Attention (default: same as query heads). "
         "Must divide num_heads. GQA requires --architecture gpt|noq_gpt|rys_gpt|trm_gpt (not CRATE/NoQCRATE).",
)
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=int, default=8, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=524288, help="total batch size in tokens")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.2, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.4, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
parser.add_argument("--jepa-lambda", type=float, default=0.25,
    help="Weight of JEPA loss (0 = disabled). Recommend 0.1-0.25 for pretraining.")
parser.add_argument("--jepa-schedule", type=str, default="constant", choices=JEPA_SCHEDULES,
    help="Lambda schedule: constant (default), linear_decay (→0), cosine_decay (→0)")
parser.add_argument("--jepa-dropout", type=float, default=0.5,
    help="Fraction of micro-batches to skip JEPA loss. 0.5 = compute parity.")
parser.add_argument("--jepa-view-min-len", type=int, default=64,
    help="Minimum tokens per view. Sequences shorter than 2x this are skipped.")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=20*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
if args.architecture == "crate":
    warnings.warn(
        "The 'crate' (CRATE-α) architecture is deprecated for new runs — it has consistently "
        "underperformed vanilla GPT in this fork. Prefer --architecture gpt. "
        "CRATE remains supported for loading old checkpoints and reproduction.",
        FutureWarning,
        stacklevel=1,
    )
user_config = vars(args).copy()  # for logging
jepa_base_lambda = args.jepa_lambda
jepa_schedule = args.jepa_schedule
jepa_dropout = args.jepa_dropout
use_jepa = jepa_base_lambda > 0.0
assert 0.0 <= jepa_dropout <= 1.0
# EMA bookkeeping
ema_decay = float(args.ema_decay)
use_ema = ema_decay > 0.0
assert 0.0 <= ema_decay < 1.0, f"--ema-decay must be in [0, 1), got {ema_decay}"
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model kwargs are derived from the desired depth of the model
num_layers = args.depth
model_dim = args.depth * args.aspect_ratio
def find_num_heads(model_dim, target_head_dim):
    # Find num_heads that divides model_dim evenly, with head_dim closest to target.
    ideal = max(1, round(model_dim / target_head_dim))
    for offset in range(model_dim):
        for candidate in [ideal + offset, ideal - offset]:
            if candidate > 0 and model_dim % candidate == 0:
                return candidate
    return 1
num_heads = find_num_heads(model_dim, args.head_dim)
if args.num_kv_heads is None:
    num_kv_heads = num_heads  # 1:1 — GQA disabled
else:
    num_kv_heads = args.num_kv_heads
    if num_kv_heads < 1 or num_kv_heads > num_heads or num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_kv_heads ({num_kv_heads}) must satisfy 1 <= num_kv_heads <= num_heads ({num_heads}) "
            f"and num_heads % num_kv_heads == 0"
        )
    if num_kv_heads < num_heads and args.architecture in ("crate", "noq_crate"):
        raise ValueError(
            "Grouped-Query Attention (--num-kv-heads < num_heads) is not supported for CRATE/NoQCRATE "
            "(tied QKV). Use --architecture gpt, noq_gpt, rys_gpt, or trm_gpt."
        )
    if num_kv_heads < num_heads and args.architecture == "tpa_gpt":
        raise ValueError(
            "Grouped-Query Attention (--num-kv-heads < num_heads) is not supported for --architecture tpa_gpt. "
            "TPA expresses GQA-style KV compression via --tpa-rank-k / --tpa-rank-v instead "
            "(see TPA paper §3.4 — GQA is the rank-G non-contextual case of TPA)."
        )
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")
print0(f"architecture: {args.architecture}")
if args.architecture == "trm_gpt":
    trm_effective = num_layers * args.trm_n_recur * args.trm_T_cycles
    print0(f"TRM: n_unique={num_layers}, n_recur={args.trm_n_recur}, T_cycles={args.trm_T_cycles}, effective_depth={trm_effective}")

# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Batch size scaling for learning rates (hyperparameters were tuned at reference batch size 2^19)
batch_lr_scale = 1.0
reference_batch_size = 2**19
batch_ratio = args.total_batch_size / reference_batch_size
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard
    # Muon: sqrt scaling is an assumption - not fully studied, but it's a second-order-ish optimizer
    batch_lr_scale = batch_ratio ** 0.5
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {args.total_batch_size:,} (reference: {reference_batch_size:,})")

# Weight decay is tuned at d12 and its scaling seems to be \propto 1/channels^2 (or equivalently, \propto 1/depth^2 due to constant aspect ratio)
if args.architecture == "trm_gpt":
    # For TRM, depth=n_unique_layers is tiny; scale by effective depth instead
    wd_depth = num_layers * args.trm_n_recur * args.trm_T_cycles
else:
    wd_depth = args.depth
weight_decay_scaled = args.weight_decay * (12 / wd_depth)**2
if wd_depth != 12:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for {'effective ' if args.architecture == 'trm_gpt' else ''}depth {wd_depth}")

# -----------------------------------------------------------------------------
# Initialize the Model

# Create a new model with random weights
if args.architecture == "trm_gpt":
    # For TRM, --depth means n_unique_layers (e.g. 2), effective depth is computed from recursion params
    model_config_kwargs = dict(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_unique_layers=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
        n_recur=args.trm_n_recur, T_cycles=args.trm_T_cycles,
    )
else:
    model_config_kwargs = dict(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
if args.architecture == "rys_gpt":
    model_config_kwargs.update(rys_block_start=args.rys_block_start, rys_block_end=args.rys_block_end, rys_num_repeats=args.rys_num_repeats)
    if args.rys_blocks:
        model_config_kwargs["rys_blocks"] = args.rys_blocks
    if args.rys_frac_recur_start > 0.0:
        model_config_kwargs["frac_recur_start"] = args.rys_frac_recur_start
if args.architecture == "tpa_gpt":
    model_config_kwargs.update(
        tpa_rank_q=args.tpa_rank_q,
        tpa_rank_k=args.tpa_rank_k,
        tpa_rank_v=args.tpa_rank_v,
    )
# --parallel-residual is only defined on nanochat.gpt.GPTConfig (the two-lane forward lives in GPT.forward).
if args.parallel_residual:
    if args.architecture != "gpt":
        raise ValueError(
            f"--parallel-residual is only supported for --architecture gpt; got {args.architecture!r}."
        )
    model_config_kwargs["parallel_residual"] = True
model_class, model_config_class = {
    "crate": (CRATE, CRATEConfig),
    "gpt": (GPT, GPTConfig),
    "noq_gpt": (NoQGPT, NoQGPTConfig),
    "noq_crate": (NoQCRATE, NoQCRATEConfig),
    "rys_gpt": (RYSGPT, RYSGPTConfig),
    "trm_gpt": (TRMGPT, TRMGPTConfig),
    "tpa_gpt": (TPAGPT, TPAGPTConfig),
}[args.architecture]
with torch.device("meta"):
    # All tensors are created as meta tensors (they have shape/dtype but no data)
    model_config = model_config_class(**model_config_kwargs)
    model = model_class(model_config)
model.to_empty(device=device) # All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    wte_key = next((k for k in model_data if k.endswith("wte.weight")), None)
    if wte_key is not None:
        checkpoint_vocab_size = model_data[wte_key].shape[0]
        if checkpoint_vocab_size > get_backbone(model).transformer.wte.weight.shape[0]:
            print0(
                f"Resizing model vocab from {get_backbone(model).transformer.wte.weight.shape[0]} "
                f"to {checkpoint_vocab_size} to match resume checkpoint"
            )
            resize_model_vocab(get_backbone(model), checkpoint_vocab_size, device)
            model_config_kwargs["vocab_size"] = checkpoint_vocab_size
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
if use_jepa:
    PRED_TOKEN_ID = ensure_pred_token_slot(orig_model, tokenizer, device)
    model_config_kwargs["vocab_size"] = get_backbone(orig_model).config.vocab_size
else:
    PRED_TOKEN_ID = None
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe
num_params = sum(p.numel() for p in model.parameters())
num_scaling_params = orig_model.num_scaling_params()
print0(f"Number of parameters: {num_params:,} (scaling: {num_scaling_params:,})")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(args.target_flops / (num_flops_per_token * args.total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio (use scaling params per Kaplan et al.)
    target_tokens = args.target_param_data_ratio * num_scaling_params
    num_iterations = target_tokens // args.total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = args.total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {args.total_batch_size * num_iterations / num_scaling_params:.2f}") # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")
if resuming and args.resume_from_step >= num_iterations:
    print0(f"WARNING: --resume-from-step ({args.resume_from_step}) >= --num-iterations ({num_iterations}). "
           f"The training loop will exit immediately. Set --num-iterations higher to continue training.")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
adam_betas = (args.adam_beta1, args.adam_beta2)
_setup_kwargs = dict(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    adam_betas=adam_betas,
    scalar_lr=args.scalar_lr * batch_lr_scale,
)
# Only architectures whose setup_optimizers accepts muon_mode get the kwarg (gpt + rys_gpt + tpa_gpt for now).
if args.architecture in ("gpt", "rys_gpt", "tpa_gpt"):
    _setup_kwargs["muon_mode"] = args.muon_mode
elif args.muon_mode != "default":
    raise ValueError(
        f"--muon-mode {args.muon_mode!r} is only supported for --architecture gpt, rys_gpt, or tpa_gpt; "
        f"got {args.architecture!r}."
    )
optimizers = model.setup_optimizers(**_setup_kwargs)
adamw_optimizer, muon_optimizer = optimizers

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data # free up the memory

# -----------------------------------------------------------------------------
# EMA shadow weights (parameter-golf trick): keep a running exponential moving
# average of the model parameters, evaluate val/bpb_ema alongside val/bpb, and
# save it as model_ema_<step>.pt next to the regular checkpoint.
ema_state = None  # name -> tensor (same dtype as the live param)
def _ema_named_params():
    """Iterate over (name, param) of the original (uncompiled) model.
    Strip torch.compile's `_orig_mod.` prefix that appears on the compiled wrapper."""
    for name, p in orig_model.named_parameters():
        yield name, p
def _init_ema():
    global ema_state
    ema_state = {name: p.detach().clone() for name, p in _ema_named_params()}
    print0(f"EMA: initialized shadow with {sum(t.numel() for t in ema_state.values()):,} params (decay={ema_decay})")
def _update_ema(decay_now):
    if ema_state is None:
        return
    with torch.no_grad():
        for name, p in _ema_named_params():
            shadow = ema_state[name]
            shadow.mul_(decay_now).add_(p.detach().to(shadow.dtype), alpha=1.0 - decay_now)
def _swap_in_ema():
    """Swap live params with EMA. Returns dict of saved live tensors for restoration."""
    if ema_state is None:
        return None
    saved = {}
    with torch.no_grad():
        for name, p in _ema_named_params():
            saved[name] = p.detach().clone()
            p.data.copy_(ema_state[name].to(p.dtype))
    return saved
def _swap_out_ema(saved):
    if saved is None:
        return
    with torch.no_grad():
        for name, p in _ema_named_params():
            p.data.copy_(saved[name])
if use_ema:
    _init_ema()

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
tokens_dir = os.path.join(base_dir, "tokenized_data")
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# Learning rate scheduler
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

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# Weight decay scheduler for Muon optimizer (linear to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)

if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
    jepa_loss_log = None
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]
    jepa_loss_log = loop_state.get("jepa_loss_log", None)
mfu = 0.0  # will be overwritten each training step; pre-init for resume-at-final-step

# -----------------------------------------------------------------------------
# Training loop
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        tokens_per_eval_step = args.device_batch_size * args.max_seq_len * ddp_world_size
        eval_steps = max(1, args.eval_tokens // tokens_per_eval_step)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        log_eval = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        }
        # Same eval pass with EMA-swapped weights so we can compare raw vs. EMA representations.
        if use_ema and ema_state is not None:
            ema_saved = _swap_in_ema()
            try:
                val_loader_ema = build_val_loader()
                with autocast_ctx:
                    val_bpb_ema = evaluate_bpb(model, val_loader_ema, eval_steps, token_bytes)
                print0(f"Step {step:05d} | Validation bpb (EMA): {val_bpb_ema:.6f}")
                log_eval["val/bpb_ema"] = val_bpb_ema
            finally:
                _swap_out_ema(ema_saved)
        wandb_run.log(log_eval)
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            [opt.state_dict() for opt in optimizers], # optimizer states
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                    "jepa_loss_log": jepa_loss_log,
                },
            },
            rank=ddp_rank,
        )
        # Also save EMA shadow weights (same dict layout as model state_dict, master rank only).
        if use_ema and master_process and ema_state is not None:
            ema_path = os.path.join(checkpoint_dir, f"model_ema_{step:06d}.pt")
            torch.save(ema_state, ema_path)
            print0(f"EMA: saved shadow weights to {ema_path}")

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # Tell the model where it is in training (used by RYS fractional recurrence).
    if hasattr(orig_model, "set_training_progress"):
        orig_model.set_training_progress(step / max(1, num_iterations))
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    step_jepa_loss = torch.tensor(0.0, device=device)
    step_jepa_count = torch.tensor(0, device=device)
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            llm_loss = model(x, y)
            total_loss = llm_loss

        if use_jepa:
            should_jepa = torch.zeros(1, device=device)
            if master_process:
                should_jepa.fill_(1.0 if torch.rand(1).item() >= jepa_dropout else 0.0)
            if ddp:
                dist.broadcast(should_jepa, src=0)

            if should_jepa.item() > 0.5:
                with autocast_ctx:
                    jepa_loss_mean, jepa_pair_count = compute_jepa_loss_for_batch(
                        orig_model, x, y, PRED_TOKEN_ID, device,
                        view_min_len=args.jepa_view_min_len, max_view_tokens=256,
                    )

                    if jepa_loss_mean is not None:
                        jepa_lambda_t = get_jepa_lambda(jepa_base_lambda, step, num_iterations, jepa_schedule)
                        total_loss = llm_loss + jepa_lambda_t * jepa_loss_mean
                        step_jepa_loss = step_jepa_loss + jepa_loss_mean.detach()
                        step_jepa_count = step_jepa_count + 1

        train_loss = llm_loss.detach() # for logging keep this as LM-only loss
        loss = total_loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizers
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
        group["weight_decay"] = muon_weight_decay
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    # Update EMA shadow weights AFTER the optimizer step.
    if use_ema:
        # Hard-copy during the warmup window so EMA doesn't lag the rapidly-changing init weights.
        if step < args.ema_warmup_steps:
            decay_now = 0.0  # → shadow := live params
        else:
            decay_now = ema_decay
        _update_ema(decay_now)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # JEPA logging
    if use_jepa and ddp:
        dist.all_reduce(step_jepa_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(step_jepa_count, op=dist.ReduceOp.SUM)

    jepa_loss_log = None
    if use_jepa and step_jepa_count.item() > 0:
        jepa_loss_log = (step_jepa_loss / step_jepa_count).item()

    # logging
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]
    jepa_lambda_now = get_jepa_lambda(jepa_base_lambda, step, num_iterations, jepa_schedule) if use_jepa else 0.0
    jepa_str = f" | jepa: {jepa_loss_log:.4f} (λ={jepa_lambda_now:.3f})" if jepa_loss_log is not None else " | jepa: skip"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f}{jepa_str} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        if use_jepa:
            log_data["jepa_lambda"] = jepa_lambda_now
        if jepa_loss_log is not None:
            log_data["jepa_loss"] = jepa_loss_log
        wandb_run.log(log_data)

    # state update
    step += 1

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training + JEPA", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": args.total_batch_size * num_iterations / num_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": args.warmup_ratio,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "JEPA loss": jepa_loss_log,
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
