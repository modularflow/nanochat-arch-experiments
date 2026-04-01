"""
Semi-supervised self-training via iterative pseudo-labeling.

Generates candidate responses, scores/filters them, and trains
on the high-quality pseudo-labels. Optionally repeats for multiple
self-training iterations.

Run on one GPU e.g. for debugging:

python -m scripts.self_train --source sft --prompt-source prompts.jsonl

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.self_train -- --source sft --prompt-task gsm8k --filter-strategy reward
"""

import argparse
import gc
import hashlib
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import wandb
import torch
import torch.distributed as dist
from contextlib import nullcontext

from nanochat.common import compute_init, compute_cleanup, get_base_dir, get_dist_info, print0, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.engine import Engine
from nanochat.jepa import (
    get_backbone, ensure_pred_token_slot,
    compute_jepa_loss, compute_jepa_loss_batched, extract_last_turn_views,
)
from nanochat.self_training import (
    PromptSource,
    PseudoLabelDataset,
    generate_pseudo_labels,
    score_by_confidence,
    filter_candidates,
)

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Semi-supervised self-training")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument("--source", type=str, default="sft", help="base|mid|sft|rl - which checkpoint to load from")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
# Prompt source (exactly one of these must be provided)
parser.add_argument("--prompt-source", type=str, default=None, help="path to JSONL file of prompts")
parser.add_argument("--prompt-task", type=str, default=None, help="task name to extract prompts from (gsm8k|arc-easy|arc-challenge|smoltalk|mmlu|all)")
parser.add_argument("--prompt-task-mode", type=str, default="mixed", help="how to combine tasks when using 'all': mixed (shuffled) | sequential (one task at a time)")
parser.add_argument("--max-prompts", type=int, default=1000, help="max number of prompts to use from the source (default 1000)")
# Generation parameters
parser.add_argument("--num-candidates", type=int, default=8, help="number of candidate responses per prompt")
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens to generate per candidate")
parser.add_argument("--temperature", type=float, default=0.8, help="sampling temperature for generation")
parser.add_argument("--gen-top-k", type=int, default=50, help="top-k sampling for generation")
# Scoring parameters
parser.add_argument("--score-batch-size", type=int, default=16, help="batch size for confidence scoring (increase to use more VRAM)")
# Filtering parameters
parser.add_argument("--filter-strategy", type=str, default="top_k", help="top_k|threshold|reward")
parser.add_argument("--top-k", type=int, default=2, help="keep top-k candidates per prompt (for top_k strategy)")
parser.add_argument("--threshold-percentile", type=float, default=75.0, help="confidence percentile cutoff (for threshold strategy)")
# Self-training iterations
parser.add_argument("--num-iterations", type=int, default=1, help="number of outer self-training iterations (generate-filter-train)")
# Training parameters (per self-training iteration)
parser.add_argument("--num-train-steps", type=int, default=200, help="SFT training steps per self-training iteration")
parser.add_argument("--device-batch-size", type=int, default=4, help="per-device batch size for generation and scoring")
parser.add_argument("--train-batch-size", type=int, default=None, help="per-device batch size for training (defaults to --device-batch-size if not set)")
parser.add_argument("--target-examples-per-step", type=int, default=32, help="target examples per optimization step")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.02, help="initial LR as fraction of base LR")
# JEPA auxiliary loss
parser.add_argument("--jepa-lambda", type=float, default=0.0, help="weight of JEPA loss (0 = disabled, recommend 0.1-0.25)")
parser.add_argument("--jepa-dropout", type=float, default=0.5, help="fraction of micro-batches to skip JEPA")
parser.add_argument("--jepa-view-max-len", type=int, default=256, help="max tokens per JEPA view to avoid OOM")
# Evaluation
parser.add_argument("--eval-every", type=int, default=50, help="evaluate val loss every N training steps")
parser.add_argument("--eval-steps", type=int, default=50, help="number of batches for val loss evaluation")
# Caching
parser.add_argument("--candidates-cache", type=str, default=None, help="path to save/load generated candidates (skips generation if file exists)")
# Output
parser.add_argument("--save-dir", type=str, default="semisup_checkpoints", help="checkpoint directory name under base_dir (e.g. semisup_code_checkpoints)")
args = parser.parse_args()
if args.train_batch_size is None:
    args.train_batch_size = args.device_batch_size
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-semisup", name=args.run, config=user_config, save_code=True)

# Load the model and tokenizer
model, tokenizer, meta = load_model(args.source, device, phase="train", model_tag=args.model_tag, step=args.model_step)
orig_model = model
engine = Engine(model, tokenizer)

# JEPA setup
use_jepa = args.jepa_lambda > 0.0
assert 0.0 <= args.jepa_dropout <= 1.0
PRED_TOKEN_ID = None
USER_START_ID = None
ASSISTANT_START_ID = None
if use_jepa:
    PRED_TOKEN_ID = ensure_pred_token_slot(orig_model, tokenizer, device)
    USER_START_ID = tokenizer.encode_special("<|user_start|>")
    ASSISTANT_START_ID = tokenizer.encode_special("<|assistant_start|>")
    print0(f"JEPA enabled: lambda={args.jepa_lambda}, dropout={args.jepa_dropout}, view_max_len={args.jepa_view_max_len}")

# Fingerprint the loaded checkpoint so we can detect stale candidate caches.
# If the source model changes (e.g. re-trained), cached candidates from an
# older model are automatically invalidated and regenerated.
_meta_str = json.dumps(meta, sort_keys=True, default=str)
_model_fingerprint = hashlib.md5(f"{args.source}:{args.model_tag}:{_meta_str}".encode()).hexdigest()[:12]
print0(f"Model fingerprint: {_model_fingerprint}")

# -----------------------------------------------------------------------------
# Build the prompt source and optional task (for reward-based filtering)
SUPPORTED_TASKS = ["gsm8k", "arc-easy", "arc-challenge", "smoltalk", "mmlu", "codestack", "all"]

def _build_task(name):
    """Instantiate a Task by name. Returns (task, has_reward)."""
    if name == "gsm8k":
        from tasks.gsm8k import GSM8K
        return GSM8K(subset="main", split="train"), True
    elif name == "arc-easy":
        from tasks.arc import ARC
        return ARC(subset="ARC-Easy", split="train"), False
    elif name == "arc-challenge":
        from tasks.arc import ARC
        return ARC(subset="ARC-Challenge", split="train"), False
    elif name == "smoltalk":
        from tasks.smoltalk import SmolTalk
        return SmolTalk(split="train", stop=10_000), False
    elif name == "mmlu":
        from tasks.mmlu import MMLU
        return MMLU(subset="auxiliary_train", split="train", stop=5_000), False
    elif name == "codestack":
        from tasks.codestack import CodeStack
        return CodeStack(split="train", stop=10_000), False
    else:
        raise ValueError(f"Unknown task: {name}. Supported: {SUPPORTED_TASKS}")

task_object = None  # set if a single reward-capable task is used
if args.prompt_source is not None:
    prompt_source = PromptSource(filepath=args.prompt_source)
    print0(f"Loaded {len(prompt_source)} prompts from JSONL: {args.prompt_source}")
elif args.prompt_task is not None:
    task_name = args.prompt_task.lower()
    if task_name == "all":
        from tasks.common import TaskMixture, TaskSequence
        all_tasks = []
        for t_name in ["gsm8k", "arc-easy", "arc-challenge", "smoltalk", "mmlu"]:
            t, _ = _build_task(t_name)
            all_tasks.append(t)
        mode = args.prompt_task_mode.lower()
        if mode == "sequential":
            combined = TaskSequence(all_tasks)
            print0(f"Loaded {len(combined)} prompts from all tasks (sequential: one task at a time)")
        else:
            combined = TaskMixture(all_tasks)
            print0(f"Loaded {len(combined)} prompts from all tasks (mixed: shuffled together)")
        prompt_source = PromptSource(task=combined)
    else:
        task_obj, has_reward = _build_task(task_name)
        if has_reward:
            task_object = task_obj
        prompt_source = PromptSource(task=task_obj)
        print0(f"Loaded {len(prompt_source)} prompts from task: {args.prompt_task}")
else:
    raise ValueError("Must provide exactly one of --prompt-source (JSONL path) or --prompt-task (task name)")

# Truncate to --max-prompts
if args.max_prompts > 0 and len(prompt_source) > args.max_prompts:
    prompt_source.prompts = prompt_source.prompts[:args.max_prompts]
    print0(f"Truncated to {len(prompt_source)} prompts (--max-prompts={args.max_prompts})")

# -----------------------------------------------------------------------------
# SFT-style data generator (same pattern as chat_sft.py)

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
        inputs = inputs.to(device)
        targets = targets.to(device)
        return inputs, targets
    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

# -----------------------------------------------------------------------------
# Training subroutine for one self-training iteration

def run_train_phase(pseudo_label_dataset, iteration_idx):
    """Run SFT-style training on the pseudo-labeled dataset. Returns final train loss."""
    num_train_steps = args.num_train_steps
    if len(pseudo_label_dataset) == 0:
        print0(f"  [Iteration {iteration_idx}] No pseudo-labels to train on, skipping.")
        return 0.0

    train_bs = args.train_batch_size
    print0(f"  [Iteration {iteration_idx}] Training on {len(pseudo_label_dataset)} pseudo-labeled examples for {num_train_steps} steps (batch_size={train_bs})")

    # Free VRAM from generation/scoring before allocating training buffers
    if device_type == "cuda":
        torch.cuda.empty_cache()

    # Gradient accumulation setup
    examples_per_step = train_bs * ddp_world_size
    assert args.target_examples_per_step % examples_per_step == 0, \
        "Target examples per step must be divisible by train_batch_size * world_size"
    grad_accum_steps = args.target_examples_per_step // examples_per_step

    # Build data loader
    train_loader = sft_data_generator(pseudo_label_dataset, batch_size=train_bs)

    # (Re-)initialize optimizers each iteration to reset momentum
    optimizers = model.setup_optimizers(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=args.weight_decay,
    )
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["lr"] * args.init_lr_frac
            group["initial_lr"] = group["lr"]

    def get_lr_multiplier(it):
        return 1.0 - it / num_train_steps

    train_loss_item = 0.0
    val_loss = float("inf")
    for step in range(num_train_steps):
        last_step = step == num_train_steps - 1

        # Periodic validation loss
        if last_step or step % args.eval_every == 0:
            model.eval()
            val_loader = sft_data_generator(pseudo_label_dataset, batch_size=train_bs)
            losses = []
            for _ in range(min(args.eval_steps, max(len(pseudo_label_dataset) // train_bs, 1))):
                val_inputs, val_targets = next(val_loader)
                with torch.no_grad(), autocast_ctx:
                    loss = model(val_inputs, val_targets)
                losses.append(loss)
            val_loss = torch.stack(losses).mean()
            if ddp:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss = val_loss.item()
            print0(f"  [Iter {iteration_idx}] Step {step:05d} | Val loss: {val_loss:.6f}")
            wandb_run.log({
                "iteration": iteration_idx,
                "train_step": step,
                "val_loss": val_loss,
            })
            model.train()

        if last_step:
            break

        # Forward/backward with gradient accumulation
        num_tokens = torch.tensor(0, device=device)
        step_jepa_loss = torch.tensor(0.0, device=device)
        step_jepa_count = torch.tensor(0, device=device)
        for micro_step in range(grad_accum_steps):
            train_inputs, train_targets = next(train_loader)
            with autocast_ctx:
                llm_loss = model(train_inputs, train_targets)
                total_loss = llm_loss

            if use_jepa:
                should_jepa = torch.zeros(1, device=device)
                if master_process:
                    should_jepa.fill_(1.0 if torch.rand(1).item() >= args.jepa_dropout else 0.0)
                if ddp:
                    dist.broadcast(should_jepa, src=0)

                if should_jepa.item() > 0.5:
                    with autocast_ctx:
                        views_a = []
                        views_b = []
                        for b in range(train_inputs.shape[0]):
                            seq = train_inputs[b]
                            user_ids, assistant_ids = extract_last_turn_views(
                                seq, USER_START_ID, ASSISTANT_START_ID
                            )
                            if user_ids is None or assistant_ids is None:
                                continue
                            views_a.append(user_ids[-args.jepa_view_max_len:])
                            views_b.append(assistant_ids[:args.jepa_view_max_len])

                        if views_a:
                            jepa_loss_mean = compute_jepa_loss_batched(
                                orig_model, views_a, views_b, PRED_TOKEN_ID, device
                            )
                            total_loss = llm_loss + args.jepa_lambda * jepa_loss_mean
                            step_jepa_loss = step_jepa_loss + jepa_loss_mean.detach()
                            step_jepa_count = step_jepa_count + 1

            train_loss = llm_loss.detach()
            loss = total_loss / grad_accum_steps
            loss.backward()
            num_tokens += (train_targets >= 0).sum()
        if ddp:
            dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)

        if use_jepa and ddp:
            dist.all_reduce(step_jepa_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(step_jepa_count, op=dist.ReduceOp.SUM)

        jepa_loss_log = None
        if use_jepa and step_jepa_count.item() > 0:
            jepa_loss_log = (step_jepa_loss / step_jepa_count).item()

        # LR schedule
        lrm = get_lr_multiplier(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm

        # Optimizer step
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        train_loss_item = train_loss.item()
        num_tokens_item = num_tokens.item()
        jepa_str = f" | jepa: {jepa_loss_log:.4f}" if jepa_loss_log is not None else (" | jepa: skip" if use_jepa else "")
        print0(f"  [Iter {iteration_idx}] Step {step:05d}/{num_train_steps:05d} | Train loss: {train_loss_item:.6f}{jepa_str} | lrm: {lrm:.6f} | tokens: {num_tokens_item:,}")
        log_data = {
            "iteration": iteration_idx,
            "train_step": step,
            "lrm": lrm,
            "train_loss": train_loss_item,
            "num_tokens": num_tokens_item,
        }
        if jepa_loss_log is not None:
            log_data["jepa_loss"] = jepa_loss_log
        wandb_run.log(log_data)

    # Explicitly free optimizer state (momentum buffers) to reclaim VRAM
    for opt in optimizers:
        del opt
    del optimizers
    gc.collect()

    return train_loss_item

# =============================================================================
# Main self-training loop
# =============================================================================

print0(f"Starting self-training: {args.num_iterations} iteration(s)")
print0(f"  Candidates per prompt: {args.num_candidates}")
print0(f"  Filter strategy: {args.filter_strategy} (top_k={args.top_k}, percentile={args.threshold_percentile})")

final_train_loss = 0.0
total_pseudo_labels = 0

for iteration in range(args.num_iterations):
    print0(f"\n{'='*60}")
    print0(f"Self-training iteration {iteration + 1}/{args.num_iterations}")
    print0(f"{'='*60}")

    # Phase 1: Generate candidate responses (or load from cache)
    cache_path = None
    if args.candidates_cache:
        cache_path = f"{args.candidates_cache}.iter{iteration}.json"

    use_cache = False
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache_obj = json.load(f)
        # Validate fingerprint: if the model has changed, stale candidates are useless
        if isinstance(cache_obj, dict) and "fingerprint" in cache_obj:
            if cache_obj["fingerprint"] == _model_fingerprint:
                pseudo_labels = cache_obj["candidates"]
                use_cache = True
                print0(f"Phase 1: Loaded {len(pseudo_labels)} cached entries from {cache_path} (fingerprint matches)")
            else:
                print0(f"Phase 1: Cache exists but model has changed (cached={cache_obj['fingerprint']}, current={_model_fingerprint}), regenerating...")
        else:
            # Legacy cache format (plain list, no fingerprint) -- treat as stale
            print0(f"Phase 1: Cache exists but has no fingerprint (legacy format), regenerating...")

    if not use_cache:
        print0(f"Phase 1: Generating {args.num_candidates} candidates per prompt...")
        model.eval()
        pseudo_labels = generate_pseudo_labels(
            engine=engine,
            prompt_source=prompt_source,
            tokenizer=tokenizer,
            num_candidates=args.num_candidates,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.gen_top_k,
            device_batch_size=args.device_batch_size,
            device=device,
            autocast_ctx=autocast_ctx,
        )
        # Save candidates with fingerprint so we can detect staleness later
        if cache_path and master_process:
            os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)
            cache_obj = {"fingerprint": _model_fingerprint, "candidates": pseudo_labels}
            with open(cache_path, "w") as f:
                json.dump(cache_obj, f)
            print0(f"  Saved candidates cache to {cache_path}")

    # Phase 2: Score candidates by confidence
    if args.filter_strategy != "reward":
        print0(f"Phase 2: Scoring {sum(len(p['candidates']) for p in pseudo_labels)} candidates by confidence...")
        pseudo_labels = score_by_confidence(
            model=model,
            tokenizer=tokenizer,
            pseudo_labels=pseudo_labels,
            device=device,
            autocast_ctx=autocast_ctx,
            batch_size=args.score_batch_size,
        )
    else:
        print0(f"Phase 2: Skipping confidence scoring (using reward-based filtering)")

    # Phase 3: Filter candidates
    print0(f"Phase 3: Filtering candidates with strategy={args.filter_strategy}...")
    conversations = filter_candidates(
        pseudo_labels=pseudo_labels,
        strategy=args.filter_strategy,
        threshold_percentile=args.threshold_percentile,
        top_k=args.top_k,
        task=task_object,
    )

    # Gather counts across ranks for logging
    local_count = torch.tensor(len(conversations), device=device)
    if ddp:
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
    total_filtered = local_count.item()
    print0(f"  Filtered down to {total_filtered} pseudo-labeled examples (across all ranks)")
    total_pseudo_labels += total_filtered

    wandb_run.log({
        "iteration": iteration,
        "num_pseudo_labels": total_filtered,
    })

    # Phase 4: Train on filtered pseudo-labels
    print0(f"Phase 4: Training on pseudo-labeled data...")
    pseudo_dataset = PseudoLabelDataset(conversations)
    model.train()
    final_train_loss = run_train_phase(pseudo_dataset, iteration)

    # Free optimizer states and training buffers before the next generation phase
    model.zero_grad(set_to_none=True)
    if device_type == "cuda":
        torch.cuda.empty_cache()
    print0(f"  Cleaned up training state, freed VRAM for next iteration.")

    # Save checkpoint after each self-training iteration
    iter_meta = {
        "iteration": iteration,
        "num_pseudo_labels": total_filtered,
        "train_loss": final_train_loss,
        "model_config": model.config.__dict__,
        "user_config": user_config,
    }
    if master_process:
        base_dir = get_base_dir()
        depth = model.config.n_layer
        output_dirname = args.model_tag if args.model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, args.save_dir, output_dirname)
        save_checkpoint(
            checkpoint_dir,
            iteration,
            model.state_dict(),
            None,
            iter_meta,
        )
        print0(f"  Saved checkpoint to {checkpoint_dir}")

    # Update fingerprint so the next iteration's cache reflects the updated model
    _meta_str = json.dumps(iter_meta, sort_keys=True, default=str)
    _model_fingerprint = hashlib.md5(f"{args.save_dir}:{args.model_tag}:{_meta_str}".encode()).hexdigest()[:12]
    print0(f"  Updated model fingerprint: {_model_fingerprint}")

print0(f"\nSelf-training complete. {args.num_iterations} iteration(s), {total_pseudo_labels} total pseudo-labels used.")

# Log to report
from nanochat.report import get_report
get_report().log(section="Semi-Supervised Self-Training", data=[
    user_config,
    {
        "Self-training iterations": args.num_iterations,
        "Total pseudo-labels": total_pseudo_labels,
        "Final training loss": final_train_loss,
    },
])

# Cleanup
wandb_run.finish()
compute_cleanup()
