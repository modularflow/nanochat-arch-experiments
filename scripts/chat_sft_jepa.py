"""
Finetune a base model to be a chat model with an auxiliary LLM-JEPA loss.
Run on one GPU e.g. for debugging:

python -m scripts.chat_sft_jepa

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft_jepa
"""

import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
)
from nanochat.engine import Engine
from nanochat.jepa import get_jepa_lambda, compute_jepa_loss_batched, JEPA_SCHEDULES
from nanochat.report import get_report
from scripts.chat_eval import run_chat_eval
from tasks.arc import ARC
from tasks.common import TaskMixture
from tasks.customjson import CustomJSON
from tasks.gsm8k import GSM8K
from tasks.smoltalk import SmolTalk
from tasks.spellingbee import SimpleSpelling, SpellingBee

PRED_TOKEN_STR = "<|pred|>"
USER_START_TOKEN = "<|user_start|>"
ASSISTANT_START_TOKEN = "<|assistant_start|>"
ASSISTANT_END_TOKEN = "<|assistant_end|>"


def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))


def get_backbone(model):
    return model.backbone if hasattr(model, "backbone") else model


def resize_model_vocab(backbone, new_vocab_size, device):
    """
    Resize embedding/unembedding tables while keeping checkpoint save/load sane.
    """
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


def ensure_pred_token(model, tokenizer, device):
    """
    Reserve one token id for <|pred|>.

    The tokenizer in this repo does not currently expose dynamic special-token
    registration, so we use the first tokenizer-unused vocab id and ensure the
    model has weights for it. If the model already has padded spare rows, we
    reuse one of them and reinitialize it.
    """
    backbone = get_backbone(model)
    pred_token_id = None

    if hasattr(tokenizer, "get_special_tokens") and PRED_TOKEN_STR in tokenizer.get_special_tokens():
        pred_token_id = tokenizer.encode_special(PRED_TOKEN_STR)
    elif hasattr(tokenizer, "add_special_token"):
        tokenizer.add_special_token(PRED_TOKEN_STR)
        pred_token_id = tokenizer.encode_special(PRED_TOKEN_STR)
    elif hasattr(tokenizer, "register_special_token"):
        tokenizer.register_special_token(PRED_TOKEN_STR)
        pred_token_id = tokenizer.encode_special(PRED_TOKEN_STR)
    else:
        pred_token_id = tokenizer.get_vocab_size()
        print0(
            f"WARNING: Could not register {PRED_TOKEN_STR} in the tokenizer; "
            f"using raw vocab slot {pred_token_id} directly."
        )

    current_vocab = backbone.transformer.wte.weight.shape[0]
    if pred_token_id >= current_vocab:
        padded_vocab = ((pred_token_id + 1 + 63) // 64) * 64
        print0(f"Resizing model vocab from {current_vocab} to {padded_vocab} for {PRED_TOKEN_STR}")
        resize_model_vocab(backbone, padded_vocab, device)
    else:
        print0(f"Reusing padded vocab slot {pred_token_id} for {PRED_TOKEN_STR}")

    with torch.no_grad():
        torch.nn.init.normal_(backbone.transformer.wte.weight[pred_token_id:pred_token_id + 1], mean=0.0, std=0.02)
        if pred_token_id < backbone.lm_head.weight.shape[0]:
            torch.nn.init.normal_(backbone.lm_head.weight[pred_token_id:pred_token_id + 1], mean=0.0, std=0.001)

    return pred_token_id


def forward_final_hidden(model, idx):
    """
    Run the shared backbone path manually and return the final hidden states.

    This works across GPT, CRATE, and SelfFlowCRATE because they all expose the
    same backbone tensors/blocks through `transformer`, `resid_lambdas`,
    `x0_lambdas`, `window_sizes`, `cos`, and `sin`.
    """
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


def extract_last_turn_views(seq, effective_len, user_start_id, assistant_start_id):
    """
    Extract the final user turn and its matching assistant turn from one sample.
    """
    seq = seq[:effective_len]

    assistant_positions = (seq == assistant_start_id).nonzero(as_tuple=True)[0]
    if len(assistant_positions) == 0:
        return None, None
    assistant_start = assistant_positions[-1].item()

    user_positions = (seq[:assistant_start] == user_start_id).nonzero(as_tuple=True)[0]
    if len(user_positions) == 0:
        return None, None
    user_start = user_positions[-1].item()

    user_ids = seq[user_start:assistant_start]
    assistant_ids = seq[assistant_start:effective_len]
    if len(user_ids) < 4 or len(assistant_ids) < 4:
        return None, None

    return user_ids, assistant_ids


def compute_jepa_loss(model, user_ids, assistant_ids, pred_token_id, device):
    """
    Compute the JEPA embedding-prediction loss.

    user_ids       : 1-D LongTensor of token ids for the user turn
    assistant_ids  : 1-D LongTensor of token ids for the assistant turn
    pred_token_id  : int, the id of the [PRED] special token
    device         : torch device
    """
    pred_id_tensor = torch.tensor([pred_token_id], dtype=torch.long, device=device)
    input_a = torch.cat([user_ids, pred_id_tensor]).unsqueeze(0)

    hidden_a = forward_final_hidden(model, input_a)
    pred_embed = hidden_a[0, -1, :]

    input_b = assistant_ids.unsqueeze(0)
    with torch.no_grad():
        hidden_b = forward_final_hidden(model, input_b)
    target_embed = hidden_b[0, -1, :].detach()

    loss = 1.0 - F.cosine_similarity(
        pred_embed.unsqueeze(0),
        target_embed.unsqueeze(0),
    )
    return loss.squeeze()


# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Supervised finetuning for chat with an auxiliary JEPA loss")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument("--source", type=str, default="mid", help="base|mid - which checkpoint to load from")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs")
parser.add_argument("--num-iterations", type=int, default=-1, help="override number of iterations (-1 = use num_epochs)")
# Batch sizes
parser.add_argument("--device-batch-size", type=int, default=4, help="per-device batch size")
parser.add_argument("--target-examples-per-step", type=int, default=32, help="target examples per optimization step")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.02, help="initial LR as fraction of base LR")
parser.add_argument("--jepa-lambda", type=float, default=0.5,
    help="Weight of the JEPA loss term (0 = disabled, pure next-token)")
parser.add_argument("--jepa-schedule", type=str, default="constant", choices=JEPA_SCHEDULES,
    help="Lambda schedule: constant (default), linear_decay (→0), cosine_decay (→0)")
parser.add_argument("--jepa-dropout", type=float, default=0.5,
    help="Fraction of micro-batches where JEPA loss is skipped (0=always on, 0.5=skip half)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=100, help="evaluate val loss every N steps")
parser.add_argument("--eval-steps", type=int, default=100, help="number of batches for val loss evaluation")
parser.add_argument("--eval-metrics-every", type=int, default=200, help="evaluate accuracy metrics every N steps")
parser.add_argument("--eval-metrics-max-problems", type=int, default=1024, help="max problems per metric evaluation")
args = parser.parse_args()
user_config = vars(args).copy()

jepa_base_lambda = args.jepa_lambda
jepa_schedule = args.jepa_schedule
jepa_dropout = args.jepa_dropout
use_jepa = jepa_base_lambda > 0.0
assert 0.0 <= jepa_dropout <= 1.0, "--jepa-dropout must be between 0 and 1"

# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanochat-sft",
    name=args.run,
    config=user_config,
    save_code=True,
)

# Load the model and tokenizer
model, tokenizer, meta = load_model(args.source, device, phase="train", model_tag=args.model_tag, step=args.model_step)
if use_jepa:
    PRED_TOKEN_ID = ensure_pred_token(model, tokenizer, device)
    USER_START_ID = tokenizer.encode_special(USER_START_TOKEN)
    ASSISTANT_START_ID = tokenizer.encode_special(ASSISTANT_START_TOKEN)
else:
    PRED_TOKEN_ID = None
    USER_START_ID = None
    ASSISTANT_START_ID = None
orig_model = model
model = torch.compile(model, dynamic=True)
engine = Engine(orig_model, tokenizer)

# -----------------------------------------------------------------------------
# Task data mixture we'll train on
identity_conversations_filepath = os.path.join(get_base_dir(), "identity_conversations.jsonl")
train_ds = TaskMixture([
    ARC(subset="ARC-Easy", split="train"),
    ARC(subset="ARC-Challenge", split="train"),
    GSM8K(subset="main", split="train"),
    SmolTalk(split="train", stop=10_000),
    CustomJSON(filepath=identity_conversations_filepath),
    SimpleSpelling(size=300, split="train"),
    SpellingBee(size=300, split="train"),
])
val_ds = SmolTalk(split="test")

# -----------------------------------------------------------------------------
# DataLoader


def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special(ASSISTANT_END_TOKEN)

    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n - 1] = ids_tensor[:-1]
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, :n - 1] = row_targets
        return inputs.to(device), targets.to(device)

    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            # Some long conversations get truncated before any supervised
            # assistant tokens remain. Those produce all -1 targets and
            # cross-entropy returns nan, so skip them here.
            if sum(mask[1:]) == 0:
                continue
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []


examples_per_step = args.device_batch_size * ddp_world_size
print0(f"Target examples per step: {args.target_examples_per_step}")
print0(f"Device batch size: {args.device_batch_size}")
print0(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}")
assert args.target_examples_per_step % examples_per_step == 0, "Target examples per step must be divisible by examples per step"
grad_accum_steps = args.target_examples_per_step // examples_per_step
print0(f"=> Setting grad accum steps: {grad_accum_steps}")

if args.num_iterations == -1:
    assert args.num_epochs > 0, "num_epochs must be positive if num_iterations is -1"
    num_iterations = (len(train_ds) // args.target_examples_per_step) * args.num_epochs
else:
    num_iterations = args.num_iterations
train_loader = sft_data_generator(train_ds, batch_size=args.device_batch_size)
build_val_loader = lambda: sft_data_generator(val_ds, batch_size=args.device_batch_size)

# -----------------------------------------------------------------------------
# Initialize the Optimizer

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

# -----------------------------------------------------------------------------
# Training loop


def get_lr_multiplier(it):
    return 1.0 - it / num_iterations


step = 0
metrics = {}
val_loss = float("nan")
train_loss_item = float("nan")
llm_loss_item = float("nan")
jepa_loss_item = None

for step in range(num_iterations):
    last_step = step == num_iterations - 1

    if last_step or step % args.eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        losses = []
        for _ in range(args.eval_steps):
            val_inputs, val_targets = next(val_loader)
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
            losses.append(loss)
        val_loss_tensor = torch.stack(losses).mean()
        if ddp:
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        val_loss = val_loss_tensor.item()
        print0(f"Step {step:05d} | Validation loss: {val_loss:.6f}")
        wandb_run.log({
            "step": step,
            "val_loss": val_loss,
        })
        model.train()

    if last_step or (step > 0 and step % args.eval_metrics_every == 0):
        model.eval()
        metrics = {}
        with torch.no_grad(), autocast_ctx:
            metrics["mmlu_acc"] = run_chat_eval(
                "MMLU",
                orig_model,
                tokenizer,
                engine,
                batch_size=args.device_batch_size * 2,
                max_problems=args.eval_metrics_max_problems,
            )
            metrics["arc_easy_acc"] = run_chat_eval(
                "ARC-Easy",
                orig_model,
                tokenizer,
                engine,
                batch_size=args.device_batch_size * 2,
                max_problems=args.eval_metrics_max_problems,
            )
        metrics_str = ", ".join(f"{k}: {v:.6f}" for k, v in metrics.items())
        print0(f"Step {step:05d} | {metrics_str}")
        wandb_run.log({
            "step": step,
            **metrics,
        })
        model.train()

    if last_step:
        break

    num_tokens = torch.tensor(0, device=device)
    step_total_loss = torch.tensor(0.0, device=device)
    step_llm_loss = torch.tensor(0.0, device=device)
    step_jepa_loss = torch.tensor(0.0, device=device)
    step_jepa_microbatches = torch.tensor(0, device=device)
    step_jepa_pairs = torch.tensor(0, device=device)

    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_loader)

        with autocast_ctx:
            llm_loss = model(train_inputs, train_targets)
            total_loss = llm_loss
            jepa_loss_mean = None
            jepa_pair_count = 0

            should_compute_jepa_tensor = torch.zeros(1, device=device)
            if master_process:
                should_compute_jepa_tensor.fill_(
                    1.0 if (use_jepa and torch.rand(1).item() >= jepa_dropout) else 0.0
                )
            if ddp:
                dist.broadcast(should_compute_jepa_tensor, src=0)
            should_compute_jepa = should_compute_jepa_tensor.item() > 0.5
            if should_compute_jepa:
                views_a = []
                views_b = []
                for b in range(train_inputs.shape[0]):
                    active_positions = (train_targets[b] >= 0).nonzero(as_tuple=True)[0]
                    if len(active_positions) == 0:
                        continue

                    effective_len = active_positions[-1].item() + 1
                    seq = train_inputs[b, :effective_len]

                    user_ids, assistant_ids = extract_last_turn_views(
                        seq,
                        effective_len,
                        USER_START_ID,
                        ASSISTANT_START_ID,
                    )
                    if user_ids is None or assistant_ids is None:
                        continue

                    views_a.append(user_ids[-512:])
                    views_b.append(assistant_ids[:512])

                if views_a:
                    jepa_pair_count = len(views_a)
                    jepa_loss_mean = compute_jepa_loss_batched(
                        orig_model, views_a, views_b, PRED_TOKEN_ID, device,
                    )
                    jepa_lambda_t = get_jepa_lambda(jepa_base_lambda, step, num_iterations, jepa_schedule)
                    total_loss = total_loss + jepa_lambda_t * jepa_loss_mean

        loss = total_loss / grad_accum_steps
        loss.backward()

        num_tokens += (train_targets >= 0).sum()
        step_total_loss += total_loss.detach()
        step_llm_loss += llm_loss.detach()
        if jepa_loss_mean is not None:
            step_jepa_loss += jepa_loss_mean.detach()
            step_jepa_microbatches += 1
            step_jepa_pairs += jepa_pair_count

    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)
        dist.all_reduce(step_total_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(step_llm_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(step_jepa_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(step_jepa_microbatches, op=dist.ReduceOp.SUM)
        dist.all_reduce(step_jepa_pairs, op=dist.ReduceOp.SUM)

    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    train_loss_item = (step_total_loss / grad_accum_steps).item()
    llm_loss_item = (step_llm_loss / grad_accum_steps).item()
    num_tokens_item = num_tokens.item()
    jepa_loss_item = None
    jepa_lambda_now = get_jepa_lambda(jepa_base_lambda, step, num_iterations, jepa_schedule) if use_jepa else 0.0
    if step_jepa_microbatches.item() > 0:
        jepa_loss_item = (step_jepa_loss / step_jepa_microbatches).item()
        print0(
            f"Step {step:05d}/{num_iterations:05d} | "
            f"Training loss: {train_loss_item:.6f} | "
            f"llm_loss: {llm_loss_item:.6f} | "
            f"jepa_loss: {jepa_loss_item:.6f} (λ={jepa_lambda_now:.3f}) | "
            f"jepa_pairs: {int(step_jepa_pairs.item())} | "
            f"lrm: {lrm:.6f} | "
            f"num_tokens: {num_tokens_item:,}"
        )
    else:
        print0(
            f"Step {step:05d}/{num_iterations:05d} | "
            f"Training loss: {train_loss_item:.6f} | "
            f"llm_loss: {llm_loss_item:.6f} | "
            f"jepa_loss: skipped | "
            f"lrm: {lrm:.6f} | "
            f"num_tokens: {num_tokens_item:,}"
        )

    log_data = {
        "step": step,
        "lrm": lrm,
        "train_loss": train_loss_item,
        "llm_loss": llm_loss_item,
        "num_tokens": num_tokens_item,
    }
    if use_jepa:
        log_data["jepa_lambda"] = jepa_lambda_now
    if jepa_loss_item is not None:
        log_data["jepa_loss"] = jepa_loss_item
        log_data["jepa_pairs"] = int(step_jepa_pairs.item())
    wandb_run.log(log_data)

# Save the model at the end of the run
if master_process:
    base_dir = get_base_dir()
    depth = orig_model.config.n_layer
    output_dirname = f"{args.model_tag}_jepa" if args.model_tag else f"d{depth}_jepa"
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
    model_config_kwargs = orig_model.config.__dict__
    save_checkpoint(
        checkpoint_dir,
        step,
        orig_model.state_dict(),
        None,
        {
            "step": step,
            "val_loss": val_loss,
            **metrics,
            "model_config": model_config_kwargs,
        },
    )
    print(f"Saved model checkpoint to {checkpoint_dir}")

# Log to report
get_report().log(section="Chat SFT + JEPA", data=[
    user_config,
    {
        "Training rows": len(train_ds),
        "Number of iterations": num_iterations,
        "Training loss": train_loss_item,
        "LLM loss": llm_loss_item,
        "JEPA loss": jepa_loss_item,
        "Validation loss": val_loss,
    },
])

# Cleanup
wandb_run.finish()
compute_cleanup()
