"""
Upload a nanochat checkpoint to HuggingFace Hub.

Usage:
    python -m scripts.upload_to_hf \
        --checkpoint-dir ~/.cache/nanochat/base_checkpoints/d12 \
        --step 20000 \
        --repo-name throbbey/crate-d12-base

Or using the --source shorthand (resolves checkpoint dirs automatically):
    python -m scripts.upload_to_hf \
        --source base \
        --model-tag d12 \
        --step 20000 \
        --repo-name throbbey/crate-d12-base
"""

import os
import json
import shutil
import argparse
import tempfile

import torch
from safetensors.torch import save_file
from huggingface_hub import HfApi

from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import load_checkpoint, find_last_step


SOURCE_DIR_MAP = {
    "base": "base_checkpoints",
    "mid": "mid_checkpoints",
    "sft": "chatsft_checkpoints",
    "rl": "chatrl_checkpoints",
    "semisup": "semisup_checkpoints",
    "semisup_code": "semisup_code_checkpoints",
    "semisup_general": "semisup_general_checkpoints",
    "semisup_math": "semisup_math_checkpoints",
}


def resolve_checkpoint_dir(source, model_tag):
    base_dir = get_base_dir()
    if source not in SOURCE_DIR_MAP:
        raise ValueError(f"Unknown source '{source}'. Choose from: {list(SOURCE_DIR_MAP)}")
    checkpoints_dir = os.path.join(base_dir, SOURCE_DIR_MAP[source])
    if model_tag:
        return os.path.join(checkpoints_dir, model_tag)
    # Auto-detect: pick the first (only) subdirectory
    subdirs = [d for d in os.listdir(checkpoints_dir)
               if os.path.isdir(os.path.join(checkpoints_dir, d))]
    if len(subdirs) == 1:
        return os.path.join(checkpoints_dir, subdirs[0])
    raise ValueError(
        f"Multiple model tags found in {checkpoints_dir}: {subdirs}. "
        "Specify --model-tag explicitly."
    )


def generate_model_card(meta_data, step, repo_name):
    model_config = meta_data.get("model_config", {})
    user_config = meta_data.get("user_config", {})
    loop_state = meta_data.get("loop_state", {})

    n_layer = model_config.get("n_layer", "?")
    n_embd = model_config.get("n_embd", "?")
    n_head = model_config.get("n_head", "?")
    vocab_size = model_config.get("vocab_size", "?")
    seq_len = model_config.get("sequence_len", "?")
    window_pattern = model_config.get("window_pattern", "?")

    val_bpb = meta_data.get("val_bpb", "N/A")
    if isinstance(val_bpb, float):
        val_bpb = f"{val_bpb:.4f}"
    train_loss = loop_state.get("smooth_train_loss", "N/A")
    if isinstance(train_loss, float):
        train_loss = f"{train_loss:.4f}"
    training_time_s = loop_state.get("total_training_time", 0)
    training_time_h = training_time_s / 3600 if training_time_s else "N/A"
    if isinstance(training_time_h, float):
        training_time_h = f"{training_time_h:.1f}"

    run_name = user_config.get("run", "unknown")
    total_batch_size = user_config.get("total_batch_size", "?")

    # Use real Unicode chars assembled outside the f-string to avoid escape issues
    ALPHA = "\u03b1"       # Greek alpha: α
    TIMES = "\u00d7"       # Multiplication sign: ×
    TREE_T = "\u251c\u2500\u2500"   # ├──
    TREE_L = "\u2514\u2500\u2500"   # └──
    TREE_I = "\u2502"               # │
    ARROW = "\u2192"                # →

    title = repo_name.split('/')[-1]

    return f"""---
tags:
- nanochat
- crate
- white-box
- sparse-coding
license: mit
---

# {title}

A **CRATE-{ALPHA}** (Coding RAte reduction TransformEr) language model trained with
[nanochat-crate-a](https://github.com/modularflow/nanochat-crate-a), a fork of
[nanochat](https://github.com/karpathy/nanochat) that integrates the CRATE
white-box transformer architecture, SDPA/Flash Attention, and a self-supervised
pseudo-labeling pipeline for domain-specific mid-training and fine-tuning.

This checkpoint serves as the **baseline** for a series of experiments exploring
self-supervised learning for mid-training and fine-tuning with the CRATE
architecture.

## What is CRATE?

CRATE is a **white-box transformer** -- unlike standard transformers where the
architecture is heuristically designed, every layer of CRATE is mathematically
derived from a principled optimization objective. Each layer alternates between
two operations:

1. **MSSA (Multi-Head Subspace Self-Attention)** -- a *compression* step that
   performs gradient descent on the *coding rate reduction* objective. Q, K, and
   V share a single tied projection matrix, which means the attention operation
   is compressing token representations into low-dimensional subspaces.

2. **ODL (Overcomplete Dictionary Learning)** -- a *sparsification* step that
   projects tokens into an overcomplete dictionary space (4{TIMES} expansion),
   applies a sparse activation, and projects back. This encourages the model to
   learn sparse, interpretable representations at every layer.

The net effect is that each forward pass solves a structured optimization
problem: *compress* and *sparsify* the representation, layer by layer. The
resulting internal representations are significantly more interpretable than
those of standard transformers.

### Why ReLU Instead of Soft-Thresholding?

The original CRATE paper (NeurIPS 2023) used ISTA-style **soft-thresholding**
as the sparse activation: `S_lambda(x) = sign(x) * max(|x| - lambda, 0)`.
This is the theoretically "correct" proximal operator for L1-regularized sparse
coding, but it caused training instability at scale.

CRATE-{ALPHA} (NeurIPS 2024) introduced three modifications that enable scaling:

| Change | Vanilla CRATE | CRATE-{ALPHA} |
|--------|--------------|------------|
| Dictionary | Complete (d {TIMES} d) | Overcomplete (d {TIMES} 4d) |
| Activation | Soft-threshold | **ReLU** with learnable bias |
| Sparse block | No residual | **Residual connection** |

**ReLU** works better for scaling because: (a) it has a well-behaved gradient
everywhere (no sign discontinuity), (b) the learnable threshold/bias allows
each neuron to adaptively set its own sparsity level during training, and
(c) ReLU is heavily optimized in GPU kernels. The resulting ODL block looks
structurally similar to a standard MLP -- but it is *derived from* sparse coding
principles rather than heuristically chosen, giving it a principled
interpretation as dictionary learning.

## Evaluation: MMLU

This model is evaluated against **MMLU** (Massive Multitask Language
Understanding), a benchmark of 57 subjects spanning STEM, humanities, social
sciences, and professional domains. MMLU tests the model's ability to answer
multiple-choice questions requiring world knowledge and reasoning -- from
abstract algebra and anatomy to US foreign policy and virology. It provides a
broad signal for how much general knowledge the model has absorbed during
pre-training.

## Baseline for Self-Supervised Experiments

This checkpoint is the starting point for a multi-stage experimental pipeline:

```
crate-d12-base (this model)
{TREE_T} {ARROW} Code self-supervised (learn structural patterns from code)
{TREE_I}      {TREE_L} {ARROW} Mid-training (adapt to chat/instruction format)
{TREE_I}             {TREE_L} {ARROW} General self-supervised (broad knowledge via SmolTalk)
{TREE_I}                    {TREE_L} {ARROW} Math self-supervised (reasoning via GSM8K)
{TREE_I}                           {TREE_L} {ARROW} Chat SFT (final instruction tuning)
{TREE_T} {ARROW} Direct mid-training (comparison branch)
{TREE_L} {ARROW} Other experimental forks
```

The self-supervised stages use **pseudo-labeling**: the model generates candidate
responses for unlabeled prompts, scores them by confidence (average log-probability)
or task reward, filters to the highest-quality candidates, and trains on the
result. This loop can be iterated multiple times, progressively improving the
model's own training signal.

The hypothesis driving the pipeline order is that learning **code structure
first** (syntax, nesting, logical flow) provides transferable structural priors
that benefit subsequent natural language learning -- the model learns "systems
of systems" thinking from code before encountering sentence structure and
general knowledge.

## Model Details

| Parameter | Value |
|-----------|-------|
| Architecture | CRATE-{ALPHA} |
| Layers | {n_layer} |
| Hidden dim | {n_embd} |
| Attention heads | {n_head} |
| Vocab size | {vocab_size} |
| Max sequence length | {seq_len} |
| Window pattern | {window_pattern} |
| ODL expansion | 4{TIMES} (overcomplete dictionary) |
| Sparse activation | ReLU with learnable threshold |
| Training step | {step:,} |
| Validation BPB | {val_bpb} |
| Smooth train loss | {train_loss} |
| Training time | {training_time_h} hours |
| Run name | {run_name} |
| Batch size (tokens) | {total_batch_size} |

## Files

- `model.safetensors` -- model weights in safetensors format
- `config.json` -- model architecture config (reconstruct with `CRATEConfig(**config)`)
- `tokenizer.pkl` -- BPE tokenizer (pickle of tiktoken Encoding)
- `token_bytes.pt` -- token byte mappings
- `meta.json` -- full training metadata from the checkpoint

## Usage

```python
from nanochat.checkpoint_manager import build_model

model, tokenizer, meta = build_model("path/to/downloaded/dir", step={step}, device=torch.device("cuda"), phase="eval")
```

## References

- Yu et al., "White-Box Transformers via Sparse Rate Reduction" (NeurIPS 2023) -- original CRATE
- Yang et al., "Scaling White-Box Transformers for Vision" (NeurIPS 2024) -- CRATE-{ALPHA}
- Hendrycks et al., "Measuring Massive Multitask Language Understanding" (ICLR 2021) -- MMLU

## License

This model is released under the **MIT License**.

Built on:
- [nanochat-crate-a](https://github.com/modularflow/nanochat-crate-a) -- CRATE integration, self-supervised pipeline, SDPA/Flash Attention
- [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy -- MIT License, Copyright (c) 2025
- [CRATE](https://github.com/Ma-Lab-Berkeley/CRATE) (White-Box Transformers via Sparse Rate Reduction) by Ma-Lab-Berkeley -- MIT License, Copyright (c) 2023
- [CRATE-{ALPHA}](https://github.com/UCSC-VLAA/CRATE-alpha) (Scaling White-Box Transformers for Vision) by UCSC-VLAA
"""


def main():
    parser = argparse.ArgumentParser(description="Upload a nanochat checkpoint to HuggingFace Hub")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint-dir", type=str,
                       help="Direct path to the checkpoint directory (e.g. ~/.cache/nanochat/base_checkpoints/d12)")
    group.add_argument("--source", type=str, choices=list(SOURCE_DIR_MAP),
                       help="Source stage name (resolves checkpoint dir automatically)")

    parser.add_argument("--model-tag", type=str, default=None,
                        help="Model tag subdirectory (e.g. d12). Required with --source if multiple tags exist.")
    parser.add_argument("--step", type=int, default=None,
                        help="Checkpoint step to upload. Defaults to the latest step.")
    parser.add_argument("--repo-name", type=str, required=True,
                        help="HuggingFace repo ID (e.g. throbbey/crate-d12-base)")
    parser.add_argument("--private", action="store_true",
                        help="Create a private repository")

    args = parser.parse_args()

    # Resolve checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = os.path.expanduser(args.checkpoint_dir)
    else:
        checkpoint_dir = resolve_checkpoint_dir(args.source, args.model_tag)

    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Resolve step
    step = args.step if args.step is not None else find_last_step(checkpoint_dir)
    print(f"Uploading step {step} from {checkpoint_dir}")

    # Load checkpoint
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device=torch.device("cpu"))

    # Strip torch.compile prefix if present
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    # Stage everything in a temp directory
    staging_dir = tempfile.mkdtemp(prefix="hf_upload_")
    try:
        print(f"Staging files in {staging_dir}")

        # 1. Save model weights as safetensors
        safetensors_path = os.path.join(staging_dir, "model.safetensors")
        save_file(model_data, safetensors_path)
        print(f"  Saved model.safetensors ({os.path.getsize(safetensors_path) / 1e6:.1f} MB)")

        # 2. Save model config
        config_path = os.path.join(staging_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(meta_data.get("model_config", {}), f, indent=2)
        print(f"  Saved config.json")

        # 3. Save full metadata
        meta_path = os.path.join(staging_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=2)
        print(f"  Saved meta.json")

        # 4. Copy tokenizer files
        base_dir = get_base_dir()
        tokenizer_dir = os.path.join(base_dir, "tokenizer")
        for tok_file in ["tokenizer.pkl", "token_bytes.pt"]:
            src = os.path.join(tokenizer_dir, tok_file)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(staging_dir, tok_file))
                print(f"  Copied {tok_file}")
            else:
                print(f"  Warning: {tok_file} not found at {src}, skipping")

        # 5. Generate README model card
        readme_path = os.path.join(staging_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(generate_model_card(meta_data, step, args.repo_name))
        print(f"  Generated README.md")

        # Upload to HuggingFace
        api = HfApi()
        print(f"\nCreating repo: {args.repo_name} (private={args.private})")
        api.create_repo(args.repo_name, private=args.private, exist_ok=True)

        print(f"Uploading files...")
        api.upload_folder(
            folder_path=staging_dir,
            repo_id=args.repo_name,
            commit_message=f"Upload CRATE checkpoint at step {step}",
        )

        print(f"\nDone! Model uploaded to: https://huggingface.co/{args.repo_name}")

    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
