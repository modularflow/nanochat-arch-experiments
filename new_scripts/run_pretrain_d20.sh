#!/bin/bash
# Pretrain a d20 vanilla GPT (nanochat.gpt) up to step 5000.
#
# Usage:
#   bash run_pretrain_d20.sh
#   WANDB_RUN="my-run-name" bash run_pretrain_d20.sh

set -euo pipefail

WANDB_RUN="${WANDB_RUN:-dummy}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"

python -m scripts.base_train \
    --run "$WANDB_RUN" \
    --depth 20 \
    --aspect-ratio 64 \
    --head-dim 128 \
    --max-seq-len 1024 \
    --window-pattern "SSSL" \
    --num-iterations 5000 \
    --device-batch-size "$DEVICE_BATCH_SIZE" \
    --total-batch-size 65536 \
    --eval-every 250 \
    --core-metric-every 2500 \
    --core-metric-max-per-task 500 \
    --sample-every 2500 \
    --save-every 1000 \
    --warmup-ratio 0.0 \
    --warmdown-ratio 0.4 \
    --model-tag d20
