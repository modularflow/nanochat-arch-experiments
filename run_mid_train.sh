#!/bin/bash
# Mid-training on a single GPU (RTX 4090 24GB)
#
# Usage:
#   bash run_mid_train.sh                                          # base d12, latest step
#   SOURCE=semisup_code MODEL_STEP=200 bash run_mid_train.sh       # from selfsup-code step 200
#   MODEL_TAG=d12 MODEL_STEP=20000 bash run_mid_train.sh           # specific base step
#   WANDB_RUN=my-run bash run_mid_train.sh                         # enable wandb

set -e

cd ~/nanochat-crate-a

SOURCE=${SOURCE:-"base"}
MODEL_TAG=${MODEL_TAG:-"d12"}
MODEL_STEP=${MODEL_STEP:-""}
WANDB_RUN=${WANDB_RUN:-"dummy"}

STEP_FLAG=""
if [ -n "$MODEL_STEP" ]; then
    STEP_FLAG="--model-step $MODEL_STEP"
fi

echo "========================================"
echo "Mid-training"
echo "  Source: ${SOURCE} / ${MODEL_TAG}"
if [ -n "$MODEL_STEP" ]; then
    echo "  Step: ${MODEL_STEP}"
else
    echo "  Step: (latest)"
fi
echo "  Device: single GPU, batch_size=8, seq_len=1024"
echo "========================================"

python -m scripts.mid_train \
    --source "$SOURCE" \
    --model-tag "$MODEL_TAG" \
    $STEP_FLAG \
    --device-batch-size 8 \
    --max-seq-len 1024 \
    --total-batch-size 65536 \
    --num-iterations -1 \
    --eval-every 150 \
    --run "$WANDB_RUN"

echo ""
echo "========================================"
echo "Mid-training complete!"
echo "Checkpoint saved to: ~/.cache/nanochat/mid_checkpoints/${MODEL_TAG}/"
echo "========================================"
