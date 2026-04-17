#!/bin/bash
# Mid-training with JEPA on a single RTX 4090 (24GB).
#
# Usage:
#   bash run_mid_train_jepa_4090.sh
#   SOURCE=base MODEL_TAG=d12-jepa-4090 bash run_mid_train_jepa_4090.sh
#   MODEL_STEP=5000 WANDB_RUN=mid-jepa bash run_mid_train_jepa_4090.sh
#   DEVICE_BATCH_SIZE=6 TOTAL_BATCH_SIZE=65536 bash run_mid_train_jepa_4090.sh

set -euo pipefail

cd ~/nanochat-crate-a

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if [ -x ".venv/bin/python3" ]; then
        PYTHON_BIN=".venv/bin/python3"
    else
        PYTHON_BIN="python3"
    fi
fi

SOURCE="${SOURCE:-base}"
MODEL_TAG="${MODEL_TAG:-d12-jepa-4090}"
MODEL_STEP="${MODEL_STEP:-}"
WANDB_RUN="${WANDB_RUN:-dummy}"

# 4090-friendly defaults for JEPA mid-training.
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-65536}"
NUM_ITERATIONS="${NUM_ITERATIONS:--1}"
JEPA_LAMBDA="${JEPA_LAMBDA:-0.25}"
JEPA_SCHEDULE="${JEPA_SCHEDULE:-constant}"
JEPA_DROPOUT="${JEPA_DROPOUT:-0.5}"
JEPA_VIEW_MAX_LEN="${JEPA_VIEW_MAX_LEN:-256}"
EVAL_EVERY="${EVAL_EVERY:-150}"

STEP_FLAG=""
if [ -n "$MODEL_STEP" ]; then
    STEP_FLAG="--model-step $MODEL_STEP"
fi

echo "========================================"
echo "Mid-training + JEPA (RTX 4090)"
echo "  Python: ${PYTHON_BIN}"
echo "  Source: ${SOURCE}"
echo "  Model tag: ${MODEL_TAG}"
if [ -n "$MODEL_STEP" ]; then
    echo "  Model step: ${MODEL_STEP}"
else
    echo "  Model step: (latest)"
fi
echo "  Device batch size: ${DEVICE_BATCH_SIZE}"
echo "  Total batch size: ${TOTAL_BATCH_SIZE}"
echo "  Max seq len: ${MAX_SEQ_LEN}"
echo "  JEPA lambda: ${JEPA_LAMBDA}"
echo "  JEPA schedule: ${JEPA_SCHEDULE}"
echo "  JEPA dropout: ${JEPA_DROPOUT}"
echo "========================================"

"${PYTHON_BIN}" -m scripts.mid_train_jepa \
    --source "${SOURCE}" \
    --model-tag "${MODEL_TAG}" \
    ${STEP_FLAG} \
    --run "${WANDB_RUN}" \
    --device-batch-size "${DEVICE_BATCH_SIZE}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --total-batch-size "${TOTAL_BATCH_SIZE}" \
    --num-iterations "${NUM_ITERATIONS}" \
    --jepa-lambda "${JEPA_LAMBDA}" \
    --jepa-schedule "${JEPA_SCHEDULE}" \
    --jepa-dropout "${JEPA_DROPOUT}" \
    --jepa-view-max-len "${JEPA_VIEW_MAX_LEN}" \
    --eval-every "${EVAL_EVERY}"

echo ""
echo "========================================"
echo "Mid-training + JEPA complete!"
echo "Checkpoint saved to: ~/.cache/nanochat/mid_checkpoints/${MODEL_TAG}/"
echo "========================================"
