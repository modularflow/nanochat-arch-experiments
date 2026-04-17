#!/bin/bash
# Chat SFT with JEPA on a single RTX 4090 (24GB).
#
# Usage:
#   bash run_chat_sft_jepa_4090.sh
#   MODEL_TAG=d12-jepa-4090 WANDB_RUN=sft-jepa bash run_chat_sft_jepa_4090.sh
#   SOURCE=mid MODEL_STEP=1200 DEVICE_BATCH_SIZE=4 bash run_chat_sft_jepa_4090.sh

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

SOURCE="${SOURCE:-mid}"
MODEL_TAG="${MODEL_TAG:-d12-jepa-4090}"
MODEL_STEP="${MODEL_STEP:-}"
WANDB_RUN="${WANDB_RUN:-dummy}"

# Conservative SFT defaults for a 24GB 4090.
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"
TARGET_EXAMPLES_PER_STEP="${TARGET_EXAMPLES_PER_STEP:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
JEPA_LAMBDA="${JEPA_LAMBDA:-0.5}"
JEPA_SCHEDULE="${JEPA_SCHEDULE:-constant}"
JEPA_DROPOUT="${JEPA_DROPOUT:-0.5}"
EVAL_EVERY="${EVAL_EVERY:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
EVAL_METRICS_EVERY="${EVAL_METRICS_EVERY:-200}"
EVAL_METRICS_MAX_PROBLEMS="${EVAL_METRICS_MAX_PROBLEMS:-1024}"

STEP_FLAG=""
if [ -n "$MODEL_STEP" ]; then
    STEP_FLAG="--model-step $MODEL_STEP"
fi

echo "========================================"
echo "Chat SFT + JEPA (RTX 4090)"
echo "  Python: ${PYTHON_BIN}"
echo "  Source: ${SOURCE}"
echo "  Model tag: ${MODEL_TAG}"
if [ -n "$MODEL_STEP" ]; then
    echo "  Model step: ${MODEL_STEP}"
else
    echo "  Model step: (latest)"
fi
echo "  Device batch size: ${DEVICE_BATCH_SIZE}"
echo "  Target examples per step: ${TARGET_EXAMPLES_PER_STEP}"
echo "  JEPA lambda: ${JEPA_LAMBDA}"
echo "  JEPA schedule: ${JEPA_SCHEDULE}"
echo "  JEPA dropout: ${JEPA_DROPOUT}"
echo "========================================"

"${PYTHON_BIN}" -m scripts.chat_sft_jepa \
    --source "${SOURCE}" \
    --model-tag "${MODEL_TAG}" \
    ${STEP_FLAG} \
    --run "${WANDB_RUN}" \
    --num-epochs "${NUM_EPOCHS}" \
    --device-batch-size "${DEVICE_BATCH_SIZE}" \
    --target-examples-per-step "${TARGET_EXAMPLES_PER_STEP}" \
    --jepa-lambda "${JEPA_LAMBDA}" \
    --jepa-schedule "${JEPA_SCHEDULE}" \
    --jepa-dropout "${JEPA_DROPOUT}" \
    --eval-every "${EVAL_EVERY}" \
    --eval-steps "${EVAL_STEPS}" \
    --eval-metrics-every "${EVAL_METRICS_EVERY}" \
    --eval-metrics-max-problems "${EVAL_METRICS_MAX_PROBLEMS}"

echo ""
echo "========================================"
echo "Chat SFT + JEPA complete!"
echo "Checkpoint saved to: ~/.cache/nanochat/chatsft_checkpoints/${MODEL_TAG}_jepa/"
echo "========================================"
