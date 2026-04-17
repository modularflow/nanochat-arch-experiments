#!/bin/bash
# Base pretraining with Grouped-Query Attention (GQA) on a single RTX 4090 (24GB).
#
# Uses the vanilla GPT stack (--architecture gpt) because CRATE uses tied QKV and
# does not support n_kv_head < n_head.
#
# This run disables the JEPA auxiliary loss (--jepa-lambda 0) so you can compare
# GQA vs full-rank KV without JEPA. For GQA + JEPA, use run_base_train_gqa_jepa_4090.sh.
#
# Usage:
#   bash run_base_train_gqa_4090.sh
#   NUM_KV_HEADS=3 bash run_base_train_gqa_4090.sh   # e.g. 6 query / 3 KV
#   MODEL_TAG=d12-gqa-4090-v2 WANDB_RUN=my-run bash run_base_train_gqa_4090.sh

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

WANDB_RUN="${WANDB_RUN:-dummy}"
MODEL_TAG="${MODEL_TAG:-d12-gqa-4090}"
ARCHITECTURE="${ARCHITECTURE:-gpt}"

# d12 → 768 dim → 6 heads @ 128d; NUM_KV_HEADS must divide 6 (e.g. 2 or 3).
NUM_KV_HEADS="${NUM_KV_HEADS:-2}"

DEPTH="${DEPTH:-12}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
HEAD_DIM="${HEAD_DIM:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-65536}"
NUM_ITERATIONS="${NUM_ITERATIONS:-5000}"

RESUME_FROM_STEP="${RESUME_FROM_STEP:--1}"

# No JEPA — pure CE + GQA
JEPA_LAMBDA="${JEPA_LAMBDA:-0}"
JEPA_SCHEDULE="${JEPA_SCHEDULE:-constant}"
JEPA_DROPOUT="${JEPA_DROPOUT:-0.5}"
JEPA_VIEW_MIN_LEN="${JEPA_VIEW_MIN_LEN:-64}"

EVAL_EVERY="${EVAL_EVERY:-250}"
EVAL_TOKENS="${EVAL_TOKENS:-131072}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
SAMPLE_EVERY="${SAMPLE_EVERY:--1}"
SAVE_EVERY="${SAVE_EVERY:-1000}"

echo "========================================"
echo "Base pretraining + GQA (RTX 4090, no JEPA)"
echo "  Python: ${PYTHON_BIN}"
echo "  Model tag: ${MODEL_TAG}"
echo "  Architecture: ${ARCHITECTURE}"
echo "  num_kv_heads: ${NUM_KV_HEADS}"
echo "  Depth: ${DEPTH}"
echo "  JEPA lambda: ${JEPA_LAMBDA} (0 = off)"
echo "========================================"

RESUME_ARG=""
if [ "${RESUME_FROM_STEP}" != "-1" ]; then
    RESUME_ARG="--resume-from-step ${RESUME_FROM_STEP}"
fi

"${PYTHON_BIN}" -m scripts.base_train_jepa \
    --run "${WANDB_RUN}" \
    --architecture "${ARCHITECTURE}" \
    --depth "${DEPTH}" \
    --aspect-ratio "${ASPECT_RATIO}" \
    --head-dim "${HEAD_DIM}" \
    --num-kv-heads "${NUM_KV_HEADS}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --window-pattern "${WINDOW_PATTERN}" \
    --num-iterations "${NUM_ITERATIONS}" \
    --device-batch-size "${DEVICE_BATCH_SIZE}" \
    --total-batch-size "${TOTAL_BATCH_SIZE}" \
    --jepa-lambda "${JEPA_LAMBDA}" \
    --jepa-schedule "${JEPA_SCHEDULE}" \
    --jepa-dropout "${JEPA_DROPOUT}" \
    --jepa-view-min-len "${JEPA_VIEW_MIN_LEN}" \
    --eval-every "${EVAL_EVERY}" \
    --eval-tokens "${EVAL_TOKENS}" \
    --core-metric-every "${CORE_METRIC_EVERY}" \
    --sample-every "${SAMPLE_EVERY}" \
    --save-every "${SAVE_EVERY}" \
    --model-tag "${MODEL_TAG}" \
    ${RESUME_ARG}

echo ""
echo "========================================"
echo "GQA base training complete!"
echo "Checkpoint: ~/.cache/nanochat/base_checkpoints/${MODEL_TAG}/"
echo "========================================"
