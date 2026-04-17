#!/bin/bash
# Focused pipeline: mid-train + SFT + eval for d12-noqgpt-4090 (No-Q GPT, no JEPA)
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

TAG="d12-noqgpt-4090"
SFT_TAG="${TAG}"
CACHE_DIR="${HOME}/.cache/nanochat"
WANDB_PREFIX="${WANDB_PREFIX:-noq}"

echo "========================================"
echo "No-Q GPT pipeline: mid-train → SFT → eval"
echo "  Base checkpoint: ${TAG}"
echo "  W&B prefix: ${WANDB_PREFIX}"
echo "========================================"

# ============================================================
# STAGE 1: Mid-train
# ============================================================
if [ -d "${CACHE_DIR}/mid_checkpoints/${TAG}" ]; then
    echo "[mid-train] EXISTS: ${TAG} — skipping"
else
    echo ""
    echo "================================================================"
    echo "[mid-train] ${TAG} (JEPA=off)"
    echo "================================================================"

    WANDB_RUN="${WANDB_PREFIX}-mid-${TAG}"
    if [ "${WANDB_PREFIX}" = "dummy" ]; then WANDB_RUN="dummy"; fi

    "${PYTHON_BIN}" -m scripts.mid_train_jepa \
        --source base \
        --model-tag "${TAG}" \
        --run "${WANDB_RUN}" \
        --device-batch-size 8 \
        --max-seq-len 1024 \
        --total-batch-size 65536 \
        --num-iterations -1 \
        --jepa-lambda 0 \
        --jepa-schedule constant \
        --jepa-dropout 0.5 \
        --eval-every 150

    echo "[mid-train] DONE: ${TAG}"
fi

# ============================================================
# STAGE 2: Chat SFT
# ============================================================
if [ -d "${CACHE_DIR}/chatsft_checkpoints/${SFT_TAG}" ]; then
    echo "[sft] EXISTS: ${SFT_TAG} — skipping"
else
    echo ""
    echo "================================================================"
    echo "[sft] ${SFT_TAG} (source=mid:${TAG}, JEPA=off)"
    echo "================================================================"

    WANDB_RUN="${WANDB_PREFIX}-sft-${TAG}"
    if [ "${WANDB_PREFIX}" = "dummy" ]; then WANDB_RUN="dummy"; fi

    "${PYTHON_BIN}" -m scripts.chat_sft_jepa \
        --source mid \
        --model-tag "${TAG}" \
        --run "${WANDB_RUN}" \
        --num-epochs 1 \
        --device-batch-size 4 \
        --target-examples-per-step 32 \
        --jepa-lambda 0 \
        --jepa-schedule constant \
        --jepa-dropout 0.5 \
        --eval-every 100 \
        --eval-steps 100 \
        --eval-metrics-every 200 \
        --eval-metrics-max-problems 1024

    echo "[sft] DONE: ${SFT_TAG}"
fi

# ============================================================
# STAGE 3: Pipeline eval — Base models
# ============================================================
echo ""
echo "================================================================"
echo "[eval] Pipeline evaluation — Base models (CORE)"
echo "================================================================"

BASE_SPECS="base:${TAG}"
for existing in d12-gpt-jepa-lindecay-4090 d12-gpt-jepa-4090 d12-gpt-4090 d12-jepa-4090; do
    if [ -d "${CACHE_DIR}/base_checkpoints/${existing}" ]; then
        BASE_SPECS="${BASE_SPECS} base:${existing}"
    fi
done

echo "  Checkpoints: ${BASE_SPECS}"
"${PYTHON_BIN}" -m scripts.pipeline_eval \
    --mode core \
    --checkpoints ${BASE_SPECS}

# ============================================================
# STAGE 4: Pipeline eval — SFT models
# ============================================================
echo ""
echo "================================================================"
echo "[eval] Pipeline evaluation — SFT models (CORE)"
echo "================================================================"

SFT_SPECS="sft:${SFT_TAG}"
for existing in d12-gpt-jepa-lindecay-4090_jepa d12-gpt-jepa-4090_jepa d12-gpt-4090; do
    if [ -d "${CACHE_DIR}/chatsft_checkpoints/${existing}" ]; then
        SFT_SPECS="${SFT_SPECS} sft:${existing}"
    fi
done

echo "  Checkpoints: ${SFT_SPECS}"
"${PYTHON_BIN}" -m scripts.pipeline_eval \
    --mode core \
    --checkpoints ${SFT_SPECS}

echo ""
echo "========================================"
echo "No-Q GPT pipeline complete!"
echo "  Mid: ${CACHE_DIR}/mid_checkpoints/${TAG}/"
echo "  SFT: ${CACHE_DIR}/chatsft_checkpoints/${SFT_TAG}/"
echo "  Eval: ${CACHE_DIR}/pipeline_eval/"
echo "========================================"
