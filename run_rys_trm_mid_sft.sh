#!/bin/bash
# Mid-train + Chat SFT for all RYS/TRM variants on a single RTX 4090.
#
# Tags (4 total):
#   - d12-trm-4090                   (TRM-GPT, no JEPA)
#   - d12-rys-4090-2                 (RYS-GPT, no JEPA)
#   - d12-trm-jepa-lindecay-4090     (TRM-GPT + JEPA)
#   - d12-rys-jepa-lindecay-4090     (RYS-GPT + JEPA)
#
# Non-JEPA runs: jepa_lambda=0 throughout.
# JEPA runs: mid λ=0.25 constant, SFT λ=0.5 constant (matching existing runs).
#
# Each stage checks for existing checkpoints and skips if present.
#
# Usage:
#   bash run_rys_trm_mid_sft.sh
#   SKIP_MID=1 bash run_rys_trm_mid_sft.sh       # SFT only
#   SKIP_SFT=1 bash run_rys_trm_mid_sft.sh       # mid-train only

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

# Mid-train defaults
MID_DEVICE_BATCH_SIZE="${MID_DEVICE_BATCH_SIZE:-8}"
MID_TOTAL_BATCH_SIZE="${MID_TOTAL_BATCH_SIZE:-65536}"
MID_NUM_ITERATIONS="${MID_NUM_ITERATIONS:--1}"
MID_EVAL_EVERY="${MID_EVAL_EVERY:-150}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
JEPA_DROPOUT="${JEPA_DROPOUT:-0.5}"

# SFT defaults
SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-4}"
SFT_TARGET_EXAMPLES="${SFT_TARGET_EXAMPLES:-32}"
SFT_NUM_EPOCHS="${SFT_NUM_EPOCHS:-1}"
SFT_EVAL_EVERY="${SFT_EVAL_EVERY:-100}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-100}"
SFT_EVAL_METRICS_EVERY="${SFT_EVAL_METRICS_EVERY:-200}"
SFT_EVAL_METRICS_MAX="${SFT_EVAL_METRICS_MAX:-1024}"

WANDB_PREFIX="${WANDB_PREFIX:-trm-rys}"

SKIP_MID="${SKIP_MID:-0}"
SKIP_SFT="${SKIP_SFT:-0}"

CACHE_DIR="${HOME}/.cache/nanochat"

# ============================================================
# Experiment definitions
# Format: model_tag|mid_jepa_lambda|mid_jepa_schedule|sft_jepa_lambda|sft_tag_suffix
# ============================================================
declare -a EXPERIMENTS=(
    "d12-trm-4090|0|constant|0|"
    "d12-rys-4090-2|0|constant|0|"
    "d12-trm-jepa-lindecay-4090|0.25|constant|0.5|_jepa"
    "d12-rys-jepa-lindecay-4090|0.25|constant|0.5|_jepa"
)

echo "========================================"
echo "RYS + TRM — Mid-train + Chat SFT (RTX 4090)"
echo "  Experiments: ${#EXPERIMENTS[@]}"
echo "  Skip mid-train: ${SKIP_MID}"
echo "  Skip SFT: ${SKIP_SFT}"
echo "========================================"
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r tag mjlam mjsched sjlam sft_suffix <<< "$exp"
    mid_str="off"; if [ "$mjlam" != "0" ]; then mid_str="λ=${mjlam} ${mjsched}"; fi
    sft_str="off"; if [ "$sjlam" != "0" ]; then sft_str="λ=${sjlam}"; fi
    echo "  ${tag}  (mid JEPA=${mid_str}, SFT JEPA=${sft_str})"
done
echo "========================================"
echo ""

# ============================================================
# STAGE 1: Mid-train
# ============================================================

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r tag mjlam mjsched sjlam sft_suffix <<< "$exp"

    if [ "${SKIP_MID}" = "1" ]; then
        echo "[mid-train] SKIPPED: ${tag} (SKIP_MID=1)"
        continue
    fi

    if [ -d "${CACHE_DIR}/mid_checkpoints/${tag}" ]; then
        echo "[mid-train] EXISTS: ${tag} — skipping"
        continue
    fi

    if [ ! -d "${CACHE_DIR}/base_checkpoints/${tag}" ]; then
        echo "[mid-train] NO BASE: ${tag} — skipping (base checkpoint not found)"
        continue
    fi

    mid_str="off"; if [ "$mjlam" != "0" ]; then mid_str="λ=${mjlam} ${mjsched}"; fi

    echo ""
    echo "================================================================"
    echo "[mid-train] ${tag}  (JEPA=${mid_str})"
    echo "================================================================"

    local_wandb="${WANDB_PREFIX}"
    if [ "${local_wandb}" != "dummy" ]; then local_wandb="${WANDB_PREFIX}-mid-${tag}"; fi

    ${PYTHON_BIN} -m scripts.mid_train_jepa \
        --source base \
        --model-tag "${tag}" \
        --run "${local_wandb}" \
        --device-batch-size "${MID_DEVICE_BATCH_SIZE}" \
        --max-seq-len "${MAX_SEQ_LEN}" \
        --total-batch-size "${MID_TOTAL_BATCH_SIZE}" \
        --num-iterations "${MID_NUM_ITERATIONS}" \
        --jepa-lambda "${mjlam}" \
        --jepa-schedule "${mjsched}" \
        --jepa-dropout "${JEPA_DROPOUT}" \
        --eval-every "${MID_EVAL_EVERY}"

    echo "[mid-train] DONE: ${tag} -> ${CACHE_DIR}/mid_checkpoints/${tag}/"
done

# ============================================================
# STAGE 2: Chat SFT
# ============================================================

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r tag mjlam mjsched sjlam sft_suffix <<< "$exp"

    sft_out_tag="${tag}${sft_suffix}"

    if [ "${SKIP_SFT}" = "1" ]; then
        echo "[sft] SKIPPED: ${sft_out_tag} (SKIP_SFT=1)"
        continue
    fi

    if [ -d "${CACHE_DIR}/chatsft_checkpoints/${sft_out_tag}" ]; then
        echo "[sft] EXISTS: ${sft_out_tag} — skipping"
        continue
    fi

    if [ ! -d "${CACHE_DIR}/mid_checkpoints/${tag}" ]; then
        echo "[sft] NO MID: ${sft_out_tag} — skipping (mid checkpoint not found for ${tag})"
        continue
    fi

    sft_str="off"; if [ "$sjlam" != "0" ]; then sft_str="λ=${sjlam}"; fi

    echo ""
    echo "================================================================"
    echo "[sft] ${sft_out_tag}  (source=mid:${tag}, JEPA=${sft_str})"
    echo "================================================================"

    local_wandb="${WANDB_PREFIX}"
    if [ "${local_wandb}" != "dummy" ]; then local_wandb="${WANDB_PREFIX}-sft-${tag}"; fi

    ${PYTHON_BIN} -m scripts.chat_sft_jepa \
        --source mid \
        --model-tag "${tag}" \
        --run "${local_wandb}" \
        --num-epochs "${SFT_NUM_EPOCHS}" \
        --device-batch-size "${SFT_DEVICE_BATCH_SIZE}" \
        --target-examples-per-step "${SFT_TARGET_EXAMPLES}" \
        --jepa-lambda "${sjlam}" \
        --jepa-schedule constant \
        --jepa-dropout "${JEPA_DROPOUT}" \
        --eval-every "${SFT_EVAL_EVERY}" \
        --eval-steps "${SFT_EVAL_STEPS}" \
        --eval-metrics-every "${SFT_EVAL_METRICS_EVERY}" \
        --eval-metrics-max-problems "${SFT_EVAL_METRICS_MAX}"

    echo "[sft] DONE: ${sft_out_tag} -> ${CACHE_DIR}/chatsft_checkpoints/${sft_out_tag}/"
done

# ============================================================
# Summary
# ============================================================

echo ""
echo "========================================"
echo "RYS + TRM mid-train + SFT complete!"
echo ""
echo "Mid checkpoints:"
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r tag mjlam mjsched sjlam sft_suffix <<< "$exp"
    if [ -d "${CACHE_DIR}/mid_checkpoints/${tag}" ]; then
        echo "  [OK]  ${tag}"
    else
        echo "  [--]  ${tag} (not found)"
    fi
done
echo ""
echo "SFT checkpoints:"
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r tag mjlam mjsched sjlam sft_suffix <<< "$exp"
    sft_out_tag="${tag}${sft_suffix}"
    if [ -d "${CACHE_DIR}/chatsft_checkpoints/${sft_out_tag}" ]; then
        echo "  [OK]  ${sft_out_tag}"
    else
        echo "  [--]  ${sft_out_tag} (not found)"
    fi
done
echo "========================================"
