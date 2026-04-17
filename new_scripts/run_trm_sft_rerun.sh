#!/bin/bash
# =============================================================================
# Re-run TRM SFT + CORE eval with the autocast fix in trm_gpt.py
#
# The original TRM SFT ran with a PyTorch autocast caching bug that caused
# shared block weights to receive zero gradients (frozen blocks during SFT).
# This script deletes the stale checkpoints, re-runs SFT from valid mid-train
# checkpoints, and evaluates the new SFT models on CORE.
# =============================================================================

set -euo pipefail

cd ~/nanochat-crate-a

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ]; then
    if [ -x "${SCRIPT_DIR}/.venv/bin/python3" ]; then
        PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python3"
    else
        PYTHON_BIN="python3"
    fi
fi

CACHE_DIR="${HOME}/.cache/nanochat"

# SFT defaults (matching the full sweep)
SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-4}"
SFT_TARGET_EXAMPLES="${SFT_TARGET_EXAMPLES:-32}"
SFT_NUM_EPOCHS="${SFT_NUM_EPOCHS:-1}"
SFT_EVAL_EVERY="${SFT_EVAL_EVERY:-100}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-100}"
SFT_EVAL_METRICS_EVERY="${SFT_EVAL_METRICS_EVERY:-200}"
SFT_EVAL_METRICS_MAX="${SFT_EVAL_METRICS_MAX:-1024}"
SFT_JEPA_LAMBDA="${SFT_JEPA_LAMBDA:-0.5}"
SFT_JEPA_SCHEDULE="${SFT_JEPA_SCHEDULE:-constant}"
SFT_JEPA_DROPOUT="${SFT_JEPA_DROPOUT:-0.5}"
WANDB_PREFIX="${WANDB_PREFIX:-5k-runs}"

PIPELINE_EVAL_MAX_PER_TASK="${PIPELINE_EVAL_MAX_PER_TASK:--1}"

# The 4 TRM base tags and their SFT output tags
declare -a TRM_TAGS=(
    "d12-s5k-trm-ce-4090"
    "d12-s5k-trm-jepa-lin-4090"
    "d12-s5k-trm-gqa2-ce-4090"
    "d12-s5k-trm-gqa2-jepa-lin-4090"
)

tag_is_jepa() { [[ "$1" == *"jepa"* ]]; }

sft_tag_for() {
    if tag_is_jepa "$1"; then echo "${1}_jepa"; else echo "$1"; fi
}

# =============================================================================
# Step 1: Delete stale TRM SFT checkpoints
# =============================================================================

echo ""
echo "================================================================================"
echo "TRM SFT Re-run (autocast fix)"
echo "================================================================================"
echo ""
echo "Step 1: Removing stale TRM SFT checkpoints..."

for tag in "${TRM_TAGS[@]}"; do
    sft_tag="$(sft_tag_for "${tag}")"
    sft_dir="${CACHE_DIR}/chatsft_checkpoints/${sft_tag}"
    if [ -d "${sft_dir}" ]; then
        echo "  Deleting: ${sft_dir}"
        rm -rf "${sft_dir}"
    else
        echo "  Not found (already clean): ${sft_dir}"
    fi
done

# =============================================================================
# Step 2: Re-run SFT for each TRM variant
# =============================================================================

echo ""
echo "================================================================================"
echo "Step 2: Re-running SFT for TRM variants"
echo "================================================================================"

declare -a SFT_SPECS=()

for tag in "${TRM_TAGS[@]}"; do
    mid_dir="${CACHE_DIR}/mid_checkpoints/${tag}"
    sft_tag="$(sft_tag_for "${tag}")"

    if [ ! -d "${mid_dir}" ]; then
        echo "  [sft] NO MID: ${tag} — skipping"
        continue
    fi

    local_wandb="${WANDB_PREFIX}"
    if [ "${local_wandb}" != "dummy" ]; then
        local_wandb="${WANDB_PREFIX}-sft-rerun-${tag}"
    fi

    if tag_is_jepa "${tag}"; then
        echo ""
        echo "  [sft] ${sft_tag}  (source=mid:${tag}, JEPA λ=${SFT_JEPA_LAMBDA})"

        "${PYTHON_BIN}" -m scripts.chat_sft_jepa \
            --source mid \
            --model-tag "${tag}" \
            --run "${local_wandb}" \
            --num-epochs "${SFT_NUM_EPOCHS}" \
            --device-batch-size "${SFT_DEVICE_BATCH_SIZE}" \
            --target-examples-per-step "${SFT_TARGET_EXAMPLES}" \
            --jepa-lambda "${SFT_JEPA_LAMBDA}" \
            --jepa-schedule "${SFT_JEPA_SCHEDULE}" \
            --jepa-dropout "${SFT_JEPA_DROPOUT}" \
            --eval-every "${SFT_EVAL_EVERY}" \
            --eval-steps "${SFT_EVAL_STEPS}" \
            --eval-metrics-every "${SFT_EVAL_METRICS_EVERY}" \
            --eval-metrics-max-problems "${SFT_EVAL_METRICS_MAX}"
    else
        echo ""
        echo "  [sft] ${sft_tag}  (source=mid:${tag}, no JEPA)"

        "${PYTHON_BIN}" -m scripts.chat_sft \
            --source mid \
            --model-tag "${tag}" \
            --run "${local_wandb}" \
            --num-epochs "${SFT_NUM_EPOCHS}" \
            --device-batch-size "${SFT_DEVICE_BATCH_SIZE}" \
            --target-examples-per-step "${SFT_TARGET_EXAMPLES}" \
            --eval-every "${SFT_EVAL_EVERY}" \
            --eval-steps "${SFT_EVAL_STEPS}" \
            --eval-metrics-every "${SFT_EVAL_METRICS_EVERY}" \
            --eval-metrics-max-problems "${SFT_EVAL_METRICS_MAX}"
    fi

    echo "  [sft] DONE: ${sft_tag}"
    SFT_SPECS+=("sft:${sft_tag}")
done

# =============================================================================
# Step 3: CORE eval on the 4 re-run TRM SFT checkpoints
# =============================================================================

if [ ${#SFT_SPECS[@]} -gt 0 ]; then
    echo ""
    echo "================================================================================"
    echo "Step 3: CORE eval on re-run TRM SFT checkpoints (${#SFT_SPECS[@]})"
    echo "================================================================================"
    printf '  %s\n' "${SFT_SPECS[@]}"

    "${PYTHON_BIN}" -m scripts.pipeline_eval \
        --mode core \
        --max-per-task "${PIPELINE_EVAL_MAX_PER_TASK}" \
        --checkpoints "${SFT_SPECS[@]}"
else
    echo ""
    echo "No TRM SFT checkpoints produced — nothing to evaluate."
fi

echo ""
echo "================================================================================"
echo "TRM SFT re-run complete."
echo "================================================================================"
