#!/bin/bash
# =============================================================================
# 10k continuation models: mid-train → SFT → eval (base CORE + SFT CORE)
#
# The 6 models from run_continue_10k.sh now have 10k base checkpoints.
# Their mid/SFT dirs from the 5k run still exist, so this script archives
# them (→ *_5k) before re-running mid-training and SFT from the 10k base.
#
# Models:
#   1. d12-s5k-gpt-ce-4090         (gpt, CE)
#   2. d12-s5k-gpt-jepa-lin-4090   (gpt, JEPA linear_decay)
#   3. d12-s5k-gpt-gqa2-ce-4090    (gpt GQA2, CE)
#   4. d12-s5k-rys-ce-4090         (rys, CE)
#   5. d12-s5k-rys-jepa-lin-4090   (rys, JEPA linear_decay)
#   6. d12-s5k-trm-ce-4090         (trm, CE)
#
# Usage:
#   tmux new -s sft-eval
#   bash run_10k_sft_eval.sh
#
# Skip phases:
#   SKIP_MID=1      — skip mid-training
#   SKIP_SFT=1      — skip SFT
#   SKIP_EVAL=1     — skip all evals
#   SKIP_SFT_EVAL=1 — skip SFT eval only (still run base eval)
# =============================================================================

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

CACHE_DIR="${HOME}/.cache/nanochat"

SKIP_MID="${SKIP_MID:-0}"
SKIP_SFT="${SKIP_SFT:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_SFT_EVAL="${SKIP_SFT_EVAL:-0}"

# --- Mid-training defaults ---
MID_DEVICE_BATCH_SIZE="${MID_DEVICE_BATCH_SIZE:-8}"
MID_MAX_SEQ_LEN="${MID_MAX_SEQ_LEN:-1024}"
MID_TOTAL_BATCH_SIZE="${MID_TOTAL_BATCH_SIZE:-65536}"
MID_NUM_ITERATIONS="${MID_NUM_ITERATIONS:--1}"
MID_EVAL_EVERY="${MID_EVAL_EVERY:-150}"
MID_JEPA_LAMBDA="${MID_JEPA_LAMBDA:-0.25}"
MID_JEPA_SCHEDULE="${MID_JEPA_SCHEDULE:-constant}"
MID_JEPA_DROPOUT="${MID_JEPA_DROPOUT:-0.5}"

# --- SFT defaults ---
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

WANDB_PREFIX="${WANDB_PREFIX:-10k-cont}"

# Format: tag|is_jepa  (is_jepa=1 means use JEPA for mid+SFT)
declare -a MODELS=(
    "d12-s5k-gpt-ce-4090|0"
    "d12-s5k-gpt-jepa-lin-4090|1"
    "d12-s5k-gpt-gqa2-ce-4090|0"
    "d12-s5k-rys-ce-4090|0"
    "d12-s5k-rys-jepa-lin-4090|1"
    "d12-s5k-trm-ce-4090|0"
)

sft_tag_for() {
    local tag="$1" is_jepa="$2"
    if [ "${is_jepa}" = "1" ]; then
        echo "${tag}_jepa"
    else
        echo "${tag}"
    fi
}

echo ""
echo "================================================================================"
echo "10k models: mid-train → SFT → eval"
echo "  SKIP_MID=${SKIP_MID}  SKIP_SFT=${SKIP_SFT}  SKIP_EVAL=${SKIP_EVAL}"
echo "================================================================================"

# =============================================================================
# Phase 1: Mid-training (base 10k → mid)
# =============================================================================

if [ "${SKIP_MID}" = "0" ]; then
    echo ""
    echo "================================================================================"
    echo "[Phase 1] Mid-training from 10k base checkpoints"
    echo "================================================================================"

    for row in "${MODELS[@]}"; do
        IFS='|' read -r tag is_jepa <<< "${row}"

        base_dir="${CACHE_DIR}/base_checkpoints/${tag}"
        mid_dir="${CACHE_DIR}/mid_checkpoints/${tag}"

        if [ ! -d "${base_dir}" ]; then
            echo "[mid] NO BASE: ${tag} — skipping"
            continue
        fi

        # Archive old 5k-based mid checkpoint if it exists
        if [ -d "${mid_dir}" ]; then
            archive="${mid_dir}_5k"
            if [ -d "${archive}" ]; then
                echo "[mid] Archive ${archive} already exists, removing old mid dir"
                rm -rf "${mid_dir}"
            else
                echo "[mid] Archiving 5k mid: ${tag} → ${tag}_5k"
                mv "${mid_dir}" "${archive}"
            fi
        fi

        local_jepa_lambda=0
        local_jepa_schedule="constant"
        if [ "${is_jepa}" = "1" ]; then
            local_jepa_lambda="${MID_JEPA_LAMBDA}"
            local_jepa_schedule="${MID_JEPA_SCHEDULE}"
        fi

        wb_run="${WANDB_PREFIX}-mid-${tag}"

        echo ""
        echo "  [mid] ${tag}  (JEPA λ=${local_jepa_lambda} ${local_jepa_schedule})"

        "${PYTHON_BIN}" -m scripts.mid_train_jepa \
            --source base \
            --model-tag "${tag}" \
            --run "${wb_run}" \
            --device-batch-size "${MID_DEVICE_BATCH_SIZE}" \
            --max-seq-len "${MID_MAX_SEQ_LEN}" \
            --total-batch-size "${MID_TOTAL_BATCH_SIZE}" \
            --num-iterations "${MID_NUM_ITERATIONS}" \
            --jepa-lambda "${local_jepa_lambda}" \
            --jepa-schedule "${local_jepa_schedule}" \
            --jepa-dropout "${MID_JEPA_DROPOUT}" \
            --eval-every "${MID_EVAL_EVERY}"

        echo "  [mid] DONE: ${tag}"
    done
else
    echo ""
    echo "[Phase 1] SKIPPED (SKIP_MID=1)"
fi

# =============================================================================
# Phase 2: Chat SFT (mid → sft)
# =============================================================================

if [ "${SKIP_SFT}" = "0" ]; then
    echo ""
    echo "================================================================================"
    echo "[Phase 2] Chat SFT from mid checkpoints"
    echo "================================================================================"

    for row in "${MODELS[@]}"; do
        IFS='|' read -r tag is_jepa <<< "${row}"

        mid_dir="${CACHE_DIR}/mid_checkpoints/${tag}"
        sft_tag="$(sft_tag_for "${tag}" "${is_jepa}")"
        sft_dir="${CACHE_DIR}/chatsft_checkpoints/${sft_tag}"

        if [ ! -d "${mid_dir}" ]; then
            echo "[sft] NO MID: ${tag} — skipping"
            continue
        fi

        # Archive old 5k-based SFT checkpoint if it exists
        if [ -d "${sft_dir}" ]; then
            archive="${sft_dir}_5k"
            if [ -d "${archive}" ]; then
                echo "[sft] Archive ${archive} already exists, removing old sft dir"
                rm -rf "${sft_dir}"
            else
                echo "[sft] Archiving 5k SFT: ${sft_tag} → ${sft_tag}_5k"
                mv "${sft_dir}" "${archive}"
            fi
        fi

        wb_run="${WANDB_PREFIX}-sft-${tag}"

        if [ "${is_jepa}" = "1" ]; then
            echo ""
            echo "  [sft] ${sft_tag}  (JEPA λ=${SFT_JEPA_LAMBDA})"

            "${PYTHON_BIN}" -m scripts.chat_sft_jepa \
                --source mid \
                --model-tag "${tag}" \
                --run "${wb_run}" \
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
            echo "  [sft] ${sft_tag}  (no JEPA)"

            "${PYTHON_BIN}" -m scripts.chat_sft \
                --source mid \
                --model-tag "${tag}" \
                --run "${wb_run}" \
                --num-epochs "${SFT_NUM_EPOCHS}" \
                --device-batch-size "${SFT_DEVICE_BATCH_SIZE}" \
                --target-examples-per-step "${SFT_TARGET_EXAMPLES}" \
                --eval-every "${SFT_EVAL_EVERY}" \
                --eval-steps "${SFT_EVAL_STEPS}" \
                --eval-metrics-every "${SFT_EVAL_METRICS_EVERY}" \
                --eval-metrics-max-problems "${SFT_EVAL_METRICS_MAX}"
        fi

        echo "  [sft] DONE: ${sft_tag}"
    done
else
    echo ""
    echo "[Phase 2] SKIPPED (SKIP_SFT=1)"
fi

# =============================================================================
# Phase 3: Pipeline eval — CORE on 10k base checkpoints
# =============================================================================

if [ "${SKIP_EVAL}" = "1" ]; then
    echo ""
    echo "[Phases 3-4] SKIPPED (SKIP_EVAL=1)"
    exit 0
fi

declare -a BASE_SPECS=()
for row in "${MODELS[@]}"; do
    IFS='|' read -r tag is_jepa <<< "${row}"
    if [ -d "${CACHE_DIR}/base_checkpoints/${tag}" ]; then
        BASE_SPECS+=("base:${tag}")
    fi
done

if [ ${#BASE_SPECS[@]} -gt 0 ]; then
    echo ""
    echo "================================================================================"
    echo "[Phase 3] Pipeline eval — CORE on 10k base checkpoints (${#BASE_SPECS[@]})"
    echo "================================================================================"
    printf '  %s\n' "${BASE_SPECS[@]}"

    "${PYTHON_BIN}" -m scripts.pipeline_eval \
        --mode core \
        --checkpoints "${BASE_SPECS[@]}"
fi

# =============================================================================
# Phase 4: Pipeline eval — CORE on SFT checkpoints
# =============================================================================

if [ "${SKIP_SFT_EVAL}" = "1" ]; then
    echo ""
    echo "[Phase 4] SKIPPED (SKIP_SFT_EVAL=1)"
    exit 0
fi

declare -a SFT_SPECS=()
for row in "${MODELS[@]}"; do
    IFS='|' read -r tag is_jepa <<< "${row}"
    sft_tag="$(sft_tag_for "${tag}" "${is_jepa}")"
    if [ -d "${CACHE_DIR}/chatsft_checkpoints/${sft_tag}" ]; then
        SFT_SPECS+=("sft:${sft_tag}")
    fi
done

if [ ${#SFT_SPECS[@]} -gt 0 ]; then
    echo ""
    echo "================================================================================"
    echo "[Phase 4] Pipeline eval — CORE on SFT checkpoints (${#SFT_SPECS[@]})"
    echo "================================================================================"
    printf '  %s\n' "${SFT_SPECS[@]}"

    "${PYTHON_BIN}" -m scripts.pipeline_eval \
        --mode core \
        --checkpoints "${SFT_SPECS[@]}"
fi

echo ""
echo "================================================================================"
echo "All done — 10k SFT + eval complete."
echo "  Base checkpoints:  ~/.cache/nanochat/base_checkpoints/<tag>/"
echo "  Mid checkpoints:   ~/.cache/nanochat/mid_checkpoints/<tag>/"
echo "  SFT checkpoints:   ~/.cache/nanochat/chatsft_checkpoints/<tag>/"
echo "  CSV output:        ~/.cache/nanochat/pipeline_eval/"
echo "================================================================================"
