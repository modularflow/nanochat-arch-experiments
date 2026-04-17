#!/bin/bash
# =============================================================================
# Full sweep: base pretrain → mid-train → chat SFT → evaluations
#
# Phase 1 — Delegates to run_arch_sweep_5k_4090.sh (base_train_jepa + optional SelfFlow).
# Phase 2 — Mid-training: for each base tag, runs mid_train_jepa (JEPA λ auto-detected from tag).
# Phase 3 — Chat SFT: for each mid-trained tag, runs chat_sft or chat_sft_jepa.
# Phase 4 — Pipeline eval (CORE) on base checkpoints.
# Phase 5 — Pipeline eval (CORE) on SFT checkpoints.
# Phase 6 — Pipeline eval (CHAT: MMLU + HumanEval) on SFT checkpoints (opt-in).
#
# SelfFlow pretrain checkpoints are included in Phase 4 CORE eval but excluded
# from mid-training and SFT (different training objective / checkpoint format).
#
# JEPA detection:
#   Tags containing "jepa" get JEPA auxiliaries in mid-train (λ=MID_JEPA_LAMBDA)
#   and SFT (λ=SFT_JEPA_LAMBDA). Non-JEPA tags ("ce") get λ=0 throughout.
#   SFT output tags: non-JEPA → {tag}, JEPA → {tag}_jepa (chat_sft_jepa behaviour).
#
# Usage:
#   bash run_full_sweep_train_and_eval.sh
#
# Train only (skip all eval):
#   SKIP_PIPELINE_EVAL=1 bash run_full_sweep_train_and_eval.sh
#
# Eval only (after a prior sweep; uses last tag lists):
#   SKIP_TRAIN=1 SKIP_MID=1 SKIP_SFT=1 bash run_full_sweep_train_and_eval.sh
#
# Skip individual phases:
#   SKIP_TRAIN=1        — skip base pretrain (use existing base checkpoints)
#   SKIP_MID=1          — skip mid-training
#   SKIP_SFT=1          — skip chat SFT
#   SKIP_PIPELINE_EVAL=1 — skip all evaluations (phases 4-6)
#   SKIP_SFT_EVAL=1     — skip SFT evaluations (phases 5-6), keep base eval
#   SKIP_CHAT_EVAL=0    — also run --mode chat on SFT checkpoints (phase 6)
#
# Match sweep knobs (same env vars as run_arch_sweep_5k_4090.sh):
#   NUM_ITERATIONS=2000 SKIP_SELFFLOW=0 SKIP_TRM_RYS=1 bash run_full_sweep_train_and_eval.sh
#
# Mid-train knobs:
#   MID_DEVICE_BATCH_SIZE=8    MID_TOTAL_BATCH_SIZE=65536
#   MID_NUM_ITERATIONS=-1      MID_EVAL_EVERY=150
#   MID_JEPA_LAMBDA=0.25       MID_JEPA_SCHEDULE=constant   MID_JEPA_DROPOUT=0.5
#
# SFT knobs:
#   SFT_DEVICE_BATCH_SIZE=4    SFT_TARGET_EXAMPLES=32    SFT_NUM_EPOCHS=1
#   SFT_JEPA_LAMBDA=0.5        SFT_JEPA_SCHEDULE=constant  SFT_JEPA_DROPOUT=0.5
#   SFT_EVAL_EVERY=100         SFT_EVAL_METRICS_EVERY=200
#
# W&B (training runs use WANDB_PREFIX from run_arch_sweep_5k_4090.sh):
#   WANDB_PREFIX=5k-runs          — base train prefix (also used for mid/sft naming)
#   WANDB_RUN=dummy               — disable W&B for all training steps
#   PIPELINE_EVAL_WANDB_PROJECT=nanochat-eval  — enables W&B for pipeline_eval
#   PIPELINE_EVAL_WANDB_RUN=5k-runs            — base name for eval runs
#
# Tag files written after each phase (under ~/.cache/nanochat/):
#   last_arch_sweep_base_tags.txt        — base pretrain tags
#   last_arch_sweep_selfflow_tag.txt     — SelfFlow pretrain tag (may be empty)
#   last_arch_sweep_sft_tags.txt         — SFT checkpoint tags (for eval)
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

# --- Phase skip flags ---
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_MID="${SKIP_MID:-0}"
SKIP_SFT="${SKIP_SFT:-0}"
SKIP_PIPELINE_EVAL="${SKIP_PIPELINE_EVAL:-0}"
SKIP_SFT_EVAL="${SKIP_SFT_EVAL:-0}"
SKIP_CHAT_EVAL="${SKIP_CHAT_EVAL:-1}"

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

# --- Eval defaults ---
PIPELINE_EVAL_MAX_PER_TASK="${PIPELINE_EVAL_MAX_PER_TASK:--1}"
CHAT_MAX_PROBLEMS="${CHAT_MAX_PROBLEMS:-}"
PIPELINE_EVAL_WANDB_PROJECT="${PIPELINE_EVAL_WANDB_PROJECT:-}"
PIPELINE_EVAL_WANDB_RUN="${PIPELINE_EVAL_WANDB_RUN:-5k-runs}"

# --- W&B prefix (shared with run_arch_sweep_5k_4090.sh) ---
WANDB_PREFIX="${WANDB_PREFIX:-5k-runs}"

# --- Helpers ---
tag_is_jepa() {
    [[ "$1" == *"jepa"* ]]
}

sft_tag_for() {
    local tag="$1"
    if tag_is_jepa "${tag}"; then
        echo "${tag}_jepa"
    else
        echo "${tag}"
    fi
}

echo ""
echo "================================================================================"
echo "Full sweep: base pretrain → mid-train → SFT → eval"
echo "  SKIP_TRAIN=${SKIP_TRAIN}  SKIP_MID=${SKIP_MID}  SKIP_SFT=${SKIP_SFT}"
echo "  SKIP_PIPELINE_EVAL=${SKIP_PIPELINE_EVAL}  SKIP_SFT_EVAL=${SKIP_SFT_EVAL}  SKIP_CHAT_EVAL=${SKIP_CHAT_EVAL}"
echo "================================================================================"
echo ""

# =============================================================================
# Phase 1: Base pretrain (delegates to run_arch_sweep_5k_4090.sh)
# =============================================================================

if [ "${SKIP_TRAIN}" = "0" ]; then
    echo "================================================================================"
    echo "[Phase 1] Base pretrain (run_arch_sweep_5k_4090.sh)"
    echo "================================================================================"
    bash "${SCRIPT_DIR}/run_arch_sweep_5k_4090.sh"
else
    echo "[Phase 1] SKIPPED (SKIP_TRAIN=1) — using existing tag lists under ${CACHE_DIR}/"
fi

BASE_TAGS_FILE="${CACHE_DIR}/last_arch_sweep_base_tags.txt"
SF_TAG_FILE="${CACHE_DIR}/last_arch_sweep_selfflow_tag.txt"
SFT_TAGS_FILE="${CACHE_DIR}/last_arch_sweep_sft_tags.txt"

if [ ! -f "${BASE_TAGS_FILE}" ]; then
    echo "ERROR: ${BASE_TAGS_FILE} not found. Run training first (SKIP_TRAIN=0)"
    exit 1
fi

# =============================================================================
# Phase 2: Mid-training (base → mid for each tag)
# =============================================================================

if [ "${SKIP_MID}" = "0" ]; then
    echo ""
    echo "================================================================================"
    echo "[Phase 2] Mid-training"
    echo "  device_batch=${MID_DEVICE_BATCH_SIZE}  total_batch=${MID_TOTAL_BATCH_SIZE}"
    echo "  JEPA tags: λ=${MID_JEPA_LAMBDA} ${MID_JEPA_SCHEDULE} | non-JEPA tags: λ=0"
    echo "================================================================================"

    while IFS= read -r tag; do
        tag="$(echo "${tag}" | tr -d '\r')"
        [ -z "${tag}" ] && continue

        base_dir="${CACHE_DIR}/base_checkpoints/${tag}"
        mid_dir="${CACHE_DIR}/mid_checkpoints/${tag}"

        if [ ! -d "${base_dir}" ]; then
            echo "[mid] NO BASE: ${tag} — skipping"
            continue
        fi
        if [ -d "${mid_dir}" ]; then
            echo "[mid] EXISTS: ${tag} — skipping"
            continue
        fi

        local_jepa_lambda=0
        local_jepa_schedule="constant"
        if tag_is_jepa "${tag}"; then
            local_jepa_lambda="${MID_JEPA_LAMBDA}"
            local_jepa_schedule="${MID_JEPA_SCHEDULE}"
        fi

        local_wandb="${WANDB_PREFIX}"
        if [ "${local_wandb}" != "dummy" ]; then
            local_wandb="${WANDB_PREFIX}-mid-${tag}"
        fi

        echo ""
        echo "  [mid] ${tag}  (JEPA λ=${local_jepa_lambda} ${local_jepa_schedule})"

        "${PYTHON_BIN}" -m scripts.mid_train_jepa \
            --source base \
            --model-tag "${tag}" \
            --run "${local_wandb}" \
            --device-batch-size "${MID_DEVICE_BATCH_SIZE}" \
            --max-seq-len "${MID_MAX_SEQ_LEN}" \
            --total-batch-size "${MID_TOTAL_BATCH_SIZE}" \
            --num-iterations "${MID_NUM_ITERATIONS}" \
            --jepa-lambda "${local_jepa_lambda}" \
            --jepa-schedule "${local_jepa_schedule}" \
            --jepa-dropout "${MID_JEPA_DROPOUT}" \
            --eval-every "${MID_EVAL_EVERY}"

        echo "  [mid] DONE: ${tag} → mid_checkpoints/${tag}/"
    done < "${BASE_TAGS_FILE}"
else
    echo ""
    echo "[Phase 2] SKIPPED (SKIP_MID=1)"
fi

# =============================================================================
# Phase 3: Chat SFT (mid → sft for each tag)
# =============================================================================

: > "${SFT_TAGS_FILE}"

if [ "${SKIP_SFT}" = "0" ]; then
    echo ""
    echo "================================================================================"
    echo "[Phase 3] Chat SFT"
    echo "  device_batch=${SFT_DEVICE_BATCH_SIZE}  epochs=${SFT_NUM_EPOCHS}"
    echo "  JEPA tags: λ=${SFT_JEPA_LAMBDA} ${SFT_JEPA_SCHEDULE} | non-JEPA tags: plain chat_sft"
    echo "================================================================================"

    while IFS= read -r tag; do
        tag="$(echo "${tag}" | tr -d '\r')"
        [ -z "${tag}" ] && continue

        mid_dir="${CACHE_DIR}/mid_checkpoints/${tag}"
        sft_tag="$(sft_tag_for "${tag}")"
        sft_dir="${CACHE_DIR}/chatsft_checkpoints/${sft_tag}"

        if [ ! -d "${mid_dir}" ]; then
            echo "[sft] NO MID: ${tag} — skipping (mid checkpoint not found)"
            continue
        fi
        if [ -d "${sft_dir}" ]; then
            echo "[sft] EXISTS: ${sft_tag} — skipping"
            echo "${sft_tag}" >> "${SFT_TAGS_FILE}"
            continue
        fi

        local_wandb="${WANDB_PREFIX}"
        if [ "${local_wandb}" != "dummy" ]; then
            local_wandb="${WANDB_PREFIX}-sft-${tag}"
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

        echo "  [sft] DONE: ${sft_tag} → chatsft_checkpoints/${sft_tag}/"
        echo "${sft_tag}" >> "${SFT_TAGS_FILE}"
    done < "${BASE_TAGS_FILE}"
else
    echo ""
    echo "[Phase 3] SKIPPED (SKIP_SFT=1)"
    # Reconstruct SFT tag list from existing checkpoints for eval phases
    while IFS= read -r tag; do
        tag="$(echo "${tag}" | tr -d '\r')"
        [ -z "${tag}" ] && continue
        sft_tag="$(sft_tag_for "${tag}")"
        if [ -d "${CACHE_DIR}/chatsft_checkpoints/${sft_tag}" ]; then
            echo "${sft_tag}" >> "${SFT_TAGS_FILE}"
        fi
    done < "${BASE_TAGS_FILE}"
fi

# =============================================================================
# Phase 4: Pipeline eval — CORE on base checkpoints
# =============================================================================

if [ "${SKIP_PIPELINE_EVAL}" = "1" ]; then
    echo ""
    echo "[Phases 4-6] SKIPPED (SKIP_PIPELINE_EVAL=1)"
    exit 0
fi

# Collect base checkpoint specs
declare -a BASE_SPECS=()
while IFS= read -r tag; do
    tag="$(echo "${tag}" | tr -d '\r')"
    [ -z "${tag}" ] && continue
    if [ -d "${CACHE_DIR}/base_checkpoints/${tag}" ]; then
        BASE_SPECS+=("base:${tag}")
    else
        echo "[eval] WARN: skip missing base checkpoint: ${tag}"
    fi
done < "${BASE_TAGS_FILE}"

# Add SelfFlow pretrain if it exists
SF_TAG=""
if [ -f "${SF_TAG_FILE}" ]; then
    read -r SF_TAG < "${SF_TAG_FILE}" || true
    SF_TAG="$(echo "${SF_TAG}" | tr -d '\r')"
fi
if [ -n "${SF_TAG}" ]; then
    sf_dir="${CACHE_DIR}/selfflow_pretrain_checkpoints/${SF_TAG}"
    if [ -d "${sf_dir}" ]; then
        BASE_SPECS+=("selfflow_pretrain:${SF_TAG}")
    else
        echo "[eval] WARN: skip missing SelfFlow checkpoint: ${sf_dir}"
    fi
fi

if [ ${#BASE_SPECS[@]} -gt 0 ]; then
    echo ""
    echo "================================================================================"
    echo "[Phase 4] Pipeline eval — CORE on base checkpoints (${#BASE_SPECS[@]})"
    echo "================================================================================"
    printf '  %s\n' "${BASE_SPECS[@]}"

    CORE_WANDB=()
    if [ -n "${PIPELINE_EVAL_WANDB_PROJECT}" ]; then
        CORE_WANDB+=(--wandb-project "${PIPELINE_EVAL_WANDB_PROJECT}")
        CORE_WANDB+=(--wandb-run "${PIPELINE_EVAL_WANDB_RUN}-base-core")
    fi

    "${PYTHON_BIN}" -m scripts.pipeline_eval \
        --mode core \
        --max-per-task "${PIPELINE_EVAL_MAX_PER_TASK}" \
        "${CORE_WANDB[@]}" \
        --checkpoints "${BASE_SPECS[@]}"
else
    echo "[Phase 4] No base checkpoints to evaluate."
fi

# =============================================================================
# Phase 5: Pipeline eval — CORE on SFT checkpoints
# =============================================================================

if [ "${SKIP_SFT_EVAL}" = "1" ]; then
    echo ""
    echo "[Phases 5-6] SKIPPED (SKIP_SFT_EVAL=1)"
else
    declare -a SFT_SPECS=()
    if [ -f "${SFT_TAGS_FILE}" ]; then
        while IFS= read -r sft_tag; do
            sft_tag="$(echo "${sft_tag}" | tr -d '\r')"
            [ -z "${sft_tag}" ] && continue
            if [ -d "${CACHE_DIR}/chatsft_checkpoints/${sft_tag}" ]; then
                SFT_SPECS+=("sft:${sft_tag}")
            else
                echo "[eval] WARN: skip missing SFT checkpoint: ${sft_tag}"
            fi
        done < "${SFT_TAGS_FILE}"
    fi

    if [ ${#SFT_SPECS[@]} -gt 0 ]; then
        echo ""
        echo "================================================================================"
        echo "[Phase 5] Pipeline eval — CORE on SFT checkpoints (${#SFT_SPECS[@]})"
        echo "================================================================================"
        printf '  %s\n' "${SFT_SPECS[@]}"

        SFT_CORE_WANDB=()
        if [ -n "${PIPELINE_EVAL_WANDB_PROJECT}" ]; then
            SFT_CORE_WANDB+=(--wandb-project "${PIPELINE_EVAL_WANDB_PROJECT}")
            SFT_CORE_WANDB+=(--wandb-run "${PIPELINE_EVAL_WANDB_RUN}-sft-core")
        fi

        "${PYTHON_BIN}" -m scripts.pipeline_eval \
            --mode core \
            --max-per-task "${PIPELINE_EVAL_MAX_PER_TASK}" \
            "${SFT_CORE_WANDB[@]}" \
            --checkpoints "${SFT_SPECS[@]}"

        # -----------------------------------------------------------------
        # Phase 6: Pipeline eval — CHAT (MMLU + HumanEval) on SFT checkpoints
        # -----------------------------------------------------------------
        if [ "${SKIP_CHAT_EVAL}" = "0" ]; then
            echo ""
            echo "================================================================================"
            echo "[Phase 6] Pipeline eval — CHAT on SFT checkpoints (MMLU + HumanEval)"
            echo "================================================================================"

            declare -a CHAT_ARGS=()
            if [ -n "${CHAT_MAX_PROBLEMS:-}" ]; then
                CHAT_ARGS+=(--max-problems "${CHAT_MAX_PROBLEMS}")
            fi
            CHAT_WANDB=()
            if [ -n "${PIPELINE_EVAL_WANDB_PROJECT}" ]; then
                CHAT_WANDB+=(--wandb-project "${PIPELINE_EVAL_WANDB_PROJECT}")
                CHAT_WANDB+=(--wandb-run "${PIPELINE_EVAL_WANDB_RUN}-sft-chat")
            fi

            "${PYTHON_BIN}" -m scripts.pipeline_eval \
                --mode chat \
                "${CHAT_ARGS[@]}" \
                "${CHAT_WANDB[@]}" \
                --checkpoints "${SFT_SPECS[@]}"
        fi
    else
        echo ""
        echo "[Phase 5-6] No SFT checkpoints to evaluate."
    fi
fi

echo ""
echo "================================================================================"
echo "All done."
echo "  Base checkpoints:  ~/.cache/nanochat/base_checkpoints/<tag>/"
echo "  Mid checkpoints:   ~/.cache/nanochat/mid_checkpoints/<tag>/"
echo "  SFT checkpoints:   ~/.cache/nanochat/chatsft_checkpoints/<tag>/"
echo "  CSV output:        ~/.cache/nanochat/pipeline_eval/"
echo "================================================================================"
