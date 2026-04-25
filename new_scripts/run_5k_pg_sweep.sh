#!/bin/bash
# =============================================================================
# 5k-step parameter-golf import sweep — full pipeline (base → mid → SFT → eval)
#
# Each row imports a different *piece* of the parameter-golf SOTA recipe into
# nanochat's existing GPT/RYS stack. After base training to 5k, every row goes
# through the same continuation as the existing arch sweep:
#   Phase 1: 5k base training        (scripts.base_train_jepa)
#   Phase 2: mid-training            (scripts.mid_train_jepa)
#   Phase 3: chat SFT                (scripts.chat_sft   or chat_sft_jepa)
#   Phase 4: CORE eval on base       (scripts.pipeline_eval --mode core)
#   Phase 5: CORE eval on SFT        (scripts.pipeline_eval --mode core)
#
# 3 ROWS (in execution order):
#   1) gpt + parallel residuals + CE                   → SFT via chat_sft
#   2) rys + fractional-activation recurrence + JEPA   → SFT via chat_sft_jepa
#   3) gpt + MuonEqR + CE                              → SFT via chat_sft
#
# Usage:
#   tmux new -s pg-sweep
#   bash new_scripts/run_5k_pg_sweep.sh
#
#   # Subset the rows
#   ROWS="1 3" bash new_scripts/run_5k_pg_sweep.sh
#
#   # Pick up where a previous launch left off
#   SKIP_EXISTING=1 bash new_scripts/run_5k_pg_sweep.sh
#
# Skip phases (any combination, all default 0 = run):
#   SKIP_BASE=1     — skip 5k base training (assumes checkpoints exist)
#   SKIP_MID=1      — skip mid-training
#   SKIP_SFT=1      — skip SFT
#   SKIP_EVAL=1     — skip ALL pipeline_eval phases (base + SFT CORE)
#   SKIP_BASE_EVAL=1 — skip base-CORE eval only
#   SKIP_SFT_EVAL=1 — skip SFT-CORE eval only
#
# Tags written to:
#   ~/.cache/nanochat/last_pg_sweep_base_tags.txt
#   ~/.cache/nanochat/last_pg_sweep_sft_tags.txt
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

# --- Phase skip flags ---
SKIP_BASE="${SKIP_BASE:-0}"
SKIP_MID="${SKIP_MID:-0}"
SKIP_SFT="${SKIP_SFT:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_BASE_EVAL="${SKIP_BASE_EVAL:-0}"
SKIP_SFT_EVAL="${SKIP_SFT_EVAL:-0}"

# --- Per-row skip if checkpoint already exists (per phase) ---
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# --- Base training: shared budget (matches run_arch_sweep_5k_4090.sh for direct comparability) ---
DEPTH="${DEPTH:-12}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
HEAD_DIM="${HEAD_DIM:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-65536}"
NUM_ITERATIONS="${NUM_ITERATIONS:-5000}"

EVAL_EVERY="${EVAL_EVERY:-250}"
EVAL_TOKENS="${EVAL_TOKENS:-131072}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
SAMPLE_EVERY="${SAMPLE_EVERY:--1}"

JEPA_DROPOUT="${JEPA_DROPOUT:-0.5}"
JEPA_VIEW_MIN_LEN="${JEPA_VIEW_MIN_LEN:-64}"

# EMA shadow weights — tracked on EVERY base row so we get the raw vs EMA delta.
EMA_DECAY="${EMA_DECAY:-0.997}"
EMA_WARMUP_STEPS="${EMA_WARMUP_STEPS:-100}"

# Fractional-activation recurrence: turn RYS recurrence on at 35% of training.
RYS_FRAC_RECUR_START="${RYS_FRAC_RECUR_START:-0.35}"

# --- Mid-training defaults (mirror run_10k_sft_eval.sh) ---
MID_DEVICE_BATCH_SIZE="${MID_DEVICE_BATCH_SIZE:-8}"
MID_MAX_SEQ_LEN="${MID_MAX_SEQ_LEN:-1024}"
MID_TOTAL_BATCH_SIZE="${MID_TOTAL_BATCH_SIZE:-65536}"
MID_NUM_ITERATIONS="${MID_NUM_ITERATIONS:--1}"  # -1 = auto from data
MID_EVAL_EVERY="${MID_EVAL_EVERY:-150}"
MID_JEPA_LAMBDA="${MID_JEPA_LAMBDA:-0.25}"
MID_JEPA_SCHEDULE="${MID_JEPA_SCHEDULE:-constant}"
MID_JEPA_DROPOUT="${MID_JEPA_DROPOUT:-0.5}"

# --- SFT defaults (mirror run_10k_sft_eval.sh) ---
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

# --- W&B ---
WANDB_PREFIX="${WANDB_PREFIX:-5k-pg-sweep}"
WANDB_RUN="${WANDB_RUN:-}"

ROWS="${ROWS:-1 2 3}"

echo ""
echo "================================================================================"
echo "Parameter-Golf import sweep — full pipeline (base → mid → SFT → eval)"
echo "  Base: steps=${NUM_ITERATIONS}  tokens/step=${TOTAL_BATCH_SIZE}  seq=${MAX_SEQ_LEN}"
echo "        device_batch=${DEVICE_BATCH_SIZE}  window=${WINDOW_PATTERN}"
echo "        ema_decay=${EMA_DECAY}  warmup=${EMA_WARMUP_STEPS}"
echo "  Mid:  device_batch=${MID_DEVICE_BATCH_SIZE}  num_iter=${MID_NUM_ITERATIONS} (-1 = auto)"
echo "  SFT:  device_batch=${SFT_DEVICE_BATCH_SIZE}  num_epochs=${SFT_NUM_EPOCHS}"
echo "  Skip: BASE=${SKIP_BASE} MID=${SKIP_MID} SFT=${SKIP_SFT} EVAL=${SKIP_EVAL} (BASE_EVAL=${SKIP_BASE_EVAL} SFT_EVAL=${SKIP_SFT_EVAL})"
echo "  Rows: ${ROWS}"
if [ "${WANDB_RUN}" = "dummy" ]; then
    echo "  W&B:  disabled (WANDB_RUN=dummy)"
elif [ -n "${WANDB_RUN}" ]; then
    echo "  W&B:  run names ${WANDB_RUN}-<phase>-<tag>"
else
    echo "  W&B:  run names ${WANDB_PREFIX}-<phase>-<tag>"
fi
echo "================================================================================"
echo ""

# -----------------------------------------------------------------------------
# Row registry. Format:
#   ROW_TAG[i]      → checkpoint tag (also model dir name)
#   ROW_ARCH[i]     → architecture: gpt | rys_gpt
#   ROW_JLAM[i]     → base-training JEPA lambda (0 = pure CE)
#   ROW_JSCHED[i]   → base-training JEPA schedule
#   ROW_EXTRA[i]    → extra CLI args for base_train_jepa (PG flags, RYS knobs, muon mode...)
#   ROW_IS_JEPA[i]  → whether to use JEPA in mid+SFT (1 = yes → chat_sft_jepa, 0 = no → chat_sft)
# -----------------------------------------------------------------------------
declare -A ROW_TAG ROW_ARCH ROW_JLAM ROW_JSCHED ROW_EXTRA ROW_IS_JEPA

ROW_TAG[1]="d12-s5k-gpt-parresid-ce-4090"
ROW_ARCH[1]="gpt"
ROW_JLAM[1]="0"
ROW_JSCHED[1]="constant"
ROW_EXTRA[1]="--parallel-residual"
ROW_IS_JEPA[1]="0"

ROW_TAG[2]="d12-s5k-rys-fracrec-jepa-lin-4090"
ROW_ARCH[2]="rys_gpt"
ROW_JLAM[2]="0.25"
ROW_JSCHED[2]="linear_decay"
ROW_EXTRA[2]="--depth ${DEPTH} --aspect-ratio ${ASPECT_RATIO} --rys-block-start 3 --rys-block-end 6 --rys-num-repeats 2 --rys-frac-recur-start ${RYS_FRAC_RECUR_START}"
ROW_IS_JEPA[2]="1"

ROW_TAG[3]="d12-s5k-gpt-muoneqr-ce-4090"
ROW_ARCH[3]="gpt"
ROW_JLAM[3]="0"
ROW_JSCHED[3]="constant"
ROW_EXTRA[3]="--muon-mode eqr"
ROW_IS_JEPA[3]="0"

# Derive an SFT-checkpoint tag from a base tag + JEPA flag.
# (Mirrors the convention in run_10k_sft_eval.sh: append `_jepa` for JEPA SFT runs.)
sft_tag_for() {
    local tag="$1" is_jepa="$2"
    if [ "${is_jepa}" = "1" ]; then
        echo "${tag}_jepa"
    else
        echo "${tag}"
    fi
}

wb_run_for() {
    # $1 = phase (base|mid|sft), $2 = tag
    local phase="$1" tag="$2"
    if [ "${WANDB_RUN}" = "dummy" ]; then
        echo "dummy"
    elif [ -n "${WANDB_RUN}" ]; then
        echo "${WANDB_RUN}-${phase}-${tag}"
    else
        echo "${WANDB_PREFIX}-${phase}-${tag}"
    fi
}

mkdir -p "${CACHE_DIR}"
BASE_TAGS_FILE="${CACHE_DIR}/last_pg_sweep_base_tags.txt"
SFT_TAGS_FILE="${CACHE_DIR}/last_pg_sweep_sft_tags.txt"

# Resolve the rows we'll process this run.
RESOLVED_ROWS=()
for row_id in ${ROWS}; do
    if [ -z "${ROW_TAG[$row_id]:-}" ]; then
        echo "[warn] unknown row id: ${row_id} (valid: 1..3)"
        continue
    fi
    RESOLVED_ROWS+=("${row_id}")
done

# =============================================================================
# Phase 1: Base training (5k each)
# =============================================================================

run_base_row() {
    local row_id="$1"
    local tag="${ROW_TAG[$row_id]}"
    local arch="${ROW_ARCH[$row_id]}"
    local jlam="${ROW_JLAM[$row_id]}"
    local jsched="${ROW_JSCHED[$row_id]}"
    local extra="${ROW_EXTRA[$row_id]}"

    if [ "${SKIP_EXISTING}" = "1" ] && [ -d "${CACHE_DIR}/base_checkpoints/${tag}" ]; then
        echo "[base row ${row_id}] skip — exists: base_checkpoints/${tag}"
        return 0
    fi

    # Rows that supply --depth in extra (RYS rows) shouldn't get the global --depth/--aspect.
    local dim_args=(--depth "${DEPTH}" --aspect-ratio "${ASPECT_RATIO}")
    if [[ "${extra}" == *"--depth"* ]]; then
        dim_args=()
    fi

    local wb_run; wb_run="$(wb_run_for base "${tag}")"

    echo ""
    echo "================================================================================"
    echo "[Phase 1 — base | row ${row_id}] ${tag}"
    echo "  arch=${arch}  jepa=(${jlam}, ${jsched})  is_jepa_sft=${ROW_IS_JEPA[$row_id]}"
    echo "  extra: ${extra}"
    echo "  wandb run: ${wb_run}"
    echo "================================================================================"

    # shellcheck disable=SC2086
    "${PYTHON_BIN}" -m scripts.base_train_jepa \
        --run "${wb_run}" \
        --architecture "${arch}" \
        "${dim_args[@]}" \
        --head-dim "${HEAD_DIM}" \
        --max-seq-len "${MAX_SEQ_LEN}" \
        --window-pattern "${WINDOW_PATTERN}" \
        --num-iterations "${NUM_ITERATIONS}" \
        --device-batch-size "${DEVICE_BATCH_SIZE}" \
        --total-batch-size "${TOTAL_BATCH_SIZE}" \
        --jepa-lambda "${jlam}" \
        --jepa-schedule "${jsched}" \
        --jepa-dropout "${JEPA_DROPOUT}" \
        --jepa-view-min-len "${JEPA_VIEW_MIN_LEN}" \
        --eval-every "${EVAL_EVERY}" \
        --eval-tokens "${EVAL_TOKENS}" \
        --core-metric-every "${CORE_METRIC_EVERY}" \
        --sample-every "${SAMPLE_EVERY}" \
        --save-every "${SAVE_EVERY}" \
        --ema-decay "${EMA_DECAY}" \
        --ema-warmup-steps "${EMA_WARMUP_STEPS}" \
        --model-tag "${tag}" \
        ${extra}
}

if [ "${SKIP_BASE}" = "0" ]; then
    : > "${BASE_TAGS_FILE}"
    for row_id in "${RESOLVED_ROWS[@]}"; do
        run_base_row "${row_id}"
        echo "${ROW_TAG[$row_id]}" >> "${BASE_TAGS_FILE}"
    done
else
    echo ""
    echo "[Phase 1 — base] SKIPPED (SKIP_BASE=1)"
    # Still record the tags we expect downstream phases to find.
    : > "${BASE_TAGS_FILE}"
    for row_id in "${RESOLVED_ROWS[@]}"; do
        echo "${ROW_TAG[$row_id]}" >> "${BASE_TAGS_FILE}"
    done
fi

# =============================================================================
# Phase 2: Mid-training (base → mid)
# =============================================================================

run_mid_row() {
    local row_id="$1"
    local tag="${ROW_TAG[$row_id]}"
    local is_jepa="${ROW_IS_JEPA[$row_id]}"

    local base_dir="${CACHE_DIR}/base_checkpoints/${tag}"
    local mid_dir="${CACHE_DIR}/mid_checkpoints/${tag}"

    if [ ! -d "${base_dir}" ]; then
        echo "[mid row ${row_id}] NO BASE: ${tag} — skipping"
        return 0
    fi
    if [ "${SKIP_EXISTING}" = "1" ] && [ -d "${mid_dir}" ]; then
        echo "[mid row ${row_id}] skip — exists: mid_checkpoints/${tag}"
        return 0
    fi

    local local_jepa_lambda=0
    local local_jepa_schedule="constant"
    if [ "${is_jepa}" = "1" ]; then
        local_jepa_lambda="${MID_JEPA_LAMBDA}"
        local_jepa_schedule="${MID_JEPA_SCHEDULE}"
    fi

    local wb_run; wb_run="$(wb_run_for mid "${tag}")"

    echo ""
    echo "================================================================================"
    echo "[Phase 2 — mid | row ${row_id}] ${tag}  (JEPA λ=${local_jepa_lambda} ${local_jepa_schedule})"
    echo "  wandb run: ${wb_run}"
    echo "================================================================================"

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
}

if [ "${SKIP_MID}" = "0" ]; then
    echo ""
    echo "================================================================================"
    echo "[Phase 2] Mid-training from 5k base checkpoints"
    echo "================================================================================"
    for row_id in "${RESOLVED_ROWS[@]}"; do
        run_mid_row "${row_id}"
    done
else
    echo ""
    echo "[Phase 2 — mid] SKIPPED (SKIP_MID=1)"
fi

# =============================================================================
# Phase 3: Chat SFT (mid → sft)
# =============================================================================

run_sft_row() {
    local row_id="$1"
    local tag="${ROW_TAG[$row_id]}"
    local is_jepa="${ROW_IS_JEPA[$row_id]}"

    local mid_dir="${CACHE_DIR}/mid_checkpoints/${tag}"
    local sft_tag; sft_tag="$(sft_tag_for "${tag}" "${is_jepa}")"
    local sft_dir="${CACHE_DIR}/chatsft_checkpoints/${sft_tag}"

    if [ ! -d "${mid_dir}" ]; then
        echo "[sft row ${row_id}] NO MID: ${tag} — skipping"
        return 0
    fi
    if [ "${SKIP_EXISTING}" = "1" ] && [ -d "${sft_dir}" ]; then
        echo "[sft row ${row_id}] skip — exists: chatsft_checkpoints/${sft_tag}"
        return 0
    fi

    local wb_run; wb_run="$(wb_run_for sft "${tag}")"

    echo ""
    echo "================================================================================"
    echo "[Phase 3 — sft | row ${row_id}] ${sft_tag}  (is_jepa=${is_jepa})"
    echo "  wandb run: ${wb_run}"
    echo "================================================================================"

    if [ "${is_jepa}" = "1" ]; then
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
}

if [ "${SKIP_SFT}" = "0" ]; then
    echo ""
    echo "================================================================================"
    echo "[Phase 3] Chat SFT from mid checkpoints"
    echo "================================================================================"
    : > "${SFT_TAGS_FILE}"
    for row_id in "${RESOLVED_ROWS[@]}"; do
        run_sft_row "${row_id}"
        echo "$(sft_tag_for "${ROW_TAG[$row_id]}" "${ROW_IS_JEPA[$row_id]}")" >> "${SFT_TAGS_FILE}"
    done
else
    echo ""
    echo "[Phase 3 — sft] SKIPPED (SKIP_SFT=1)"
    : > "${SFT_TAGS_FILE}"
    for row_id in "${RESOLVED_ROWS[@]}"; do
        echo "$(sft_tag_for "${ROW_TAG[$row_id]}" "${ROW_IS_JEPA[$row_id]}")" >> "${SFT_TAGS_FILE}"
    done
fi

# =============================================================================
# Phase 4: pipeline_eval — CORE on base checkpoints
# =============================================================================

if [ "${SKIP_EVAL}" = "1" ]; then
    echo ""
    echo "[Phases 4-5 — eval] SKIPPED (SKIP_EVAL=1)"
    echo ""
    echo "================================================================================"
    echo "Sweep finished (eval skipped)."
    echo "  Base tags: ${BASE_TAGS_FILE}"
    echo "  SFT tags:  ${SFT_TAGS_FILE}"
    echo "================================================================================"
    exit 0
fi

if [ "${SKIP_BASE_EVAL}" = "0" ]; then
    declare -a BASE_SPECS=()
    for row_id in "${RESOLVED_ROWS[@]}"; do
        local_tag="${ROW_TAG[$row_id]}"
        if [ -d "${CACHE_DIR}/base_checkpoints/${local_tag}" ]; then
            BASE_SPECS+=("base:${local_tag}")
        fi
    done

    if [ ${#BASE_SPECS[@]} -gt 0 ]; then
        echo ""
        echo "================================================================================"
        echo "[Phase 4] pipeline_eval — CORE on 5k base checkpoints (${#BASE_SPECS[@]})"
        echo "================================================================================"
        printf '  %s\n' "${BASE_SPECS[@]}"
        "${PYTHON_BIN}" -m scripts.pipeline_eval \
            --mode core \
            --checkpoints "${BASE_SPECS[@]}"
    else
        echo "[Phase 4] No base checkpoints to eval; skipping."
    fi
else
    echo ""
    echo "[Phase 4 — base CORE eval] SKIPPED (SKIP_BASE_EVAL=1)"
fi

# =============================================================================
# Phase 5: pipeline_eval — CORE on SFT checkpoints
# =============================================================================

if [ "${SKIP_SFT_EVAL}" = "0" ]; then
    declare -a SFT_SPECS=()
    for row_id in "${RESOLVED_ROWS[@]}"; do
        local_sft_tag="$(sft_tag_for "${ROW_TAG[$row_id]}" "${ROW_IS_JEPA[$row_id]}")"
        if [ -d "${CACHE_DIR}/chatsft_checkpoints/${local_sft_tag}" ]; then
            SFT_SPECS+=("sft:${local_sft_tag}")
        fi
    done

    if [ ${#SFT_SPECS[@]} -gt 0 ]; then
        echo ""
        echo "================================================================================"
        echo "[Phase 5] pipeline_eval — CORE on SFT checkpoints (${#SFT_SPECS[@]})"
        echo "================================================================================"
        printf '  %s\n' "${SFT_SPECS[@]}"
        "${PYTHON_BIN}" -m scripts.pipeline_eval \
            --mode core \
            --checkpoints "${SFT_SPECS[@]}"
    else
        echo "[Phase 5] No SFT checkpoints to eval; skipping."
    fi
else
    echo ""
    echo "[Phase 5 — SFT CORE eval] SKIPPED (SKIP_SFT_EVAL=1)"
fi

echo ""
echo "================================================================================"
echo "PG sweep finished."
echo "  Base checkpoints:  ~/.cache/nanochat/base_checkpoints/<tag>/"
echo "  EMA shadow:        ~/.cache/nanochat/base_checkpoints/<tag>/model_ema_<step>.pt"
echo "  Mid checkpoints:   ~/.cache/nanochat/mid_checkpoints/<tag>/"
echo "  SFT checkpoints:   ~/.cache/nanochat/chatsft_checkpoints/<tag(_jepa)>/"
echo "  CORE eval CSVs:    ~/.cache/nanochat/pipeline_eval/"
echo ""
echo "  Tags:"
echo "    ${BASE_TAGS_FILE}"
echo "    ${SFT_TAGS_FILE}"
echo "================================================================================"
