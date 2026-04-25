#!/bin/bash
# =============================================================================
# Chinchilla-optimal d12 sweep (5 factorial rows) — full pipeline on H100 NVL.
#
# 5 rows chosen as an orthogonal ablation over {attention, objective, backbone}.
# Every row trains FROM SCRATCH at NUM_ITERATIONS=37000 (≈ 20 tok/param for
# d12, the Chinchilla-optimal ratio). No resume from s5k checkpoints — resuming
# would introduce schedule discontinuities (LR, weight-decay, and JEPA-lambda
# all rewrite against the new horizon at resume, giving non-monotone trajectories).
#
# 5 ROWS (in execution order):
#   1) gpt + CE                               → SFT via chat_sft
#      Vanilla baseline. Anchor for any claim.
#   2) gpt + JEPA (linear)                    → SFT via chat_sft_jepa
#      JEPA-lin isolated on vanilla. Tests "does JEPA help at Chinchilla?"
#   3) gpt + GQA2 + CE                        → SFT via chat_sft
#      Grouped-Query Attention (2 KV heads). Inference-efficiency test.
#   4) rys_gpt + JEPA (linear)                → SFT via chat_sft_jepa
#      Best performer at 5k (base 0.093, Δ SFT +0.001). Tests if top-of-leaderboard
#      behavior holds at scale.
#   5) tpa_gpt T6 + JEPA (linear)             → SFT via chat_sft_jepa
#      Tensor Product Attention, Q rank=6, KV rank=2 (-31% attn params vs MHA).
#      5k base was strong (0.089) but SFT collapsed (Δ=-0.043). Chinchilla will
#      disambiguate whether the collapse was an undercook artifact or structural.
#
# DESIGNED FOR: 5× H100 NVL (94 GB HBM3, 3.9 TB/s, Hopper — FA3 native).
#   device_batch_size 64 × seq_len 1024 × grad_accum 1 = 65536 tokens/step (clean).
#   Per-run wall clock estimate: ~5.5h (37k steps × ~0.53 s/step on H100 NVL).
#   Total wall clock with 1 GPU per row in parallel: ~6h including SFT+eval.
#   Total cost estimate at $3.07/hr × 5 GPUs × ~6h ≈ $92.
#
# PARALLEL EXECUTION (1 run per GPU):
#   This script uses the standard ROWS= env knob. Launch 5 copies in parallel,
#   each pinned to a single GPU, each restricted to one row:
#
#     CUDA_VISIBLE_DEVICES=0 ROWS="1" bash new_scripts/run_chinchilla_d12_h100.sh &
#     sleep 60
#     CUDA_VISIBLE_DEVICES=1 ROWS="2" bash new_scripts/run_chinchilla_d12_h100.sh &
#     sleep 60
#     CUDA_VISIBLE_DEVICES=2 ROWS="3" bash new_scripts/run_chinchilla_d12_h100.sh &
#     sleep 60
#     CUDA_VISIBLE_DEVICES=3 ROWS="4" bash new_scripts/run_chinchilla_d12_h100.sh &
#     sleep 60
#     CUDA_VISIBLE_DEVICES=4 ROWS="5" bash new_scripts/run_chinchilla_d12_h100.sh &
#     wait
#
#   The 60s stagger avoids 5-way torch.compile contention on startup. The
#   eval_bundle + tokenizer + shards are shared via page cache; once the first
#   run warms them up, subsequent runs get cheap reads.
#
# SERIAL EXECUTION (single GPU):
#   bash new_scripts/run_chinchilla_d12_h100.sh         # all 5 rows serially
#   ROWS="1 3" bash new_scripts/run_chinchilla_d12_h100.sh
#
# SMOKE TEST (local 4090 or cheap L4):
#   NUM_ITERATIONS=50 MID_NUM_ITERATIONS=30 \
#   SFT_NUM_EPOCHS=1 SFT_EVAL_EVERY=20 SFT_EVAL_METRICS_EVERY=1000000 \
#   EVAL_EVERY=25 SAVE_EVERY=25 \
#   WANDB_RUN=dummy \
#   ROWS="1" bash new_scripts/run_chinchilla_d12_h100.sh
#
# Skip phases (any combination, all default 0 = run):
#   SKIP_BASE=1 SKIP_MID=1 SKIP_SFT=1 SKIP_EVAL=1
#   SKIP_BASE_EVAL=1 SKIP_SFT_EVAL=1
#
# Per-row skip-if-exists:
#   SKIP_EXISTING=1 bash new_scripts/run_chinchilla_d12_h100.sh
#
# Tags written to:
#   ~/.cache/nanochat/last_chinchilla_sweep_base_tags.txt
#   ~/.cache/nanochat/last_chinchilla_sweep_sft_tags.txt
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

# --- Per-row skip if checkpoint already exists ---
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# --- Base training: Chinchilla-optimal budget for d12 ---
# NUM_ITERATIONS=37000 × TOTAL_BATCH_SIZE=65536 = 2.43B tokens (≈ 20 tok/param for ~120M params).
DEPTH="${DEPTH:-12}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
HEAD_DIM="${HEAD_DIM:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
# H100 NVL has 94 GB VRAM — device_batch 64 fits with headroom, making
# grad_accum = TOTAL_BATCH_SIZE / (DEVICE_BATCH * MAX_SEQ_LEN) = 65536 / (64*1024) = 1 exactly.
# On a 24 GB card (4090 / L4), drop to DEVICE_BATCH_SIZE=32 (grad_accum=2).
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-64}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-65536}"
NUM_ITERATIONS="${NUM_ITERATIONS:-37000}"

EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_TOKENS="${EVAL_TOKENS:-131072}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
SAMPLE_EVERY="${SAMPLE_EVERY:--1}"

JEPA_DROPOUT="${JEPA_DROPOUT:-0.5}"
JEPA_VIEW_MIN_LEN="${JEPA_VIEW_MIN_LEN:-64}"

EMA_DECAY="${EMA_DECAY:-0.997}"
EMA_WARMUP_STEPS="${EMA_WARMUP_STEPS:-100}"

# GQA: 6 heads @ d12 → 2 KV heads (consistent with arch sweep)
NUM_KV_HEADS="${NUM_KV_HEADS:-2}"

# TPA rank preset (T6 — Q rank == n_head, full Q expressiveness; heavy KV compression)
TPA_RQ="${TPA_RQ:-6}"
TPA_RK="${TPA_RK:-2}"
TPA_RV="${TPA_RV:-2}"

# RYS knobs (consistent with arch sweep rys-jepa-lin row)
RYS_BLOCK_START="${RYS_BLOCK_START:-3}"
RYS_BLOCK_END="${RYS_BLOCK_END:-6}"
RYS_NUM_REPEATS="${RYS_NUM_REPEATS:-2}"

# --- Mid-training defaults ---
MID_DEVICE_BATCH_SIZE="${MID_DEVICE_BATCH_SIZE:-16}"
MID_MAX_SEQ_LEN="${MID_MAX_SEQ_LEN:-1024}"
MID_TOTAL_BATCH_SIZE="${MID_TOTAL_BATCH_SIZE:-65536}"
MID_NUM_ITERATIONS="${MID_NUM_ITERATIONS:--1}"  # -1 = auto from data
MID_EVAL_EVERY="${MID_EVAL_EVERY:-150}"
MID_JEPA_LAMBDA="${MID_JEPA_LAMBDA:-0.25}"
MID_JEPA_SCHEDULE="${MID_JEPA_SCHEDULE:-constant}"
MID_JEPA_DROPOUT="${MID_JEPA_DROPOUT:-0.5}"

# --- SFT defaults ---
SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-8}"
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
WANDB_PREFIX="${WANDB_PREFIX:-chinchilla-d12}"
WANDB_RUN="${WANDB_RUN:-}"

ROWS="${ROWS:-1 2 3 4 5}"

# Convention: "s37k" prefix so these tags sort alongside existing s2k/s5k runs.
TAG_PREFIX="${TAG_PREFIX:-d12-s37k}"
TAG_SUFFIX="${TAG_SUFFIX:-h100}"

echo ""
echo "================================================================================"
echo "Chinchilla-optimal d12 sweep (5 rows, orthogonal ablation) — full pipeline"
echo "  Base: steps=${NUM_ITERATIONS}  tokens/step=${TOTAL_BATCH_SIZE}  seq=${MAX_SEQ_LEN}"
echo "        device_batch=${DEVICE_BATCH_SIZE}  window=${WINDOW_PATTERN}"
echo "        ema_decay=${EMA_DECAY}  warmup=${EMA_WARMUP_STEPS}"
echo "  Mid:  device_batch=${MID_DEVICE_BATCH_SIZE}  num_iter=${MID_NUM_ITERATIONS} (-1 = auto)"
echo "  SFT:  device_batch=${SFT_DEVICE_BATCH_SIZE}  num_epochs=${SFT_NUM_EPOCHS}"
echo "  Skip: BASE=${SKIP_BASE} MID=${SKIP_MID} SFT=${SKIP_SFT} EVAL=${SKIP_EVAL}"
echo "        (BASE_EVAL=${SKIP_BASE_EVAL} SFT_EVAL=${SKIP_SFT_EVAL})"
echo "  Rows: ${ROWS}"
echo "  Tag prefix: ${TAG_PREFIX}   suffix: ${TAG_SUFFIX}"
if [ "${WANDB_RUN}" = "dummy" ]; then
    echo "  W&B:  disabled (WANDB_RUN=dummy)"
elif [ -n "${WANDB_RUN}" ]; then
    echo "  W&B:  run names ${WANDB_RUN}-<phase>-<tag>"
else
    echo "  W&B:  run names ${WANDB_PREFIX}-<phase>-<tag>"
fi
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "================================================================================"
echo ""

# -----------------------------------------------------------------------------
# Row registry. Format:
#   ROW_TAG[i]      → checkpoint tag (also model dir name)
#   ROW_ARCH[i]     → architecture: gpt | rys_gpt | tpa_gpt
#   ROW_JLAM[i]     → base-training JEPA lambda (0 = pure CE)
#   ROW_JSCHED[i]   → base-training JEPA schedule
#   ROW_EXTRA[i]    → extra CLI args for base_train_jepa
#   ROW_IS_JEPA[i]  → whether to use JEPA in mid+SFT (1 → chat_sft_jepa, 0 → chat_sft)
# -----------------------------------------------------------------------------
declare -A ROW_TAG ROW_ARCH ROW_JLAM ROW_JSCHED ROW_EXTRA ROW_IS_JEPA

ROW_TAG[1]="${TAG_PREFIX}-gpt-ce-${TAG_SUFFIX}"
ROW_ARCH[1]="gpt"
ROW_JLAM[1]="0"
ROW_JSCHED[1]="constant"
ROW_EXTRA[1]=""
ROW_IS_JEPA[1]="0"

ROW_TAG[2]="${TAG_PREFIX}-gpt-jepa-lin-${TAG_SUFFIX}"
ROW_ARCH[2]="gpt"
ROW_JLAM[2]="0.25"
ROW_JSCHED[2]="linear_decay"
ROW_EXTRA[2]=""
ROW_IS_JEPA[2]="1"

ROW_TAG[3]="${TAG_PREFIX}-gpt-gqa${NUM_KV_HEADS}-ce-${TAG_SUFFIX}"
ROW_ARCH[3]="gpt"
ROW_JLAM[3]="0"
ROW_JSCHED[3]="constant"
ROW_EXTRA[3]="--num-kv-heads ${NUM_KV_HEADS}"
ROW_IS_JEPA[3]="0"

ROW_TAG[4]="${TAG_PREFIX}-rys-jepa-lin-${TAG_SUFFIX}"
ROW_ARCH[4]="rys_gpt"
ROW_JLAM[4]="0.25"
ROW_JSCHED[4]="linear_decay"
ROW_EXTRA[4]="--depth ${DEPTH} --aspect-ratio ${ASPECT_RATIO} --rys-block-start ${RYS_BLOCK_START} --rys-block-end ${RYS_BLOCK_END} --rys-num-repeats ${RYS_NUM_REPEATS}"
ROW_IS_JEPA[4]="1"

ROW_TAG[5]="${TAG_PREFIX}-tpa-T6-jepa-lin-${TAG_SUFFIX}"
ROW_ARCH[5]="tpa_gpt"
ROW_JLAM[5]="0.25"
ROW_JSCHED[5]="linear_decay"
ROW_EXTRA[5]="--tpa-rank-q ${TPA_RQ} --tpa-rank-k ${TPA_RK} --tpa-rank-v ${TPA_RV}"
ROW_IS_JEPA[5]="1"

sft_tag_for() {
    local tag="$1" is_jepa="$2"
    if [ "${is_jepa}" = "1" ]; then
        echo "${tag}_jepa"
    else
        echo "${tag}"
    fi
}

wb_run_for() {
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
BASE_TAGS_FILE="${CACHE_DIR}/last_chinchilla_sweep_base_tags.txt"
SFT_TAGS_FILE="${CACHE_DIR}/last_chinchilla_sweep_sft_tags.txt"

RESOLVED_ROWS=()
for row_id in ${ROWS}; do
    if [ -z "${ROW_TAG[$row_id]:-}" ]; then
        echo "[warn] unknown row id: ${row_id} (valid: 1..5)"
        continue
    fi
    RESOLVED_ROWS+=("${row_id}")
done

# =============================================================================
# Phase 1: Base training (NUM_ITERATIONS each, from scratch)
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

    # Rows that supply --depth in extra (RYS) shouldn't get the global --depth/--aspect.
    local dim_args=(--depth "${DEPTH}" --aspect-ratio "${ASPECT_RATIO}")
    if [[ "${extra}" == *"--depth"* ]]; then
        dim_args=()
    fi

    local wb_run; wb_run="$(wb_run_for base "${tag}")"

    echo ""
    echo "================================================================================"
    echo "[Phase 1 — base | row ${row_id}] ${tag}"
    echo "  arch=${arch}  jepa=(${jlam}, ${jsched})  is_jepa_sft=${ROW_IS_JEPA[$row_id]}"
    if [ -n "${extra}" ]; then echo "  extra: ${extra}"; fi
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
    echo "[Phase 2] Mid-training from base checkpoints"
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
        echo "[Phase 4] pipeline_eval — CORE on base checkpoints (${#BASE_SPECS[@]})"
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
echo "Chinchilla d12 sweep finished."
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
