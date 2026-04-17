#!/bin/bash
# =============================================================================
# 5k-step architecture sweep (single RTX 4090) — shared data + optimizer budget
#
# WHAT IS ACTUALLY "APPLES TO APPLES"?
# -----------------------------------------------------------------------------
# Aligned across ALL base_train_jepa rows below:
#   - Optimizer steps: NUM_ITERATIONS (default 5000)
#   - Tokens per step: TOTAL_BATCH_SIZE (default 65536)  → same train token count
#   - Microbatch geometry: DEVICE_BATCH_SIZE × MAX_SEQ_LEN × grad_accum = TOTAL_BATCH_SIZE
#   - Width for "d12 stack" models: depth 12 × aspect 64 = 768 (unless extra overrides TRM)
#   - Windowing: WINDOW_PATTERN
#   - JEPA aux when enabled: same JEPA_DROPOUT / JEPA_VIEW_MIN_LEN
#
# TIER A — Tight comparisons (same backbone family: 12 layers, 768 dim, standard block):
#   gpt, noq_gpt, + JEPA / GQA flags. Parameter count differs slightly (No-Q, GQA KV).
#
# TIER B — Same training recipe, different compute graph (still 768 wide where noted):
#   trm_gpt: 2 unique layers × 384 dim, recursion → NOT same #params or FLOPs/step as Tier A.
#   rys_gpt: 12 layers with repeated mid-block → NOT same FLOPs/step as plain GPT.
#   Compare Tier B mainly within {trm, rys} and against Tier A only qualitatively.
#
# TIER C — Self-Flow (different objective than plain CE/JEPA in base_train_jepa):
#   scripts.self_flow_pretrain — default --backbone gpt (vanilla GPT + rep-alignment + EMA + corruptions).
#   Use --backbone crate for the legacy CRATE stack. Optional --jepa-lambda adds JEPA on top.
#   Eval: selfflow_pretrain:<model-tag>
#
# Usage:
#   bash run_arch_sweep_5k_4090.sh
#   NUM_ITERATIONS=2000 bash run_arch_sweep_5k_4090.sh
#   SKIP_SELFFLOW=1 SKIP_TRM_RYS=1 bash run_arch_sweep_5k_4090.sh
#
# Skip groups:
#   SKIP_TIER_A=1     — skip gpt / noq / gqa rows
#   SKIP_TRM_RYS=1    — skip trm_gpt and rys_gpt rows
#   SKIP_SELFFLOW=1   — skip SelfFlow pretrain block
#   SKIP_EXISTING=1   — skip if ~/.cache/nanochat/base_checkpoints/<tag> already exists
#
# Weights & Biases (default: log each experiment as its own run):
#   WANDB_PREFIX=5k-runs     — default run name prefix (final name: ${WANDB_PREFIX}-${model_tag})
#   WANDB_RUN=dummy        — disable W&B for all training steps
#   WANDB_RUN=my-sweep     — use my-sweep-${model_tag} as each run name instead of WANDB_PREFIX
#
# After a successful run, tag lists are written for run_full_sweep_train_and_eval.sh:
#   ~/.cache/nanochat/last_arch_sweep_base_tags.txt
#   ~/.cache/nanochat/last_arch_sweep_selfflow_tag.txt
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

# --- Shared with run_base_train_jepa_4090.sh / run_noq_experiments_4090.sh ------------
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

# GQA: 6 heads @ d12 → 2 or 3 KV heads are valid
NUM_KV_HEADS="${NUM_KV_HEADS:-2}"

# W&B: unset WANDB_RUN → use WANDB_PREFIX per experiment. WANDB_RUN=dummy → off.
WANDB_PREFIX="${WANDB_PREFIX:-5k-runs}"
WANDB_RUN="${WANDB_RUN:-}"

# SelfFlow: EMA teacher uses more VRAM; default 16 with same TOTAL_BATCH_SIZE (more grad accum)
SELFFLOW_DEVICE_BATCH_SIZE="${SELFFLOW_DEVICE_BATCH_SIZE:-16}"

SKIP_TIER_A="${SKIP_TIER_A:-0}"
SKIP_TRM_RYS="${SKIP_TRM_RYS:-0}"
SKIP_SELFFLOW="${SKIP_SELFFLOW:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

CACHE_DIR="${HOME}/.cache/nanochat"

echo ""
echo "================================================================================"
echo "Architecture sweep — read comparability tiers in script header"
echo "  steps=${NUM_ITERATIONS}  tokens/step=${TOTAL_BATCH_SIZE}  seq=${MAX_SEQ_LEN}"
echo "  device_batch=${DEVICE_BATCH_SIZE}  window=${WINDOW_PATTERN}"
if [ "${WANDB_RUN}" = "dummy" ]; then
    echo "  W&B: disabled (WANDB_RUN=dummy)"
elif [ -n "${WANDB_RUN}" ]; then
    echo "  W&B: run names ${WANDB_RUN}-<tag>"
else
    echo "  W&B: run names ${WANDB_PREFIX}-<tag>"
fi
echo "================================================================================"
echo ""

# -----------------------------------------------------------------------------
# base_train_jepa experiments
# Format: tier|model_tag|architecture|--extra-args (single string)|jepa_lambda|jepa_schedule
# extra args: TRM/RYS override depth; GQA adds --num-kv-heads
# -----------------------------------------------------------------------------

declare -a JEPA_EXPS=()

if [ "${SKIP_TIER_A}" = "0" ]; then
    # Tier A — d12 × 64 = 768
    JEPA_EXPS+=("A|d12-s5k-gpt-ce-4090|gpt||0|constant")
    JEPA_EXPS+=("A|d12-s5k-gpt-jepa-lin-4090|gpt||0.25|linear_decay")
    JEPA_EXPS+=("A|d12-s5k-gpt-jepa-const-4090|gpt||0.25|constant")
    JEPA_EXPS+=("A|d12-s5k-noq-ce-4090|noq_gpt||0|constant")
    JEPA_EXPS+=("A|d12-s5k-noq-jepa-lin-4090|noq_gpt||0.25|linear_decay")
    JEPA_EXPS+=("A|d12-s5k-gpt-gqa${NUM_KV_HEADS}-ce-4090|gpt|--num-kv-heads ${NUM_KV_HEADS}|0|constant")
    JEPA_EXPS+=("A|d12-s5k-gpt-gqa${NUM_KV_HEADS}-jepa-lin-4090|gpt|--num-kv-heads ${NUM_KV_HEADS}|0.25|linear_decay")
    JEPA_EXPS+=("A|d12-s5k-noq-gqa${NUM_KV_HEADS}-ce-4090|noq_gpt|--num-kv-heads ${NUM_KV_HEADS}|0|constant")
    JEPA_EXPS+=("A|d12-s5k-noq-gqa${NUM_KV_HEADS}-jepa-lin-4090|noq_gpt|--num-kv-heads ${NUM_KV_HEADS}|0.25|linear_decay")
fi

if [ "${SKIP_TRM_RYS}" = "0" ]; then
    # Tier B — TRM: width 768, NOT same depth stack as Tier A
    JEPA_EXPS+=("B|d12-s5k-trm-ce-4090|trm_gpt|--depth 2 --aspect-ratio 384 --trm-n-recur 3 --trm-T-cycles 2|0|constant")
    JEPA_EXPS+=("B|d12-s5k-trm-jepa-lin-4090|trm_gpt|--depth 2 --aspect-ratio 384 --trm-n-recur 3 --trm-T-cycles 2|0.25|linear_decay")
    JEPA_EXPS+=("B|d12-s5k-trm-gqa${NUM_KV_HEADS}-ce-4090|trm_gpt|--depth 2 --aspect-ratio 384 --trm-n-recur 3 --trm-T-cycles 2 --num-kv-heads ${NUM_KV_HEADS}|0|constant")
    JEPA_EXPS+=("B|d12-s5k-trm-gqa${NUM_KV_HEADS}-jepa-lin-4090|trm_gpt|--depth 2 --aspect-ratio 384 --trm-n-recur 3 --trm-T-cycles 2 --num-kv-heads ${NUM_KV_HEADS}|0.25|linear_decay")
    # Tier B — RYS
    JEPA_EXPS+=("B|d12-s5k-rys-ce-4090|rys_gpt|--depth 12 --aspect-ratio 64 --rys-block-start 3 --rys-block-end 6 --rys-num-repeats 2|0|constant")
    JEPA_EXPS+=("B|d12-s5k-rys-jepa-lin-4090|rys_gpt|--depth 12 --aspect-ratio 64 --rys-block-start 3 --rys-block-end 6 --rys-num-repeats 2|0.25|linear_decay")
    JEPA_EXPS+=("B|d12-s5k-rys-gqa${NUM_KV_HEADS}-ce-4090|rys_gpt|--depth 12 --aspect-ratio 64 --rys-block-start 3 --rys-block-end 6 --rys-num-repeats 2 --num-kv-heads ${NUM_KV_HEADS}|0|constant")
    JEPA_EXPS+=("B|d12-s5k-rys-gqa${NUM_KV_HEADS}-jepa-lin-4090|rys_gpt|--depth 12 --aspect-ratio 64 --rys-block-start 3 --rys-block-end 6 --rys-num-repeats 2 --num-kv-heads ${NUM_KV_HEADS}|0.25|linear_decay")
fi

run_jepa_exp() {
    local tier="$1"
    local tag="$2"
    local arch="$3"
    local extra="$4"
    local jlam="$5"
    local jsched="$6"

    if [ "${SKIP_EXISTING}" = "1" ] && [ -d "${CACHE_DIR}/base_checkpoints/${tag}" ]; then
        echo "[skip] exists: base_checkpoints/${tag}"
        return 0
    fi

    # TRM/RYS rows pass --depth/--aspect-ratio in extra; avoid duplicate CLI flags.
    local dim_args=(--depth "${DEPTH}" --aspect-ratio "${ASPECT_RATIO}")
    if [[ "${extra}" == *"--depth"* ]]; then
        dim_args=()
    fi

    echo ""
    echo "================================================================================"
    echo "[Tier ${tier}] ${tag}"
    echo "  arch=${arch}  jepa_lambda=${jlam}  jepa_schedule=${jsched}"
    if [ -n "${extra}" ]; then echo "  extra: ${extra}"; fi
    local wb_run
    if [ "${WANDB_RUN}" = "dummy" ]; then
        wb_run="dummy"
    elif [ -n "${WANDB_RUN}" ]; then
        wb_run="${WANDB_RUN}-${tag}"
    else
        wb_run="${WANDB_PREFIX}-${tag}"
    fi
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
        --model-tag "${tag}" \
        ${extra}
}

for row in "${JEPA_EXPS[@]}"; do
    IFS='|' read -r tier tag arch extra jlam jsched <<< "${row}"
    run_jepa_exp "${tier}" "${tag}" "${arch}" "${extra}" "${jlam}" "${jsched}"
done

# Record base tags for downstream pipeline_eval (run_full_sweep_train_and_eval.sh)
BASE_TAGS_FILE="${CACHE_DIR}/last_arch_sweep_base_tags.txt"
: > "${BASE_TAGS_FILE}"
for row in "${JEPA_EXPS[@]}"; do
    IFS='|' read -r tier tag arch extra jlam jsched <<< "${row}"
    echo "${tag}" >> "${BASE_TAGS_FILE}"
done

# -----------------------------------------------------------------------------
# Tier C — SelfFlow pretrain (default GPT backbone; see SELFFLOW_* env vars)
# -----------------------------------------------------------------------------

SELFFLOW_JEPA_LAMBDA="${SELFFLOW_JEPA_LAMBDA:-0}"
SELFFLOW_JEPA_SCHEDULE="${SELFFLOW_JEPA_SCHEDULE:-constant}"
SF_TAG=""

if [ "${SKIP_SELFFLOW}" = "0" ]; then
    SF_TAG="${SELFFLOW_MODEL_TAG:-d12-s5k-selfflow-4090}"
    if [ "${SKIP_EXISTING}" = "1" ] && [ -d "${CACHE_DIR}/selfflow_pretrain_checkpoints/${SF_TAG}" ]; then
        echo "[skip] exists: selfflow_pretrain_checkpoints/${SF_TAG}"
    else
        echo ""
        echo "================================================================================"
        echo "[Tier C] SelfFlow pretrain (backbone=${SELFFLOW_BACKBONE:-gpt}, JEPA λ=${SELFFLOW_JEPA_LAMBDA})"
        echo "  tag=${SF_TAG}  device_batch=${SELFFLOW_DEVICE_BATCH_SIZE} (VRAM; EMA teacher)"
        echo "================================================================================"
        local_sf_wb="${WANDB_PREFIX}-${SF_TAG}"
        if [ "${WANDB_RUN}" = "dummy" ]; then
            local_sf_wb="dummy"
        elif [ -n "${WANDB_RUN}" ]; then
            local_sf_wb="${WANDB_RUN}-${SF_TAG}"
        fi
        echo "  wandb run: ${local_sf_wb}"
        "${PYTHON_BIN}" -m scripts.self_flow_pretrain \
            --run "${local_sf_wb}" \
            --backbone "${SELFFLOW_BACKBONE:-gpt}" \
            --depth "${DEPTH}" \
            --aspect-ratio "${ASPECT_RATIO}" \
            --head-dim "${HEAD_DIM}" \
            --max-seq-len "${MAX_SEQ_LEN}" \
            --window-pattern "${WINDOW_PATTERN}" \
            --num-iterations "${NUM_ITERATIONS}" \
            --device-batch-size "${SELFFLOW_DEVICE_BATCH_SIZE}" \
            --total-batch-size "${TOTAL_BATCH_SIZE}" \
            --jepa-lambda "${SELFFLOW_JEPA_LAMBDA}" \
            --jepa-schedule "${SELFFLOW_JEPA_SCHEDULE}" \
            --eval-every "${EVAL_EVERY}" \
            --eval-tokens "${EVAL_TOKENS}" \
            --sample-every -1 \
            --save-every "${SAVE_EVERY}" \
            --model-tag "${SF_TAG}"
    fi
fi

echo "${SF_TAG}" > "${CACHE_DIR}/last_arch_sweep_selfflow_tag.txt"

echo ""
echo "================================================================================"
echo "Sweep finished."
echo "  base_train_jepa checkpoints:  ~/.cache/nanochat/base_checkpoints/<tag>/"
if [ "${SKIP_SELFFLOW}" = "0" ]; then
    echo "  SelfFlow pretrain:           ~/.cache/nanochat/selfflow_pretrain_checkpoints/${SELFFLOW_MODEL_TAG:-d12-s5k-selfflow-4090}/"
fi
echo ""
echo "Pipeline eval (example):"
echo "  python -m scripts.pipeline_eval --mode core --checkpoints base:d12-s5k-gpt-ce-4090 base:d12-s5k-gpt-jepa-lin-4090"
if [ "${SKIP_SELFFLOW}" = "0" ]; then
    echo "  python -m scripts.pipeline_eval --mode core --checkpoints selfflow_pretrain:${SELFFLOW_MODEL_TAG:-d12-s5k-selfflow-4090}"
fi
echo "================================================================================"
