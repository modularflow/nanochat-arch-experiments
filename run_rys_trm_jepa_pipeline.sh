#!/bin/bash
# RYS-JEPA + TRM-JEPA full pipeline on a single RTX 4090 (24GB).
#
# Trains both JEPA variants through the complete pipeline:
#   1. Base pretrain (18000 steps)
#   2. Mid-train (1 epoch)
#   3. Chat SFT (1 epoch)
#   4. Pipeline eval (CORE) on base checkpoints — all existing baselines included
#   5. Pipeline eval (CORE) on SFT checkpoints — all existing baselines included
#
# Experiments (2 total):
#   - TRM-GPT + JEPA linear_decay  (d12-trm-jepa-lindecay-4090)
#   - RYS-GPT + JEPA linear_decay  (d12-rys-jepa-lindecay-4090)
#
# Architecture configs (matched to existing non-JEPA runs):
#   TRM-GPT: depth=2 unique blocks, aspect-ratio=384 → dim=768;
#     n_recur=3 × T_cycles=2 × n_unique=2 = 12 effective layers.
#   RYS-GPT: depth=12, rys_block_start=3, rys_block_end=6, repeats=2;
#     9 unique blocks, middle 3 traversed twice = 12 effective layers.
#
# Usage:
#   bash run_rys_trm_jepa_pipeline.sh                          # full pipeline
#   SKIP_PRETRAIN=1 bash run_rys_trm_jepa_pipeline.sh          # skip pretrain
#   SKIP_MID=1 bash run_rys_trm_jepa_pipeline.sh               # skip mid-train
#   SKIP_SFT=1 bash run_rys_trm_jepa_pipeline.sh               # skip chat SFT
#   SKIP_EVAL=1 bash run_rys_trm_jepa_pipeline.sh              # skip evals
#   SKIP_TRM=1 bash run_rys_trm_jepa_pipeline.sh               # TRM-JEPA only
#   SKIP_RYS=1 bash run_rys_trm_jepa_pipeline.sh               # RYS-JEPA only
#   NUM_ITERATIONS=2000 bash run_rys_trm_jepa_pipeline.sh      # short validation

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

# ============================================================
# Hyperparams — matched to existing d12 18k runs
# ============================================================
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-65536}"
NUM_ITERATIONS="${NUM_ITERATIONS:-18000}"

EVAL_EVERY="${EVAL_EVERY:-250}"
EVAL_TOKENS="${EVAL_TOKENS:-131072}"
SAVE_EVERY="${SAVE_EVERY:-2000}"

# Mid-train defaults
MID_DEVICE_BATCH_SIZE="${MID_DEVICE_BATCH_SIZE:-8}"
MID_TOTAL_BATCH_SIZE="${MID_TOTAL_BATCH_SIZE:-65536}"
MID_NUM_ITERATIONS="${MID_NUM_ITERATIONS:--1}"
MID_EVAL_EVERY="${MID_EVAL_EVERY:-150}"

# SFT defaults
SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-4}"
SFT_TARGET_EXAMPLES="${SFT_TARGET_EXAMPLES:-32}"
SFT_NUM_EPOCHS="${SFT_NUM_EPOCHS:-1}"
SFT_EVAL_EVERY="${SFT_EVAL_EVERY:-100}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-100}"
SFT_EVAL_METRICS_EVERY="${SFT_EVAL_METRICS_EVERY:-200}"
SFT_EVAL_METRICS_MAX="${SFT_EVAL_METRICS_MAX:-1024}"

WANDB_PREFIX="${WANDB_PREFIX:-trm-rys-jepa}"

# JEPA config
JEPA_LAMBDA="${JEPA_LAMBDA:-0.25}"
JEPA_SCHEDULE="${JEPA_SCHEDULE:-linear_decay}"
JEPA_DROPOUT="${JEPA_DROPOUT:-0.5}"
MID_JEPA_LAMBDA="${MID_JEPA_LAMBDA:-0.25}"
MID_JEPA_SCHEDULE="${MID_JEPA_SCHEDULE:-constant}"
SFT_JEPA_LAMBDA="${SFT_JEPA_LAMBDA:-0.5}"

# Feature flags
SKIP_PRETRAIN="${SKIP_PRETRAIN:-0}"
SKIP_MID="${SKIP_MID:-0}"
SKIP_SFT="${SKIP_SFT:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_TRM="${SKIP_TRM:-0}"
SKIP_RYS="${SKIP_RYS:-0}"

CACHE_DIR="${HOME}/.cache/nanochat"

# ============================================================
# Experiment definitions
# Format: architecture|model_tag|jepa_lambda|sft_tag_suffix|jepa_schedule|extra_pretrain_args
# ============================================================
declare -a EXPERIMENTS=()

if [ "${SKIP_TRM}" = "0" ]; then
    EXPERIMENTS+=("trm_gpt|d12-trm-jepa-lindecay-4090|${JEPA_LAMBDA}|_jepa|${JEPA_SCHEDULE}|--depth 2 --aspect-ratio 384 --trm-n-recur 3 --trm-T-cycles 2")
fi

if [ "${SKIP_RYS}" = "0" ]; then
    EXPERIMENTS+=("rys_gpt|d12-rys-jepa-lindecay-4090|${JEPA_LAMBDA}|_jepa|${JEPA_SCHEDULE}|--depth 12 --aspect-ratio 64 --rys-block-start 3 --rys-block-end 6 --rys-num-repeats 2")
fi

if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
    echo "No experiments selected. Set SKIP_TRM=0 or SKIP_RYS=0."
    exit 0
fi

WANDB_STATUS="ENABLED (project=nanochat, prefix=${WANDB_PREFIX})"
if [ "${WANDB_PREFIX}" = "dummy" ]; then WANDB_STATUS="DISABLED (set WANDB_PREFIX to enable)"; fi

echo "========================================"
echo "RYS-JEPA + TRM-JEPA — Full Pipeline (RTX 4090)"
echo "  Experiments: ${#EXPERIMENTS[@]}"
echo "  Pretrain steps: ${NUM_ITERATIONS}"
echo "  JEPA: λ=${JEPA_LAMBDA}, sched=${JEPA_SCHEDULE}"
echo "  W&B: ${WANDB_STATUS}"
echo "  Skip pretrain: ${SKIP_PRETRAIN}"
echo "  Skip mid-train: ${SKIP_MID}"
echo "  Skip SFT: ${SKIP_SFT}"
echo "  Skip eval: ${SKIP_EVAL}"
echo "========================================"
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r arch tag jlam sft_suffix jsched extra <<< "$exp"
    echo "  ${tag}  (arch=${arch}, JEPA=λ=${jlam}, sched=${jsched})"
done
echo "========================================"
echo ""

BASE_TAGS=()
SFT_SPECS=()

# ============================================================
# STAGE 1: Base Pretrain
# ============================================================

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r arch tag jlam sft_suffix jsched extra <<< "$exp"
    BASE_TAGS+=("${tag}")

    if [ "${SKIP_PRETRAIN}" = "1" ]; then
        echo "[pretrain] SKIPPED: ${tag} (SKIP_PRETRAIN=1)"
        continue
    fi

    if [ -d "${CACHE_DIR}/base_checkpoints/${tag}" ]; then
        echo "[pretrain] EXISTS: ${tag} — skipping (delete dir to re-train)"
        continue
    fi

    echo ""
    echo "================================================================"
    echo "[pretrain] ${tag}  (arch=${arch}, JEPA=λ=${jlam} sched=${jsched}, steps=${NUM_ITERATIONS})"
    echo "================================================================"

    local_wandb="${WANDB_PREFIX}"
    if [ "${local_wandb}" != "dummy" ]; then local_wandb="${WANDB_PREFIX}-${tag}"; fi

    ${PYTHON_BIN} -m scripts.base_train_jepa \
        --run "${local_wandb}" \
        --architecture "${arch}" \
        ${extra} \
        --head-dim 128 \
        --max-seq-len "${MAX_SEQ_LEN}" \
        --window-pattern "${WINDOW_PATTERN}" \
        --num-iterations "${NUM_ITERATIONS}" \
        --device-batch-size "${DEVICE_BATCH_SIZE}" \
        --total-batch-size "${TOTAL_BATCH_SIZE}" \
        --jepa-lambda "${jlam}" \
        --jepa-schedule "${jsched}" \
        --jepa-dropout "${JEPA_DROPOUT}" \
        --eval-every "${EVAL_EVERY}" \
        --eval-tokens "${EVAL_TOKENS}" \
        --core-metric-every -1 \
        --sample-every -1 \
        --save-every "${SAVE_EVERY}" \
        --model-tag "${tag}"

    echo "[pretrain] DONE: ${tag} -> ${CACHE_DIR}/base_checkpoints/${tag}/"
done

# ============================================================
# STAGE 2: Mid-train
# ============================================================

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r arch tag jlam sft_suffix jsched extra <<< "$exp"

    if [ "${SKIP_MID}" = "1" ]; then
        echo "[mid-train] SKIPPED: ${tag} (SKIP_MID=1)"
        continue
    fi

    if [ -d "${CACHE_DIR}/mid_checkpoints/${tag}" ]; then
        echo "[mid-train] EXISTS: ${tag} — skipping (delete dir to re-train)"
        continue
    fi

    echo ""
    echo "================================================================"
    echo "[mid-train] ${tag}  (JEPA=λ=${MID_JEPA_LAMBDA}, sched=${MID_JEPA_SCHEDULE})"
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
        --jepa-lambda "${MID_JEPA_LAMBDA}" \
        --jepa-schedule "${MID_JEPA_SCHEDULE}" \
        --jepa-dropout "${JEPA_DROPOUT}" \
        --eval-every "${MID_EVAL_EVERY}"

    echo "[mid-train] DONE: ${tag} -> ${CACHE_DIR}/mid_checkpoints/${tag}/"
done

# ============================================================
# STAGE 3: Chat SFT
# ============================================================

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r arch tag jlam sft_suffix jsched extra <<< "$exp"

    sft_out_tag="${tag}${sft_suffix}"
    SFT_SPECS+=("sft:${sft_out_tag}")

    if [ "${SKIP_SFT}" = "1" ]; then
        echo "[sft] SKIPPED: ${sft_out_tag} (SKIP_SFT=1)"
        continue
    fi

    if [ -d "${CACHE_DIR}/chatsft_checkpoints/${sft_out_tag}" ]; then
        echo "[sft] EXISTS: ${sft_out_tag} — skipping (delete dir to re-train)"
        continue
    fi

    echo ""
    echo "================================================================"
    echo "[sft] ${sft_out_tag}  (source=mid:${tag}, JEPA=λ=${SFT_JEPA_LAMBDA})"
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
        --jepa-lambda "${SFT_JEPA_LAMBDA}" \
        --jepa-schedule constant \
        --jepa-dropout "${JEPA_DROPOUT}" \
        --eval-every "${SFT_EVAL_EVERY}" \
        --eval-steps "${SFT_EVAL_STEPS}" \
        --eval-metrics-every "${SFT_EVAL_METRICS_EVERY}" \
        --eval-metrics-max-problems "${SFT_EVAL_METRICS_MAX}"

    echo "[sft] DONE: ${sft_out_tag} -> ${CACHE_DIR}/chatsft_checkpoints/${sft_out_tag}/"
done

# ============================================================
# STAGE 4: Pipeline eval — Base models (CORE)
# ============================================================

if [ "${SKIP_EVAL}" = "0" ]; then
    echo ""
    echo "================================================================"
    echo "[eval] Pipeline evaluation — Base models (CORE)"
    echo "================================================================"

    BASE_EVAL_SPECS=""
    for tag in "${BASE_TAGS[@]}"; do
        if [ -d "${CACHE_DIR}/base_checkpoints/${tag}" ]; then
            BASE_EVAL_SPECS="${BASE_EVAL_SPECS} base:${tag}"
        fi
    done

    # Non-JEPA RYS/TRM baselines for direct comparison
    for existing_tag in d12-trm-4090 d12-rys-4090-2; do
        if [ -d "${CACHE_DIR}/base_checkpoints/${existing_tag}" ]; then
            BASE_EVAL_SPECS="${BASE_EVAL_SPECS} base:${existing_tag}"
        fi
    done

    # All other existing baselines
    for existing_tag in d12-gpt-4090 d12-gpt-jepa-4090 d12-gpt-jepa-lindecay-4090 d12-jepa-4090 d12-noqgpt-4090 d12-noqgpt-jepa-4090; do
        if [ -d "${CACHE_DIR}/base_checkpoints/${existing_tag}" ]; then
            BASE_EVAL_SPECS="${BASE_EVAL_SPECS} base:${existing_tag}"
        fi
    done

    if [ -n "${BASE_EVAL_SPECS}" ]; then
        echo "  Checkpoints:${BASE_EVAL_SPECS}"
        ${PYTHON_BIN} -m scripts.pipeline_eval \
            --mode core \
            --checkpoints ${BASE_EVAL_SPECS}
    else
        echo "  No base checkpoints found — skipping."
    fi

    # ============================================================
    # STAGE 5: Pipeline eval — SFT models (CORE)
    # ============================================================

    echo ""
    echo "================================================================"
    echo "[eval] Pipeline evaluation — SFT models (CORE)"
    echo "================================================================"

    SFT_EVAL_SPECS=""
    for spec in "${SFT_SPECS[@]}"; do
        sft_tag="${spec#sft:}"
        if [ -d "${CACHE_DIR}/chatsft_checkpoints/${sft_tag}" ]; then
            SFT_EVAL_SPECS="${SFT_EVAL_SPECS} ${spec}"
        fi
    done

    # Existing SFT baselines for comparison
    for existing_sft in d12-gpt-4090 d12-gpt-jepa-4090_jepa d12-gpt-jepa-lindecay-4090_jepa d12-jepa-4090_jepa d12-noqgpt-4090 d12-noqgpt-4090_jepa; do
        if [ -d "${CACHE_DIR}/chatsft_checkpoints/${existing_sft}" ]; then
            SFT_EVAL_SPECS="${SFT_EVAL_SPECS} sft:${existing_sft}"
        fi
    done

    if [ -n "${SFT_EVAL_SPECS}" ]; then
        echo "  Checkpoints:${SFT_EVAL_SPECS}"
        ${PYTHON_BIN} -m scripts.pipeline_eval \
            --mode core \
            --checkpoints ${SFT_EVAL_SPECS}
    else
        echo "  No SFT checkpoints found — skipping."
    fi
fi

# ============================================================
# Summary
# ============================================================

echo ""
echo "========================================"
echo "RYS-JEPA + TRM-JEPA pipeline complete!"
echo ""
echo "Base checkpoints:"
for tag in "${BASE_TAGS[@]}"; do
    if [ -d "${CACHE_DIR}/base_checkpoints/${tag}" ]; then
        echo "  [OK]  ${tag}"
    else
        echo "  [--]  ${tag} (not found)"
    fi
done
echo ""
echo "Mid checkpoints:"
for tag in "${BASE_TAGS[@]}"; do
    if [ -d "${CACHE_DIR}/mid_checkpoints/${tag}" ]; then
        echo "  [OK]  ${tag}"
    else
        echo "  [--]  ${tag} (not found)"
    fi
done
echo ""
echo "SFT checkpoints:"
for spec in "${SFT_SPECS[@]}"; do
    sft_tag="${spec#sft:}"
    if [ -d "${CACHE_DIR}/chatsft_checkpoints/${sft_tag}" ]; then
        echo "  [OK]  ${sft_tag}"
    else
        echo "  [--]  ${sft_tag} (not found)"
    fi
done
echo ""
echo "Eval results: ${CACHE_DIR}/pipeline_eval/"
echo "========================================"
