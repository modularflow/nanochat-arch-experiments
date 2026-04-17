#!/bin/bash
# Full No-Q Attention pipeline on a single RTX 4090 (24GB).
#
# Runs 4 architectures through the complete pipeline:
#   1. Base pretrain (18000 steps) — matches existing d12-gpt-* runs
#   2. Mid-train
#   3. Chat SFT
#   4. Pipeline eval (CORE) on base checkpoints
#   5. Pipeline eval (CORE) on SFT checkpoints
#
# Experiments (6 total, 3 per arch family):
#   - No-Q GPT + JEPA constant       (d12-noqgpt-jepa-4090)
#   - No-Q GPT, no JEPA              (d12-noqgpt-4090)
#   - No-Q GPT + JEPA linear_decay   (d12-noqgpt-jepa-lindecay-4090)
#   - No-Q CRATE + JEPA constant     (d12-noqcrate-jepa-4090)
#   - No-Q CRATE, no JEPA            (d12-noqcrate-4090)
#   - No-Q CRATE + JEPA linear_decay (d12-noqcrate-jepa-lindecay-4090)
#
# Usage:
#   bash run_noq_experiments_4090.sh                          # full pipeline (all 6)
#   SKIP_PRETRAIN=1 bash run_noq_experiments_4090.sh          # skip pretrain (reuse checkpoints)
#   SKIP_MID=1 bash run_noq_experiments_4090.sh               # skip mid-train
#   SKIP_SFT=1 bash run_noq_experiments_4090.sh               # skip chat SFT
#   SKIP_EVAL=1 bash run_noq_experiments_4090.sh              # skip pipeline evals
#   SKIP_NOQGPT=1 bash run_noq_experiments_4090.sh            # skip all No-Q GPT runs
#   SKIP_NOQCRATE=1 bash run_noq_experiments_4090.sh          # skip all No-Q CRATE runs
#   SKIP_LINDECAY=1 bash run_noq_experiments_4090.sh          # skip linear_decay variants
#   NUM_ITERATIONS=2000 bash run_noq_experiments_4090.sh      # short validation run

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
DEPTH="${DEPTH:-12}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
HEAD_DIM="${HEAD_DIM:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-65536}"
NUM_ITERATIONS="${NUM_ITERATIONS:-18000}"

JEPA_LAMBDA="${JEPA_LAMBDA:-0.25}"
JEPA_SCHEDULE="${JEPA_SCHEDULE:-constant}"
JEPA_DROPOUT="${JEPA_DROPOUT:-0.5}"
JEPA_VIEW_MIN_LEN="${JEPA_VIEW_MIN_LEN:-64}"

EVAL_EVERY="${EVAL_EVERY:-250}"
EVAL_TOKENS="${EVAL_TOKENS:-131072}"
SAVE_EVERY="${SAVE_EVERY:-2000}"

# Mid-train defaults
MID_DEVICE_BATCH_SIZE="${MID_DEVICE_BATCH_SIZE:-8}"
MID_TOTAL_BATCH_SIZE="${MID_TOTAL_BATCH_SIZE:-65536}"
MID_NUM_ITERATIONS="${MID_NUM_ITERATIONS:--1}"
MID_JEPA_LAMBDA="${MID_JEPA_LAMBDA:-0.25}"
MID_JEPA_SCHEDULE="${MID_JEPA_SCHEDULE:-constant}"
MID_EVAL_EVERY="${MID_EVAL_EVERY:-150}"

# SFT defaults
SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-4}"
SFT_TARGET_EXAMPLES="${SFT_TARGET_EXAMPLES:-32}"
SFT_NUM_EPOCHS="${SFT_NUM_EPOCHS:-1}"
SFT_JEPA_LAMBDA="${SFT_JEPA_LAMBDA:-0.5}"
SFT_EVAL_EVERY="${SFT_EVAL_EVERY:-100}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-100}"
SFT_EVAL_METRICS_EVERY="${SFT_EVAL_METRICS_EVERY:-200}"
SFT_EVAL_METRICS_MAX="${SFT_EVAL_METRICS_MAX:-1024}"

WANDB_PREFIX="${WANDB_PREFIX:-noq}"

# Feature flags
SKIP_PRETRAIN="${SKIP_PRETRAIN:-0}"
SKIP_MID="${SKIP_MID:-0}"
SKIP_SFT="${SKIP_SFT:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_NOQGPT="${SKIP_NOQGPT:-0}"
SKIP_NOQCRATE="${SKIP_NOQCRATE:-0}"

CACHE_DIR="${HOME}/.cache/nanochat"

# ============================================================
# Experiment definitions: (architecture, base_tag, jepa_lambda, sft_tag_suffix, jepa_schedule)
#
# sft_tag_suffix: appended to base_tag for the SFT checkpoint dir name.
#   JEPA runs get "_jepa" (matching existing convention: d12-gpt-jepa-4090_jepa).
#   Non-JEPA runs get "" (matching existing: d12-gpt-4090).
# jepa_schedule: per-experiment schedule for base pretrain (constant, linear_decay, cosine_decay).
# ============================================================
SKIP_LINDECAY="${SKIP_LINDECAY:-0}"

declare -a EXPERIMENTS=()

if [ "${SKIP_NOQGPT}" = "0" ]; then
    EXPERIMENTS+=("noq_gpt|d12-noqgpt-jepa-4090|${JEPA_LAMBDA}|_jepa|constant")
    EXPERIMENTS+=("noq_gpt|d12-noqgpt-4090|0||constant")
    if [ "${SKIP_LINDECAY}" = "0" ]; then
        EXPERIMENTS+=("noq_gpt|d12-noqgpt-jepa-lindecay-4090|${JEPA_LAMBDA}|_jepa|linear_decay")
    fi
fi

if [ "${SKIP_NOQCRATE}" = "0" ]; then
    EXPERIMENTS+=("noq_crate|d12-noqcrate-jepa-4090|${JEPA_LAMBDA}|_jepa|constant")
    EXPERIMENTS+=("noq_crate|d12-noqcrate-4090|0||constant")
    if [ "${SKIP_LINDECAY}" = "0" ]; then
        EXPERIMENTS+=("noq_crate|d12-noqcrate-jepa-lindecay-4090|${JEPA_LAMBDA}|_jepa|linear_decay")
    fi
fi

if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
    echo "No experiments selected. Set SKIP_NOQGPT=0 or SKIP_NOQCRATE=0."
    exit 0
fi

WANDB_STATUS="ENABLED (project=nanochat, prefix=${WANDB_PREFIX})"
if [ "${WANDB_PREFIX}" = "dummy" ]; then WANDB_STATUS="DISABLED (set WANDB_PREFIX to enable)"; fi

echo "========================================"
echo "No-Q Attention — Full Pipeline (RTX 4090)"
echo "  Experiments: ${#EXPERIMENTS[@]}"
echo "  Pretrain steps: ${NUM_ITERATIONS}"
echo "  W&B: ${WANDB_STATUS}"
echo "  Skip pretrain: ${SKIP_PRETRAIN}"
echo "  Skip mid-train: ${SKIP_MID}"
echo "  Skip SFT: ${SKIP_SFT}"
echo "  Skip eval: ${SKIP_EVAL}"
echo "========================================"
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r arch tag jlam sft_suffix jsched <<< "$exp"
    local_jepa_str="off"
    if [ "$jlam" != "0" ]; then local_jepa_str="λ=${jlam}, sched=${jsched}"; fi
    echo "  ${tag}  (arch=${arch}, JEPA=${local_jepa_str})"
done
echo "========================================"
echo ""

BASE_TAGS=()
SFT_SPECS=()

# ============================================================
# STAGE 1: Base Pretrain
# ============================================================

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r arch tag jlam sft_suffix jsched <<< "$exp"
    BASE_TAGS+=("${tag}")

    if [ "${SKIP_PRETRAIN}" = "1" ]; then
        echo "[pretrain] SKIPPED: ${tag} (SKIP_PRETRAIN=1)"
        continue
    fi

    if [ -d "${CACHE_DIR}/base_checkpoints/${tag}" ]; then
        echo "[pretrain] EXISTS: ${tag} — skipping (delete dir to re-train)"
        continue
    fi

    local_jepa_str="off"
    if [ "$jlam" != "0" ]; then local_jepa_str="λ=${jlam}, sched=${jsched}"; fi

    echo ""
    echo "================================================================"
    echo "[pretrain] ${tag}  (arch=${arch}, JEPA=${local_jepa_str}, steps=${NUM_ITERATIONS})"
    echo "================================================================"

    local_wandb="${WANDB_PREFIX}"
    if [ "${local_wandb}" != "dummy" ]; then local_wandb="${WANDB_PREFIX}-${tag}"; fi

    "${PYTHON_BIN}" -m scripts.base_train_jepa \
        --run "${local_wandb}" \
        --architecture "${arch}" \
        --depth "${DEPTH}" \
        --aspect-ratio "${ASPECT_RATIO}" \
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
    IFS='|' read -r arch tag jlam sft_suffix jsched <<< "$exp"

    if [ "${SKIP_MID}" = "1" ]; then
        echo "[mid-train] SKIPPED: ${tag} (SKIP_MID=1)"
        continue
    fi

    if [ -d "${CACHE_DIR}/mid_checkpoints/${tag}" ]; then
        echo "[mid-train] EXISTS: ${tag} — skipping (delete dir to re-train)"
        continue
    fi

    # Mid-train JEPA lambda: use same as base for JEPA runs, 0 for non-JEPA
    local_mid_jlam="${MID_JEPA_LAMBDA}"
    if [ "$jlam" = "0" ]; then local_mid_jlam="0"; fi

    local_jepa_str="off"
    if [ "$local_mid_jlam" != "0" ]; then local_jepa_str="λ=${local_mid_jlam}"; fi

    echo ""
    echo "================================================================"
    echo "[mid-train] ${tag}  (JEPA=${local_jepa_str})"
    echo "================================================================"

    local_wandb="${WANDB_PREFIX}"
    if [ "${local_wandb}" != "dummy" ]; then local_wandb="${WANDB_PREFIX}-mid-${tag}"; fi

    "${PYTHON_BIN}" -m scripts.mid_train_jepa \
        --source base \
        --model-tag "${tag}" \
        --run "${local_wandb}" \
        --device-batch-size "${MID_DEVICE_BATCH_SIZE}" \
        --max-seq-len "${MAX_SEQ_LEN}" \
        --total-batch-size "${MID_TOTAL_BATCH_SIZE}" \
        --num-iterations "${MID_NUM_ITERATIONS}" \
        --jepa-lambda "${local_mid_jlam}" \
        --jepa-schedule "${MID_JEPA_SCHEDULE}" \
        --jepa-dropout "${JEPA_DROPOUT}" \
        --eval-every "${MID_EVAL_EVERY}"

    echo "[mid-train] DONE: ${tag} -> ${CACHE_DIR}/mid_checkpoints/${tag}/"
done

# ============================================================
# STAGE 3: Chat SFT
# ============================================================

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r arch tag jlam sft_suffix jsched <<< "$exp"

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

    # SFT JEPA lambda: use SFT default for JEPA runs, 0 for non-JEPA
    local_sft_jlam="${SFT_JEPA_LAMBDA}"
    if [ "$jlam" = "0" ]; then local_sft_jlam="0"; fi

    local_jepa_str="off"
    if [ "$local_sft_jlam" != "0" ]; then local_jepa_str="λ=${local_sft_jlam}"; fi

    echo ""
    echo "================================================================"
    echo "[sft] ${sft_out_tag}  (source=mid:${tag}, JEPA=${local_jepa_str})"
    echo "================================================================"

    local_wandb="${WANDB_PREFIX}"
    if [ "${local_wandb}" != "dummy" ]; then local_wandb="${WANDB_PREFIX}-sft-${tag}"; fi

    "${PYTHON_BIN}" -m scripts.chat_sft_jepa \
        --source mid \
        --model-tag "${tag}" \
        --run "${local_wandb}" \
        --num-epochs "${SFT_NUM_EPOCHS}" \
        --device-batch-size "${SFT_DEVICE_BATCH_SIZE}" \
        --target-examples-per-step "${SFT_TARGET_EXAMPLES}" \
        --jepa-lambda "${local_sft_jlam}" \
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

    # Include existing baselines for comparison
    for existing_tag in d12-gpt-jepa-lindecay-4090 d12-gpt-jepa-4090 d12-gpt-4090 d12-jepa-4090; do
        if [ -d "${CACHE_DIR}/base_checkpoints/${existing_tag}" ]; then
            BASE_EVAL_SPECS="${BASE_EVAL_SPECS} base:${existing_tag}"
        fi
    done

    if [ -n "${BASE_EVAL_SPECS}" ]; then
        echo "  Checkpoints:${BASE_EVAL_SPECS}"
        "${PYTHON_BIN}" -m scripts.pipeline_eval \
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

    # Include existing SFT baselines for comparison
    for existing_sft in d12-gpt-jepa-lindecay-4090_jepa d12-gpt-jepa-4090_jepa d12-gpt-4090 d12-jepa-4090_jepa; do
        if [ -d "${CACHE_DIR}/chatsft_checkpoints/${existing_sft}" ]; then
            SFT_EVAL_SPECS="${SFT_EVAL_SPECS} sft:${existing_sft}"
        fi
    done

    if [ -n "${SFT_EVAL_SPECS}" ]; then
        echo "  Checkpoints:${SFT_EVAL_SPECS}"
        "${PYTHON_BIN}" -m scripts.pipeline_eval \
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
echo "No-Q pipeline complete!"
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
