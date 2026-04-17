#!/bin/bash
# =============================================================================
# Continue 6 selected models from 5k → 10k steps
#
# Models selected based on 5k sweep results:
#   1. RYS JEPA-lin   — best post-SFT, no forgetting
#   2. GPT-GQA2 CE    — best base, inference-efficient
#   3. GPT JEPA-lin   — perfect SFT retention
#   4. GPT CE         — vanilla baseline
#   5. RYS CE         — isolate architecture vs loss
#   6. TRM CE         — top TRM, best CommonsenseQA (reasoning hypothesis)
#
# Resume mechanics:
#   --resume-from-step 5000 loads the 5k checkpoint (weights + optimizer + dataloader state)
#   --num-iterations 10000 sets the new horizon for LR/JEPA schedules
#
# Schedule effects (warmdown_ratio=0.4, warmup_ratio=0.0):
#   LR:  step 5000 resumes at lrm=1.0, warmdown begins at step 6000
#   JEPA (linear_decay): λ at step 5000 = 0.125, decays to 0 by step 10000
#   Weight decay: linear to 0 over 10k, so ~0.5× at step 5000
#
# Usage:
#   bash run_continue_10k.sh
#   SKIP_EXISTING=1 bash run_continue_10k.sh   # skip if step 10000 already saved
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

# Shared settings (must match original sweep)
HEAD_DIM="${HEAD_DIM:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-65536}"

RESUME_STEP=5000
TARGET_ITERATIONS=10000

EVAL_EVERY="${EVAL_EVERY:-250}"
EVAL_TOKENS="${EVAL_TOKENS:-131072}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
SAMPLE_EVERY="${SAMPLE_EVERY:--1}"

JEPA_DROPOUT="${JEPA_DROPOUT:-0.5}"
JEPA_VIEW_MIN_LEN="${JEPA_VIEW_MIN_LEN:-64}"

WANDB_PREFIX="${WANDB_PREFIX:-10k-cont}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# Format: tag|architecture|extra_args|jepa_lambda|jepa_schedule
declare -a MODELS=(
    "d12-s5k-gpt-ce-4090|gpt|--depth 12 --aspect-ratio 64|0|constant"
    "d12-s5k-gpt-jepa-lin-4090|gpt|--depth 12 --aspect-ratio 64|0.25|linear_decay"
    "d12-s5k-gpt-gqa2-ce-4090|gpt|--depth 12 --aspect-ratio 64 --num-kv-heads 2|0|constant"
    "d12-s5k-rys-ce-4090|rys_gpt|--depth 12 --aspect-ratio 64 --rys-block-start 3 --rys-block-end 6 --rys-num-repeats 2|0|constant"
    "d12-s5k-rys-jepa-lin-4090|rys_gpt|--depth 12 --aspect-ratio 64 --rys-block-start 3 --rys-block-end 6 --rys-num-repeats 2|0.25|linear_decay"
    "d12-s5k-trm-ce-4090|trm_gpt|--depth 2 --aspect-ratio 384 --trm-n-recur 3 --trm-T-cycles 2|0|constant"
)

echo ""
echo "================================================================================"
echo "Continue to ${TARGET_ITERATIONS} steps — 6 selected models"
echo "  resume_from=${RESUME_STEP}  target=${TARGET_ITERATIONS}"
echo "  tokens/step=${TOTAL_BATCH_SIZE}  seq=${MAX_SEQ_LEN}  device_batch=${DEVICE_BATCH_SIZE}"
echo "================================================================================"
echo ""

for row in "${MODELS[@]}"; do
    IFS='|' read -r tag arch extra jlam jsched <<< "${row}"

    ckpt_dir="${CACHE_DIR}/base_checkpoints/${tag}"

    if [ ! -d "${ckpt_dir}" ]; then
        echo "[SKIP] No checkpoint dir: ${ckpt_dir}"
        continue
    fi

    target_ckpt=$(printf "${ckpt_dir}/model_%06d.pt" "${TARGET_ITERATIONS}")
    if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${target_ckpt}" ]; then
        echo "[SKIP] Already at ${TARGET_ITERATIONS}: ${tag}"
        continue
    fi

    # Auto-detect the latest checkpoint to resume from
    latest_step="${RESUME_STEP}"
    for f in "${ckpt_dir}"/model_*.pt; do
        [ -f "$f" ] || continue
        s=$(basename "$f" | sed 's/model_0*\([0-9]*\)\.pt/\1/')
        if [ "$s" -gt "$latest_step" ] 2>/dev/null; then
            latest_step="$s"
        fi
    done

    wb_run="${WANDB_PREFIX}-${tag}"

    echo ""
    echo "================================================================================"
    echo "${tag}  →  ${TARGET_ITERATIONS} steps"
    echo "  arch=${arch}  jepa_lambda=${jlam} (${jsched})"
    echo "  resume_from=${latest_step}  wandb: ${wb_run}"
    echo "================================================================================"

    # shellcheck disable=SC2086
    "${PYTHON_BIN}" -m scripts.base_train_jepa \
        --run "${wb_run}" \
        --architecture "${arch}" \
        ${extra} \
        --head-dim "${HEAD_DIM}" \
        --max-seq-len "${MAX_SEQ_LEN}" \
        --window-pattern "${WINDOW_PATTERN}" \
        --num-iterations "${TARGET_ITERATIONS}" \
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
        --resume-from-step "${latest_step}"

    echo ""
    echo "  DONE: ${tag} → step ${TARGET_ITERATIONS}"
done

echo ""
echo "================================================================================"
echo "All 10k continuations complete."
echo "  Checkpoints: ~/.cache/nanochat/base_checkpoints/<tag>/"
echo ""
echo "Next: run SFT + eval on the 10k checkpoints"
echo "================================================================================"
