#!/bin/bash
# =============================================================================
# Continue 5 selected architectures from 5k → 18k steps
#
# Picks (see 5k leaderboard + SFT deltas):
#   1. d12-s5k-gpt-gqa2-ce-4090        — top CORE @ 5k (0.0957), baseline
#   2. d12-s5k-rys-jepa-lin-4090       — #3 @ 5k, positive Δsft, rys_gpt
#   3. d12-s5k-tpa-T6-jepa-lin-4090    — tensor product attention, big s2k→sft lift
#   4. d12-s5k-trm-gqa2-jepa-lin-4090  — recurrent TRM, largest Δsft (+0.031)
#   5. d12-s5k-svd-r64-jepa-lin-4090   — low-rank SVD attn, ~40% cheaper/step
#
# Resume mechanics:
#   --resume-from-step <latest>  loads weights + optimizer + dataloader state
#   --num-iterations 18000       sets the horizon for LR + weight-decay schedules
#
# JEPA schedule for the *-jepa-lin-4090 rows:
#   --jepa-schedule linear_decay_cyclic --jepa-period 5000
#   => lambda restarts at base (0.25) every 5000 steps and decays linearly to 0.
#      At step  5000 : λ=0.25 (fresh cycle, continuing from where 5k run ended)
#      At step 10000 : λ=0.25 (fresh cycle)
#      At step 15000 : λ=0.25 (fresh cycle)
#      At step 18000 : λ=0.10   (3000/5000 into the final partial cycle)
#   This preserves the 0→base→0 shape the 5k runs were trained with, three times,
#   instead of stretching a single decay across the extended horizon.
#
# LR / weight decay schedules (warmup_ratio=0.0, warmdown_ratio=0.4):
#   Constant LR from step 5000 until the warmdown point (0.6 * 18000 = 10800),
#   then linear warmdown to 0 over the final 40% of the run.
#   Muon weight decay decays linearly to 0 over the full 18k horizon.
#
# Usage:
#   tmux new -s cont18k
#   bash new_scripts/run_continue_18k.sh
#
#   # skip rows whose step 18000 checkpoint already exists
#   SKIP_EXISTING=1 bash new_scripts/run_continue_18k.sh
#
#   # run a single row (1-indexed into MODELS below)
#   ROWS="3" bash new_scripts/run_continue_18k.sh
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

# Shared settings (must match original 5k runs)
HEAD_DIM="${HEAD_DIM:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-65536}"

RESUME_STEP=5000
TARGET_ITERATIONS="${TARGET_ITERATIONS:-18000}"
JEPA_CYCLE_PERIOD="${JEPA_CYCLE_PERIOD:-5000}"

EVAL_EVERY="${EVAL_EVERY:-500}"
EVAL_TOKENS="${EVAL_TOKENS:-131072}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
SAMPLE_EVERY="${SAMPLE_EVERY:--1}"

JEPA_DROPOUT="${JEPA_DROPOUT:-0.5}"
JEPA_VIEW_MIN_LEN="${JEPA_VIEW_MIN_LEN:-64}"

WANDB_PREFIX="${WANDB_PREFIX:-18k-cont}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
ROWS="${ROWS:-1 2 3 4 5}"

# Format: tag|architecture|device_batch_size|extra_args|jepa_lambda|jepa_schedule
# device_batch_size and extra args mirror each row's original 5k training meta
# so the resumed run matches memory / compute shape exactly.
declare -a MODELS=(
    "d12-s5k-gpt-gqa2-ce-4090|gpt|32|--depth 12 --aspect-ratio 64 --num-kv-heads 2|0|constant"
    "d12-s5k-rys-jepa-lin-4090|rys_gpt|32|--depth 12 --aspect-ratio 64 --rys-block-start 3 --rys-block-end 6 --rys-num-repeats 2|0.25|linear_decay_cyclic"
    "d12-s5k-tpa-T6-jepa-lin-4090|tpa_gpt|16|--depth 12 --aspect-ratio 64 --tpa-rank-q 6 --tpa-rank-k 2 --tpa-rank-v 2|0.25|linear_decay_cyclic"
    "d12-s5k-trm-gqa2-jepa-lin-4090|trm_gpt|32|--depth 2 --aspect-ratio 384 --trm-n-recur 3 --trm-T-cycles 2 --num-kv-heads 2|0.25|linear_decay_cyclic"
    "d12-s5k-svd-r64-jepa-lin-4090|svd_gpt|32|--depth 12 --aspect-ratio 64 --svd-rank 64|0.25|linear_decay_cyclic"
)

echo ""
echo "================================================================================"
echo "Continue to ${TARGET_ITERATIONS} steps — 5 selected architectures"
echo "  resume_from=${RESUME_STEP}  target=${TARGET_ITERATIONS}"
echo "  tokens/step=${TOTAL_BATCH_SIZE}  seq=${MAX_SEQ_LEN}"
echo "  jepa-lin rows: schedule=linear_decay_cyclic  period=${JEPA_CYCLE_PERIOD}"
echo "  rows=${ROWS}"
echo "================================================================================"
echo ""

row_idx=0
for row in "${MODELS[@]}"; do
    row_idx=$((row_idx + 1))
    if ! [[ " ${ROWS} " == *" ${row_idx} "* ]]; then
        continue
    fi

    IFS='|' read -r tag arch dbs extra jlam jsched <<< "${row}"

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

    # Auto-detect the latest checkpoint to resume from (so reruns pick up where
    # they left off rather than re-starting at the 5k checkpoint).
    latest_step="${RESUME_STEP}"
    for f in "${ckpt_dir}"/model_*.pt; do
        [ -f "$f" ] || continue
        s=$(basename "$f" | sed 's/model_0*\([0-9]*\)\.pt/\1/')
        if [ "$s" -ge "$latest_step" ] 2>/dev/null && [ "$s" -lt "$TARGET_ITERATIONS" ] 2>/dev/null; then
            latest_step="$s"
        fi
    done

    # Only pass --jepa-period for cyclic schedules (the arg is accepted unconditionally,
    # but keep the CLI minimal and explicit for non-cyclic rows).
    period_arg=""
    if [[ "${jsched}" == *_cyclic ]]; then
        period_arg="--jepa-period ${JEPA_CYCLE_PERIOD}"
    fi

    wb_run="${WANDB_PREFIX}-${tag}"

    echo ""
    echo "================================================================================"
    echo "[row ${row_idx}] ${tag}  →  ${TARGET_ITERATIONS} steps"
    echo "  arch=${arch}  device_batch=${dbs}"
    echo "  jepa_lambda=${jlam} (${jsched}${period_arg:+, period=${JEPA_CYCLE_PERIOD}})"
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
        --device-batch-size "${dbs}" \
        --total-batch-size "${TOTAL_BATCH_SIZE}" \
        --jepa-lambda "${jlam}" \
        --jepa-schedule "${jsched}" \
        ${period_arg} \
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
echo "All 18k continuations complete."
echo "  Checkpoints: ~/.cache/nanochat/base_checkpoints/<tag>/model_018000.pt"
echo ""
echo "Next: mid + SFT + pipeline_eval on the 18k checkpoints"
echo "================================================================================"
