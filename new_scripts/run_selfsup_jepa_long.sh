#!/bin/bash
# Longer self-supervised training: JEPA vs no-JEPA comparison (RTX 4090).
#
# Runs two experiments from the same SFT checkpoint, then evaluates all three:
#   1. Baseline SFT (no additional training)
#   2. Long self-sup WITHOUT JEPA
#   3. Long self-sup WITH JEPA
#
# Uses existing source mappings for eval compatibility:
#   - No-JEPA saves to semisup_checkpoints    → loaded as semisup:TAG
#   - JEPA saves to semisup_code_checkpoints  → loaded as semisup_code:TAG
#
# Usage:
#   bash run_selfsup_jepa_long.sh
#   MODEL_TAG=d12-gpt-jepa-lindecay-4090_jepa bash run_selfsup_jepa_long.sh

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

SOURCE="${SOURCE:-sft}"
MODEL_TAG="${MODEL_TAG:-d12-gpt-jepa-lindecay-4090_jepa}"
WANDB_PREFIX="${WANDB_PREFIX:-selfsup-long}"

# 4090-safe: generation doesn't use JEPA so batch can be higher;
# training batch kept low to leave headroom for JEPA forward passes.
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-8}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"

# Longer run: 5 iterations x 400 steps = 2000 total training steps
MAX_PROMPTS="${MAX_PROMPTS:-3000}"
NUM_CANDIDATES="${NUM_CANDIDATES:-8}"
NUM_ITERATIONS="${NUM_ITERATIONS:-5}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-400}"

echo "========================================"
echo "Long Self-Supervised: JEPA vs No-JEPA"
echo "  Source: ${SOURCE}:${MODEL_TAG}"
echo "  Prompts: ${MAX_PROMPTS}, Candidates: ${NUM_CANDIDATES}"
echo "  Iterations: ${NUM_ITERATIONS}, Steps/iter: ${NUM_TRAIN_STEPS}"
echo "  Total training steps: $((NUM_ITERATIONS * NUM_TRAIN_STEPS))"
echo "========================================"

COMMON_ARGS=(
    --source "${SOURCE}"
    --model-tag "${MODEL_TAG}"
    --prompt-task all
    --prompt-task-mode mixed
    --filter-strategy top_k
    --top-k 2
    --device-batch-size "${GEN_BATCH_SIZE}"
    --train-batch-size "${TRAIN_BATCH_SIZE}"
    --score-batch-size "${SCORE_BATCH_SIZE}"
    --num-candidates "${NUM_CANDIDATES}"
    --max-prompts "${MAX_PROMPTS}"
    --num-iterations "${NUM_ITERATIONS}"
    --num-train-steps "${NUM_TRAIN_STEPS}"
)

# --- Run 1: No JEPA (baseline self-sup) ---
echo ""
echo "========== RUN 1: No JEPA =========="
"${PYTHON_BIN}" -m scripts.self_train \
    --run "${WANDB_PREFIX}-no-jepa" \
    "${COMMON_ARGS[@]}" \
    --save-dir semisup_checkpoints \
    --candidates-cache "$HOME/.cache/nanochat/candidates_long_nojepa_${MODEL_TAG}" \
    --jepa-lambda 0.0

# --- Run 2: With JEPA ---
echo ""
echo "========== RUN 2: JEPA lambda=0.25 =========="
"${PYTHON_BIN}" -m scripts.self_train \
    --run "${WANDB_PREFIX}-jepa" \
    "${COMMON_ARGS[@]}" \
    --save-dir semisup_code_checkpoints \
    --candidates-cache "$HOME/.cache/nanochat/candidates_long_jepa_${MODEL_TAG}" \
    --jepa-lambda 0.25 \
    --jepa-dropout 0.5 \
    --jepa-view-max-len 128

# --- Evaluate all three ---
echo ""
echo "========== Evaluation =========="

EVAL_CHECKPOINTS="sft:${MODEL_TAG} semisup:${MODEL_TAG} semisup_code:${MODEL_TAG}"

echo "--- CORE ---"
"${PYTHON_BIN}" -m scripts.pipeline_eval \
    --mode core \
    --checkpoints ${EVAL_CHECKPOINTS}

echo ""
echo "--- Chat ---"
"${PYTHON_BIN}" -m scripts.pipeline_eval \
    --mode chat \
    --checkpoints ${EVAL_CHECKPOINTS}

echo ""
echo "========================================"
echo "Done! Three-way comparison:"
echo "  sft:${MODEL_TAG}          = baseline (no self-sup)"
echo "  semisup:${MODEL_TAG}      = long self-sup, no JEPA"
echo "  semisup_code:${MODEL_TAG} = long self-sup + JEPA"
echo "========================================"
