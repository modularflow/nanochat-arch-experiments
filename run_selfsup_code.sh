#!/bin/bash
# Code self-supervised training using Python code instructions
#
# Loads from a base (or semisup_code) checkpoint and trains the model
# on self-generated code completions, filtered by confidence.
#
# Usage:
#   bash run_selfsup_code.sh                          # d12 baseline
#   MODEL_TAG=h100-crate-a bash run_selfsup_code.sh   # d20 model
#   SOURCE=semisup_code bash run_selfsup_code.sh       # resume from previous run
#   MODEL_STEP=20000 bash run_selfsup_code.sh          # specific checkpoint step

set -e
cd "$(dirname "$0")"

MODEL_TAG=${MODEL_TAG:-"d12"}
SOURCE=${SOURCE:-"base"}
MODEL_STEP=${MODEL_STEP:-""}
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-4}
NUM_CANDIDATES=${NUM_CANDIDATES:-8}
MAX_PROMPTS=${MAX_PROMPTS:-1000}
NUM_ITERATIONS=${NUM_ITERATIONS:-2}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-200}
JEPA_LAMBDA=${JEPA_LAMBDA:-0.0}
JEPA_DROPOUT=${JEPA_DROPOUT:-0.5}
JEPA_VIEW_MAX_LEN=${JEPA_VIEW_MAX_LEN:-256}

STEP_FLAG=""
if [ -n "$MODEL_STEP" ]; then
    STEP_FLAG="--model-step $MODEL_STEP"
fi

echo "========================================"
echo "Code Self-Supervised Training"
echo "  Source: ${SOURCE} / ${MODEL_TAG}"
if [ -n "$MODEL_STEP" ]; then
    echo "  Step: ${MODEL_STEP}"
fi
echo "  Saves to: semisup_code_checkpoints/${MODEL_TAG}"
echo "  Prompts: ${MAX_PROMPTS} (Python code instructions)"
echo "  Candidates: ${NUM_CANDIDATES}, Iterations: ${NUM_ITERATIONS}"
echo "  Train batch: ${TRAIN_BATCH_SIZE}, Score batch: ${SCORE_BATCH_SIZE}"
if [ "$(echo "$JEPA_LAMBDA > 0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
    echo "  JEPA: lambda=${JEPA_LAMBDA}, dropout=${JEPA_DROPOUT}"
fi
echo "========================================"

python -m scripts.self_train \
    --source "$SOURCE" \
    --model-tag "$MODEL_TAG" \
    $STEP_FLAG \
    --prompt-task codestack \
    --filter-strategy top_k \
    --top-k 2 \
    --device-batch-size "$DEVICE_BATCH_SIZE" \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --score-batch-size "$SCORE_BATCH_SIZE" \
    --num-candidates "$NUM_CANDIDATES" \
    --max-prompts "$MAX_PROMPTS" \
    --num-iterations "$NUM_ITERATIONS" \
    --num-train-steps "$NUM_TRAIN_STEPS" \
    --save-dir semisup_code_checkpoints \
    --candidates-cache "$HOME/.cache/nanochat/candidates_cache_code_${MODEL_TAG}" \
    --jepa-lambda "$JEPA_LAMBDA" \
    --jepa-dropout "$JEPA_DROPOUT" \
    --jepa-view-max-len "$JEPA_VIEW_MAX_LEN"

echo "========================================"
echo "Code self-supervised training complete."
echo "Checkpoint: ~/.cache/nanochat/semisup_code_checkpoints/${MODEL_TAG}/"
echo "========================================"
