#!/bin/bash
# Code self-supervised training using The Stack (Python subset)
#
# Loads from a base (or semisup_code) checkpoint and trains the model
# on self-generated code completions, filtered by confidence.
#
# Usage:
#   bash run_selfsup_code.sh
#   MODEL_TAG=h100-crate-a bash run_selfsup_code.sh
#   SOURCE=semisup_code bash run_selfsup_code.sh  # resume from previous code self-sup

set -e
cd "$(dirname "$0")"

MODEL_TAG=${MODEL_TAG:-"h100-crate-a"}
SOURCE=${SOURCE:-"base"}
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-2}
NUM_CANDIDATES=${NUM_CANDIDATES:-8}
MAX_PROMPTS=${MAX_PROMPTS:-1000}
NUM_ITERATIONS=${NUM_ITERATIONS:-1}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-200}

echo "========================================"
echo "Code Self-Supervised Training"
echo "  Source: ${SOURCE} / ${MODEL_TAG}"
echo "  Saves to: semisup_code_checkpoints/${MODEL_TAG}"
echo "  Prompts: ${MAX_PROMPTS} from The Stack (Python)"
echo "  Candidates: ${NUM_CANDIDATES}, Iterations: ${NUM_ITERATIONS}"
echo "========================================"

python -m scripts.self_train \
    --source "$SOURCE" \
    --model-tag "$MODEL_TAG" \
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
    --candidates-cache "$HOME/.cache/nanochat/candidates_cache_code"

echo "Code self-supervised training complete."
echo "Checkpoint: ~/.cache/nanochat/semisup_code_checkpoints/${MODEL_TAG}/"
