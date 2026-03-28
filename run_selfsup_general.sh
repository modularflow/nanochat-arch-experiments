#!/bin/bash
# General knowledge self-supervised training using SmolTalk
#
# Loads from a mid (or semisup_general) checkpoint and trains the model
# on self-generated conversational responses, filtered by confidence.
#
# Usage:
#   bash run_selfsup_general.sh
#   MODEL_TAG=h100-crate-a bash run_selfsup_general.sh
#   SOURCE=semisup_general bash run_selfsup_general.sh  # resume

set -e
cd "$(dirname "$0")"

MODEL_TAG=${MODEL_TAG:-"d12"}
SOURCE=${SOURCE:-"mid"}
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-2}
NUM_CANDIDATES=${NUM_CANDIDATES:-8}
MAX_PROMPTS=${MAX_PROMPTS:-1000}
NUM_ITERATIONS=${NUM_ITERATIONS:-1}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-200}
JEPA_LAMBDA=${JEPA_LAMBDA:-0.0}
JEPA_DROPOUT=${JEPA_DROPOUT:-0.5}
JEPA_VIEW_MAX_LEN=${JEPA_VIEW_MAX_LEN:-256}

echo "========================================"
echo "General Knowledge Self-Supervised Training"MODEL_STEP=813 bash
echo "  Source: ${SOURCE} / ${MODEL_TAG}"
echo "  Saves to: semisup_general_checkpoints/${MODEL_TAG}"
echo "  Prompts: ${MAX_PROMPTS} from SmolTalk"
echo "  Candidates: ${NUM_CANDIDATES}, Iterations: ${NUM_ITERATIONS}"
echo "========================================"

python -m scripts.self_train \
    --source "$SOURCE" \
    --model-tag "$MODEL_TAG" \
    --prompt-task smoltalk \
    --filter-strategy top_k \
    --top-k 2 \
    --device-batch-size "$DEVICE_BATCH_SIZE" \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --score-batch-size "$SCORE_BATCH_SIZE" \
    --num-candidates "$NUM_CANDIDATES" \
    --max-prompts "$MAX_PROMPTS" \
    --num-iterations "$NUM_ITERATIONS" \
    --num-train-steps "$NUM_TRAIN_STEPS" \
    --save-dir semisup_general_checkpoints \
    --candidates-cache "$HOME/.cache/nanochat/candidates_cache_general" \
    --jepa-lambda "$JEPA_LAMBDA" \
    --jepa-dropout "$JEPA_DROPOUT" \
    --jepa-view-max-len "$JEPA_VIEW_MAX_LEN"

echo "General knowledge self-supervised training complete."
echo "Checkpoint: ~/.cache/nanochat/semisup_general_checkpoints/${MODEL_TAG}/"
