#!/bin/bash
# Math self-supervised training using GSM8K with reward-based filtering
#
# Loads from a semisup_general (or semisup_math) checkpoint and trains
# on self-generated math solutions that are verified correct by GSM8K's
# answer-checking reward function.
#
# Usage:
#   bash run_selfsup_math.sh
#   MODEL_TAG=h100-crate-a bash run_selfsup_math.sh
#   SOURCE=semisup_math bash run_selfsup_math.sh  # resume

set -e
cd "$(dirname "$0")"

MODEL_TAG=${MODEL_TAG:-"h100-crate-a"}
SOURCE=${SOURCE:-"semisup_general"}
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-2}
NUM_CANDIDATES=${NUM_CANDIDATES:-8}
MAX_PROMPTS=${MAX_PROMPTS:-1000}
NUM_ITERATIONS=${NUM_ITERATIONS:-2}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-200}
JEPA_LAMBDA=${JEPA_LAMBDA:-0.0}
JEPA_DROPOUT=${JEPA_DROPOUT:-0.5}
JEPA_VIEW_MAX_LEN=${JEPA_VIEW_MAX_LEN:-256}

echo "========================================"
echo "Math Self-Supervised Training"
echo "  Source: ${SOURCE} / ${MODEL_TAG}"
echo "  Saves to: semisup_math_checkpoints/${MODEL_TAG}"
echo "  Prompts: ${MAX_PROMPTS} from GSM8K"
echo "  Candidates: ${NUM_CANDIDATES}, Iterations: ${NUM_ITERATIONS}"
echo "  Filter: reward (exact answer match)"
echo "========================================"

python -m scripts.self_train \
    --source "$SOURCE" \
    --model-tag "$MODEL_TAG" \
    --prompt-task gsm8k \
    --filter-strategy reward \
    --device-batch-size "$DEVICE_BATCH_SIZE" \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --score-batch-size "$SCORE_BATCH_SIZE" \
    --num-candidates "$NUM_CANDIDATES" \
    --max-prompts "$MAX_PROMPTS" \
    --num-iterations "$NUM_ITERATIONS" \
    --num-train-steps "$NUM_TRAIN_STEPS" \
    --save-dir semisup_math_checkpoints \
    --candidates-cache "$HOME/.cache/nanochat/candidates_cache_math" \
    --jepa-lambda "$JEPA_LAMBDA" \
    --jepa-dropout "$JEPA_DROPOUT" \
    --jepa-view-max-len "$JEPA_VIEW_MAX_LEN"

echo "Math self-supervised training complete."
echo "Checkpoint: ~/.cache/nanochat/semisup_math_checkpoints/${MODEL_TAG}/"
