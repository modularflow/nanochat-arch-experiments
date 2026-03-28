#!/bin/bash
# Self-Flow self-distillation training for CRATE.
#
# Trains the student with heavy token corruption while the EMA teacher sees
# a lightly-corrupted version of the same tokens. A projection head maps the
# student's intermediate hidden state toward the teacher's deeper hidden state.
#
# This script is fully independent of the other training stages (self_train,
# base_train, etc.) -- it won't touch their checkpoints or processes.
#
# Usage:
#   bash run_selfflow.sh                              # defaults
#   MODEL_TAG=d12 MODEL_STEP=20000 bash run_selfflow.sh
#   PROMPT_TASK=codestack bash run_selfflow.sh        # train on code
#   SOURCE=selfflow bash run_selfflow.sh              # resume from prior selfflow ckpt

set -e
cd "$(dirname "$0")"

SOURCE=${SOURCE:-"base"}
MODEL_TAG=${MODEL_TAG:-"d12"}
MODEL_STEP=${MODEL_STEP:-""}
PROMPT_TASK=${PROMPT_TASK:-"smoltalk"}
MAX_EXAMPLES=${MAX_EXAMPLES:-5000}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-4}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-500}
STUDENT_CORRUPTION=${STUDENT_CORRUPTION:-0.75}
TEACHER_CORRUPTION=${TEACHER_CORRUPTION:-0.25}
DISTILL_WEIGHT=${DISTILL_WEIGHT:-1.0}
EMA_DECAY=${EMA_DECAY:-0.999}

STEP_FLAG=""
if [ -n "$MODEL_STEP" ]; then
    STEP_FLAG="--model-step $MODEL_STEP"
fi

echo "========================================"
echo "Self-Flow Training"
echo "  Source: ${SOURCE} / ${MODEL_TAG}"
if [ -n "$MODEL_STEP" ]; then
    echo "  Step: ${MODEL_STEP}"
fi
echo "  Task: ${PROMPT_TASK} (${MAX_EXAMPLES} examples)"
echo "  Student corruption: ${STUDENT_CORRUPTION}"
echo "  Teacher corruption: ${TEACHER_CORRUPTION}"
echo "  Distill weight: ${DISTILL_WEIGHT}, EMA decay: ${EMA_DECAY}"
echo "  Steps: ${NUM_TRAIN_STEPS}, Batch: ${TRAIN_BATCH_SIZE}"
echo "  Saves to: selfflow_checkpoints/${MODEL_TAG}"
echo "========================================"

python -m scripts.self_flow_train \
    --source "$SOURCE" \
    --model-tag "$MODEL_TAG" \
    $STEP_FLAG \
    --prompt-task "$PROMPT_TASK" \
    --max-examples "$MAX_EXAMPLES" \
    --device-batch-size "$DEVICE_BATCH_SIZE" \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --num-train-steps "$NUM_TRAIN_STEPS" \
    --student-corruption "$STUDENT_CORRUPTION" \
    --teacher-corruption "$TEACHER_CORRUPTION" \
    --distill-weight "$DISTILL_WEIGHT" \
    --ema-decay "$EMA_DECAY"

echo "========================================"
echo "Self-Flow training complete."
echo "Checkpoint: ~/.cache/nanochat/selfflow_checkpoints/${MODEL_TAG}/"
echo "========================================"
