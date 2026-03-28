#!/bin/bash
# Self-Flow pretraining from scratch.
#
# Trains a SelfFlowCRATE model with dual-timestep scheduling, pluggable
# corruption, per-token conditioning, and multi-scale representation
# alignment -- all as first-class training objectives alongside LM loss.
#
# Usage:
#   bash run_selfflow_pretrain.sh
#   WANDB_RUN="my-run" bash run_selfflow_pretrain.sh
#   DEPTH=20 CORRUPTION=token_replacement bash run_selfflow_pretrain.sh

set -euo pipefail

WANDB_RUN="${WANDB_RUN:-dummy}"
DEPTH="${DEPTH:-12}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
NUM_ITERATIONS="${NUM_ITERATIONS:-5000}"
CORRUPTION="${CORRUPTION:-embedding_interpolation}"
REP_LOSS="${REP_LOSS:-cosine}"
REP_LOSS_WEIGHT="${REP_LOSS_WEIGHT:-1.0}"
NOISE_DIST="${NOISE_DIST:-uniform}"
MASK_RATIO="${MASK_RATIO:-0.5}"
EMA_DECAY="${EMA_DECAY:-0.999}"
ADVERSARIAL="${ADVERSARIAL:-none}"
ADV_WEIGHT="${ADV_WEIGHT:-0.5}"
ADV_LR="${ADV_LR:-0.0003}"
FORGET="${FORGET:-none}"
FORGET_LAYERS="${FORGET_LAYERS:-all}"
FORGET_WEIGHT="${FORGET_WEIGHT:-0.1}"
FORGET_LR="${FORGET_LR:-0.001}"

echo "========================================"
echo "Self-Flow Pretraining (ground-up)"
echo "  Depth: ${DEPTH}"
echo "  Corruption: ${CORRUPTION}"
echo "  Rep loss: ${REP_LOSS} (weight=${REP_LOSS_WEIGHT})"
echo "  Noise dist: ${NOISE_DIST}, mask ratio: ${MASK_RATIO}"
echo "  EMA decay: ${EMA_DECAY}"
echo "  Adversarial: ${ADVERSARIAL} (weight=${ADV_WEIGHT}, lr=${ADV_LR})"
echo "  Forget: ${FORGET} (layers=${FORGET_LAYERS}, weight=${FORGET_WEIGHT}, lr=${FORGET_LR})"
echo "  Iterations: ${NUM_ITERATIONS}"
echo "========================================"

python -m scripts.self_flow_pretrain \
    --run "$WANDB_RUN" \
    --depth "$DEPTH" \
    --aspect-ratio 64 \
    --head-dim 128 \
    --max-seq-len 1024 \
    --window-pattern "SSSL" \
    --corruption-strategy "$CORRUPTION" \
    --rep-loss-type "$REP_LOSS" \
    --rep-loss-weight "$REP_LOSS_WEIGHT" \
    --noise-distribution "$NOISE_DIST" \
    --mask-ratio "$MASK_RATIO" \
    --ema-decay "$EMA_DECAY" \
    --adversarial "$ADVERSARIAL" \
    --adv-weight "$ADV_WEIGHT" \
    --adv-lr "$ADV_LR" \
    --forget "$FORGET" \
    --forget-layers "$FORGET_LAYERS" \
    --forget-weight "$FORGET_WEIGHT" \
    --forget-lr "$FORGET_LR" \
    --num-iterations "$NUM_ITERATIONS" \
    --device-batch-size "$DEVICE_BATCH_SIZE" \
    --total-batch-size 65536 \
    --eval-every 250 \
    --sample-every 2500 \
    --save-every 1000 \
    --warmup-ratio 0.0 \
    --warmdown-ratio 0.4 \
    --model-tag "d${DEPTH}"

echo "========================================"
echo "Self-Flow pretraining complete."
echo "Checkpoint: ~/.cache/nanochat/selfflow_pretrain_checkpoints/d${DEPTH}/"
echo "========================================"
