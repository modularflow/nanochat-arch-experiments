#!/bin/bash
# Full Domain-Specific Self-Training Pipeline
#
# Chains all training stages in order:
#   1. [Skip] Pre-train (assumes base checkpoint already exists)
#   2. Code self-supervised    (base -> semisup_code_checkpoints)
#   3. Mid-training            (semisup_code -> mid_checkpoints)
#   4. General self-supervised (mid -> semisup_general_checkpoints)
#   5. Math self-supervised    (semisup_general -> semisup_math_checkpoints)
#   6. Chat SFT                (semisup_math -> chatsft_checkpoints)
#
# Each stage checks for existing checkpoints and skips if already complete.
#
# Usage:
#   bash run_full_pipeline.sh
#   MODEL_TAG=h100-crate-a DEVICE_BATCH_SIZE=8 bash run_full_pipeline.sh

set -e
cd "$(dirname "$0")"

# Configuration (override via environment variables)
MODEL_TAG=${MODEL_TAG:-"h100-crate-a"}
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-2}
NUM_CANDIDATES=${NUM_CANDIDATES:-8}
MAX_PROMPTS=${MAX_PROMPTS:-1000}
CACHE_DIR=${HOME}/.cache/nanochat

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

stage_exists() {
    local dir="$1"
    if [ -d "${CACHE_DIR}/${dir}/${MODEL_TAG}" ] && \
       [ -n "$(ls -A "${CACHE_DIR}/${dir}/${MODEL_TAG}" 2>/dev/null)" ]; then
        return 0
    fi
    return 1
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Full Domain-Specific Self-Training Pipeline${NC}"
echo -e "${BLUE}  Model tag:  ${MODEL_TAG}${NC}"
echo -e "${BLUE}  Batch size: device=${DEVICE_BATCH_SIZE}, train=${TRAIN_BATCH_SIZE}${NC}"
echo -e "${BLUE}  Prompts:    ${MAX_PROMPTS} per stage${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# -----------------------------------------------------------------------
# Stage 1: Pre-training (skip -- assume base checkpoint exists)
# -----------------------------------------------------------------------
echo -e "${GREEN}[1/6] Pre-training${NC}"
if stage_exists "base_checkpoints"; then
    echo -e "  ${GREEN}Base checkpoint found. Skipping.${NC}"
else
    echo -e "  ${RED}ERROR: No base checkpoint found at ${CACHE_DIR}/base_checkpoints/${MODEL_TAG}/${NC}"
    echo -e "  ${RED}Run base pre-training first, e.g.:${NC}"
    echo -e "  ${RED}  python -m scripts.base_train --model-tag ${MODEL_TAG}${NC}"
    exit 1
fi
echo ""

# -----------------------------------------------------------------------
# Stage 2: Code self-supervised (base -> semisup_code_checkpoints)
# -----------------------------------------------------------------------
echo -e "${YELLOW}[2/6] Code Self-Supervised Training${NC}"
if stage_exists "semisup_code_checkpoints"; then
    echo -e "  ${GREEN}Code self-sup checkpoint found. Skipping.${NC}"
else
    echo -e "  ${YELLOW}Running code self-supervised training...${NC}"
    SOURCE=base \
    MODEL_TAG="$MODEL_TAG" \
    DEVICE_BATCH_SIZE="$DEVICE_BATCH_SIZE" \
    TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    SCORE_BATCH_SIZE="$SCORE_BATCH_SIZE" \
    NUM_CANDIDATES="$NUM_CANDIDATES" \
    MAX_PROMPTS="$MAX_PROMPTS" \
        bash run_selfsup_code.sh
    echo -e "  ${GREEN}Code self-supervised training complete.${NC}"
fi
echo ""

# -----------------------------------------------------------------------
# Stage 3: Mid-training (semisup_code -> mid_checkpoints)
# -----------------------------------------------------------------------
echo -e "${YELLOW}[3/6] Mid-Training${NC}"
if stage_exists "mid_checkpoints"; then
    echo -e "  ${GREEN}Mid-training checkpoint found. Skipping.${NC}"
else
    echo -e "  ${YELLOW}Running mid-training...${NC}"
    python -m scripts.mid_train \
        --model-tag "$MODEL_TAG" \
        --device-batch-size "$DEVICE_BATCH_SIZE" \
        --max-seq-len 1024 \
        --total-batch-size 65536 \
        --num-iterations -1
    echo -e "  ${GREEN}Mid-training complete.${NC}"
fi
echo ""

# -----------------------------------------------------------------------
# Stage 4: General self-supervised (mid -> semisup_general_checkpoints)
# -----------------------------------------------------------------------
echo -e "${YELLOW}[4/6] General Knowledge Self-Supervised Training${NC}"
if stage_exists "semisup_general_checkpoints"; then
    echo -e "  ${GREEN}General self-sup checkpoint found. Skipping.${NC}"
else
    echo -e "  ${YELLOW}Running general self-supervised training...${NC}"
    SOURCE=mid \
    MODEL_TAG="$MODEL_TAG" \
    DEVICE_BATCH_SIZE="$DEVICE_BATCH_SIZE" \
    TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    SCORE_BATCH_SIZE="$SCORE_BATCH_SIZE" \
    NUM_CANDIDATES="$NUM_CANDIDATES" \
    MAX_PROMPTS="$MAX_PROMPTS" \
        bash run_selfsup_general.sh
    echo -e "  ${GREEN}General self-supervised training complete.${NC}"
fi
echo ""

# -----------------------------------------------------------------------
# Stage 5: Math self-supervised (semisup_general -> semisup_math_checkpoints)
# -----------------------------------------------------------------------
echo -e "${YELLOW}[5/6] Math Self-Supervised Training${NC}"
if stage_exists "semisup_math_checkpoints"; then
    echo -e "  ${GREEN}Math self-sup checkpoint found. Skipping.${NC}"
else
    echo -e "  ${YELLOW}Running math self-supervised training...${NC}"
    SOURCE=semisup_general \
    MODEL_TAG="$MODEL_TAG" \
    DEVICE_BATCH_SIZE="$DEVICE_BATCH_SIZE" \
    TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    SCORE_BATCH_SIZE="$SCORE_BATCH_SIZE" \
    MAX_PROMPTS="$MAX_PROMPTS" \
        bash run_selfsup_math.sh
    echo -e "  ${GREEN}Math self-supervised training complete.${NC}"
fi
echo ""

# -----------------------------------------------------------------------
# Stage 6: Chat SFT (semisup_math -> chatsft_checkpoints)
# -----------------------------------------------------------------------
echo -e "${YELLOW}[6/6] Chat SFT${NC}"
if stage_exists "chatsft_checkpoints"; then
    echo -e "  ${GREEN}Chat SFT checkpoint found. Skipping.${NC}"
else
    echo -e "  ${YELLOW}Running chat SFT...${NC}"
    python -m scripts.chat_sft \
        --source semisup_math \
        --model-tag "$MODEL_TAG" \
        --device-batch-size 4
    echo -e "  ${GREEN}Chat SFT complete.${NC}"
fi
echo ""

# -----------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Pipeline complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Checkpoints saved under ${CACHE_DIR}/:"
echo "  base_checkpoints/${MODEL_TAG}/"
echo "  semisup_code_checkpoints/${MODEL_TAG}/"
echo "  mid_checkpoints/${MODEL_TAG}/"
echo "  semisup_general_checkpoints/${MODEL_TAG}/"
echo "  semisup_math_checkpoints/${MODEL_TAG}/"
echo "  chatsft_checkpoints/${MODEL_TAG}/"
echo ""
echo "To chat with your model:"
echo "  python -m scripts.chat_cli --model-tag ${MODEL_TAG}"
echo ""
echo "To evaluate:"
echo "  python -m scripts.chat_eval --model-tag ${MODEL_TAG}"
