#!/bin/bash
set -euo pipefail
cd ~/nanochat-crate-a

PYTHON_BIN=".venv/bin/python3"

echo "================================================================"
echo "[eval] CORE — Base models (10 checkpoints)"
echo "================================================================"

${PYTHON_BIN} -m scripts.pipeline_eval \
    --mode core \
    --checkpoints \
        base:d12-trm-4090 \
        base:d12-trm-jepa-lindecay-4090 \
        base:d12-rys-4090-2 \
        base:d12-rys-jepa-lindecay-4090 \


echo ""
echo "================================================================"
echo "[eval] CORE — SFT models (8 checkpoints)"
echo "================================================================"

${PYTHON_BIN} -m scripts.pipeline_eval \
    --mode core \
    --checkpoints \
        sft:d12-trm-4090_jepa \
        sft:d12-trm-jepa-lindecay-4090_jepa \
        sft:d12-rys-4090-2_jepa \
        sft:d12-rys-jepa-lindecay-4090_jepa \


echo ""
echo "================================================================"
echo "All evaluations complete!"
echo "Results: ~/.cache/nanochat/pipeline_eval/"
echo "================================================================"
