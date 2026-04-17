#!/bin/bash
set -euo pipefail
cd ~/nanochat-crate-a

PYTHON_BIN=".venv/bin/python3"

echo "================================================================"
echo "[eval] CHAT — SFT models (8 checkpoints)"
echo "================================================================"

${PYTHON_BIN} -m scripts.pipeline_eval \
    --mode chat \
    --checkpoints \
        sft:d12-gpt-4090 \
        sft:d12-gpt-jepa-lindecay-4090_jepa \
        sft:d12-gpt-jepa-4090_jepa \
        sft:d12-noqgpt-4090_jepa \
        sft:d12-trm-4090_jepa \
        sft:d12-trm-jepa-lindecay-4090_jepa \
        sft:d12-rys-4090-2_jepa \
        sft:d12-rys-jepa-lindecay-4090_jepa

echo ""
echo "================================================================"
echo "All chat evaluations complete!"
echo "Results: ~/.cache/nanochat/pipeline_eval/"
echo "================================================================"
