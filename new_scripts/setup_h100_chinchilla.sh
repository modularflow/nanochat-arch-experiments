#!/bin/bash
# =============================================================================
# One-shot environment setup for a fresh 5× H100 NVL rental box.
#
# Modeled on Karpathy's speedrun.sh, but specialized for the d12 Chinchilla
# sweep (~2.43B tokens per run × 5 runs, all sharing the same shards on disk).
#
# This script handles:
#   1) uv + .venv + dependency sync (GPU extras)
#   2) Pretraining shard download (NUM_SHARDS=100 by default, ~10 GB)
#   3) Tokenizer training on 2B chars, vocab=65536
#   4) Tokenizer evaluation
#   5) Midtraining assets — identity_conversations.jsonl + words_alpha.txt
#   6) Prefetch eval_bundle.zip (~162 MB) to avoid 5-way concurrent download
#   7) Prefetch HuggingFace datasets (smoltalk / MMLU / GSM8K) to the shared HF
#      cache so that 5 parallel runs don't race on HF cold-start downloads.
#
# Then (optionally) kicks off the parallel sweep.
#
# Usage (fresh box):
#   git clone <repo> && cd nanochat-crate-a
#   bash new_scripts/setup_h100_chinchilla.sh
#
#   # Setup + immediately launch the sweep:
#   LAUNCH=1 bash new_scripts/setup_h100_chinchilla.sh
#
#   # Setup only a subset of rows worth of data (e.g. smoke test on 1 GPU):
#   NUM_SHARDS=20 bash new_scripts/setup_h100_chinchilla.sh
#
#   # Pick up partway through:
#   SKIP_VENV=1 SKIP_TOKENIZER=1 bash new_scripts/setup_h100_chinchilla.sh
#
# Skip-phase flags (all default 0 = run):
#   SKIP_VENV=1       — skip uv install + venv sync
#   SKIP_SHARDS=1     — skip pretraining shard download
#   SKIP_TOKENIZER=1  — skip tokenizer train + eval
#   SKIP_MID_ASSETS=1 — skip identity_conversations + words_alpha prefetch
#   SKIP_EVAL_BUNDLE=1 — skip eval_bundle.zip prefetch
#   SKIP_HF_PREFETCH=1 — skip HF dataset prefetch (smoltalk/mmlu/gsm8k)
#
# Shard-count sizing:
#   d12 × NUM_ITERATIONS=37000 × TOTAL_BATCH_SIZE=65536 = 2.43B tokens per run.
#   @ ~4.8 chars/tok ⇒ 11.66B chars per run = 47 shards of 250M chars.
#   DataLoader wastes ~35% to cropping ⇒ 73 shards per run.
#   5 parallel runs share the same shard pool (OS page cache handles it).
#   ⇒ NUM_SHARDS=100 gives comfortable headroom. Override for smoke tests.
# =============================================================================

set -euo pipefail

# Resolve to the repo root regardless of where we're invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# --- Config ---
NUM_SHARDS="${NUM_SHARDS:-100}"
TOKENIZER_MAX_CHARS="${TOKENIZER_MAX_CHARS:-2000000000}"
TOKENIZER_VOCAB_SIZE="${TOKENIZER_VOCAB_SIZE:-65536}"
INITIAL_SHARDS_FOR_TOKENIZER="${INITIAL_SHARDS_FOR_TOKENIZER:-8}"

SKIP_VENV="${SKIP_VENV:-0}"
SKIP_SHARDS="${SKIP_SHARDS:-0}"
SKIP_TOKENIZER="${SKIP_TOKENIZER:-0}"
SKIP_MID_ASSETS="${SKIP_MID_ASSETS:-0}"
SKIP_EVAL_BUNDLE="${SKIP_EVAL_BUNDLE:-0}"
SKIP_HF_PREFETCH="${SKIP_HF_PREFETCH:-0}"

LAUNCH="${LAUNCH:-0}"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "${NANOCHAT_BASE_DIR}"

echo ""
echo "================================================================================"
echo "H100 NVL Chinchilla sweep — environment setup"
echo "  Repo:        ${REPO_DIR}"
echo "  Base dir:    ${NANOCHAT_BASE_DIR}"
echo "  Shards:      ${NUM_SHARDS}  (initial ${INITIAL_SHARDS_FOR_TOKENIZER} for tokenizer)"
echo "  Tokenizer:   vocab=${TOKENIZER_VOCAB_SIZE}  max_chars=${TOKENIZER_MAX_CHARS}"
echo "  Skip flags:  VENV=${SKIP_VENV} SHARDS=${SKIP_SHARDS} TOKENIZER=${SKIP_TOKENIZER}"
echo "               MID_ASSETS=${SKIP_MID_ASSETS} EVAL_BUNDLE=${SKIP_EVAL_BUNDLE}"
echo "               HF_PREFETCH=${SKIP_HF_PREFETCH}"
echo "  LAUNCH:      ${LAUNCH}  (1 = chain into run_chinchilla_parallel_5gpu.sh)"
echo "================================================================================"
echo ""

# -----------------------------------------------------------------------------
# 1) Python venv setup with uv
# -----------------------------------------------------------------------------

if [ "${SKIP_VENV}" = "0" ]; then
    echo "--- [1/7] uv + venv + dependencies ---"
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    # Ensure uv is on PATH if we just installed it.
    export PATH="$HOME/.local/bin:$PATH"

    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "    venv python: $(which python)"
    echo "    python ver:  $(python --version)"
    echo ""
else
    echo "--- [1/7] uv + venv + dependencies (SKIPPED) ---"
    # Still activate the venv for the rest of the script.
    if [ -d ".venv" ]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
    fi
    echo ""
fi

# From here on, `python` should be the project venv python.
PYTHON_BIN="${PYTHON_BIN:-python}"

# -----------------------------------------------------------------------------
# GPU / CUDA sanity check. Non-fatal — just surface info.
# -----------------------------------------------------------------------------

if command -v nvidia-smi &> /dev/null; then
    echo "--- GPU info ---"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
    echo ""
    echo "--- PyTorch / CUDA / FA3 sanity check ---"
    # FA3 is NOT installed as a pip package in this repo. It's loaded lazily via
    # HuggingFace's `kernels` package (see nanochat/gpt.py):
    #    from kernels import get_kernel
    #    flash_attn = get_kernel('varunneal/flash-attention-3').flash_attn_interface
    # The kernels package downloads a prebuilt FA3 binary from HF Hub matching
    # the installed torch/CUDA version on first use. Gated on SM >= 9.0 (Hopper+).
    # This sanity check reproduces that exact path to verify the download works.
    "${PYTHON_BIN}" -c "
import os
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import torch
print(f'torch:   {torch.__version__}')
print(f'cuda:    {torch.version.cuda}')
print(f'devices: {torch.cuda.device_count()}')
any_hopper = False
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    hopper = p.major >= 9
    any_hopper = any_hopper or hopper
    tag = '(Hopper+, FA3 eligible)' if hopper else '(Ada/Ampere, SDPA fallback)'
    print(f'  [{i}] {p.name}  sm_{p.major}{p.minor}  {p.total_memory/1e9:.1f} GB  {tag}')

if any_hopper:
    try:
        from kernels import get_kernel
        fa3 = get_kernel('varunneal/flash-attention-3').flash_attn_interface
        print(f'flash-attn 3: OK via kernels.get_kernel')
        print(f'              module: {fa3.__name__}')
        print(f'              has flash_attn_func: {hasattr(fa3, \"flash_attn_func\")}')
    except Exception as e:
        print(f'flash-attn 3: FAILED to load via kernels ({e.__class__.__name__}: {e})')
        print(f'              training will fall back to PyTorch SDPA (slower).')
        print(f'              Check HF Hub network egress / HF_TOKEN if this is gated.')
else:
    print('flash-attn 3: not attempted (no Hopper+ GPU detected).')
"
    echo ""
else
    echo "WARN: nvidia-smi not found — are you on a GPU box?"
    echo ""
fi

# -----------------------------------------------------------------------------
# 2) Pretraining shards
#    First ${INITIAL_SHARDS_FOR_TOKENIZER} synchronously (needed by tokenizer),
#    then the remaining ${NUM_SHARDS} in the background while tokenizer trains.
# -----------------------------------------------------------------------------

DATASET_DOWNLOAD_PID=""

if [ "${SKIP_SHARDS}" = "0" ]; then
    echo "--- [2/7] Pretraining shards (~${NUM_SHARDS} total) ---"
    # Synchronous: first batch needed by tokenizer.
    "${PYTHON_BIN}" -m nanochat.dataset -n "${INITIAL_SHARDS_FOR_TOKENIZER}"

    # Background: the rest (nanochat.dataset is idempotent — re-requesting 100
    # when 8 are already on disk downloads the missing 92).
    "${PYTHON_BIN}" -m nanochat.dataset -n "${NUM_SHARDS}" &
    DATASET_DOWNLOAD_PID=$!
    echo "    full-shard download running in background (pid ${DATASET_DOWNLOAD_PID})"
    echo ""
else
    echo "--- [2/7] Shard download (SKIPPED) ---"
    echo ""
fi

# -----------------------------------------------------------------------------
# 3) Tokenizer training
# -----------------------------------------------------------------------------

TOKENIZER_DIR="${NANOCHAT_BASE_DIR}/tokenizer"

if [ "${SKIP_TOKENIZER}" = "0" ]; then
    if [ -d "${TOKENIZER_DIR}" ] && [ -n "$(ls -A "${TOKENIZER_DIR}" 2>/dev/null)" ]; then
        echo "--- [3/7] Tokenizer (already present at ${TOKENIZER_DIR}, skipping train) ---"
    else
        echo "--- [3/7] Training tokenizer (vocab=${TOKENIZER_VOCAB_SIZE}, max_chars=${TOKENIZER_MAX_CHARS}) ---"
        "${PYTHON_BIN}" -m scripts.tok_train \
            --max-chars="${TOKENIZER_MAX_CHARS}" \
            --vocab-size="${TOKENIZER_VOCAB_SIZE}"
        echo ""
        echo "    evaluating tokenizer..."
        "${PYTHON_BIN}" -m scripts.tok_eval
    fi
    echo ""
else
    echo "--- [3/7] Tokenizer (SKIPPED) ---"
    echo ""
fi

# -----------------------------------------------------------------------------
# 4) Wait for background shard download (must finish before we can train)
# -----------------------------------------------------------------------------

if [ -n "${DATASET_DOWNLOAD_PID}" ]; then
    echo "--- [4/7] Waiting on background shard download (pid ${DATASET_DOWNLOAD_PID}) ---"
    wait "${DATASET_DOWNLOAD_PID}" || {
        echo "ERROR: shard download failed. Re-run with SKIP_VENV=1 SKIP_TOKENIZER=1 to retry."
        exit 1
    }
    echo "    shard download complete."
    echo ""
else
    echo "--- [4/7] No background shard download to wait for ---"
    echo ""
fi

# -----------------------------------------------------------------------------
# 5) Midtraining assets (identity_conversations.jsonl + words_alpha.txt)
#    These normally auto-download on first use with file locks, but we pre-fetch
#    them so that 5 parallel runs don't race on the same network/disk writes.
# -----------------------------------------------------------------------------

if [ "${SKIP_MID_ASSETS}" = "0" ]; then
    echo "--- [5/7] Midtraining assets ---"

    IDENTITY_PATH="${NANOCHAT_BASE_DIR}/identity_conversations.jsonl"
    if [ ! -f "${IDENTITY_PATH}" ]; then
        echo "    downloading identity_conversations.jsonl ..."
        curl -fSL -o "${IDENTITY_PATH}" \
            https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    else
        echo "    identity_conversations.jsonl already present."
    fi

    WORDS_PATH="${NANOCHAT_BASE_DIR}/words_alpha.txt"
    if [ ! -f "${WORDS_PATH}" ]; then
        echo "    downloading words_alpha.txt ..."
        curl -fSL -o "${WORDS_PATH}" \
            https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt
    else
        echo "    words_alpha.txt already present."
    fi
    echo ""
else
    echo "--- [5/7] Midtraining assets (SKIPPED) ---"
    echo ""
fi

# -----------------------------------------------------------------------------
# 6) Prefetch eval_bundle.zip
#    pipeline_eval auto-downloads on first use via download_file_with_lock,
#    but with 5 runs finishing near-simultaneously the lock contention is ugly.
#    Fetch once here while the venv python is handy.
# -----------------------------------------------------------------------------

if [ "${SKIP_EVAL_BUNDLE}" = "0" ]; then
    EVAL_BUNDLE_DIR="${NANOCHAT_BASE_DIR}/eval_bundle"
    if [ -d "${EVAL_BUNDLE_DIR}" ] && [ -n "$(ls -A "${EVAL_BUNDLE_DIR}" 2>/dev/null)" ]; then
        echo "--- [6/7] eval_bundle already present, skipping ---"
    else
        echo "--- [6/7] Prefetching eval_bundle (~162 MB) ---"
        "${PYTHON_BIN}" -c "
from nanochat.common import get_base_dir, download_file_with_lock
from scripts.base_eval import EVAL_BUNDLE_URL, place_eval_bundle
import os
if not os.path.exists(os.path.join(get_base_dir(), 'eval_bundle')):
    download_file_with_lock(EVAL_BUNDLE_URL, 'eval_bundle.zip', postprocess_fn=place_eval_bundle)
print('eval_bundle ready at', os.path.join(get_base_dir(), 'eval_bundle'))
"
    fi
    echo ""
else
    echo "--- [6/7] eval_bundle (SKIPPED) ---"
    echo ""
fi

# -----------------------------------------------------------------------------
# 7) Prefetch HuggingFace datasets used by mid-training + SFT
#    Done via a short python script so load_dataset caches to $HF_HOME.
#    Pre-populating avoids a thundering herd when 5 parallel mid_train jobs
#    all start and try to pull smoltalk at once.
# -----------------------------------------------------------------------------

if [ "${SKIP_HF_PREFETCH}" = "0" ]; then
    echo "--- [7/7] Prefetching HuggingFace datasets (smoltalk / mmlu / gsm8k) ---"
    "${PYTHON_BIN}" - <<'PY'
from datasets import load_dataset

targets = [
    ("HuggingFaceTB/smol-smoltalk", {"split": "train"}),
    ("HuggingFaceTB/smol-smoltalk", {"split": "test"}),
    ("cais/mmlu",                   {"name": "auxiliary_train", "split": "train"}),
    ("openai/gsm8k",                {"name": "main", "split": "train"}),
]

for name, kw in targets:
    label = f"{name}  ({kw})"
    try:
        ds = load_dataset(name, **kw)
        print(f"  OK  {label}  -> {len(ds):,} rows")
    except Exception as e:
        # Some names may drift on HF; non-fatal — they'll be lazily fetched later.
        print(f"  WARN  {label}  -> {e.__class__.__name__}: {e}")
PY
    echo ""
else
    echo "--- [7/7] HF dataset prefetch (SKIPPED) ---"
    echo ""
fi

# -----------------------------------------------------------------------------
# Summary + optional launch
# -----------------------------------------------------------------------------

echo "================================================================================"
echo "Setup complete."
echo "  tokenizer:  ${TOKENIZER_DIR}"
echo "  shards:     ${NANOCHAT_BASE_DIR}/base_data/   ($(ls "${NANOCHAT_BASE_DIR}/base_data" 2>/dev/null | wc -l) files)"
echo "  eval_bundle:${NANOCHAT_BASE_DIR}/eval_bundle/"
echo "  mid assets: ${NANOCHAT_BASE_DIR}/{identity_conversations.jsonl, words_alpha.txt}"
echo "  HF cache:   ${HF_HOME:-$HOME/.cache/huggingface}"
echo ""
echo "Disk usage (under ${NANOCHAT_BASE_DIR}):"
du -sh "${NANOCHAT_BASE_DIR}"/* 2>/dev/null | sort -h
echo "================================================================================"
echo ""

if [ "${LAUNCH}" = "1" ]; then
    echo "LAUNCH=1 → chaining into run_chinchilla_parallel_5gpu.sh"
    echo ""
    exec bash new_scripts/run_chinchilla_parallel_5gpu.sh
else
    echo "Next step — launch the 5-GPU parallel sweep:"
    echo ""
    echo "  tmux new -s chinchilla"
    echo "  bash new_scripts/run_chinchilla_parallel_5gpu.sh"
    echo ""
    echo "Or do a single-GPU pre-flight first (recommended):"
    echo ""
    echo "  CUDA_VISIBLE_DEVICES=0 \\"
    echo "  NUM_ITERATIONS=100 MID_NUM_ITERATIONS=50 \\"
    echo "  EVAL_EVERY=50 SAVE_EVERY=100 \\"
    echo "  SFT_EVAL_EVERY=50 SFT_EVAL_STEPS=20 SFT_EVAL_METRICS_EVERY=1000000 \\"
    echo "  TAG_PREFIX=d12-preflight TAG_SUFFIX=h100 \\"
    echo "  WANDB_RUN=preflight \\"
    echo "  ROWS=\"5\" \\"
    echo "  bash new_scripts/run_chinchilla_d12_h100.sh"
fi
