#!/bin/bash
# =============================================================================
# Parallel launcher: 5 rows × 1 H100 NVL each, full pipeline.
#
# Fires 5 instances of run_chinchilla_d12_h100.sh in the background, each
# pinned to a single GPU via CUDA_VISIBLE_DEVICES, each restricted to one
# ROW via the ROWS= env knob. Staggers startup by STAGGER_SECONDS (default
# 60s) to avoid torch.compile / disk I/O contention during warmup.
#
# Usage:
#   tmux new -s chinchilla
#   bash new_scripts/run_chinchilla_parallel_5gpu.sh
#
# Logs go to logs/chinchilla/row<N>.log — watch them with:
#   tail -f logs/chinchilla/row*.log
#
# Override any env var that run_chinchilla_d12_h100.sh understands, e.g.:
#   NUM_ITERATIONS=18000 bash new_scripts/run_chinchilla_parallel_5gpu.sh
#
# Specific launcher knobs:
#   STAGGER_SECONDS=60 — delay between starting each GPU's job (default 60)
#   NUM_GPUS=5         — expected GPU count, used to sanity-check rows
#   SKIP_WAIT=1        — exit immediately after launching (don't wait for wait)
# =============================================================================

set -euo pipefail

cd ~/nanochat-crate-a

STAGGER_SECONDS="${STAGGER_SECONDS:-60}"
NUM_GPUS="${NUM_GPUS:-5}"
SKIP_WAIT="${SKIP_WAIT:-0}"

mkdir -p logs/chinchilla

# Pre-flight: verify we have ≥5 GPUs visible.
if command -v nvidia-smi >/dev/null 2>&1; then
    visible_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    echo "Detected ${visible_gpus} GPUs via nvidia-smi (expected ${NUM_GPUS})."
    if [ "${visible_gpus}" -lt "${NUM_GPUS}" ]; then
        echo "ERROR: Need at least ${NUM_GPUS} GPUs. Found ${visible_gpus}."
        exit 1
    fi
else
    echo "WARN: nvidia-smi not found; skipping GPU count check."
fi

echo ""
echo "================================================================================"
echo "Launching 5 parallel Chinchilla runs (1 GPU each, ${STAGGER_SECONDS}s stagger)"
echo "  Logs: logs/chinchilla/row<N>.log"
echo "================================================================================"

declare -a PIDS=()

for i in 1 2 3 4 5; do
    gpu_idx=$((i - 1))
    logfile="logs/chinchilla/row${i}.log"

    echo "[$(date +%T)] Starting ROW=${i} on GPU ${gpu_idx} -> ${logfile}"

    CUDA_VISIBLE_DEVICES="${gpu_idx}" \
    ROWS="${i}" \
    bash new_scripts/run_chinchilla_d12_h100.sh \
        > "${logfile}" 2>&1 &

    PIDS+=("$!")

    # Stagger all but the last launch to avoid thundering-herd on compile + disk.
    if [ "$i" -lt 5 ]; then
        sleep "${STAGGER_SECONDS}"
    fi
done

echo ""
echo "All 5 launched. PIDs: ${PIDS[*]}"
echo "Tail logs with:  tail -f logs/chinchilla/row*.log"
echo "Monitor GPUs:    watch -n 5 nvidia-smi"

if [ "${SKIP_WAIT}" = "1" ]; then
    echo ""
    echo "SKIP_WAIT=1 — exiting now. Use 'wait ${PIDS[*]}' to block on these jobs."
    exit 0
fi

echo ""
echo "Blocking on all 5 jobs (Ctrl-C to detach; jobs keep running in tmux)..."
echo ""

# Exit code aggregation: report per-row success/fail.
any_fail=0
for i in "${!PIDS[@]}"; do
    row=$((i + 1))
    pid="${PIDS[$i]}"
    if wait "${pid}"; then
        echo "[$(date +%T)] ROW ${row} (pid ${pid}) OK"
    else
        echo "[$(date +%T)] ROW ${row} (pid ${pid}) FAILED — see logs/chinchilla/row${row}.log"
        any_fail=1
    fi
done

echo ""
if [ "${any_fail}" = "1" ]; then
    echo "One or more rows failed. Inspect logs/chinchilla/row*.log for details."
    exit 1
fi

echo "All 5 rows completed successfully."
echo ""
echo "Next steps:"
echo "  1. Verify checkpoints:"
echo "     ls ~/.cache/nanochat/base_checkpoints/ | grep s37k"
echo "     ls ~/.cache/nanochat/chatsft_checkpoints/ | grep s37k"
echo "  2. Pipeline eval CSVs:"
echo "     ls ~/.cache/nanochat/pipeline_eval/ | grep s37k"
echo "  3. Consolidate into leaderboard (use /tmp/consolidate_tpa.py as a template)."
