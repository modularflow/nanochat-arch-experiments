#!/bin/bash
# =============================================================================
# Pull Chinchilla-sweep results off the rental box onto this local machine.
#
# Runs LOCALLY. SSHs to the rental, rsyncs artifacts to ~/.cache/nanochat/
# (or wherever you point it). Idempotent — re-running only transfers deltas.
#
# Usage:
#   # Default: research pull (final checkpoints + EMA + CSVs + logs, ~11 GB)
#   bash new_scripts/fetch_chinchilla_results.sh user@rental.example.com
#
#   # Minimal (eval CSVs + logs + tags only, ~10 MB) — run this FIRST while
#   # training is still going to sanity-check the SSH setup cheaply:
#   MINIMAL=1 bash new_scripts/fetch_chinchilla_results.sh user@rental.example.com
#
#   # Full (all checkpoints + all intermediates + optim states, ~100 GB):
#   FULL=1 bash new_scripts/fetch_chinchilla_results.sh user@rental.example.com
#
#   # Dry-run (build + show the rsync commands but don't transfer):
#   DRY_RUN=1 bash new_scripts/fetch_chinchilla_results.sh user@rental.example.com
#
#   # Custom SSH port / identity / remote path:
#   SSH_PORT=2222 SSH_KEY=~/.ssh/hyperbolic_rsa \
#     REMOTE_BASE=/root/.cache/nanochat \
#     bash new_scripts/fetch_chinchilla_results.sh user@rental.example.com
#
#   # Custom tag pattern (default: "d12-s37k-*" — matches the sweep's 5 rows):
#   TAG_PATTERN="d12-preflight-*" bash new_scripts/fetch_chinchilla_results.sh user@rental.example.com
#
# What gets pulled by mode:
#   MINIMAL:
#     ~/.cache/nanochat/pipeline_eval/comparison_core_*<TAG>*.csv
#     ~/.cache/nanochat/last_chinchilla_sweep_*_tags.txt
#     ~/nanochat-crate-a/logs/chinchilla/*.log
#
#   Default (research):
#     MINIMAL plus:
#     ~/.cache/nanochat/base_checkpoints/<TAG>/meta_*.json
#     ~/.cache/nanochat/base_checkpoints/<TAG>/model_*.pt          (all steps)
#     ~/.cache/nanochat/base_checkpoints/<TAG>/model_ema_*.pt      (EMA shadow)
#     ~/.cache/nanochat/mid_checkpoints/<TAG>/meta_*.json + model_*.pt
#     ~/.cache/nanochat/chatsft_checkpoints/<TAG>/meta_*.json + model_*.pt
#     (optim_*_rank*.pt files are EXCLUDED — not needed for inference/eval)
#
#   FULL:
#     Default plus optim_*.pt files (resumable / for further fine-tuning)
# =============================================================================

set -euo pipefail

SSH_TARGET="${1:-}"
if [ -z "${SSH_TARGET}" ]; then
    echo "Usage: bash $0 <ssh_target> [flags via env vars]" >&2
    echo "Example: bash $0 root@123.45.67.89" >&2
    exit 1
fi

# --- Config ---
MINIMAL="${MINIMAL:-0}"
FULL="${FULL:-0}"
DRY_RUN="${DRY_RUN:-0}"

SSH_PORT="${SSH_PORT:-22}"
SSH_KEY="${SSH_KEY:-}"
REMOTE_BASE="${REMOTE_BASE:-\$HOME/.cache/nanochat}"
REMOTE_REPO="${REMOTE_REPO:-\$HOME/nanochat-crate-a}"
LOCAL_BASE="${LOCAL_BASE:-$HOME/.cache/nanochat}"
LOCAL_REPO="${LOCAL_REPO:-$HOME/nanochat-crate-a}"

TAG_PATTERN="${TAG_PATTERN:-d12-s37k-*}"

# --- Build SSH/rsync args ---
SSH_ARGS="-p ${SSH_PORT}"
if [ -n "${SSH_KEY}" ]; then
    SSH_ARGS="${SSH_ARGS} -i ${SSH_KEY}"
fi
RSYNC_SSH="ssh ${SSH_ARGS}"

# -z = compress on the wire; -P = --partial --progress;
# -a = archive (recursive, preserve perms/times/links); -h = human-readable
RSYNC_BASE_OPTS=(-azh -P)
if [ "${DRY_RUN}" = "1" ]; then
    RSYNC_BASE_OPTS+=(--dry-run --verbose)
fi

mkdir -p "${LOCAL_BASE}/base_checkpoints" \
         "${LOCAL_BASE}/mid_checkpoints" \
         "${LOCAL_BASE}/chatsft_checkpoints" \
         "${LOCAL_BASE}/pipeline_eval" \
         "${LOCAL_REPO}/logs/chinchilla"

echo ""
echo "================================================================================"
echo "Chinchilla fetch — pulling from ${SSH_TARGET}"
echo "  Mode:         $( [ "${MINIMAL}" = "1" ] && echo MINIMAL || ([ "${FULL}" = "1" ] && echo FULL || echo RESEARCH))"
echo "  Remote base:  ${REMOTE_BASE}"
echo "  Remote repo:  ${REMOTE_REPO}"
echo "  Local base:   ${LOCAL_BASE}"
echo "  Local repo:   ${LOCAL_REPO}"
echo "  Tag pattern:  ${TAG_PATTERN}"
echo "  Dry-run:      ${DRY_RUN}"
echo "================================================================================"
echo ""

# -----------------------------------------------------------------------------
# Sanity: verify we can reach the remote and that the sweep artifacts exist.
# -----------------------------------------------------------------------------

echo "--- Remote sanity check ---"
# shellcheck disable=SC2029
ssh ${SSH_ARGS} "${SSH_TARGET}" "bash -s" <<EOF
set -e
echo "  host:     \$(hostname)"
echo "  user:     \$(whoami)"
base_dir="${REMOTE_BASE}"
base_dir="\${base_dir/#\\\$HOME/\$HOME}"
echo "  base dir: \${base_dir}"
if [ ! -d "\${base_dir}" ]; then
    echo "  WARN: base dir does not exist yet."
    exit 0
fi
echo ""
echo "  Matching tags under base_checkpoints:"
(cd "\${base_dir}" && ls -d base_checkpoints/${TAG_PATTERN} 2>/dev/null | sed 's|^|    |') || echo "    (none)"
echo ""
echo "  Matching tags under chatsft_checkpoints:"
(cd "\${base_dir}" && ls -d chatsft_checkpoints/${TAG_PATTERN}* 2>/dev/null | sed 's|^|    |') || echo "    (none)"
echo ""
echo "  Matching eval CSVs:"
(cd "\${base_dir}/pipeline_eval" 2>/dev/null && ls comparison_core_*${TAG_PATTERN}*.csv 2>/dev/null | wc -l | xargs -I{} echo "    {} files") || echo "    (pipeline_eval dir missing)"
EOF
echo ""

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Expand remote path that may contain $HOME. We do this client-side so rsync's
# --relative flag anchors everything under ~/ correctly.
RB="${REMOTE_BASE}"     # e.g. "\$HOME/.cache/nanochat" (literal $HOME for ssh)
RR="${REMOTE_REPO}"

# -----------------------------------------------------------------------------
# MINIMAL: eval CSVs, tag files, logs (always pulled, tiny)
# -----------------------------------------------------------------------------

echo "--- Pulling eval CSVs matching '${TAG_PATTERN}' ---"
# Use --include/--exclude to select only CSVs matching our tag pattern.
rsync "${RSYNC_BASE_OPTS[@]}" \
    -e "${RSYNC_SSH}" \
    --include="comparison_core_*${TAG_PATTERN//\*/}*.csv" \
    --include="leaderboard_*.csv" \
    --exclude="*" \
    "${SSH_TARGET}:${RB}/pipeline_eval/" \
    "${LOCAL_BASE}/pipeline_eval/"
echo ""

echo "--- Pulling tag files + manifests ---"
rsync "${RSYNC_BASE_OPTS[@]}" \
    -e "${RSYNC_SSH}" \
    --include="last_chinchilla_sweep_*_tags.txt" \
    --include="last_*_tags.txt" \
    --exclude="*/" \
    --exclude="*" \
    "${SSH_TARGET}:${RB}/" \
    "${LOCAL_BASE}/"
echo ""

echo "--- Pulling parallel-launcher logs ---"
rsync "${RSYNC_BASE_OPTS[@]}" \
    -e "${RSYNC_SSH}" \
    --include="row*.log" \
    --include="*.log" \
    --exclude="*" \
    "${SSH_TARGET}:${RR}/logs/chinchilla/" \
    "${LOCAL_REPO}/logs/chinchilla/"
echo ""

if [ "${MINIMAL}" = "1" ]; then
    echo "MINIMAL mode — stopping here."
    echo ""
    echo "Pulled to:"
    echo "  ${LOCAL_BASE}/pipeline_eval/"
    echo "  ${LOCAL_BASE}/last_chinchilla_sweep_*_tags.txt"
    echo "  ${LOCAL_REPO}/logs/chinchilla/"
    exit 0
fi

# -----------------------------------------------------------------------------
# RESEARCH / FULL: checkpoints
# -----------------------------------------------------------------------------

# Pattern to determine which checkpoint files to transfer.
# Default (RESEARCH): model + EMA weights + metadata, no optim states.
# FULL: include optim_*.pt too.
CKPT_INCLUDES=(
    --include="${TAG_PATTERN}/"
    --include="${TAG_PATTERN}/meta_*.json"
    --include="${TAG_PATTERN}/model_*.pt"
    --include="${TAG_PATTERN}/*.json"
    --include="${TAG_PATTERN}/*.txt"
)
if [ "${FULL}" = "1" ]; then
    CKPT_INCLUDES+=(--include="${TAG_PATTERN}/optim_*.pt")
fi
CKPT_EXCLUDES=(
    --exclude="optim_*.pt"   # explicit: dropped unless FULL=1 re-included it
    --exclude="*"
)

for phase_dir in base_checkpoints mid_checkpoints chatsft_checkpoints; do
    echo "--- Pulling ${phase_dir}/${TAG_PATTERN} ---"
    rsync "${RSYNC_BASE_OPTS[@]}" \
        -e "${RSYNC_SSH}" \
        "${CKPT_INCLUDES[@]}" \
        "${CKPT_EXCLUDES[@]}" \
        "${SSH_TARGET}:${RB}/${phase_dir}/" \
        "${LOCAL_BASE}/${phase_dir}/"
    echo ""
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY_RUN complete. Re-run without DRY_RUN=1 to actually transfer."
    exit 0
fi

echo "================================================================================"
echo "Fetch complete. Local footprint:"
echo ""
echo "  Checkpoints matching '${TAG_PATTERN}':"
for phase_dir in base_checkpoints mid_checkpoints chatsft_checkpoints; do
    if [ -d "${LOCAL_BASE}/${phase_dir}" ]; then
        while IFS= read -r d; do
            [ -z "$d" ] && continue
            sz=$(du -sh "$d" 2>/dev/null | cut -f1)
            echo "    ${sz}  ${d#${LOCAL_BASE}/}"
        done < <(find "${LOCAL_BASE}/${phase_dir}" -maxdepth 1 -mindepth 1 -type d -name "${TAG_PATTERN}" 2>/dev/null)
    fi
done
echo ""
csv_count=$(find "${LOCAL_BASE}/pipeline_eval" -maxdepth 1 -name "comparison_core_*${TAG_PATTERN//\*/}*.csv" 2>/dev/null | wc -l)
echo "  Eval CSVs:   ${csv_count} files in ${LOCAL_BASE}/pipeline_eval/"
log_count=$(find "${LOCAL_REPO}/logs/chinchilla" -maxdepth 1 -name "*.log" 2>/dev/null | wc -l)
echo "  Logs:        ${log_count} files in ${LOCAL_REPO}/logs/chinchilla/"
echo ""
echo "  Total pulled (local):"
du -sh "${LOCAL_BASE}" 2>/dev/null | sed 's/^/    /'
echo ""
echo "Next: consolidate into the leaderboard. Use /tmp/consolidate_tpa.py as a"
echo "template, or adapt for the d12-s37k tag pattern."
echo "================================================================================"
