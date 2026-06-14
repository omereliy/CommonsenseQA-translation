#!/usr/bin/env bash
# Pull the cluster's results/ tree into a dated local subdir (never deletes remotely).
# Each sync lands in its own results/cluster-YYYYMMDD/ so sweeps stay separable.
#
# Usage:
#   bash .claude/skills/cluster-ops/scripts/sync.sh                 # → results/cluster-YYYYMMDD/
#   bash .claude/skills/cluster-ops/scripts/sync.sh results/my-run  # explicit dest
# Env: REMOTE_HOST (default slurm), REMOTE_RESULTS (default ~/CommonsenseQA-translation/results)
set -euo pipefail
REMOTE_HOST="${REMOTE_HOST:-slurm}"
REMOTE_RESULTS="${REMOTE_RESULTS:-CommonsenseQA-translation/results}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"   # scripts → cluster-ops → skills → .claude → repo
DEST="${1:-$REPO_ROOT/results/cluster-$(date +%Y%m%d)}"

mkdir -p "$DEST"
echo "Syncing ${REMOTE_HOST}:${REMOTE_RESULTS}/ → $DEST"
rsync -az -e "ssh -o BatchMode=yes" \
  --exclude='*.tmp' \
  "${REMOTE_HOST}:${REMOTE_RESULTS}/" "$DEST/"

n=$(find "$DEST" -name outputs.jsonl 2>/dev/null | wc -l | tr -d ' ')
echo "Done — $n outputs.jsonl under $DEST"
echo "Analyze: PYTHONPATH=src python -m scripts.analyze --results-dir \"$DEST\""
