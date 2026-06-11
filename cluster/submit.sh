#!/bin/bash
# Submit the generative arm as a SLURM array: one task per (model, think) cell.
# Each task serves the model once and evaluates all condition variants.
#
#   bash cluster/submit.sh                 # all models x {off,on}
#   bash cluster/submit.sh Qwen3.5:4B      # one model x {off,on}
#   bash cluster/submit.sh --think off     # all models, think=off only
#   bash cluster/submit.sh --limit 50      # smoke: first 50 items/variant
#   bash cluster/submit.sh --dry-run
set -eo pipefail

PROJ_ROOT="${PROJ_ROOT:-$HOME/CommonsenseQA-translation}"
source "$PROJ_ROOT/cluster/defaults.sh"

MODELS=(); THINKS=("${CSQA_THINK_MODES[@]}"); DRY=0; EXTRA=""; GPU_TYPE="rtx_6000"; TIME="1-00:00:00"
while [ $# -gt 0 ]; do
    case "$1" in
        --think) THINKS=($2); shift 2 ;;
        --limit) EXTRA="${EXTRA},LIMIT=$2"; shift 2 ;;
        --gpu-type) GPU_TYPE="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -*) echo "unknown flag $1" >&2; exit 1 ;;
        *) MODELS+=("$1"); shift ;;
    esac
done
[ ${#MODELS[@]} -eq 0 ] && MODELS=("${CSQA_MODELS[@]}")

CELLS=()
for m in "${MODELS[@]}"; do
    vllm_lookup "$m" || exit 1
    for t in "${THINKS[@]}"; do CELLS+=("${m}|${t}"); done
done
CELLS_LIST=$(IFS='^'; echo "${CELLS[*]}"); N=${#CELLS[@]}
echo "Cells ($N):"; printf '  %s\n' "${CELLS[@]}"

mkdir -p "$PROJ_ROOT/cluster/logs"
SUBMIT=(sbatch --array=0-$((N-1)) --gpus="${GPU_TYPE}:1" --constraint="$GPU_TYPE" --time="$TIME"
        --export="ALL,PROJ_ROOT=${PROJ_ROOT},CELLS_LIST=${CELLS_LIST}${EXTRA}"
        "$PROJ_ROOT/cluster/serve_and_eval.sbatch")
[ "$DRY" -eq 1 ] && { printf '%q ' "${SUBMIT[@]}"; echo; exit 0; }
"${SUBMIT[@]}"
