#!/bin/bash
# Submit the generative arm as a SLURM array: one task per (model, think) cell.
# Each task serves the model once and evaluates all condition variants.
#
#   bash cluster/submit.sh                          # all models x {off,on}
#   bash cluster/submit.sh Qwen3.5:4B               # one model x {off,on}
#   bash cluster/submit.sh --think off              # all models, think=off only
#   bash cluster/submit.sh --limit 50               # smoke: first 50 items/variant
#   bash cluster/submit.sh --gpu-type rtx_pro_6000  # escape hatch: 96 GB pool
#   bash cluster/submit.sh --dry-run
#
# GPU routing (mirrors pddl-copilot-experiments/submit_with_rtx.sh):
#   rtx_6000      (48 GB, --mem=48G)  default — the full roster fits at
#                                     gpu-memory-utilization=0.85.
#   rtx_pro_6000  (96 GB, --mem=80G)  opt-in escape hatch — use when the
#                                     rtx_6000 pool is queue-saturated.
# The CLI --gpus/--constraint/--mem override the matching #SBATCH directives
# in serve_and_eval.sbatch (CLI wins), so the pool swap needs no sbatch edit.
set -eo pipefail

PROJ_ROOT="${PROJ_ROOT:-$HOME/CommonsenseQA-translation}"
source "$PROJ_ROOT/cluster/defaults.sh"

MODELS=(); THINKS=("${CSQA_THINK_MODES[@]}"); DRY=0; EXTRA=""; GPU_TYPE="rtx_6000"
TIME=""; TIME_SET=0; SMOKE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --think) THINKS=($2); shift 2 ;;
        --limit) EXTRA="${EXTRA},LIMIT=$2"; SMOKE=1; shift 2 ;;
        --gpu-type) GPU_TYPE="$2"; shift 2 ;;
        --time) TIME="$2"; TIME_SET=1; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -*) echo "unknown flag $1" >&2; exit 1 ;;
        *) MODELS+=("$1"); shift ;;
    esac
done
[ ${#MODELS[@]} -eq 0 ] && MODELS=("${CSQA_MODELS[@]}")

# Walltime → backfill behaviour (lesson from pddl-copilot-experiments/submit_planbench.sh):
# a short request BACKFILLS into idle-GPU gaps; a 1-day request can't slot into a busy
# pool and just pends on (Priority). So a smoke (--limit) defaults to a short 02:00:00,
# full runs to 1 day. An explicit --time always wins.
if [ "$TIME_SET" -eq 0 ]; then
    if [ "$SMOKE" -eq 1 ]; then TIME="02:00:00"; else TIME="1-00:00:00"; fi
fi

# GPU pool → host-mem cap. The SLURM feature name equals the GPU-type token, so
# --constraint=$GPU_TYPE is correct for any class (idiom from submit_planbench.sh).
#   rtx_6000 / rtx_pro_6000 : 48 / 96 GB VRAM — fit the whole roster incl. AWQ heavies.
#   rtx_3090 / rtx_4090     : 24 GB — small models only (Qwen3.5 0.8B/4B); the fast
#                             escape when the 48/96 GB pools are saturated.
case "$GPU_TYPE" in
    rtx_6000)          MEM="48G" ;;
    rtx_pro_6000)      MEM="80G" ;;
    rtx_3090|rtx_4090) MEM="32G" ;;
    *) echo "Error: --gpu-type must be rtx_6000|rtx_pro_6000|rtx_3090|rtx_4090 (got: $GPU_TYPE)" >&2; exit 1 ;;
esac

CELLS=()
for m in "${MODELS[@]}"; do
    vllm_lookup "$m" || exit 1
    for t in "${THINKS[@]}"; do CELLS+=("${m}|${t}"); done
done
CELLS_LIST=$(IFS='^'; echo "${CELLS[*]}"); N=${#CELLS[@]}
echo "Cells ($N) on ${GPU_TYPE}:1 (mem=$MEM, time=$TIME):"; printf '  %s\n' "${CELLS[@]}"

mkdir -p "$PROJ_ROOT/cluster/logs"
SUBMIT=(sbatch --array=0-$((N-1)) --gpus="${GPU_TYPE}:1" --constraint="$GPU_TYPE"
        --mem="$MEM" --time="$TIME"
        --export="ALL,PROJ_ROOT=${PROJ_ROOT},CELLS_LIST=${CELLS_LIST}${EXTRA}"
        "$PROJ_ROOT/cluster/serve_and_eval.sbatch")
[ "$DRY" -eq 1 ] && { printf '%q ' "${SUBMIT[@]}"; echo; exit 0; }
"${SUBMIT[@]}"
