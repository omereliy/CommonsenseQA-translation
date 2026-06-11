#!/bin/bash
### Cross-lingual CommonsenseQA — generative arm on BGU. One array task = one
### (model, think) cell: self-deploy vLLM, then run the config-driven eval over
### all condition variants (en-en / en-x × langs). Adapted from the PDDL sweep's
### run_condition_vllm_rtx.sbatch (tool/condition machinery removed; no tools).
###
#SBATCH --partition=main
#SBATCH --job-name=csqa_xling
#SBATCH --output=cluster/logs/%x-%J.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --constraint=rtx_6000
#SBATCH --mem=48G
#SBATCH --tmp=80G
#SBATCH --time=1-00:00:00

set -eo pipefail

# Repo root on the cluster (no spaces — rsync the project here, e.g. ~/CommonsenseQA-translation)
PROJ_ROOT="${PROJ_ROOT:-$HOME/CommonsenseQA-translation}"
source "$PROJ_ROOT/cluster/defaults.sh"

echo "JOBID=$SLURM_JOBID NODE=$SLURM_JOB_NODELIST  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

[ -z "${CELLS_LIST:-}" ] && { echo "ERROR: CELLS_LIST unset — submit via cluster/submit.sh" >&2; exit 1; }
IFS='^' read -ra _CELLS <<< "$CELLS_LIST"
SELECTED="${_CELLS[${SLURM_ARRAY_TASK_ID:-0}]:-}"
[ -z "$SELECTED" ] && { echo "ERROR: cell index out of range" >&2; exit 1; }
IFS='|' read -r MODEL THINK <<< "$SELECTED"
echo "Cell: MODEL=$MODEL THINK=$THINK"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"; CONCURRENCY="${CONCURRENCY:-8}"
ENV_NAME="${ENV_NAME:-csqa}"; CONFIG="${CONFIG:-configs/default.yaml}"
LIMIT_ARG=(); [ -n "${LIMIT:-}" ] && LIMIT_ARG=(--limit "$LIMIT")

vllm_lookup "$MODEL" || exit 1
REASONING_PARSER_FLAG="$(vllm_reasoning_parser_flag)"
MAX_BATCH_FLAG=""; [ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && MAX_BATCH_FLAG="--max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS"
MODEL_TAG_FS=$(echo "$MODEL" | tr '/:.' '___')

WORK=""
if [ -n "${SLURM_JOB_USER:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    SB="/scratch/${SLURM_JOB_USER}/${SLURM_JOB_ID}"; mkdir -p "$SB/hf-cache" 2>/dev/null && WORK="$SB"
fi
[ -z "$WORK" ] && { WORK="/tmp/csqa-${SLURM_JOB_ID:-$$}"; mkdir -p "$WORK/hf-cache"; }

VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.20.2}"; SIF="$WORK/vllm.sif"
if [ -f "$HOME/vllm.sif" ]; then cp "$HOME/vllm.sif" "$SIF"
else apptainer build --force "$SIF" "$VLLM_IMAGE" 2>&1 | tail -3; cp "$SIF" "$HOME/vllm.sif"; fi

free_port() { local p="$1"; while ss -Hltn "sport = :$p" 2>/dev/null | grep -q LISTEN; do p=$((p+1)); done; echo "$p"; }
PORT="$(free_port $(( 11500 + (${SLURM_JOBID:-0} % 100) )))"
SERVE_LOG="$PROJ_ROOT/cluster/logs/${SLURM_JOBID:-local}-vllm-${MODEL_TAG_FS}.log"

echo "Serving $HF_MODEL on :$PORT (reasoning: ${REASONING_PARSER_FLAG:-none})"
apptainer exec --nv --bind "$WORK/hf-cache:/root/.cache/huggingface" --writable-tmpfs "$SIF" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$HF_MODEL" --host 0.0.0.0 --port "$PORT" --max-model-len 16384 \
        $REASONING_PARSER_FLAG --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --enable-prefix-caching $MAX_BATCH_FLAG > "$SERVE_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true; cd ~; rm -rf "$WORK"' EXIT

ready=0
for i in $(seq 1 600); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "ready ($((i*2))s)"; ready=1; break; }
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM died:"; tail -40 "$SERVE_LOG"; exit 2; }
    sleep 2
done
[ "$ready" -ne 1 ] && { echo "vLLM not ready in 20 min:"; tail -40 "$SERVE_LOG"; exit 3; }

module load anaconda
source activate "$ENV_NAME"
cd "$PROJ_ROOT"

python3 -m scripts.run_eval --config "$CONFIG" --arm generative \
    --model-tag "$MODEL" --model-hf "$HF_MODEL" \
    --base-url "http://localhost:$PORT" --think "$THINK" \
    --concurrency "$CONCURRENCY" "${LIMIT_ARG[@]}"
RC=$?
echo "End $(date) rc=$RC"
exit $RC
