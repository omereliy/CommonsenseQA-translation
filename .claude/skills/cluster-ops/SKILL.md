---
name: cluster-ops
description: Operate the BGU CIS SLURM cluster for the CommonsenseQA answer-translation generative sweep — preflight GPU pools, submit the vLLM-served Qwen lineup with the rtx_6000→rtx_pro_6000 escape-hatch strategy, monitor queue + pending reason, and sync results back. Operations only; metrics/flip-rate/significance live in src/csqa_xlang/analysis + scripts/analyze.py.
argument-hint: [preflight | submit | status | sync]
---

> User asked for: $ARGUMENTS — pick the matching recipe below.

## Why this skill exists

Triggers (so the skill auto-matches): "cluster status", "what's running", "why is it
pending", "submit the sweep", "run the generative arm on the cluster", "use the gpu
partition", "rtx_6000 is full", "use the pro GPUs", "sync results", "cancel jobs".

The generative arm self-deploys a vLLM OpenAI server on one GPU per (model, think) cell,
then runs `scripts.run_eval` against `localhost:<port>` over the prebuilt condition
variants. The cluster state is persistent but the working set isn't — this skill pins
the conventions and the **GPU-pool submission strategy** in one place. Read it before
running SSH/rsync/sbatch ad-hoc.

This serving layer is reused from the sibling `pddl-copilot-experiments` harness; the
`cluster/` scripts here are the trimmed (no-tools) adaptation. When in doubt about a
serve flag or a model's parser, the sibling's `cluster-experimenting/lib/defaults.sh`
is the upstream source of truth.

## Cluster & repo conventions

- **Login node**: `omereliy@slurm.bgu.ac.il` — SSH is pre-authed (alias `slurm` in
  `~/.ssh/config`). All recipes below run `ssh slurm '<remote cmd>'`.
- **Remote repo root**: `~/CommonsenseQA-translation` (rsync the project here; no
  spaces in the path so SLURM is happy).
- **Conda env**: `csqa` (lean: `openai` + `pyyaml` + `pip install -e .`). Created via
  the steps in `cluster/`’s sibling `setup_env.sh` pattern; the generative arm is
  **torch-free** (encoder arms `xlmr`/`esim` import torch lazily and run separately).
- **Submit path**: `cluster/submit.sh` is the only submit entry. One array task = one
  `(model, think)` cell; each serves the model once and evaluates all variants.
- **Model roster**: `CSQA_MODELS` in `cluster/defaults.sh` (Qwen3.5 0.8/4/9B,
  qwen3.6:35b, gemma4:26b-a4b) — mirrors the sibling's vLLM-verified lineup.
- **Variants**: `data/variants/{en-en__en,en-x__{ru,es,he}}.json` — prebuilt locally
  and rsync'd; the cluster eval reads them directly (no HF access needed on the node).
- **Logs**: `cluster/logs/csqa_xling-<jobid>.out` (job) and
  `cluster/logs/<jobid>-vllm-<model>.log` (serve).
- **Results**: `results/**/outputs.jsonl` (+ run manifest) per (model, variant);
  `scripts/analyze.py` rglobs `outputs.jsonl` to build `summary.csv` + `flips.csv`.
- **vLLM server**: per-job unique port on the compute node (Apptainer, no TLS),
  exported as the `--base-url` to `run_eval` by `serve_and_eval.sbatch`. The image is
  pinned `vllm/vllm-openai:v0.20.2`, cached at `~/vllm.sif`.

## GPU-pool submission strategy (the "same strategy" as the experiments repo)

Two interchangeable GPU pools carry the same `rtx_6000`/`rtx_pro_6000` SLURM feature
name as their GPU-type token, so `cluster/submit.sh` sets `--gpus`, `--constraint`,
and `--mem` together from one `--gpu-type`:

| Pool           | VRAM  | `--mem` | When to use                                            |
|----------------|-------|---------|--------------------------------------------------------|
| `rtx_6000`     | 48 GB | 48G     | **default** — whole roster fits at gpu-mem-util 0.85   |
| `rtx_pro_6000` | 96 GB | 80G     | **escape hatch** — when rtx_6000 is queue-saturated    |

The CLI flags override the matching `#SBATCH` directives in `serve_and_eval.sbatch`
(CLI wins), so swapping pools needs no sbatch edit.

**Always preflight, then pick the pool with capacity:**

```bash
bash .claude/skills/cluster-ops/scripts/preflight.sh      # reports both pools + a recommendation
```

- If **rtx_6000 has free GPUs** → submit default (no `--gpu-type`).
- If **rtx_6000 is saturated but rtx_pro_6000 has capacity** → add `--gpu-type rtx_pro_6000`.
- If **both are saturated** → see *Walltime & backfill* below before concluding it's
  hopeless; raising priority (`Nice`) needs admin on this cluster.

### Walltime & backfill (the lesson from `submit_planbench.sh`)

A **short `--time` request backfills into idle-GPU gaps**; a 1-day request can't slot in
ahead of a higher-priority reservation and just pends on `Priority` — even when the
scheduler has earmarked a node for it. So when a pool is busy, the lever that actually
gets you scheduled is a *shorter walltime*, not a different pool.

`cluster/submit.sh` encodes this: a **smoke (`--limit`) defaults to `--time 02:00:00`**;
a full run defaults to 1 day. Pass `--time HH:MM:SS` to override (keep it just above the
real runtime so it stays backfillable). The 0.8B smoke finishes in minutes — a 1-day
request was why an earlier submit sat on `Priority` indefinitely.

**Small models on free 24 GB cards.** When rtx_6000/pro are both saturated, the small
Qwen3.5 (0.8B/4B) fit on a free `rtx_3090`/`rtx_4090` (24 GB): `--gpu-type rtx_3090`
(lower `GPU_MEM_UTIL=0.80` via the sbatch env for the larger ones). The AWQ heavies
(qwen3.6:35b, gemma4:26b-a4b) still need rtx_6000. `preflight.sh` reports the rtx_6000
pools; check 3090/4090 directly with `sinfo -O Gres,GresUsed -p main | grep rtx_3090`.

## Recipes

### preflight — is the cluster ready, and which pool?
```bash
bash .claude/skills/cluster-ops/scripts/preflight.sh
```
Checks connectivity, repo + `csqa` env + `vllm.sif` presence, and free-GPU counts for
both pools; prints a recommended `--gpu-type`.

### submit — launch the generative sweep
Smoke first (smallest model, few items) to confirm serve+eval end-to-end:
```bash
ssh slurm 'cd ~/CommonsenseQA-translation && bash cluster/submit.sh Qwen3.5:0.8B --think off --limit 20 --dry-run'   # preview
ssh slurm 'cd ~/CommonsenseQA-translation && bash cluster/submit.sh Qwen3.5:0.8B --think off --limit 20 [--gpu-type rtx_pro_6000]'
```
Full sweep (all models × {off,on}); add `--gpu-type rtx_pro_6000` per the strategy above:
```bash
ssh slurm 'cd ~/CommonsenseQA-translation && bash cluster/submit.sh [--gpu-type rtx_pro_6000]'
```
Re-running is safe: `run_eval` skips any completed (model, variant) cell (cache hit),
so a resubmit resumes rather than recomputes.

If you edited `cluster/*` or rebuilt `data/variants/` locally, re-sync first:
```bash
rsync -az -e ssh --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='data/translated/' --exclude='data/raw/' \
  ./src ./scripts ./configs ./cluster ./requirements.txt ./pyproject.toml ./data \
  slurm:CommonsenseQA-translation/
```

### status — queue + pending reason + progress
```bash
ssh slurm 'squeue -u omereliy -o "%.12i %.18j %.8T %.10M %.18R"
           sacct -X -u omereliy --starttime today --format=JobID,JobName%18,State,Elapsed,ExitCode | tail -20'
```
Pending-REASON cheat-sheet: `Priority`/`Resources` = pool busy (backfill-waiting, our
fault: none); `(None)` = scheduler hasn't placed it yet (usually imminent);
`QOSMax*`/`AssocGrp*` = a per-user/account limit, not contention.

Tail a running job + its serve log:
```bash
ssh slurm 'tail -30 ~/CommonsenseQA-translation/cluster/logs/csqa_xling-<jobid>.out'
```

### sync — pull results back, then analyze locally
```bash
bash .claude/skills/cluster-ops/scripts/sync.sh           # → results/cluster-YYYYMMDD/
PYTHONPATH=src python -m scripts.analyze --results-dir results/cluster-YYYYMMDD   # summary.csv + flips.csv
```

## Safety

- **Destructive ops need explicit user consent**: `scancel` on someone's jobs, `rm` on
  logs/results. `scancel -u omereliy` kills *all* the user's jobs — never run it
  without confirmation.
- **Never mutate** `run_eval.py`, `serve_and_eval.sbatch`, or `submit.sh` from a
  status/sync recipe — those are submission logic, edited deliberately, not ops-time.
- **Preflight before a big submit**: a saturated pool wastes a queue slot and the
  pending job blocks nothing but your own patience.
