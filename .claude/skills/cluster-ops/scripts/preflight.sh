#!/usr/bin/env bash
# Cluster preflight for the CSQA generative sweep: connectivity, repo + csqa env +
# vllm.sif presence, and free-GPU capacity for BOTH pools (rtx_6000 / rtx_pro_6000)
# with a recommended --gpu-type per the skill's submission strategy.
#
# Usage: bash .claude/skills/cluster-ops/scripts/preflight.sh
# Env:   REMOTE_HOST (default: slurm)
set -euo pipefail
REMOTE_HOST="${REMOTE_HOST:-slurm}"

ssh -o BatchMode=yes "$REMOTE_HOST" bash -s <<'REMOTE'
set -e
echo "== host =="
echo "  $(whoami)@$(hostname)"

echo "== repo / env / image =="
[ -d ~/CommonsenseQA-translation ] && echo "  repo:     present (~/CommonsenseQA-translation)" || echo "  repo:     MISSING — rsync the project first"
[ -f ~/CommonsenseQA-translation/data/variants/en-en__en.json ] && echo "  variants: present" || echo "  variants: MISSING — build + rsync data/variants/"
module load anaconda 2>/dev/null || true
if conda env list 2>/dev/null | awk '{print $1}' | grep -qx csqa; then echo "  env:      csqa present"; else echo "  env:      csqa MISSING — create it"; fi
[ -f ~/vllm.sif ] && echo "  vllm.sif: cached" || echo "  vllm.sif: absent (first job builds + caches it, ~5 min)"

echo "== GPU pool capacity (partition main) =="
python3 - <<'PY'
import subprocess, re
# Fixed-width -O columns so we can slice each field unambiguously (one line per node).
widths = [("NodeHost", 22), ("Gres", 70), ("GresUsed", 70), ("StateLong", 18)]
spec = ",".join(f"{n}:{w}" for n, w in widths)
out = subprocess.run(["sinfo", "-h", "-N", "-O", spec, "-p", "main"],
                     capture_output=True, text=True).stdout
offs, acc = [], 0
for _, w in widths:
    offs.append((acc, acc + w)); acc += w
pools = ("rtx_6000", "rtx_pro_6000")
free = {p: 0 for p in pools}; total = {p: 0 for p in pools}
for line in out.splitlines():
    gres = line[offs[1][0]:offs[1][1]]
    used = line[offs[2][0]:offs[2][1]]
    state = line[offs[3][0]:offs[3][1]].strip().lower()
    sched_out = any(s in state for s in ("drain", "down", "resv", "maint", "boot"))
    for p in pools:
        mt = re.search(rf"gpu:{p}:(\d+)", gres)
        if not mt:
            continue
        t = int(mt.group(1))
        mu = re.search(rf"gpu:{p}:(\d+)", used)
        u = int(mu.group(1)) if mu else 0
        total[p] += t
        free[p] += 0 if sched_out else max(0, t - u)
for p in pools:
    print(f"  {p:14} free {free[p]:3d} / {total[p]:3d} GPUs")
# Recommendation per the strategy: default rtx_6000; escape to pro when 6000 is dry.
if free["rtx_6000"] > 0:
    print("  -> recommend: default (rtx_6000 has capacity)")
elif free["rtx_pro_6000"] > 0:
    print("  -> recommend: --gpu-type rtx_pro_6000  (rtx_6000 saturated, pro has capacity)")
else:
    print("  -> both pools saturated: submit anyway and backfill-wait on Priority, or retry later")
PY
REMOTE
