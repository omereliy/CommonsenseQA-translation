"""Stage 4: aggregate cached results → accuracy + flip/McNemar tables (+ CSVs).

Reads results/ outputs only (never re-queries). Writes results/summary.csv and
results/flips.csv for the paper's tables/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from csqa_xlang.analysis import aggregate
from csqa_xlang.config import load_config


def main() -> None:
    # The table headers use → / ← ; force UTF-8 stdout so a latin-1 locale
    # (common under WSL/SLURM) doesn't crash the run after the CSVs are written.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--results-dir")
    args = ap.parse_args()
    cfg = load_config(args.config, seed=False)
    results_root = args.results_dir or cfg.get("paths", {}).get("results", "results")

    summary, flips = aggregate(results_root)

    print("\n== Accuracy per (model, think, condition, lang) ==")
    h = f"{'model':22} {'think':5} {'cond':6} {'lang':4} {'n':>5} {'acc':>6} {'95% CI':>15} {'unparsed':>8}"
    print(h); print("-" * len(h))
    for r in summary:
        print(f"{r['model']:22} {r['think']:5} {r['condition']:6} {r['lang']:4} "
              f"{r['n']:5d} {r['accuracy']:6.3f} [{r['ci_low']:.3f},{r['ci_high']:.3f}] {r['unparsed']:8d}")

    if flips:
        print("\n== Flips: en-en → en-x (core diagnostic) ==")
        h2 = f"{'model':22} {'think':5} {'lang':4} {'n':>5} {'flip':>6} {'→gold':>6} {'←gold':>6} {'acc_en':>7} {'acc_x':>7} {'McNemar p':>10}"
        print(h2); print("-" * len(h2))
        for r in flips:
            print(f"{r['model']:22} {r['think']:5} {r['lang']:4} {r['n']:5d} "
                  f"{r['flip_rate']:6.3f} {r['toward_gold']:6d} {r['away_gold']:6d} "
                  f"{r['acc_en']:7.3f} {r['acc_x']:7.3f} {r['mcnemar_p']:10.5f}")
    print(f"\nwrote {Path(results_root) / 'summary.csv'} and flips.csv")


if __name__ == "__main__":
    main()
