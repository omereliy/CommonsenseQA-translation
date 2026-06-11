"""Read cached results/ outputs and emit the paper's numbers.

Produces two tidy CSVs under results/:
  summary.csv — accuracy + Wilson CI per (model, think, condition, lang)
  flips.csv   — flip rate + direction + McNemar p, en-en baseline → en-x, per
                (model, think, target lang)
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from csqa_xlang.analysis.metrics import accuracy, flip_rate, mcnemar, wilson


def load_records(results_root: str | Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(Path(results_root).rglob("outputs.jsonl")):
        with p.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def _key(r):  # one evaluated cell
    return (r["model"], r.get("think", "na"), r["condition"], r["lang"])


def summarize(rows: list[dict]) -> list[dict]:
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[_key(r)].append(r)
    out = []
    for (model, think, cond, lang), rs in sorted(cells.items()):
        k = sum(x["correct"] for x in rs)
        lo, hi = wilson(k, len(rs))
        out.append({"model": model, "think": think, "condition": cond, "lang": lang,
                    "n": len(rs), "accuracy": round(accuracy(rs), 4),
                    "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                    "unparsed": sum(x["pred"] is None for x in rs)})
    return out


def flip_table(rows: list[dict]) -> list[dict]:
    # baseline = en-en (per model, think); compared against each en-x lang.
    grp: dict[tuple, dict[tuple, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grp[(r["model"], r.get("think", "na"))][(r["condition"], r["lang"])].append(r)
    out = []
    for (model, think), conds in sorted(grp.items()):
        base = conds.get(("en-en", "en"))
        if not base:
            continue
        for (cond, lang), rs in sorted(conds.items()):
            if cond != "en-x":
                continue
            fr = flip_rate(base, rs)
            mc = mcnemar(base, rs)
            out.append({"model": model, "think": think, "lang": lang,
                        "n": fr["n"], "flip_rate": round(fr["flip_rate"], 4),
                        "toward_gold": fr["toward_gold"], "away_gold": fr["away_gold"],
                        "acc_en": round(accuracy(base), 4), "acc_x": round(accuracy(rs), 4),
                        "mcnemar_p": round(mc["p_value"], 5)})
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def aggregate(results_root: str | Path) -> tuple[list[dict], list[dict]]:
    rows = load_records(results_root)
    if not rows:
        raise SystemExit(f"no results under {results_root} — run an eval arm first")
    summary, flips = summarize(rows), flip_table(rows)
    root = Path(results_root)
    _write_csv(root / "summary.csv", summary)
    _write_csv(root / "flips.csv", flips)
    return summary, flips
