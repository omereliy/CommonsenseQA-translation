"""Re-parse check: re-derive predictions from cached raw text with the CURRENT
parser and report whether they match the predictions baked into outputs.jsonl.

Why this exists
---------------
`extract_letter` (eval/prompt.py) used to have a last-resort fallback that
string-matched a choice's *text* in the model output. That violated load-bearing
convention #2 (scoring must never depend on answer wording) and was removed. But
`outputs.jsonl` stores the `pred`/`correct` computed by whatever parser ran at
eval time — `analyze.py` only aggregates those cached fields, it never re-parses.
So the committed summary.csv / flips.csv still embed the OLD parser's decisions.

This script re-runs the current parser over the cached `raw` text (no model, no
GPU) and surfaces exactly the items the fallback used to rescue: rows where the
stored prediction is a letter but the current parser yields None. It is read-only
over the dataset and writes its report to results/reparse_check/ — it does NOT
overwrite outputs.jsonl, summary.csv, or flips.csv.

Direction of any change is bounded: the current parser is a strict subset of the
old one (old = letter-tiers + fallback; new = letter-tiers only), so re-parsing
can only turn predictions into None (scored wrong). Accuracy can only drop and
unparsed can only rise — never the reverse. A `rescued_removed` count of 0 for a
cell means its published numbers are already fallback-free and stand as-is.

Usage
-----
    python -m scripts.reparse_check [--results-dir results]

Run it wherever the raw outputs.jsonl live (locally only think=off cells are
present; sync the cluster outputs down — or run this on the cluster — to cover
think=on, which is where the fallback could actually have fired).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from csqa_xlang.analysis.metrics import accuracy, flip_rate, mcnemar
from csqa_xlang.eval.base import score
from csqa_xlang.eval.prompt import extract_letter


def _key(r: dict) -> tuple:
    return (r["model"], r.get("think", "na"), r["condition"], r["lang"])


def load_and_reparse(results_root: Path) -> list[dict]:
    """Load every cached row and attach the CURRENT parser's verdict from `raw`."""
    rows: list[dict] = []
    for p in sorted(results_root.rglob("outputs.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                # The new parser ignores `choices`; raw is the cached model text.
                new_pred = extract_letter(r.get("raw", ""))
                r["new_pred"] = new_pred
                r["new_correct"] = score(new_pred, r.get("gold"))
                rows.append(r)
    return rows


def per_cell(rows: list[dict]) -> list[dict]:
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[_key(r)].append(r)
    out = []
    for (model, think, cond, lang), rs in sorted(cells.items()):
        stored_unparsed = sum(r["pred"] is None for r in rs)
        new_unparsed = sum(r["new_pred"] is None for r in rs)
        # The fallback footprint: stored a letter, current parser yields None.
        rescued = sum(r["pred"] is not None and r["new_pred"] is None for r in rs)
        # Anything else the re-parse disagrees on (should be 0; flag if not).
        other_drift = sum(
            r["pred"] != r["new_pred"]
            and not (r["pred"] is not None and r["new_pred"] is None)
            for r in rs
        )
        stored_acc = sum(r["correct"] for r in rs) / len(rs)
        new_acc = sum(r["new_correct"] for r in rs) / len(rs)
        out.append({
            "model": model, "think": think, "condition": cond, "lang": lang,
            "n": len(rs),
            "stored_acc": round(stored_acc, 4), "new_acc": round(new_acc, 4),
            "acc_delta": round(new_acc - stored_acc, 4),
            "stored_unparsed": stored_unparsed, "new_unparsed": new_unparsed,
            "rescued_removed": rescued, "other_drift": other_drift,
        })
    return out


def reparsed_flip_table(rows: list[dict]) -> list[dict]:
    """Recompute the en-en -> en-x flip table using the CURRENT parser's preds."""
    grp: dict[tuple, dict[tuple, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grp[(r["model"], r.get("think", "na"))][(r["condition"], r["lang"])].append(r)
    out = []
    for (model, think), conds in sorted(grp.items()):
        base = conds.get(("en-en", "en"))
        if not base:
            continue
        # Use the re-parsed prediction/correctness for the metric inputs.
        base_np = [{"id": r["id"], "pred": r["new_pred"], "correct": r["new_correct"]} for r in base]
        for (cond, lang), rs in sorted(conds.items()):
            if cond != "en-x":
                continue
            other_np = [{"id": r["id"], "pred": r["new_pred"], "correct": r["new_correct"]} for r in rs]
            fr = flip_rate(base_np, other_np)
            mc = mcnemar(base_np, other_np)
            out.append({
                "model": model, "think": think, "lang": lang, "n": fr["n"],
                "flip_rate": round(fr["flip_rate"], 4),
                "toward_gold": fr["toward_gold"], "away_gold": fr["away_gold"],
                "acc_en": round(accuracy(base_np), 4), "acc_x": round(accuracy(other_np), 4),
                "mcnemar_p": round(mc["p_value"], 5),
            })
    return out


def _read_csv(path: Path) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {(r["model"], r["think"], r.get("condition", "en-x"), r["lang"]): r
                for r in csv.DictReader(f)}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    root = Path(args.results_dir)
    rows = load_and_reparse(root)
    if not rows:
        raise SystemExit(f"no outputs.jsonl under {root} — nothing to re-parse")

    cells = per_cell(rows)
    flips = reparsed_flip_table(rows)

    out_dir = root / "reparse_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "reparse_by_cell.csv", cells)
    _write_csv(out_dir / "reparse_flips.csv", flips)

    # Cross-check against the committed CSVs (read-only).
    committed_summary = _read_csv(root / "summary.csv")

    total_rescued = sum(c["rescued_removed"] for c in cells)
    total_drift = sum(c["other_drift"] for c in cells)
    think_modes = sorted({c["think"] for c in cells})

    print("\n== Re-parse check (current parser vs cached predictions) ==")
    print(f"cells found: {len(cells)}   think modes present: {think_modes}")
    print(f"TOTAL fallback-rescued predictions removed: {total_rescued}")
    print(f"TOTAL other parse drift (expected 0):       {total_drift}\n")

    h = (f"{'model':14} {'think':5} {'cond':6} {'lang':4} {'n':>5} "
         f"{'stored_acc':>10} {'new_acc':>8} {'Δacc':>7} {'rescued':>8} {'drift':>6}")
    print(h)
    print("-" * len(h))
    for c in cells:
        flag = "  <-- CHANGES" if c["rescued_removed"] or c["other_drift"] else ""
        cm = committed_summary.get((c["model"], c["think"], c["condition"], c["lang"]))
        csv_note = ""
        if cm and f'{float(cm["accuracy"]):.4f}' != f'{c["stored_acc"]:.4f}':
            csv_note = f"  (committed summary.csv acc={cm['accuracy']})"
        print(f"{c['model']:14} {c['think']:5} {c['condition']:6} {c['lang']:4} "
              f"{c['n']:5d} {c['stored_acc']:10.4f} {c['new_acc']:8.4f} "
              f"{c['acc_delta']:+7.4f} {c['rescued_removed']:8d} {c['other_drift']:6d}"
              f"{flag}{csv_note}")

    if "on" not in think_modes:
        print("\nNOTE: no think=on cells present here. think=off has 0 unparsed, so the "
              "fallback could never fire — this run cannot exercise the case it targets.\n"
              "Sync the cluster's think=on outputs.jsonl (or run this on the cluster) to "
              "check where it could actually have mattered.")

    print(f"\nwrote {out_dir / 'reparse_by_cell.csv'} and {out_dir / 'reparse_flips.csv'} "
          "(no existing files modified)")


if __name__ == "__main__":
    main()
