"""Analyze the MIXED-language answer runs.

Reads cached outputs only (never re-queries). For each encoder model it reports:
  * accuracy on the mixed variant (+ Wilson 95% CI), vs the en-en baseline and
    the three single-language en-x runs (ru/he/es, Google backend);
  * flip rate en-en -> mixed (the core diagnostic), direction split, McNemar p;
  * a breakdown of mixed accuracy by the language the GOLD option was drawn in.

Baselines/en-x come from the canonical results/ tree (xlmr-ep6 / mbert-ep6); the
mixed runs come from this task's results/ dir. Writes mixed_summary.csv,
mixed_flips.csv, mixed_by_gold_lang.csv next to this file.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from csqa_xlang.analysis.metrics import accuracy, flip_rate, mcnemar, wilson

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TAGS = ["xlmr-ep6", "mbert-ep6"]
LANGS = ["ru", "he", "es"]


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _records(root: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(root.rglob("outputs.jsonl")):
        rows.extend(_read(p))
    return rows


def _cell(rows, model, condition, lang):
    return [r for r in rows if r["model"] == model
            and r["condition"] == condition and r["lang"] == lang]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    baseline_rows = _records(REPO / "results")          # en-en + en-x ep6 runs
    mixed_rows = _records(HERE / "results")             # this task's runs
    assignment = json.loads(
        (HERE / "data" / "mixed_assignment.json").read_text(encoding="utf-8"))

    summary, flips, by_gold = [], [], []
    for tag in TAGS:
        en = _cell(baseline_rows, tag, "en-en", "en")
        mix = _cell(mixed_rows, tag, "mixed", "mix")
        if not en or not mix:
            print(f"!! missing data for {tag} (en-en={len(en)}, mixed={len(mix)}) — "
                  f"run build_mixed.py then run_mixed.py first")
            continue

        # accuracy rows: en-en, each en-x, mixed
        for cond, lang, rows in (
            [("en-en", "en", en)]
            + [("en-x", lg, _cell(baseline_rows, tag, "en-x", lg)) for lg in LANGS]
            + [("mixed", "mix", mix)]
        ):
            if not rows:
                continue
            k = sum(r["correct"] for r in rows)
            lo, hi = wilson(k, len(rows))
            summary.append({"model": tag, "condition": cond, "lang": lang,
                            "n": len(rows), "correct": k,
                            "accuracy": round(accuracy(rows), 4),
                            "ci_low": round(lo, 4), "ci_high": round(hi, 4)})

        # flips en-en -> {each en-x, mixed}
        for cond, lang, rows in (
            [("en-x", lg, _cell(baseline_rows, tag, "en-x", lg)) for lg in LANGS]
            + [("mixed", "mix", mix)]
        ):
            if not rows:
                continue
            fr = flip_rate(en, rows); mc = mcnemar(en, rows)
            flips.append({"model": tag, "vs": f"{cond}/{lang}", "n": fr["n"],
                          "flips": fr["flips"],
                          "flip_rate": round(fr["flip_rate"], 4),
                          "toward_gold": fr["toward_gold"], "away_gold": fr["away_gold"],
                          "acc_en": round(accuracy(en), 4), "acc_x": round(accuracy(rows), 4),
                          "mcnemar_p": round(mc["p_value"], 5)})

        # mixed accuracy split by the language the GOLD option was drawn in
        for lg in LANGS:
            sub = [r for r in mix if assignment[r["id"]][r["gold"]] == lg]
            if not sub:
                continue
            k = sum(r["correct"] for r in sub); lo, hi = wilson(k, len(sub))
            by_gold.append({"model": tag, "gold_lang": lg, "n": len(sub),
                            "correct": k, "accuracy": round(accuracy(sub), 4),
                            "ci_low": round(lo, 4), "ci_high": round(hi, 4)})

    _write_csv(HERE / "mixed_summary.csv", summary)
    _write_csv(HERE / "mixed_flips.csv", flips)
    _write_csv(HERE / "mixed_by_gold_lang.csv", by_gold)

    print("\n== Accuracy ==")
    h = f"{'model':10} {'cond':6} {'lang':5} {'n':>5} {'acc':>6} {'95% CI':>16}"
    print(h); print("-" * len(h))
    for r in summary:
        print(f"{r['model']:10} {r['condition']:6} {r['lang']:5} {r['n']:5d} "
              f"{r['accuracy']:6.3f} [{r['ci_low']:.3f},{r['ci_high']:.3f}]")

    print("\n== Flips: en-en -> condition (core diagnostic) ==")
    h2 = (f"{'model':10} {'vs':10} {'n':>5} {'flip':>6} {'->gold':>7} {'<-gold':>7} "
          f"{'acc_en':>7} {'acc_x':>7} {'McNemar p':>10}")
    print(h2); print("-" * len(h2))
    for r in flips:
        print(f"{r['model']:10} {r['vs']:10} {r['n']:5d} {r['flip_rate']:6.3f} "
              f"{r['toward_gold']:7d} {r['away_gold']:7d} {r['acc_en']:7.3f} "
              f"{r['acc_x']:7.3f} {r['mcnemar_p']:10.5f}")

    print("\n== Mixed accuracy by GOLD-option language ==")
    h3 = f"{'model':10} {'gold_lang':9} {'n':>5} {'acc':>6} {'95% CI':>16}"
    print(h3); print("-" * len(h3))
    for r in by_gold:
        print(f"{r['model']:10} {r['gold_lang']:9} {r['n']:5d} {r['accuracy']:6.3f} "
              f"[{r['ci_low']:.3f},{r['ci_high']:.3f}]")

    print(f"\nwrote mixed_summary.csv, mixed_flips.csv, mixed_by_gold_lang.csv -> {HERE}")


if __name__ == "__main__":
    main()
