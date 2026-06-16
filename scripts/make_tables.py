"""Stage 5b: emit Markdown (+ optional LaTeX booktabs) tables from the CSVs.

Reads results/summary.csv + results/flips.csv and writes to results/tables/:
  accuracy.md            per-(model, think) accuracy across conditions, Wilson CIs
  flips.md               HEADLINE: flip rate en-en→en-x per lang, McNemar p
  flips.tex / accuracy.tex   booktabs versions for the paper appendix (--latex)

Robust to partial/missing data (a missing CSV → that table is skipped). Use
--demo to render against a synthetic fixture.

  PYTHONPATH=src .venv/bin/python -m scripts.make_tables
  PYTHONPATH=src .venv/bin/python -m scripts.make_tables --demo --latex
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from scripts._analysis_io import (DEMO_BANNER, LANG_ORDER, load_csvs,
                                   make_demo_csvs, order_models)


def _stars(p):
    if p is None or p != p:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# Cells where more than this fraction of generations were unparseable (no A–E
# letter emitted) are flagged: the accuracy reflects an output-FORMAT failure,
# not reasoning, and must be excluded from model comparisons. Threshold 0.5
# isolates the Qwen3.5:0.8B think-on row (85–95% unparsed) from every other
# cell (next-highest is ~23%).
UNPARSE_FLAG = 0.5


def _unreliable(r) -> bool:
    n = r["n"] if r["n"] else 0
    return bool(n) and (r["unparsed"] / n) > UNPARSE_FLAG


# --------------------------------------------------------------------------- #
def accuracy_md(summary) -> str:
    """One row per (model, think): en-en accuracy then each en-x lang, with CIs."""
    if summary.empty:
        return ""
    langs = [l for l in LANG_ORDER if l in set(summary["lang"])]
    head = ["model", "think", "en-en"] + [f"en-{l}" for l in langs]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for model in order_models(summary["model"]):
        for think in sorted(set(summary[summary["model"] == model]["think"])):
            cells = [model, think]
            cells.append(_acc_cell(summary, model, think, "en-en", "en"))
            for lang in langs:
                cells.append(_acc_cell(summary, model, think, "en-x", lang))
            lines.append("| " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"
    if "†" in md:
        md += ("\n> **†** Over 50% of generations were unparseable (the model never "
               "emitted an A–E letter). These accuracies reflect an output-**format** "
               "failure, not commonsense reasoning, and should be **excluded** from "
               "model comparisons. Affects only `Qwen3.5:0.8B` with `think=on`.\n")
    return md


def _acc_cell(summary, model, think, cond, lang) -> str:
    row = summary[(summary["model"] == model) & (summary["think"] == think)
                  & (summary["condition"] == cond) & (summary["lang"] == lang)]
    if not len(row):
        return "—"
    r = row.iloc[0]
    flag = " †" if _unreliable(r) else ""
    return f"{r['accuracy']:.3f} [{r['ci_low']:.3f},{r['ci_high']:.3f}]{flag}"


def flips_md(flips) -> str:
    """The core diagnostic table."""
    if flips.empty:
        return ""
    head = ["model", "think", "lang", "n", "flip_rate", "→gold", "←gold",
            "acc_en", "acc_x", "McNemar p"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    fl = flips.copy()
    fl = fl.sort_values(by=["model", "think", "lang"],
                        key=lambda s: s.map(lambda v: order_models([v])[0])
                        if s.name == "model" else s)
    for r in fl.itertuples():
        p = f"{r.mcnemar_p:.4f}{_stars(r.mcnemar_p)}"
        lines.append("| " + " | ".join(str(x) for x in [
            r.model, r.think, f"en-{r.lang}", r.n, f"{r.flip_rate:.3f}",
            r.toward_gold, r.away_gold, f"{r.acc_en:.3f}", f"{r.acc_x:.3f}", p,
        ]) + " |")
    return "\n".join(lines) + "\n"


# --- LaTeX (booktabs) ------------------------------------------------------- #
def flips_tex(flips) -> str:
    if flips.empty:
        return ""
    fl = flips.sort_values(by=["model", "think", "lang"],
                           key=lambda s: s.map(lambda v: order_models([v])[0])
                           if s.name == "model" else s)
    body = []
    for r in fl.itertuples():
        body.append(
            f"{_tex(r.model)} & {r.think} & en-{r.lang} & {r.n} & "
            f"{r.flip_rate:.3f} & {r.toward_gold} & {r.away_gold} & "
            f"{r.acc_en:.3f} & {r.acc_x:.3f} & {r.mcnemar_p:.4f}{_stars(r.mcnemar_p)} \\\\")
    return ("\\begin{table}[t]\n\\centering\n\\small\n"
            "\\begin{tabular}{lllrrrrrrr}\n\\toprule\n"
            "Model & Think & Lang & $n$ & Flip & $\\rightarrow$g & $\\leftarrow$g & "
            "Acc$_{en}$ & Acc$_{x}$ & McNemar $p$ \\\\\n\\midrule\n"
            + "\n".join(body)
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Flip rate (en-en $\\to$ en-x) per model and answer language. "
            "$\\rightarrow$g/$\\leftarrow$g = flips toward/away from gold. "
            "* $p<.05$, ** $p<.01$, *** $p<.001$ (McNemar).}\n"
            "\\label{tab:flips}\n\\end{table}\n")


def accuracy_tex(summary) -> str:
    if summary.empty:
        return ""
    langs = [l for l in LANG_ORDER if l in set(summary["lang"])]
    col_spec = "ll" + "r" * (1 + len(langs))
    header = "Model & Think & en-en & " + " & ".join(f"en-{l}" for l in langs) + " \\\\"
    body = []
    for model in order_models(summary["model"]):
        for think in sorted(set(summary[summary["model"] == model]["think"])):
            cells = [_acc_tex(summary, model, think, "en-en", "en")]
            cells += [_acc_tex(summary, model, think, "en-x", l) for l in langs]
            body.append(f"{_tex(model)} & {think} & " + " & ".join(cells) + " \\\\")
    return ("\\begin{table}[t]\n\\centering\n\\small\n"
            f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n{header}\n\\midrule\n"
            + "\n".join(body)
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Accuracy per condition (Wilson 95\\% CI). en-en is the "
            "English baseline; en-x keeps the English question and translates the "
            "answer choices into language $x$. $^\\dagger$: $>$50\\% of generations "
            "were unparseable (no A--E letter); accuracy reflects an output-format "
            "failure, not reasoning, and is excluded from comparisons.}"
            "\n\\label{tab:accuracy}\n\\end{table}\n")


def _acc_tex(summary, model, think, cond, lang) -> str:
    row = summary[(summary["model"] == model) & (summary["think"] == think)
                  & (summary["condition"] == cond) & (summary["lang"] == lang)]
    if not len(row):
        return "--"
    r = row.iloc[0]
    flag = "$^\\dagger$" if _unreliable(r) else ""
    return f"{r['accuracy']:.3f}{flag}"


def _tex(s: str) -> str:
    return str(s).replace("_", "\\_").replace("&", "\\&")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", help="table dir (default: <results-dir>/tables)")
    ap.add_argument("--latex", action="store_true", help="also write .tex (booktabs)")
    ap.add_argument("--demo", action="store_true", help="synthetic fixture")
    args = ap.parse_args()

    if args.demo:
        src = Path(tempfile.mkdtemp(prefix="csqa_demo_"))
        make_demo_csvs(src)
        out_dir = Path(args.out or "results/tables_demo")
        print(f"[demo] {DEMO_BANNER}")
    else:
        src = Path(args.results_dir)
        out_dir = Path(args.out or src / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, flips = load_csvs(src)
    if summary.empty and flips.empty:
        raise SystemExit(f"no summary.csv/flips.csv under {src} — run "
                         f"`python -m scripts.analyze` first, or pass --demo")

    note = f"<!-- {DEMO_BANNER} -->\n\n" if args.demo else ""
    written = []

    for name, content in [("accuracy.md", accuracy_md(summary)),
                          ("flips.md", flips_md(flips))]:
        if content:
            title = "Accuracy by condition" if "accuracy" in name else \
                "Flip rate (en-en → en-x) — core diagnostic"
            (out_dir / name).write_text(f"{note}## {title}\n\n{content}")
            written.append(out_dir / name); print(f"  wrote {out_dir / name}")

    if args.latex:
        for name, content in [("accuracy.tex", accuracy_tex(summary)),
                              ("flips.tex", flips_tex(flips))]:
            if content:
                pre = f"% {DEMO_BANNER}\n" if args.demo else ""
                (out_dir / name).write_text(pre + content)
                written.append(out_dir / name); print(f"  wrote {out_dir / name}")

    print(f"\n{len(written)} table file(s) → {out_dir}")


if __name__ == "__main__":
    main()
