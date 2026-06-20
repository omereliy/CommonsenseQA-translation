"""Figures for the mixed-language answer experiment.

Reads the three CSVs written by analyze_mixed.py (mixed_summary.csv,
mixed_flips.csv, mixed_by_gold_lang.csv) and emits PNG figures into
report/figures/ (fig16-fig19, following the report's sequential convention).
Every percentage drawn on a bar is annotated with its raw count (correct/n, or
flipped/n). matplotlib only; reuses the project house style.

  python tasks/mixed_answers/plot_mixed.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "report" / "figures"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
    "axes.axisbelow": True, "figure.autolayout": True,
})
# house colours + a dedicated colour for the mixed condition
COND_COLOR = {"en-en": "#444444", "en-x/ru": "#4C72B0", "en-x/es": "#DD8452",
              "en-x/he": "#55A868", "mixed/mix": "#8172B3"}
LANG_COLOR = {"en": "#444444", "ru": "#4C72B0", "es": "#DD8452",
              "he": "#55A868", "mix": "#8172B3"}
LANG_NAME = {"en": "English", "ru": "Russian", "es": "Spanish",
             "he": "Hebrew", "mix": "mixed"}
MODELS = ["xlmr-ep6", "mbert-ep6"]
MODEL_NAME = {"xlmr-ep6": "XLM-R", "mbert-ep6": "mBERT"}


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}.png")


def _annotate(ax, x, top, pct, count_txt, ytop):
    """percentage + raw count above a bar."""
    ax.text(x, top + 0.012 * ytop, f"{pct*100:.1f}%\n{count_txt}",
            ha="center", va="bottom", fontsize=7.6, linespacing=1.15)


# --------------------------------------------------------------------------- #
def fig_accuracy(summary):
    """Grouped bars: per model, accuracy for en-en, en-x ru/he/es, mixed."""
    order = [("en-en", "en"), ("en-x", "ru"), ("en-x", "he"),
             ("en-x", "es"), ("mixed", "mix")]
    labels = ["en-en", "en-x ru", "en-x he", "en-x es", "MIXED"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.8 / len(order)
    ymax = 0.62
    for j, ((cond, lang), lab) in enumerate(zip(order, labels)):
        xs, accs, los, his = [], [], [], []
        for i, m in enumerate(MODELS):
            row = summary[(summary.model == m) & (summary.condition == cond)
                          & (summary.lang == lang)]
            if not len(row):
                continue
            r = row.iloc[0]
            x = i + (j - (len(order) - 1) / 2) * w
            xs.append(x); accs.append(r.accuracy)
            los.append(r.accuracy - r.ci_low); his.append(r.ci_high - r.accuracy)
            _annotate(ax, x, r.ci_high, r.accuracy, f"{int(r.correct)}/{int(r.n)}", ymax)
        ax.bar(xs, accs, w, yerr=[los, his], capsize=2,
               color=LANG_COLOR.get(lang), label=lab,
               error_kw={"elinewidth": 0.9, "alpha": 0.7})
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_NAME[m] for m in MODELS])
    ax.set_ylabel("accuracy"); ax.set_ylim(0, ymax)
    ax.set_title("Accuracy by condition — English question, answers translated "
                 "(n=1221)\nMIXED = each option independently ru/he/es (⅓ each)",
                 fontsize=11)
    ax.legend(ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.06))
    _save(fig, "fig16_mixed_accuracy_by_condition")


def fig_flip_rate(flips):
    """Grouped bars: flip rate en-en -> each condition, per model."""
    order = ["en-x/ru", "en-x/he", "en-x/es", "mixed/mix"]
    labels = ["en-x ru", "en-x he", "en-x es", "MIXED"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    w = 0.8 / len(order)
    ymax = 0.72
    for j, (vs, lab) in enumerate(zip(order, labels)):
        xs, vals = [], []
        for i, m in enumerate(MODELS):
            row = flips[(flips.model == m) & (flips["vs"] == vs)]
            if not len(row):
                continue
            r = row.iloc[0]
            x = i + (j - (len(order) - 1) / 2) * w
            xs.append(x); vals.append(r.flip_rate)
            _annotate(ax, x, r.flip_rate, r.flip_rate, f"{int(r.flips)}/{int(r.n)}", ymax)
        ax.bar(xs, vals, w, color=COND_COLOR.get(vs), label=lab)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_NAME[m] for m in MODELS])
    ax.set_ylabel("flip rate (pred changes vs en-en)"); ax.set_ylim(0, ymax)
    ax.set_title("Prediction flip rate: en-en → condition (n=1221)\n"
                 "fraction of items whose A–E choice changes when answers are "
                 "translated", fontsize=11)
    ax.legend(ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.06))
    _save(fig, "fig17_mixed_flip_rate")


def fig_flip_direction(flips):
    """Stacked toward/away-from-gold flip counts, per (model, condition)."""
    order = ["en-x/ru", "en-x/he", "en-x/es", "mixed/mix"]
    labels = ["ru", "he", "es", "MIX"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, m in zip(axes, MODELS):
        sub = flips[flips.model == m]
        xs = range(len(order))
        toward = [int(sub[sub["vs"] == v].iloc[0].toward_gold) for v in order]
        away = [int(sub[sub["vs"] == v].iloc[0].away_gold) for v in order]
        n = int(sub.iloc[0].n)
        b1 = ax.bar(xs, toward, 0.6, color="#55A868", label="→ gold (became correct)")
        b2 = ax.bar(xs, away, 0.6, bottom=toward, color="#C44E52",
                    label="← gold (became wrong)")
        for x, t, a in zip(xs, toward, away):
            ax.text(x, t / 2, f"{t}\n{t/n*100:.1f}%", ha="center", va="center",
                    fontsize=7.5, color="white")
            ax.text(x, t + a / 2, f"{a}\n{a/n*100:.1f}%", ha="center", va="center",
                    fontsize=7.5, color="white")
        ax.set_xticks(list(xs)); ax.set_xticklabels(labels)
        ax.set_title(MODEL_NAME[m]); ax.set_xlabel("condition (vs en-en)")
    axes[0].set_ylabel("flips by direction (count)")
    axes[0].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Flip direction: net movement is AWAY from gold under translation "
                 "(n=1221)", fontsize=11)
    _save(fig, "fig18_mixed_flip_direction")


def fig_by_gold_lang(by_gold):
    """Mixed accuracy split by the language the GOLD option was drawn in."""
    langs = ["ru", "he", "es"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    w = 0.8 / len(langs)
    ymax = 0.56
    for j, lg in enumerate(langs):
        xs, accs, los, his = [], [], [], []
        for i, m in enumerate(MODELS):
            row = by_gold[(by_gold.model == m) & (by_gold.gold_lang == lg)]
            if not len(row):
                continue
            r = row.iloc[0]
            x = i + (j - (len(langs) - 1) / 2) * w
            xs.append(x); accs.append(r.accuracy)
            los.append(r.accuracy - r.ci_low); his.append(r.ci_high - r.accuracy)
            _annotate(ax, x, r.ci_high, r.accuracy, f"{int(r.correct)}/{int(r.n)}", ymax)
        ax.bar(xs, accs, w, yerr=[los, his], capsize=2, color=LANG_COLOR[lg],
               label=f"gold in {LANG_NAME[lg]}",
               error_kw={"elinewidth": 0.9, "alpha": 0.7})
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_NAME[m] for m in MODELS])
    ax.set_ylabel("accuracy on mixed items"); ax.set_ylim(0, ymax)
    ax.set_title("MIXED accuracy by the GOLD option's language\n"
                 "same English question & mixed distractors — success tracks the "
                 "correct answer's language", fontsize=11)
    ax.legend(ncol=3, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.06))
    _save(fig, "fig19_mixed_by_gold_lang")


def main() -> None:
    summary = pd.read_csv(HERE / "mixed_summary.csv")
    flips = pd.read_csv(HERE / "mixed_flips.csv")
    by_gold = pd.read_csv(HERE / "mixed_by_gold_lang.csv")
    fig_accuracy(summary)
    fig_flip_rate(flips)
    fig_flip_direction(flips)
    fig_by_gold_lang(by_gold)


if __name__ == "__main__":
    main()
