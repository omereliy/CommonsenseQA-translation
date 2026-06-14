"""Stage 5a: turn analyze.py's CSVs into publication-quality figures.

Reads results/summary.csv + results/flips.csv (written by scripts.analyze) and
emits PNG+PDF figures to results/figures/. matplotlib only; self-contained style.
Robust to partial/missing data — figures that lack their inputs are skipped with
a note rather than crashing. Use --demo to render against a synthetic fixture.

  PYTHONPATH=src .venv/bin/python -m scripts.plots                 # real results/
  PYTHONPATH=src .venv/bin/python -m scripts.plots --demo          # synthetic
  PYTHONPATH=src .venv/bin/python -m scripts.plots --results-dir results/cluster-20260614

Figures:
  fig1_accuracy_by_condition  per-model accuracy (en-en vs en-x langs) + Wilson CIs
  fig2_flip_rate_by_lang      HEADLINE: flip rate en-en→en-x per lang, per model
  fig3_flip_direction         flips split toward_gold / away_gold (stacked)
  fig4_flip_rate_vs_size      flip rate vs model parameter size (scaling trend)
  fig5_think_off_vs_on        flip rate think=off vs think=on, paired per model
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt

from scripts._analysis_io import (DEMO_BANNER, LANG_ORDER, MODEL_SIZE_B,
                                   load_csvs, make_demo_csvs, order_models)

# --- house style ----------------------------------------------------------- #
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
    "axes.axisbelow": True, "figure.autolayout": True,
})
LANG_COLOR = {"en": "#444444", "ru": "#4C72B0", "es": "#DD8452", "he": "#55A868"}


def _save(fig, out_dir: Path, name: str, demo: bool) -> Path:
    if demo:
        fig.text(0.5, 0.5, "SYNTHETIC", fontsize=48, color="red", alpha=0.10,
                 ha="center", va="center", rotation=30, zorder=0)
    p = out_dir / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
def fig_accuracy_by_condition(summary, out_dir, demo):
    """Grouped bars: per model, accuracy for en-en and each en-x lang, CIs."""
    df = summary[summary["think"] == _pref_think(summary)]
    if df.empty:
        return None
    models = order_models(df["model"])
    cats = ["en"] + LANG_ORDER  # en-en baseline then the en-x langs
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 2, 4.2))
    w = 0.8 / len(cats)
    for j, lang in enumerate(cats):
        accs, los, his = [], [], []
        for m in models:
            cond = "en-en" if lang == "en" else "en-x"
            row = df[(df["model"] == m) & (df["condition"] == cond) & (df["lang"] == lang)]
            if len(row):
                r = row.iloc[0]
                accs.append(r["accuracy"]); los.append(r["accuracy"] - r["ci_low"])
                his.append(r["ci_high"] - r["accuracy"])
            else:
                accs.append(float("nan")); los.append(0); his.append(0)
        xs = [i + (j - (len(cats) - 1) / 2) * w for i in range(len(models))]
        ax.bar(xs, accs, w, yerr=[los, his], capsize=2, color=LANG_COLOR.get(lang),
               label=("en-en" if lang == "en" else f"en-{lang}"),
               error_kw={"elinewidth": 0.9, "alpha": 0.7})
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
    ax.set_title(f"Accuracy by condition (think={_pref_think(summary)})")
    ax.legend(title="answer lang", ncol=len(cats), fontsize=9, loc="lower right")
    return _save(fig, out_dir, "fig1_accuracy_by_condition", demo)


def fig_flip_rate_by_lang(flips, out_dir, demo):
    """HEADLINE figure: flip rate en-en→en-x, grouped by model, coloured by lang."""
    df = flips[flips["think"] == _pref_think(flips)]
    if df.empty:
        return None
    models = order_models(df["model"])
    langs = [l for l in LANG_ORDER if l in set(df["lang"])]
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 2, 4.2))
    w = 0.8 / max(len(langs), 1)
    for j, lang in enumerate(langs):
        rates, stars = [], []
        for m in models:
            row = df[(df["model"] == m) & (df["lang"] == lang)]
            if len(row):
                rates.append(row.iloc[0]["flip_rate"])
                stars.append(_sig_star(row.iloc[0].get("mcnemar_p")))
            else:
                rates.append(float("nan")); stars.append("")
        xs = [i + (j - (len(langs) - 1) / 2) * w for i in range(len(models))]
        bars = ax.bar(xs, rates, w, color=LANG_COLOR.get(lang), label=f"en-{lang}")
        for b, s in zip(bars, stars):
            if s:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, s,
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Flip rate (en-en → en-x)")
    ax.set_title(f"Prediction flips when only the answer language changes "
                 f"(think={_pref_think(flips)})")
    ax.legend(title="answer lang", fontsize=9)
    ax.margins(y=0.12)
    fig.text(0.01, 0.01, "* p<0.05  ** p<0.01  *** p<0.001 (McNemar)", fontsize=8,
             color="#666")
    return _save(fig, out_dir, "fig2_flip_rate_by_lang", demo)


def fig_flip_direction(flips, out_dir, demo):
    """Per (model, lang): stacked toward_gold / away_gold counts as fractions of n."""
    df = flips[flips["think"] == _pref_think(flips)]
    if df.empty:
        return None
    df = df.sort_values(by=["lang", "model"], key=lambda s: s.map(
        lambda v: order_models([v])[0] if s.name == "model" else v))
    rows = list(df.itertuples())
    labels = [f"{r.model}\nen-{r.lang}" for r in rows]
    n = [r.n for r in rows]
    toward = [r.toward_gold / x if x else 0 for r, x in zip(rows, n)]
    away = [r.away_gold / x if x else 0 for r, x in zip(rows, n)]
    other = [max(0.0, r.flip_rate - t - a) for r, t, a in zip(rows, toward, away)]
    fig, ax = plt.subplots(figsize=(1.0 * len(rows) + 2, 4.4))
    x = range(len(rows))
    ax.bar(x, toward, color="#2a9d4a", label="toward gold (→ correct)")
    ax.bar(x, away, bottom=toward, color="#c0392b", label="away from gold (→ wrong)")
    ax.bar(x, other, bottom=[t + a for t, a in zip(toward, away)],
           color="#bbbbbb", label="wrong→wrong")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("Fraction of items (flipped)")
    ax.set_title(f"Flip direction (think={_pref_think(flips)})")
    ax.legend(fontsize=9)
    return _save(fig, out_dir, "fig3_flip_direction", demo)


def fig_flip_rate_vs_size(flips, out_dir, demo):
    """Scaling trend: flip rate vs model parameter size, one line per lang."""
    df = flips[flips["think"] == _pref_think(flips)]
    df = df[df["model"].isin(MODEL_SIZE_B)]
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for lang in [l for l in LANG_ORDER if l in set(df["lang"])]:
        sub = df[df["lang"] == lang].copy()
        sub["size"] = sub["model"].map(MODEL_SIZE_B)
        sub = sub.sort_values("size")
        ax.plot(sub["size"], sub["flip_rate"], "o-", color=LANG_COLOR.get(lang),
                label=f"en-{lang}")
    ax.set_xscale("log")
    ax.set_xlabel("Model size (B params, log scale)")
    ax.set_ylabel("Flip rate (en-en → en-x)")
    ax.set_title(f"Does scale reduce language sensitivity? (think={_pref_think(flips)})")
    ax.legend(title="answer lang", fontsize=9)
    return _save(fig, out_dir, "fig4_flip_rate_vs_size", demo)


def fig_think_off_vs_on(flips, out_dir, demo):
    """Mean flip rate (over langs) per model, think=off vs think=on."""
    if set(flips["think"]) < {"off", "on"} or flips.empty:
        return None
    g = (flips.groupby(["model", "think"])["flip_rate"].mean().unstack("think"))
    if "off" not in g or "on" not in g:
        return None
    g = g.reindex(order_models(g.index))
    models = list(g.index)
    fig, ax = plt.subplots(figsize=(1.4 * len(models) + 2, 4.2))
    w = 0.38
    xs = range(len(models))
    ax.bar([i - w / 2 for i in xs], g["off"], w, color="#8888cc", label="think=off")
    ax.bar([i + w / 2 for i in xs], g["on"], w, color="#cc8888", label="think=on")
    ax.set_xticks(list(xs)); ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Mean flip rate (over langs)")
    ax.set_title("Does <think> reduce flips?")
    ax.legend(fontsize=9)
    return _save(fig, out_dir, "fig5_think_off_vs_on", demo)


# --------------------------------------------------------------------------- #
def _pref_think(df):
    """Prefer think=off for the per-model panels (deterministic baseline); fall
    back to whatever's present."""
    if df.empty:
        return "off"
    thinks = set(df["think"])
    return "off" if "off" in thinks else sorted(thinks)[0]


def _sig_star(p):
    if p is None or p != p:  # NaN
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", help="figure dir (default: <results-dir>/figures)")
    ap.add_argument("--demo", action="store_true",
                    help="render against a synthetic fixture (marked SYNTHETIC)")
    args = ap.parse_args()

    if args.demo:
        src = Path(tempfile.mkdtemp(prefix="csqa_demo_"))
        make_demo_csvs(src)
        out_dir = Path(args.out or "results/figures_demo")
        print(f"[demo] {DEMO_BANNER}")
    else:
        src = Path(args.results_dir)
        out_dir = Path(args.out or src / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, flips = load_csvs(src)
    if summary.empty and flips.empty:
        raise SystemExit(f"no summary.csv/flips.csv under {src} — run "
                         f"`python -m scripts.analyze` first, or pass --demo")

    made = []
    for name, fn, frame in [
        ("fig1", fig_accuracy_by_condition, summary),
        ("fig2", fig_flip_rate_by_lang, flips),
        ("fig3", fig_flip_direction, flips),
        ("fig4", fig_flip_rate_vs_size, flips),
        ("fig5", fig_think_off_vs_on, flips),
    ]:
        try:
            p = fn(frame, out_dir, args.demo)
        except Exception as e:  # one bad figure shouldn't sink the rest
            print(f"  {name}: skipped ({type(e).__name__}: {e})")
            continue
        if p:
            made.append(p); print(f"  wrote {p}")
        else:
            print(f"  {name}: skipped (insufficient data)")

    print(f"\n{len(made)} figure(s) → {out_dir}")


if __name__ == "__main__":
    main()
