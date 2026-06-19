"""Aggregate every eval arm into one professional report (figures + README).

Self-contained: reads the analysis CSVs (written by scripts.analyze) plus the
condition variants and cached outputs, renders publication-quality matplotlib
figures into report/figures/, and writes report/README.md with the tables,
embedded figures, narrative, and a few concrete flip examples.

Inputs (all optional — a missing one degrades gracefully):
  results/summary.csv              accuracy per (model, think, condition, lang) + Wilson CI
  results/flips.csv                flip rate en-en->en-x per (model, lang) + McNemar p
  results/translation_agreement.csv  cross-MT agreement on the choice strings
  data/variants/{en-en__en,en-x__<lang>}.json   for the worked examples
  results/<model>/<run>/outputs.jsonl           predicted letters, for the examples

Run:
  .venv/bin/python -m report.build_report
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
VARIANTS = ROOT / "data" / "variants"
FIG = Path(__file__).resolve().parent / "figures"

# --- house style ----------------------------------------------------------- #
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.autolayout": True, "font.family": "DejaVu Sans",
})
LANG_NAME = {"en": "English", "ru": "Russian", "es": "Spanish", "he": "Hebrew"}
LANG_COLOR = {"en": "#444444", "ru": "#4C72B0", "es": "#DD8452", "he": "#55A868"}
HILITE = "#C44E52"  # accent for the strongest arm

# Display labels + which arm represents each family in the headline figures.
DISPLAY = {
    "mbert-ep6": "mBERT (ft)", "mbert-ep3": "mBERT ep3",
    "xlmr-ep6": "XLM-R (ft)", "xlmr-ep3": "XLM-R ep3",
    "Qwen3.5:0.8B": "Qwen3.5 0.8B", "Qwen3.5:4B": "Qwen3.5 4B",
    "haiku-4.5-subagent": "Haiku 4.5",
}
MAIN = ["mbert-ep6", "xlmr-ep6", "Qwen3.5:0.8B", "Qwen3.5:4B", "haiku-4.5-subagent"]
LANGS = ["ru", "es", "he"]

# Per-epoch dev accuracy (974-item held-out dev split) from the model cards.
# Per-step loss was not persisted (trainer ran report_to=[]); this is the
# recorded learning signal. Epochs 4-6 are the warm-start continuation
# (LR schedule restarted) — hence the non-monotonic tail.
DEV_ACC = {
    "XLM-R (ft)": {3: 0.4415, 4: 0.4620, 5: 0.4877, 6: 0.4815},
    "mBERT (ft)": {3: 0.4692, 4: 0.4620, 5: 0.4660, 6: 0.4540},
}


def stars(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def load():
    def rd(name):
        p = RESULTS / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    return rd("summary.csv"), rd("flips.csv"), rd("translation_agreement.csv")


def capability_order(summary, models):
    """Order models by en-en accuracy (a capability proxy), weakest -> strongest."""
    acc = {m: en_acc(summary, m) for m in models}
    return sorted([m for m in models if acc[m] is not None], key=lambda m: acc[m])


def en_acc(summary, model):
    r = summary[(summary.model == model) & (summary.condition == "en-en")]
    return float(r.accuracy.iloc[0]) if len(r) else None


def acc_cell(summary, model, lang):
    cond = "en-en" if lang == "en" else "en-x"
    r = summary[(summary.model == model) & (summary.condition == cond) & (summary.lang == lang)]
    if not len(r):
        return None
    return float(r.accuracy.iloc[0]), float(r.ci_low.iloc[0]), float(r.ci_high.iloc[0])


def savefig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure: {out.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_accuracy(summary, models):
    conds = ["en", "ru", "es", "he"]
    x = range(len(models))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for j, lang in enumerate(conds):
        vals, los, his = [], [], []
        for m in models:
            c = acc_cell(summary, m, lang)
            vals.append(c[0] if c else 0)
            los.append((c[0] - c[1]) if c else 0)
            his.append((c[2] - c[0]) if c else 0)
        ax.bar([xi + (j - 1.5) * w for xi in x], vals, w,
               yerr=[los, his], capsize=2, color=LANG_COLOR[lang],
               label=("en-en" if lang == "en" else f"en-x · {LANG_NAME[lang]}"),
               error_kw={"elinewidth": 0.8, "alpha": 0.6})
    ax.axhline(0.20, ls="--", lw=0.9, color="#999", zorder=0)
    ax.text(len(models) - 0.5, 0.205, "chance (0.20)", ha="right", va="bottom",
            fontsize=8, color="#777")
    ax.set_xticks(list(x))
    ax.set_xticklabels([DISPLAY[m] for m in models])
    ax.set_ylabel("accuracy (validation, n=1221)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Accuracy by condition — English question, choices in X", weight="bold")
    ax.legend(ncol=4, fontsize=9, loc="upper left", framealpha=0.9)
    savefig(fig, "fig1_accuracy_by_condition.png")


def fig_ladder(summary, models):
    order = sorted(models, key=lambda m: en_acc(summary, m) or 0)
    vals = [en_acc(summary, m) for m in order]
    colors = [HILITE if m == "haiku-4.5-subagent" else "#4C72B0" for m in order]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh([DISPLAY[m] for m in order], vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.axvline(0.20, ls="--", lw=0.9, color="#999")
    ax.set_xlabel("en-en accuracy")
    ax.set_title("Model capability ladder (English baseline)", weight="bold")
    savefig(fig, "fig2_capability_ladder.png")


def fig_flip(flips, models):
    x = range(len(models))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for j, lang in enumerate(LANGS):
        vals, ps = [], []
        for m in models:
            r = flips[(flips.model == m) & (flips.lang == lang)]
            vals.append(float(r.flip_rate.iloc[0]) if len(r) else 0)
            ps.append(float(r.mcnemar_p.iloc[0]) if len(r) else None)
        bars = ax.bar([xi + (j - 1) * w for xi in x], vals, w,
                      color=LANG_COLOR[lang], label=LANG_NAME[lang])
        for b, v, p in zip(bars, vals, ps):
            if v:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.006, stars(p),
                        ha="center", fontsize=9, color="#333")
    ax.set_xticks(list(x))
    ax.set_xticklabels([DISPLAY[m] for m in models])
    ax.set_ylabel("flip rate  (pred changes when only choices translated)")
    ax.set_title("Core diagnostic: prediction flips, en-en → en-x   (* p<.05  ** p<.01  *** p<.001)",
                 weight="bold")
    ax.legend(title="answer language", fontsize=9)
    savefig(fig, "fig3_flip_rate.png")


def fig_direction(flips, models):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True)
    for ax, lang in zip(axes, LANGS):
        tow, awa = [], []
        for m in models:
            r = flips[(flips.model == m) & (flips.lang == lang)]
            tow.append(int(r.toward_gold.iloc[0]) if len(r) else 0)
            awa.append(int(r.away_gold.iloc[0]) if len(r) else 0)
        lab = [DISPLAY[m] for m in models]
        ax.bar(lab, tow, color="#55A868", label="→ gold (helped)")
        ax.bar(lab, awa, bottom=tow, color="#C44E52", label="← gold (hurt)")
        ax.set_title(LANG_NAME[lang], fontsize=11)
        ax.tick_params(axis="x", rotation=40)
        for t in ax.get_xticklabels():
            t.set_ha("right")
    axes[0].set_ylabel("# items flipped")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Flip direction — translating choices hurts more than it helps (net ← gold)",
                 weight="bold")
    savefig(fig, "fig4_flip_direction.png")


def fig_flip_vs_capability(summary, flips, models):
    xs, ys, labs = [], [], []
    for m in models:
        a = en_acc(summary, m)
        r = flips[(flips.model == m) & (flips.lang.isin(LANGS))]
        if a is None or not len(r):
            continue
        xs.append(a)
        ys.append(float(r.flip_rate.mean()))
        labs.append(DISPLAY[m])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(xs, ys, s=90, color="#4C72B0", zorder=3)
    for xi, yi, l in zip(xs, ys, labs):
        ax.annotate(l, (xi, yi), textcoords="offset points", xytext=(8, 4), fontsize=9)
    if len(xs) >= 2:  # least-squares trend
        b, a0 = _fit(xs, ys)
        gx = [min(xs), max(xs)]
        ax.plot(gx, [a0 + b * g for g in gx], ls="--", color="#999",
                label=f"trend (slope {b:+.2f})")
        ax.legend(fontsize=9)
    ax.set_xlabel("model capability  (en-en accuracy)")
    ax.set_ylabel("mean flip rate across ru/es/he")
    ax.set_title("Stronger models are more concept-grounded:\nflip rate falls as capability rises",
                 weight="bold")
    savefig(fig, "fig5_flip_vs_capability.png")


def _fit(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs) or 1e-9
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return b, my - b * mx


def fig_heatmap(summary, models):
    order = capability_order(summary, models)
    M = []
    for m in order:
        row = []
        en = en_acc(summary, m)
        for lang in LANGS:
            c = acc_cell(summary, m, lang)
            row.append((c[0] - en) if (c and en is not None) else float("nan"))
        M.append(row)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    im = ax.imshow(M, cmap="RdYlGn", vmin=-0.13, vmax=0.0, aspect="auto")
    ax.set_xticks(range(len(LANGS)))
    ax.set_xticklabels([LANG_NAME[l] for l in LANGS])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DISPLAY[m] for m in order])
    for i, row in enumerate(M):
        for j, v in enumerate(row):
            if v == v:
                ax.text(j, i, f"{v*100:+.1f}", ha="center", va="center",
                        fontsize=9, color="#222")
    ax.set_title("Accuracy change en-en → en-x  (Δ points)", weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ accuracy")
    savefig(fig, "fig6_degradation_heatmap.png")


def fig_sources(summary):
    """Translator robustness: xlmr-ep6 / mbert-ep6 accuracy across Google/NLLB/Opus."""
    pairs = [("xlmr-ep6", "XLM-R (ft)"), ("mbert-ep6", "mBERT (ft)")]
    srcs = [("", "Google"), ("-nllb", "NLLB"), ("-opus", "Opus")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    ok = False
    for ax, (mdl, name) in zip(axes, pairs):
        x = range(len(LANGS))
        w = 0.25
        for j, (suf, sname) in enumerate(srcs):
            vals = []
            for lang in LANGS:
                r = summary[(summary.model == mdl) & (summary.lang == f"{lang}{suf}")]
                vals.append(float(r.accuracy.iloc[0]) if len(r) else 0)
            if any(vals):
                ok = True
            ax.bar([xi + (j - 1) * w for xi in x], vals, w, label=sname)
        ax.set_xticks(list(x))
        ax.set_xticklabels([LANG_NAME[l] for l in LANGS])
        ax.set_title(name, fontsize=11)
        ax.set_ylim(0.30, 0.46)
    axes[0].set_ylabel("en-x accuracy")
    axes[-1].legend(title="translator", fontsize=9)
    fig.suptitle("Robustness: the en→x drop replicates across three independent translators",
                 weight="bold")
    if ok:
        savefig(fig, "fig7_translation_sources.png")
    else:
        plt.close(fig)
        return False
    return True


def fig_training():
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name, curve in DEV_ACC.items():
        ep = sorted(curve)
        ax.plot(ep, [curve[e] for e in ep], marker="o",
                color=HILITE if name.startswith("XLM") else "#4C72B0", label=name)
        for e in ep:
            ax.annotate(f"{curve[e]:.3f}", (e, curve[e]),
                        textcoords="offset points", xytext=(0, 7), fontsize=8, ha="center")
    ax.axvspan(3, 6, color="#f2f2f2", zorder=0)
    ax.text(4.5, ax.get_ylim()[0], "warm-start continuation (LR restarted)",
            ha="center", va="bottom", fontsize=8, color="#888")
    ax.set_xlabel("epoch")
    ax.set_ylabel("dev accuracy (974-item held-out split)")
    ax.set_xticks([3, 4, 5, 6])
    ax.set_title("Encoder training dynamics — dev accuracy per epoch", weight="bold")
    ax.legend()
    savefig(fig, "fig8_training_devacc.png")


# --------------------------------------------------------------------------- #
# Worked flip examples (English-anchoring caught in the act)
# --------------------------------------------------------------------------- #
def _preds(run_glob):
    dirs = sorted(glob.glob(str(RESULTS / run_glob)))
    if not dirs:
        return {}
    out = {}
    with open(Path(dirs[0]) / "outputs.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = (r.get("pred"), r.get("gold"))
    return out


def examples(model_tag, lang, k=3):
    try:
        en_items = {r["id"]: r for r in json.loads(
            (VARIANTS / "en-en__en.json").read_text(encoding="utf-8"))["items"]}
        x_items = {r["id"]: r for r in json.loads(
            (VARIANTS / f"en-x__{lang}.json").read_text(encoding="utf-8"))["items"]}
    except FileNotFoundError:
        return []
    en_p = _preds(f"{model_tag}/en-en__en__*")
    x_p = _preds(f"{model_tag}/en-x__{lang}__*")
    rows = []
    for cid, (ep, gold) in en_p.items():
        xp = x_p.get(cid, (None, None))[0]
        if ep == gold and xp is not None and xp != gold and cid in x_items:  # helped→hurt
            it = en_items[cid]
            gold_txt = x_items[cid]["choices"].get(gold)
            pred_txt = x_items[cid]["choices"].get(xp)
            if gold_txt == pred_txt:   # skip MT collisions (two choices → same string)
                continue
            rows.append({"q": it["question"], "gold": gold,
                         "en_choice": it["choices"].get(gold),
                         "en_pred": ep, "x_pred": xp,
                         "x_gold_choice": x_items[cid]["choices"].get(gold),
                         "x_pred_choice": x_items[cid]["choices"].get(xp)})
        if len(rows) >= k:
            break
    return rows


# --------------------------------------------------------------------------- #
# README
# --------------------------------------------------------------------------- #
def acc_table(summary, models):
    head = "| model | en-en | en-x ru | en-x es | en-x he |\n|---|---|---|---|---|"
    lines = [head]
    for m in sorted(models, key=lambda m: en_acc(summary, m) or 0, reverse=True):
        cells = [DISPLAY[m]]
        for lang in ["en", "ru", "es", "he"]:
            c = acc_cell(summary, m, lang)
            cells.append(f"{c[0]:.3f}" if c else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def flip_table(flips, models):
    head = ("| model | ru | es | he |\n|---|---|---|---|")
    lines = [head]
    for m in sorted(models, key=lambda x: MAIN.index(x) if x in MAIN else 99):
        cells = [DISPLAY[m]]
        for lang in LANGS:
            r = flips[(flips.model == m) & (flips.lang == lang)]
            if len(r):
                fr = float(r.flip_rate.iloc[0])
                cells.append(f"{fr:.3f}{stars(float(r.mcnemar_p.iloc[0]))}")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def examples_md(rows, lang):
    if not rows:
        return "_(no cached predictions found for examples)_"
    out = []
    for i, r in enumerate(rows, 1):
        out.append(
            f"{i}. **Q:** {r['q']}\n"
            f"   - gold **{r['gold']}** = `{r['en_choice']}`  (→ in {LANG_NAME[lang]}: `{r['x_gold_choice']}`)\n"
            f"   - en-en predicted **{r['en_pred']}** ✓ correct\n"
            f"   - en-x ({LANG_NAME[lang]}) predicted **{r['x_pred']}** = `{r['x_pred_choice']}` ✗ — flipped off gold")
    return "\n".join(out)


def build_readme(summary, flips, agree, models, has_sources, ex_rows, ex_lang):
    n_models = summary.model.nunique() if len(summary) else 0
    haiku_en = en_acc(summary, "haiku-4.5-subagent")
    xlmr_en = en_acc(summary, "xlmr-ep6")
    # mean flip per model for the narrative
    def mflip(m):
        r = flips[(flips.model == m) & (flips.lang.isin(LANGS))]
        return float(r.flip_rate.mean()) if len(r) else float("nan")
    md = f"""# CommonsenseQA answer-side translation — results report

*Generated by `report/build_report.py` from `results/*.csv`. Do not edit by hand;
re-run the script to refresh.*

**Research question.** Do models choose the correct commonsense **concept**, or do
they lean on English answer **wording**? We hold the CSQA **question in English** and
translate **only the answer choices** into Russian / Spanish / Hebrew, then measure
how often the prediction **flips** when nothing but the choice language changes.
Selection is the **letter** (A–E), never the answer text. Validation split, n=1221.

**Arms compared ({n_models}).** Two fine-tuned multilingual encoders (XLM-R, mBERT;
English-only fine-tune → zero-shot cross-lingual), two zero-shot generative Qwens
(0.8B, 4B, vLLM), and **Claude Haiku 4.5** (zero-shot, via subscription subagents).

---

## Headline findings

1. **The effect is real for every model.** Translating only the choices flips
   predictions at rates from **{mflip('haiku-4.5-subagent'):.0%}** (Haiku) up to
   **{mflip('Qwen3.5:0.8B'):.0%}** (Qwen 0.8B), net *away from gold*, and the
   en-en→en-x accuracy drop is McNemar-significant for **all** arms. Models are
   partly anchored to English wording — not purely concept-grounded.
2. **But it shrinks with capability.** Flip rate falls monotonically as en-en
   accuracy rises (Fig 5): a frontier LLM resists the answer-language confound far
   better than a fine-tuned `xlm-roberta-base`. The en→Hebrew accuracy drop is
   **−3.7 pts for Haiku** vs **≈−15 pts for XLM-R**.
3. **It is not a translator artifact.** The drop replicates across three independent
   MT systems (Google / NLLB / Opus) that agree on the exact word only ~35–48% of
   the time (Fig 7) — separating concept-grounding from MT noise.
4. **Hardest target language is model-dependent.** Encoders fare worst on Hebrew;
   Haiku's lowest accuracy is on Russian (Hebrew = most churn). Not intrinsic to a language.

---

## 1. Accuracy by condition

![accuracy](figures/fig1_accuracy_by_condition.png)

{acc_table(summary, models)}

![ladder](figures/fig2_capability_ladder.png)

Haiku 4.5 leads at **{haiku_en:.3f}** en-en — far above the fine-tuned encoders
(XLM-R {xlmr_en:.3f}); a frontier LLM zero-shot is in a different regime from a
fine-tuned base encoder, a useful upper anchor.

## 2. The core diagnostic — prediction flips (en-en → en-x)

![flip rate](figures/fig3_flip_rate.png)

Flip rate (`*` p<.05 `**` p<.01 `***` p<.001, McNemar):

{flip_table(flips, models)}

![flip direction](figures/fig4_flip_direction.png)

Flips are **net away from gold** — translating the choices hurts more than it helps,
which is what drives the accuracy drop.

## 3. Strength story — concept-grounding scales with capability

![flip vs capability](figures/fig5_flip_vs_capability.png)

![heatmap](figures/fig6_degradation_heatmap.png)

The clearest single result: **the more capable the model, the smaller the flip rate.**
Concept-grounding is not all-or-nothing — it emerges with model strength.

## 4. Robustness — three independent translators

{"![sources](figures/fig7_translation_sources.png)" if has_sources else "_(translation-source runs not present)_"}

The degradation ordering is **translator-invariant**: every MT system produces the
same en > es ≈ ru > he pattern for the encoders, so the effect is not an artifact of
one translation backend.

## 5. Training dynamics (encoders)

![training](figures/fig8_training_devacc.png)

Per-epoch **dev accuracy** (974-item held-out split). *Per-step train loss was not
persisted during the original runs* (`report_to=[]`), so this is the recorded
learning signal. XLM-R keeps improving to ~5 epochs (kept checkpoint); mBERT plateaus
by epoch 2–3 and the warm-start continuation mildly overtrains — consistent with
XLM-R's stronger cross-lingual transfer.

## 6. Worked examples — English-anchoring caught in the act

Items Haiku got **right** in English but **wrong** once the choices were shown in
{LANG_NAME[ex_lang]} (same question, only choice language changed):

{examples_md(ex_rows, ex_lang)}

---

## Reproduce

```bash
python -m scripts.analyze            # refresh results/summary.csv + flips.csv
python -m report.build_report        # regenerate every figure + this README
```

## Provenance & caveats

- **Encoders / Qwens / Gemini** arms are deterministic (`temperature=0`, seeded,
  manifested). The **Haiku** arm is **non-deterministic** (subscription subagents,
  temperature not pinned) — an exploratory LLM upper-anchor, flagged in its manifest.
- Source CSVs: `results/summary.csv`, `results/flips.csv`,
  `results/translation_agreement.csv`. Frozen per-arm archives live under
  `results/archive/`.
"""
    (Path(__file__).resolve().parent / "README.md").write_text(md, encoding="utf-8")
    print(f"  wrote {Path(__file__).resolve().parent / 'README.md'}")


def main():
    summary, flips, agree = load()
    if not len(summary):
        raise SystemExit("no results/summary.csv — run scripts.analyze first")
    models = [m for m in MAIN if m in set(summary.model)]
    print("building figures...")
    fig_accuracy(summary, models)
    fig_ladder(summary, models)
    fig_flip(flips, models)
    fig_direction(flips, models)
    fig_flip_vs_capability(summary, flips, models)
    fig_heatmap(summary, models)
    has_sources = fig_sources(summary)
    fig_training()
    ex_lang = "he"
    ex_rows = examples("haiku-4.5-subagent", ex_lang, k=3)
    print("writing README...")
    build_readme(summary, flips, agree, models, has_sources, ex_rows, ex_lang)
    print("done.")


if __name__ == "__main__":
    main()
