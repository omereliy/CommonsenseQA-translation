"""Stage 5c: assemble a simple Marp-compatible Markdown slide deck.

Writes results/deck/slides.md — plain Markdown with Marp front-matter and `---`
slide separators. It embeds the figures (built by scripts.plots, written next to
the deck so the relative paths resolve) and pulls a few key numbers from the
CSVs. No heavy deps. Render to PDF/HTML with Marp if installed:

  marp results/deck/slides.md -o results/deck/slides.pdf
  marp results/deck/slides.md -o results/deck/slides.html

Usage:
  PYTHONPATH=src .venv/bin/python -m scripts.build_deck            # real results/
  PYTHONPATH=src .venv/bin/python -m scripts.build_deck --demo     # synthetic
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from scripts._analysis_io import (DEMO_BANNER, LANG_NAME, LANG_ORDER, load_csvs,
                                   make_demo_csvs)
from scripts import plots as plots_mod

FRONT_MATTER = """---
marp: true
paginate: true
theme: default
---
"""


def _fmt_pct(x):
    return f"{100 * x:.1f}%"


def _headline_numbers(summary, flips):
    """A few sentences of takeaway derived from the CSVs (guarded for missing)."""
    bits = []
    if not flips.empty:
        off = flips[flips["think"] == "off"] if "off" in set(flips["think"]) else flips
        by_lang = off.groupby("lang")["flip_rate"].mean()
        for lang in [l for l in LANG_ORDER if l in by_lang.index]:
            bits.append(f"**{LANG_NAME.get(lang, lang)}**: {_fmt_pct(by_lang[lang])} "
                        f"of predictions flip when only the answer language changes")
        # most-affected model
        worst = off.loc[off["flip_rate"].idxmax()]
        bits.append(f"Largest single effect: **{worst['model']}** on en-{worst['lang']} "
                    f"→ {_fmt_pct(worst['flip_rate'])} flip rate")
    return bits


def build_markdown(summary, flips, fig_names, demo) -> str:
    s = [FRONT_MATTER]
    banner = f"\n> {DEMO_BANNER}\n" if demo else ""

    # Title
    s.append("# Do LLMs Choose Concepts or English Words?\n\n"
             "### Answer-Side Translation in CommonsenseQA\n"
             f"{banner}\n"
             "Tal Malul · Omer Eliyahu · Mark Feldman · Daniel Koyfman  \n"
             "*Computational Semantics & NLU — BGU, 2025–2026*")

    # RQ
    s.append("---\n\n## Research question\n\n"
             "Do language models choose the correct commonsense **concept**, or do "
             "they lean on the English answer **wording**?\n\n"
             "We keep CommonsenseQA **questions in English** and translate **only the "
             "answer choices** into a target language. The model reads an English "
             "question but must map it to candidate concepts written in another "
             "language — isolating cross-lingual concept grounding.")

    # Method (1 slide)
    s.append("---\n\n## Method (one slide)\n\n"
             "- **Conditions:** `en-en` (English Q + English choices, baseline) vs "
             "`en-x` (English Q, choices in *x* ∈ {ru, es, he}).\n"
             "- **Letter answers (A–E)**, never answer text — so string matching can't "
             "masquerade as concept grounding.\n"
             "- **Deterministic decoding** (temp=0, fixed seed); every run writes a "
             "manifest; raw outputs cached.\n"
             "- **Models:** Qwen3.5 {0.8B, 4B, 9B}, qwen3.6:35b, gemma4:26b-a4b, "
             "think ∈ {off, on}.\n"
             "- **Metrics:** per-condition accuracy (Wilson CI) and the core "
             "**flip rate** `en-en → en-x`, split toward/away gold + McNemar.")

    # Headline flip-rate figure
    s.append("---\n\n## Headline: flips when only the answer language changes\n\n"
             + _img("fig2_flip_rate_by_lang", fig_names)
             + "\nIf the model grounded the *concept*, translating the choices "
             "shouldn't change its pick. Every flip is a choice driven by **wording**, "
             "not concept.")

    # Per-language takeaways
    bits = _headline_numbers(summary, flips)
    if bits:
        s.append("---\n\n## Per-language results\n\n"
                 + "\n".join(f"- {b}" for b in bits))

    # Accuracy figure
    s.append("---\n\n## Accuracy by condition\n\n"
             + _img("fig1_accuracy_by_condition", fig_names)
             + "\nAccuracy drops as the answer language moves away from English — "
             "but accuracy alone hides *which* items moved. The flip rate does not.")

    # Direction
    s.append("---\n\n## Which way do the flips go?\n\n"
             + _img("fig3_flip_direction", fig_names)
             + "\nFlips split into **toward gold** (the translation helped) vs **away "
             "from gold** (the English wording was a crutch). A net move away is the "
             "strongest evidence of wording-reliance.")

    # Scaling + think
    s.append("---\n\n## Does scale or <think> help?\n\n"
             + _img("fig4_flip_rate_vs_size", fig_names)
             + _img("fig5_think_off_vs_on", fig_names))

    # Takeaway
    s.append("---\n\n## Takeaway — concept vs wording\n\n"
             "- A non-trivial fraction of predictions **flip** when nothing changes "
             "but the *language of the answer choices*.\n"
             "- That is direct evidence the model is partly keying on **English "
             "wording**, not a language-agnostic concept.\n"
             "- The effect grows for scripts more distant from English (he ≳ ru ≳ es).")

    # Caveats / next steps
    s.append("---\n\n## Caveats & next steps\n\n"
             "- MT noise on short ConceptNet phrases — a Hebrew human spot-check is "
             "still open.\n"
             "- Zero-shot generative arm vs fine-tuned XLM-R: report the **degradation "
             "pattern**, not absolute cross-arm accuracy.\n"
             "- Extensions wired but not yet run: `x-x` / `x-en`, mixed answer sets, "
             "partial question-noun translation.")

    if demo:
        s.append(f"---\n\n## ⚠️ {DEMO_BANNER}\n\n"
                 "Replace `results/` with a real sweep and rerun "
                 "`scripts.analyze → scripts.plots → scripts.build_deck`.")
    return "\n\n".join(s) + "\n"


def _img(name, fig_names) -> str:
    """Embed a figure if it was produced; otherwise a visible placeholder."""
    if name in fig_names:
        return f"![w:900]({name}.png)\n"
    return f"*(figure `{name}` unavailable — insufficient data)*\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", help="deck dir (default: <results-dir>/deck)")
    ap.add_argument("--demo", action="store_true", help="synthetic fixture")
    args = ap.parse_args()

    if args.demo:
        src = Path(tempfile.mkdtemp(prefix="csqa_demo_"))
        make_demo_csvs(src)
        out_dir = Path(args.out or "results/deck_demo")
        print(f"[demo] {DEMO_BANNER}")
    else:
        src = Path(args.results_dir)
        out_dir = Path(args.out or src / "deck")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, flips = load_csvs(src)
    if summary.empty and flips.empty:
        raise SystemExit(f"no summary.csv/flips.csv under {src} — run "
                         f"`python -m scripts.analyze` first, or pass --demo")

    # Build figures straight into the deck dir so the slides' relative paths work.
    fig_names = set()
    for fn, frame in [(plots_mod.fig_accuracy_by_condition, summary),
                      (plots_mod.fig_flip_rate_by_lang, flips),
                      (plots_mod.fig_flip_direction, flips),
                      (plots_mod.fig_flip_rate_vs_size, flips),
                      (plots_mod.fig_think_off_vs_on, flips)]:
        try:
            p = fn(frame, out_dir, args.demo)
        except Exception as e:
            print(f"  figure skipped ({type(e).__name__}: {e})")
            continue
        if p:
            fig_names.add(p.stem)

    md = build_markdown(summary, flips, fig_names, args.demo)
    slides = out_dir / "slides.md"
    slides.write_text(md)
    print(f"  wrote {slides} ({len(fig_names)} figure(s) embedded)")
    print(f"\nrender with:  marp {slides} -o {out_dir / 'slides.pdf'}")


if __name__ == "__main__":
    main()
