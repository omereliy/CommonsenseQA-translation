---
name: analyzer
description: Turn the CommonsenseQA answer-translation eval results into publication-quality figures, Markdown/LaTeX tables, and a simple slide deck. Read-only over results — consumes the two CSVs that scripts.analyze writes (summary.csv + flips.csv) and never recomputes metrics or re-queries models. Pairs with the cluster-ops skill: cluster-ops gets outputs.jsonl onto disk and runs scripts.analyze; this skill presents them.
argument-hint: [plots | tables | deck | demo]
---

> User asked for: $ARGUMENTS — pick the matching recipe below.

## Why this skill exists

Triggers (so the skill auto-matches): "plot the results", "make the figures",
"flip-rate chart", "the headline plot", "make the paper table", "markdown/LaTeX
table of accuracy", "build the slides", "make a deck", "summarize the findings",
"what's the headline number", "show me a preview before results land".

**Skill boundary.** This skill is **read-only** over experiment state. It takes
`results/summary.csv` + `results/flips.csv` (written by `scripts.analyze`) and
emits PNG/PDF figures, Markdown/LaTeX tables, and a Marp Markdown deck into
`results/{figures,tables,deck}/`. It never edits `run_eval.py`, the metric code
in `src/csqa_xlang/analysis/`, or result JSONs. **Metric computation lives
upstream** — accuracy / Wilson / flip-rate / McNemar are in
`src/csqa_xlang/analysis/{metrics,aggregate}.py`; if a number looks wrong, fix it
there, not here. For SSH / submit / sync, delegate to `cluster-ops`.

## Conventions

All paths are relative to the repo root
`/Users/omereliyahu/study-folder/spring2026/Computational Semantics/CommonsenseQA-translation`.
Local run convention is `PYTHONPATH=src .venv/bin/python -m scripts.<name>`.

- **Inputs (the only inputs):** `results/summary.csv` and `results/flips.csv`.
  - `summary.csv`: `model, think, condition, lang, n, accuracy, ci_low, ci_high, unparsed`
  - `flips.csv`: `model, think, lang, n, flip_rate, toward_gold, away_gold, acc_en, acc_x, mcnemar_p`
  - Produce them first: `PYTHONPATH=src .venv/bin/python -m scripts.analyze`
    (or `--results-dir results/cluster-YYYYMMDD` after a `cluster-ops` sync).
- **Outputs** land under the results dir: `figures/`, `tables/`, `deck/`. All of
  `results/**` is git-ignored (derived) — figures/tables/deck rebuild from the
  CSVs + these scripts, so don't commit them.
- **The headline is the FLIP RATE** `en-en → en-x`, not accuracy. Accuracy hides
  *which* items moved; the flip rate is the core diagnostic (concept vs wording).
  Direction split (toward_gold / away_gold) and McNemar `p` qualify it.
- **Robust to partial/missing data.** The sweep runs incrementally; a missing or
  half-filled CSV is normal. Every script degrades gracefully — a figure/table
  whose inputs aren't present yet is **skipped with a note**, never a crash.
- **Preview before real data with `--demo`.** Each script can render against a
  synthetic fixture (`scripts/_analysis_io.py:make_demo_csvs`). Demo output is
  written to `results/{figures,tables,deck}_demo/` and clearly marked SYNTHETIC
  (red watermark on plots, banner on tables/deck). Use it to eyeball layout while
  the cluster sweep is still running. **Never present demo numbers as results.**
- **Model order / sizes** (for the scaling figure) live in
  `scripts/_analysis_io.py:MODEL_SIZE_B`. Sizes aren't carried in the CSV — edit
  that map when the roster changes.

## Helper scripts (all under `scripts/`)

### `scripts/plots.py` — five figures → `results/figures/`

matplotlib only, headless (`Agg`), self-contained style. Each figure picks
`think=off` for the per-model panels (the deterministic baseline) unless only
`on` is present.

- `fig1_accuracy_by_condition` — grouped bars, per model: en-en vs each en-x lang, Wilson CI whiskers.
- `fig2_flip_rate_by_lang` — **HEADLINE**: flip rate en-en→en-x, grouped by model, coloured by lang, McNemar significance stars.
- `fig3_flip_direction` — per (model, lang) stacked bar: toward_gold / away_gold / wrong→wrong as fractions of n.
- `fig4_flip_rate_vs_size` — flip rate vs model parameter size (log x), one line per lang — the scaling trend.
- `fig5_think_off_vs_on` — mean flip rate (over langs) per model, think=off vs think=on, paired bars.

```bash
PYTHONPATH=src .venv/bin/python -m scripts.plots                       # real results/
PYTHONPATH=src .venv/bin/python -m scripts.plots --demo                # synthetic preview
PYTHONPATH=src .venv/bin/python -m scripts.plots --results-dir results/cluster-20260614
PYTHONPATH=src .venv/bin/python -m scripts.plots --out /tmp/figs       # override out dir
```

Both `.png` (for the deck/slides) and `.pdf` (for the paper) are written per figure.

### `scripts/make_tables.py` — Markdown (+ optional LaTeX) → `results/tables/`

- `accuracy.md` — one row per (model, think): en-en accuracy then each en-x lang, each `acc [lo,hi]`.
- `flips.md` — the core diagnostic: flip_rate, →gold/←gold counts, acc_en, acc_x, McNemar p (with stars).
- `accuracy.tex` / `flips.tex` — booktabs versions for the paper appendix (`--latex`).

```bash
PYTHONPATH=src .venv/bin/python -m scripts.make_tables                 # md only
PYTHONPATH=src .venv/bin/python -m scripts.make_tables --latex         # + booktabs .tex
PYTHONPATH=src .venv/bin/python -m scripts.make_tables --demo --latex  # synthetic preview
```

### `scripts/build_deck.py` — simple Marp deck → `results/deck/slides.md`

Plain Markdown with Marp front-matter and `---` slide separators. Builds the
figures **into the deck dir** so the slides' relative `![](fig*.png)` paths
resolve, then assembles: title, RQ, method (1 slide), the flip-rate headline,
per-language numbers (pulled from the CSVs), accuracy, flip direction, scale/think,
the concept-vs-wording takeaway, and caveats/next steps.

```bash
PYTHONPATH=src .venv/bin/python -m scripts.build_deck                  # real results/
PYTHONPATH=src .venv/bin/python -m scripts.build_deck --demo           # synthetic preview
```

Render to PDF/HTML with Marp if installed (optional, not a dependency):

```bash
marp results/deck/slides.md -o results/deck/slides.pdf
marp results/deck/slides.md -o results/deck/slides.html
```

### `scripts/_analysis_io.py` — shared loader + synthetic fixture

Not a CLI. Holds the CSV loader (`load_csvs` → empty frames when files are
absent), the model order/size map, and `make_demo_csvs` (the `--demo` fixture).
All three scripts import from here, so partial-data handling and the demo path
live in one place.

## Recipes

### "Make all the figures, tables, and the deck" (standard end-of-sweep flow)

Assumes results are already on disk and `scripts.analyze` has run (steps 1–2 are
`cluster-ops` territory).

```bash
# 1. (cluster-ops) sync results → results/cluster-YYYYMMDD/
# 2. aggregate raw outputs → the two CSVs this skill consumes
PYTHONPATH=src .venv/bin/python -m scripts.analyze --results-dir results/cluster-YYYYMMDD
# 3. figures, tables, deck (point each at the same dir)
PYTHONPATH=src .venv/bin/python -m scripts.plots       --results-dir results/cluster-YYYYMMDD
PYTHONPATH=src .venv/bin/python -m scripts.make_tables --results-dir results/cluster-YYYYMMDD --latex
PYTHONPATH=src .venv/bin/python -m scripts.build_deck  --results-dir results/cluster-YYYYMMDD
```

Then report to the user with the figure paths and 3–5 key numbers from `flips.md`
— lead with the flip rate per language, framed as concept-vs-wording.

### "Preview the deliverables before the sweep finishes"

The full eval may still be running (CSV absent or partial). Render the synthetic
preview to check layout and wording now:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.plots       --demo
PYTHONPATH=src .venv/bin/python -m scripts.make_tables --demo --latex
PYTHONPATH=src .venv/bin/python -m scripts.build_deck  --demo
# inspect results/{figures,tables,deck}_demo/ — every output is marked SYNTHETIC.
```

### "Pull the headline out of a sweep"

1. Read `results/tables/flips.md` — the flip rate per (model, think, lang).
2. Lead with the per-language flip rate (the concept-vs-wording read): a high
   flip rate means the model's pick changed when *only the answer language*
   changed → evidence it keyed on English **wording**, not the concept.
3. Check the direction split — a net move **away from gold** is the strongest
   wording-reliance signal. Qualify every claim with the McNemar `p`.
4. Use `fig2` (headline) + `fig3` (direction) in any write-up; reserve absolute
   accuracy (`fig1`) for context, not the headline.

## Things this skill does NOT do

- Compute metrics — accuracy/Wilson/flip-rate/McNemar live in
  `src/csqa_xlang/analysis/`; fix numbers there. This skill only presents the CSVs.
- Touch the cluster (SSH/submit/sync) or read raw `outputs.jsonl` — that's
  `cluster-ops` + `scripts.analyze`. Compose the two via the standard recipe.
- Commit anything under `results/**` (git-ignored, derived — rebuilds from CSV +
  scripts).
