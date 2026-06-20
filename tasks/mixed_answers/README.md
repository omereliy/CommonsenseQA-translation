# Task: mixed-language answer choices (encoder arms)

**Question.** When the five answer choices of one item are a *mixture* of
languages — each option independently Russian, Hebrew, or Spanish (1/3 each) —
does a model still pick the right concept, or does its success depend on which
language the correct option happens to be written in?

English question is kept throughout; only the choices are translated (the
load-bearing convention in `CLAUDE.md`). This is the `mixed` extension from the
condition table, realized per-*option* rather than per-item.

## Pipeline

```bash
python tasks/mixed_answers/build_mixed.py     # -> data/mixed__mix.json (+ assignment sidecar)
python tasks/mixed_answers/run_mixed.py       # XLM-R + mBERT (epoch6 ckpts) -> results/
python tasks/mixed_answers/analyze_mixed.py   # -> mixed_{summary,flips,by_gold_lang}.csv
```

- `build_mixed.py` — draws each option's language with `random.Random(cfg.seed)`
  (seed=42, reproducible) from `data/translated/choices_{ru,he,es}.json`; writes a
  standard `Variant` (`condition="mixed"`, `language="mix"`) plus
  `mixed_assignment.json` = `{id: {label: lang}}`.
- `run_mixed.py` — reuses the same encoder MC arms + `write_run` as
  `scripts/run_eval.py`; tags runs `xlmr-ep6` / `mbert-ep6` so they compare
  directly to the canonical baselines under `results/`.
- `analyze_mixed.py` — accuracy (+ Wilson CI), flip rate / direction / McNemar
  vs the `en-en` baseline, and a breakdown of mixed accuracy by the language the
  **gold** option was drawn in. Reads cached outputs only. CSVs carry exact
  counts (`correct`, `flips`) alongside the rates.
- `plot_mixed.py` — reads the three CSVs → PNGs in `report/figures/` (fig16–19,
  following the report's sequential naming). Every bar is annotated with both its
  percentage **and** its raw count.

## Figures → `report/figures/`

```bash
python tasks/mixed_answers/plot_mixed.py
```

- `fig16_mixed_accuracy_by_condition` — accuracy for en-en, en-x ru/he/es, MIXED,
  per model, with Wilson CIs; bars labelled `acc% · correct/n`.
- `fig17_mixed_flip_rate` — flip rate en-en → each condition; bars labelled
  `flip% · flipped/n`.
- `fig18_mixed_flip_direction` — stacked toward-gold (green) vs away-from-gold
  (red) flip counts; segments labelled `count · %`.
- `fig19_mixed_by_gold_lang` — MIXED accuracy split by the gold option's language
  (the headline figure); bars labelled `acc% · correct/n`.

Per-option language share in the built variant: ru .323 / he .332 / es .345.

## Results (validation, n=1221)

| model | en-en | en-x ru | en-x he | en-x es | **mixed** |
|-------|------:|--------:|--------:|--------:|----------:|
| XLM-R  | 0.510 | 0.424 | 0.358 | 0.408 | **0.385** |
| mBERT  | 0.457 | 0.389 | 0.344 | 0.380 | **0.372** |

Flip rate `en-en → mixed`: XLM-R 0.530, mBERT 0.577 — on par with the
single-language flips (0.50–0.61) and overwhelmingly **away** from gold
(XLM-R 291 away / 138 toward; mBERT 276 / 172). All McNemar p < 0.001.

Mixed accuracy by the **gold option's** language:

| model | gold ru | gold he | gold es |
|-------|--------:|--------:|--------:|
| XLM-R | 0.386 | 0.355 | 0.414 |
| mBERT | 0.378 | 0.282 | 0.456 |

## Takeaways

1. **Mixed ≈ the mean of the single-language conditions.** XLM-R mixed 0.385 vs
   mean(ru,he,es)=0.397; mBERT mixed 0.372 vs mean 0.371. Mixing languages within
   an item adds no extra penalty beyond translating the choices at all.
2. **Success tracks the gold option's language**, reproducing the en>es≈ru>he
   ordering *within* a single mixed item (mBERT: 0.456 when the right answer is
   Spanish vs 0.282 when Hebrew — a ~17-pt gap). The model is not doing
   language-invariant concept grounding; it grounds on per-option wording, and
   Hebrew script/morphology is where the encoders slip most (consistent with the
   en→he degradation seen elsewhere in the project).
3. Heavy away-from-gold flips confirm the drop is the choice language changing the
   prediction, not noise.
