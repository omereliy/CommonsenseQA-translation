# `training/xlmr_csqa/` — XLM-R fine-tune (English CSQA)

Self-contained training run for the **XLM-R cross-lingual transfer arm**. It
fine-tunes `xlm-roberta-base` on **English** CommonsenseQA and writes the
checkpoint the eval reads from (`checkpoints/xlmr-csqa`).

## Why English-only

The research question is whether the model grounds the **concept** or the
**English word**. We fine-tune on English only (the model never sees translated
choices), then evaluate the `en-x` variants. Because training was English-only,
`en-x` accuracy measures **zero-shot cross-lingual transfer** — how much grounding
survives when the *answer choices* switch to ru / es / he. This mirrors the
standard XLM-R transfer protocol (Conneau et al. 2020) and CSQA's own end-to-end
fine-tune baseline (Talmor et al. 2019); it is a **full fine-tune** (all ~270M
params + one `Linear(768→1)` multiple-choice head), not a frozen-head probe.

## The 80-10-10 split (this directory)

`make_split.py` carves the **9741-item English train set** into:

| split | ~size | role |
|-------|-------|------|
| `train` (80%) | ~7793 | fine-tune |
| `dev`   (10%) | ~974  | per-epoch eval → keep the best-accuracy epoch |
| `test`  (10%) | ~974  | English in-domain sanity check after training |

Deterministic: items are sorted by `id` then shuffled with a fixed seed, so the
split is byte-identical on every machine. Output: `splits/{train,dev,test}.jsonl`
(git-ignored — regenerate with the script).

> **Not** the cross-lingual eval set. The official CSQA **`validation`** split
> (1221 items) is untouched here; the en-en ↔ en-x flip evaluation runs on it via
> `scripts.run_eval` (load-bearing convention — see `CLAUDE.md`). Keeping the two
> separate prevents train/eval leakage.

## Run

```bash
# 1. build the split (once; deterministic)
python -m training.xlmr_csqa.make_split

# 2. full fine-tune -> checkpoints/xlmr-csqa  (~20-40 min on an RTX 4070)
python -m training.xlmr_csqa.train
#    smoke test first:  python -m training.xlmr_csqa.train --epochs 1 --limit 200

# 3. cross-lingual eval: en-en + en-x__{es,he,ru} -> results/ + manifests
python -m scripts.run_eval --arm xlmr --ckpt checkpoints/xlmr-csqa

# 4. accuracy + flip rate (en-en -> en-x) + McNemar
python -m scripts.analyze
```

`train.py` selects the best epoch on `dev`, saves to `checkpoints/xlmr-csqa`, and
prints the held-out **English test** accuracy. Defaults: 3 epochs, batch 16,
lr 1e-5, maxlen 96 (see `src/csqa_xlang/eval/xlmr.py`).
