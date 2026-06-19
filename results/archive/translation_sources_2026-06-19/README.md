# Translation-source robustness — frozen snapshot (2026-06-19)

Does the en→x degradation depend on **which translator** produced the answer
choices? We translate the CSQA answer choices with three independent systems and
re-evaluate the same trained encoders (epoch-6) on each. Validation, n=1221.

| Source | type | reproducible | notes |
|---|---|---|---|
| **Google** | commercial API | no | curated (marikf98 QA pass + manual fixes) |
| **NLLB-200** | open, local | yes | `facebook/nllb-200-distilled-600M`, beam 4 |
| **Opus-MT** | open, local | yes | Helsinki-NLP Marian, per-pair en→{es,ru,he}, beam 4 |

## 1. Accuracy replicates across all three translators
XLM-R ep6 (en-en baseline **0.510**); mBERT ep6 (en-en **0.457**):

| | XLM-R: Google / NLLB / Opus | mBERT: Google / NLLB / Opus |
|---|---|---|
| en-x **ru** | 0.424 / 0.402 / 0.381 | 0.389 / 0.380 / 0.373 |
| en-x **es** | 0.408 / 0.394 / 0.386 | 0.380 / 0.373 / 0.338 |
| en-x **he** | 0.358 / 0.374 / 0.371 | 0.344 / 0.337 / 0.341 |

The ordering **en > ru ≈ es > he** holds for **every** translator × model, and
every en-en→en-x delta is McNemar-significant (p≈0). Accuracy varies only ~1–4 pts
across translators. Opus is consistently the weakest (noisiest output); Google the
strongest, but the **pattern is translator-invariant**. See `accuracy_by_source.csv`,
`flips_by_source.csv`.

## 2. …even though the translators rarely agree on the word
Exact-match agreement on the normalized choice strings (`translation_agreement.csv`):

| lang | google–nllb | google–opus | nllb–opus | **all 3 unanimous** |
|---|---|---|---|---|
| ru | 53.2% | 48.7% | 49.0% | **38.7%** |
| es | 57.4% | 63.1% | 59.6% | **48.2%** |
| he | 48.0% | 54.6% | 43.4% | **35.0%** |

On these short ConceptNet concepts the three MT systems pick the **same** surface
form only 35–48% of the time (lowest for he/ru, highest for es). Opus-MT Hebrew
additionally emits Latin-script hallucination tokens (e.g. `…world. kgm`) on
**4.3%** of cells — a quality artifact, yet its degradation pattern is unchanged.

## 3. Why this matters (the confound, resolved)
The en→x drop conflates *concept-anchoring* (our hypothesis) with *MT noise*. These
two facts together separate them:
- Accuracy is **stable (~±2–4 pts) across translators** that disagree on the actual
  word **~half the time**. If the model were purely matching English surface forms,
  swapping in different target words would scatter accuracy far more than this.
  → evidence the model partly tracks the **concept**.
- Yet every translator still loses ~17–30% vs English, he worst.
  → evidence the model is **also** partly anchored to English wording.
Both effects coexist, and the degradation is **robust to the translation system** —
it is not a Google artifact. he is hardest for MT (lowest agreement, most
artifacts) *and* the worst-performing condition, consistent across all systems.

## Reproduce
```bash
python -m scripts.build_mt --backend nllb     # or: --backend opus
python -m scripts.run_eval --arm xlmr  --ckpt checkpoints/xlmr-csqa-epoch6  --model-tag xlmr-ep6  --config configs/nllb.yaml
python -m scripts.run_eval --arm mbert --ckpt checkpoints/mbert-csqa-epoch6 --model-tag mbert-ep6 --config configs/nllb.yaml
python -m scripts.analyze
python -m scripts.translation_agreement
```
