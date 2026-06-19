# XLM-R cross-lingual analysis — frozen snapshot (2026-06-19)

Immutable record of the XLM-R encoder arm for the final report. The live
`results/{summary,flips}.csv` are a rolling aggregate that `scripts.analyze`
overwrites on every run (and re-mixes with other arms); the CSVs **here** are the
frozen xlmr-only slice, paired with model cards, so they survive later runs.

**Arm:** `xlm-roberta-base` fine-tuned on **English** CSQA (zero-shot cross-lingual
transfer protocol), evaluated on the en-en baseline and the en-x answer-translated
variants (ru/es/he). Question stays English; only answer choices translated; gold
label fixed. Selection is letter-based (A–E). Eval split = validation (n=1221).

## Files
- `summary_xlmr.csv` — accuracy + Wilson 95% CI per (model, condition, lang).
- `flips_xlmr.csv` — flip rate + direction (toward/away gold) + McNemar p, en-en→en-x.
- `model_card_epoch3.md`, `model_card_epoch6.md` — checkpoint provenance (sha256, config, dev/test acc, reproduce commands).

## Headline: accuracy (validation, n=1221)
| Condition | epoch-3 | epoch-6 | rel. drop vs en-en (ep3 → ep6) |
|---|---|---|---|
| en-en | 0.483 | 0.510 | — |
| en-x **ru** | 0.401 | 0.424 | −17.1% → −16.9% |
| en-x **es** | 0.390 | 0.408 | −19.3% → −20.0% |
| en-x **he** | 0.341 | 0.358 | **−29.5% → −29.9%** |

## Flips + significance (en-en → en-x)
| lang | flip ep3 | flip ep6 | →gold | ←gold (ep6) | McNemar p |
|---|---|---|---|---|---|
| ru | 0.542 | 0.513 | 155 | 260 | ≈0 |
| es | 0.508 | 0.496 | 144 | 269 | ≈0 |
| he | **0.575** | **0.547** | 126 | **312** | ≈0 |

## Conclusion (for the report)
Translating **only the answer choices** drops accuracy by ~17% (ru), ~20% (es),
~30% (he). The ordering **en > ru ≈ es > he** and the magnitudes are **stable
across 3 vs 6 epochs**, every delta is McNemar-significant (p≈0), and flips are
net harmful (`←gold ≫ →gold`, most lopsided for he). This is evidence the model
partly anchors on **English answer wording** rather than the language-agnostic
**concept** — strongest for Hebrew (non-Latin script, most distant from English,
noisiest MT). The +2–3 pt uniform lift from epoch-3→6 raises all conditions
without closing the gap, so the effect is not an undertraining artifact.

Caveat for the writeup: the degradation conflates concept-anchoring with XLM-R's
weaker representation for he vs es/ru and noisier Hebrew MT.
