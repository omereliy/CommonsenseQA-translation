# mBERT cross-lingual analysis — frozen snapshot (2026-06-19)

Immutable record of the mBERT encoder arm for the final report — the
multilingual-encoder counterpart to XLM-R (see `../xlmr_2026-06-19/`). Same
protocol, **same 80-10-10 English split** (`training/xlmr_csqa/splits/`), so the
two encoders are directly comparable.

**Arm:** `bert-base-multilingual-cased` (104 languages incl. es/ru/he) fine-tuned
on **English** CSQA, evaluated on en-en + en-x (ru/es/he). Question stays English;
only answer choices translated; gold fixed; letter-based selection; validation
split (n=1221).

## Files
- `summary_mbert.csv` — accuracy + Wilson 95% CI per (model, condition, lang).
- `flips_mbert.csv` — flip rate + direction + McNemar p, en-en→en-x.
- `model_card_epoch3.md`, `model_card_epoch6.md` — provenance (sha256, dev/test acc).

## Accuracy (validation, n=1221)
| Condition | ep3 | ep6 | rel. drop vs en-en (ep3 / ep6) |
|---|---|---|---|
| en-en | 0.459 | 0.457 | — |
| en-x **ru** | 0.380 | 0.389 | −17.1% / −14.9% |
| en-x **es** | 0.377 | 0.380 | −17.9% / −16.8% |
| en-x **he** | 0.338 | 0.344 | **−26.3% / −24.7%** |

## Flips + significance (en-en → en-x)
| lang | flip ep3 | flip ep6 | →gold | ←gold (ep6) | McNemar p |
|---|---|---|---|---|---|
| ru | 0.507 | 0.551 | 172 | 255 | 7e-5 |
| es | 0.507 | 0.537 | 165 | 259 | 1e-5 |
| he | **0.593** | **0.608** | 161 | **299** | ≈0 |

## Conclusion (for the report)
mBERT reproduces the **same degradation ordering as XLM-R**: en > ru ≈ es > he,
with he worst (~25–26% relative drop), every en-en→en-x delta McNemar-significant,
and flips net-harmful (`←gold ≫ →gold`, most lopsided for he). That two independent
multilingual encoders show the same pattern strengthens the claim that it is a
property of cross-lingual concept grounding, not a single model's quirk.

Two differences vs XLM-R worth reporting:
1. **mBERT is weaker** (en-en 0.46 vs XLM-R 0.51) and **more unstable** (higher
   flip rates) — consistent with XLM-R being the stronger multilingual encoder.
2. **mBERT plateaus early**: ep3→ep6 gives no gain (it slightly overtrains),
   whereas XLM-R gains +2–3 pts. So mBERT's best checkpoint is epoch-3.
