# Model card — XLM-R CSQA, epoch-3

| field | value |
|---|---|
| checkpoint dir | `checkpoints/xlmr-csqa-epoch3/` (git-ignored; 1.1 GB) |
| weights file | `mc_model.pt` |
| sha256 | `1cf41785239003ad19e1dea50fc7c06b234ae08001c0d6297ac4ac310c055c94` |
| base model | `xlm-roberta-base` (277.5M params) |
| head | custom CLS → Dropout → `Linear(768, 1)`, softmax over 5 choices |
| training data | English CSQA train, 80% split = 7793 items (seed 42) |
| dev / test split | 974 / 974 (held out; disjoint) |
| epochs | 3 | 
| batch size / lr / maxlen | 16 / 1e-5 / 96 |
| **English dev acc** | **0.4415** |
| **English test acc** | **0.4733** |

## Why this checkpoint exists
The "before" reference for the soundness check: trained 3 epochs from pretrained
`xlm-roberta-base`, then frozen before continuing to epoch-6. Used to verify the
en→x degradation pattern is stable across training length.

## Reproduce
```bash
python -m training.xlmr_csqa.make_split          # seed 42 → splits/
python -m training.xlmr_csqa.train --epochs 3 --ckpt checkpoints/xlmr-csqa-epoch3
```
Note: exact weights may not be bit-reproducible (CUDA nondeterminism); the
accuracy figures are stable to ~±0.5pt.

## Cross-lingual evaluation (validation, n=1221)
See `summary_xlmr.csv` / `flips_xlmr.csv` (rows `xlmr-ep3`). Regenerate:
```bash
python -m scripts.run_eval --arm xlmr --ckpt checkpoints/xlmr-csqa-epoch3 --model-tag xlmr-ep3
python -m scripts.analyze
```
