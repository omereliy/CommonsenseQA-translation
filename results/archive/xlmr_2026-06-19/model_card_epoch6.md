# Model card — XLM-R CSQA, epoch-6 (warm-start continuation)

| field | value |
|---|---|
| checkpoint dir | `checkpoints/xlmr-csqa-epoch6/` (git-ignored; 1.1 GB) |
| weights file | `mc_model.pt` |
| sha256 | `519540d9b6e5232db387be65f139e9254b23047c0e42458bf5621fc2c202f270` |
| base model | `xlm-roberta-base` (277.5M params) |
| head | custom CLS → Dropout → `Linear(768, 1)`, softmax over 5 choices |
| training data | English CSQA train, 80% split = 7793 items (seed 42) |
| dev / test split | 974 / 974 (held out; disjoint) |
| epochs | 6 total = epoch-3 + **3 continued** (warm-start, LR schedule restarted) |
| saved weights | **best-on-dev = total epoch 5** (dev dipped at epoch 6) |
| batch size / lr / maxlen | 16 / 1e-5 / 96 |
| **English dev acc** | **0.4877** |
| **English test acc** | **0.5072** |

## Why this checkpoint exists
The "after" point of the soundness check. Continued 3 more epochs from epoch-3 to
test whether more training changes the cross-lingual picture. Dev accuracy per
continued epoch: 0.462 (ep4) → **0.4877 (ep5, kept)** → 0.4815 (ep6) — i.e. it
peaked at ~5 epochs. All conditions rose ~uniformly (+2–3 pts) vs epoch-3 without
closing the en→x gap, confirming the degradation is not an undertraining artifact.

## Reproduce
```bash
python -m training.xlmr_csqa.train --epochs 3 \
  --init-from checkpoints/xlmr-csqa-epoch3 --ckpt checkpoints/xlmr-csqa-epoch6
```

## Cross-lingual evaluation (validation, n=1221)
See `summary_xlmr.csv` / `flips_xlmr.csv` (rows `xlmr-ep6`). Regenerate:
```bash
python -m scripts.run_eval --arm xlmr --ckpt checkpoints/xlmr-csqa-epoch6 --model-tag xlmr-ep6
python -m scripts.analyze
```
