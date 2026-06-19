# Model card — mBERT CSQA, epoch-3

| field | value |
|---|---|
| checkpoint dir | `checkpoints/mbert-csqa-epoch3/` (git-ignored; local only) |
| weights file | `mc_model.pt` (+ `arch.json` = `{"base":"bert-base-multilingual-cased"}`) |
| sha256 | `8dbf7b104a4077de24e9302371a272b4f985d6428c61eb00dab76cffe8720474` |
| base model | `bert-base-multilingual-cased` (104 langs incl. es/ru/he; ~178M params) |
| head | custom CLS → Dropout → `Linear(768, 1)`, softmax over 5 choices (segment ids used) |
| training data | English CSQA train, **same 80% split as XLM-R** = 7793 items (seed 42) |
| dev / test split | 974 / 974 (held out; disjoint) — `training/xlmr_csqa/splits/` |
| epochs | 3 |
| batch size / lr / maxlen | 16 / 1e-5 / 96 |
| **English dev acc** | **0.4692** (best-on-dev = epoch 2) |
| **English test acc** | **0.4733** |

## Note
Best mBERT checkpoint: dev/test peak at epoch-3 (0.469 / 0.473); the warm-start
continuation to epoch-6 did **not** help (dev dropped to 0.454). mBERT converges
faster and plateaus lower than XLM-R.

## Reproduce
```bash
python -m training.xlmr_csqa.train --base bert-base-multilingual-cased \
  --epochs 3 --ckpt checkpoints/mbert-csqa-epoch3
python -m scripts.run_eval --arm mbert --ckpt checkpoints/mbert-csqa-epoch3 --model-tag mbert-ep3
python -m scripts.analyze
```
