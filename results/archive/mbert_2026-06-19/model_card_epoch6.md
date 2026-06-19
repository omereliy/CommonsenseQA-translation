# Model card — mBERT CSQA, epoch-6 (warm-start continuation)

| field | value |
|---|---|
| checkpoint dir | `checkpoints/mbert-csqa-epoch6/` (git-ignored; local only) |
| weights file | `mc_model.pt` (+ `arch.json` = `{"base":"bert-base-multilingual-cased"}`) |
| sha256 | `46d659dbd76f627a10ad0f84e8c00bc573e21562f50c9ea6dd906bb7a7064aec` |
| base model | `bert-base-multilingual-cased` (104 langs incl. es/ru/he; ~178M params) |
| head | custom CLS → Dropout → `Linear(768, 1)`, softmax over 5 choices (segment ids used) |
| training data | English CSQA train, **same 80% split as XLM-R** = 7793 items (seed 42) |
| dev / test split | 974 / 974 (held out; disjoint) — `training/xlmr_csqa/splits/` |
| epochs | 6 total = epoch-3 + **3 continued** (warm-start, LR schedule restarted) |
| saved weights | best-on-dev of continued epochs (dev 0.462→0.466→**0.454 best**) |
| batch size / lr / maxlen | 16 / 1e-5 / 96 |
| **English dev acc** | **0.4538** |
| **English test acc** | **0.4528** |

## Note
The continuation did **not** improve mBERT — dev/test are slightly *below* epoch-3
(0.454/0.453 vs 0.469/0.473). mBERT had already converged by ~epoch 2–3; the extra
epochs mildly overtrained. Contrast with XLM-R, which gained +2–3 pts from ep3→ep6.
Kept for the same soundness comparison: the en→x degradation ordering is unchanged.

## Reproduce
```bash
python -m training.xlmr_csqa.train --base bert-base-multilingual-cased --epochs 3 \
  --init-from checkpoints/mbert-csqa-epoch3 --ckpt checkpoints/mbert-csqa-epoch6
python -m scripts.run_eval --arm mbert --ckpt checkpoints/mbert-csqa-epoch6 --model-tag mbert-ep6
python -m scripts.analyze
```
