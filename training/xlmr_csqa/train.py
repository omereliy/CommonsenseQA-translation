"""Full fine-tune of a multilingual encoder on the English CSQA train split.

Reads the 80-10-10 split written by ``make_split.py``:
  - train.jsonl  (80%) — fine-tune all encoder params + the MC head
  - dev.jsonl    (10%) — per-epoch eval; best-accuracy epoch is kept
  - test.jsonl   (10%) — English in-domain sanity check after training

English-only training (the cross-lingual transfer protocol): the model never sees
translated choices. The en-x flip evaluation runs separately via
``scripts.run_eval --arm {xlmr,mbert}`` on the official validation variants.

``--base`` selects the encoder (xlm-roberta-base or bert-base-multilingual-cased);
both use the SAME split for a fair comparison.

Usage:
    python -m training.xlmr_csqa.train                                  # XLM-R, 3 epochs
    python -m training.xlmr_csqa.train --base bert-base-multilingual-cased \
        --ckpt checkpoints/mbert-csqa-epoch3                            # mBERT
    python -m training.xlmr_csqa.train --epochs 1 --limit 200           # smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from csqa_xlang.data import CSQAItem
from csqa_xlang.eval import encoder_mc
from csqa_xlang.eval.base import score

HERE = Path(__file__).resolve().parent


def load_split(path: Path, limit: int | None = None) -> list[CSQAItem]:
    items: list[CSQAItem] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            items.append(CSQAItem(id=r["id"], question=r["question"],
                                  choices=r["choices"], answer_key=r["answer_key"]))
            if limit and len(items) >= limit:
                break
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default=str(HERE / "splits"))
    ap.add_argument("--base", default="xlm-roberta-base",
                    help="encoder: xlm-roberta-base | bert-base-multilingual-cased")
    ap.add_argument("--ckpt", default="checkpoints/xlmr-csqa")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--init-from", default=None,
                    help="warm-start from an existing ckpt dir (continue training)")
    ap.add_argument("--limit", type=int, default=None, help="cap items/split (smoke)")
    args = ap.parse_args()

    sd = Path(args.splits)
    train = load_split(sd / "train.jsonl", args.limit)
    dev = load_split(sd / "dev.jsonl", args.limit)
    test = load_split(sd / "test.jsonl", args.limit)
    print(f"base={args.base} train={len(train)} dev={len(dev)} test={len(test)} -> ckpt {args.ckpt}")

    encoder_mc.train(args.ckpt, args.base, train_items=train, eval_items=dev,
                     epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                     init_from=args.init_from)

    # English in-domain sanity check on the held-out test split.
    preds = encoder_mc.predict(test, args.ckpt, args.base)
    acc = sum(score(p.pred, p.gold) for p in preds) / max(len(preds), 1)
    print(f"\nEnglish held-out test accuracy: {acc:.4f} (n={len(preds)})")


if __name__ == "__main__":
    main()
