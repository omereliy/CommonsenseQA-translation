"""Deterministic 80-10-10 split of the ENGLISH CommonsenseQA *train* set.

This split is the XLM-R fine-tune's *own* train / dev / test — it is NOT the
cross-lingual evaluation set. The official ``validation`` split (1221 items) is
left untouched; that is what the en-en / en-x flip eval runs on (load-bearing
convention, see CLAUDE.md). Here we only carve the 9741-item train set so the
fine-tune has a held-out dev (model selection) and an English in-domain test.

Reproducibility: items are sorted by id (canonical order, independent of HF's
ordering) and shuffled with a fixed seed, so the split is byte-identical on every
machine. Re-run to regenerate ``splits/{train,dev,test}.jsonl``.

Usage:
    python -m training.xlmr_csqa.make_split            # seed 42, 80/10/10
    python -m training.xlmr_csqa.make_split --seed 7 --out training/xlmr_csqa/splits
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from csqa_xlang.data import load_csqa

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", default="0.8,0.1,0.1",
                    help="train,dev,test fractions (must sum to 1.0)")
    ap.add_argument("--out", default=str(HERE / "splits"))
    args = ap.parse_args()

    r_train, r_dev, r_test = (float(x) for x in args.ratios.split(","))
    assert abs(r_train + r_dev + r_test - 1.0) < 1e-9, "ratios must sum to 1.0"

    items = load_csqa("train")                 # English train, 9741 items
    items.sort(key=lambda it: it.id)           # canonical order before shuffle
    random.Random(args.seed).shuffle(items)

    n = len(items)
    n_train = round(r_train * n)
    n_dev = round(r_dev * n)
    splits = {
        "train": items[:n_train],
        "dev": items[n_train:n_train + n_dev],
        "test": items[n_train + n_dev:],
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for it in rows:
                f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
        print(f"{name:5s} {len(rows):5d} ({len(rows)/n:.1%}) -> {path}")
    print(f"total {n} items, seed={args.seed}, ratios={args.ratios}")


if __name__ == "__main__":
    main()
