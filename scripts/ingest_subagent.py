"""Assemble Claude-Code subagent answers into a scored run (outputs.jsonl + manifest).

The Haiku-subagent arm answers items off-pipeline (subscription-billed agents,
batched), producing an {id: letter} map. This turns that map into the same
Prediction records every other arm writes, so scripts.analyze computes accuracy +
flip rate identically. The manifest flags the run as NON-deterministic
(subscription subagent; temperature not pinned) — read it as an exploratory arm,
not a bit-reproducible one (unlike the encoder/vLLM/Gemini arms).

Usage:
  python -m scripts.ingest_subagent \
      --variant data/variants/en-en__en.json \
      --answers /tmp/csqa_batches/en-en/answers.json \
      --model-tag haiku-4.5-subagent --model claude-haiku-4-5 --batch-size 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from csqa_xlang.eval import PROMPT_TEMPLATE, Prediction, score, write_run
from csqa_xlang.eval.prompt import LABELS
from csqa_xlang.variants import load_variant


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--answers", required=True, help="JSON {id: letter} (or {answers:[{id,letter}]})")
    ap.add_argument("--model-tag", default="haiku-4.5-subagent")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    raw = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "answers" in raw:
        raw = raw["answers"]
    if isinstance(raw, list):  # [{id,letter}, ...]
        amap = {r["id"]: r.get("letter") for r in raw}
    else:                      # {id: letter}
        amap = dict(raw)

    variant = load_variant(Path(args.variant))
    preds: list[Prediction] = []
    for it in variant.items:
        letter = (amap.get(it.id) or "").strip().upper()[:1]
        pred = letter if letter in LABELS else None
        preds.append(Prediction(id=it.id, gold=it.answer_key, pred=pred,
                                correct=score(pred, it.answer_key)))

    n = len(preds)
    answered = sum(p.pred is not None for p in preds)
    acc = sum(p.correct for p in preds) / n if n else 0.0
    rd = write_run(
        args.results_dir, model_tag=args.model_tag, model_snapshot=args.model,
        variant=variant, predictions=preds, prompt_template=PROMPT_TEMPLATE,
        think=None,
        decoding={"provider": "claude-code-subagent", "model": args.model,
                  "deterministic": False, "temperature": None,
                  "batch_size": args.batch_size,
                  "note": "subscription subagent; temperature not pinned"},
    )
    print(f"{args.model_tag} {variant.condition}/{variant.language}: "
          f"acc={acc:.3f}  answered={answered}/{n}  unparsed={n - answered}")
    print(f"wrote {rd}")


if __name__ == "__main__":
    main()
