"""Evaluate XLM-R and mBERT on the MIXED-language answer variant.

Reuses the exact encoder MC arms and run-persistence used by scripts/run_eval.py,
so results sit in the same outputs.jsonl + manifest.json layout and are directly
comparable to the existing en-en / en-x ep6 runs. We pin the *epoch6* checkpoints
and tag runs "xlmr-ep6" / "mbert-ep6" to match the canonical baselines under
results/ (the en-en baseline used for flip rates lives there).

  python tasks/mixed_answers/run_mixed.py            # both arms
  python tasks/mixed_answers/run_mixed.py --arm xlmr
"""

from __future__ import annotations

import argparse
from pathlib import Path

from csqa_xlang.config import load_config
from csqa_xlang.eval import write_run
from csqa_xlang.variants import load_variant

HERE = Path(__file__).resolve().parent

ARMS = {
    "xlmr": {"module": "xlmr", "ckpt": "checkpoints/xlmr-csqa-epoch6",
             "tag": "xlmr-ep6", "tmpl": "xlmr:[q]+[choice] MC-head"},
    "mbert": {"module": "mbert", "ckpt": "checkpoints/mbert-csqa-epoch6",
              "tag": "mbert-ep6", "tmpl": "mbert:[q]+[choice] MC-head"},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["xlmr", "mbert", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    load_config("configs/default.yaml")                # seed

    variant = load_variant(HERE / "data" / "mixed__mix.json")
    items = variant.items[: args.limit] if args.limit else variant.items
    results_root = HERE / "results"
    arms = ["xlmr", "mbert"] if args.arm == "both" else [args.arm]

    import importlib
    for arm in arms:
        spec = ARMS[arm]
        mod = importlib.import_module(f"csqa_xlang.eval.{spec['module']}")
        print(f"[{arm}] predicting {len(items)} items with {spec['ckpt']} ...", flush=True)
        preds = mod.predict(items, spec["ckpt"])
        rd = write_run(results_root, model_tag=spec["tag"], model_snapshot="finetuned-en",
                       variant=variant, predictions=preds, prompt_template=spec["tmpl"],
                       think=None, decoding={"type": "mc-head", "argmax": True})
        acc = sum(p.correct for p in preds) / max(len(preds), 1)
        print(f"[{arm}] mixed acc={acc:.4f} (n={len(preds)}) -> {rd}", flush=True)


if __name__ == "__main__":
    main()
