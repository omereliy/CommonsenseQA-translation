"""Stage 3: evaluate a model arm over the condition variants → results/ (+ manifest).

Arms:
  generative  vLLM-served Qwen lineup. Needs a served endpoint: --base-url or
              $LLM_BASE_URL. On the cluster, one array task = one (model, think)
              cell supplies --model-tag/--model-hf/--think; locally it iterates
              cfg['eval']['generative']['models'] x think_modes.
  xlmr        fine-tuned XLM-R checkpoint (--ckpt); evaluates all variants.
  esim        English-only anchor; evaluates the en-en variant only.

A completed (model, variant) run is a cache hit (skipped unless --force).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from csqa_xlang.config import load_config
from csqa_xlang.eval import PROMPT_TEMPLATE, esim, is_cached, run_generative, write_run, xlmr
from csqa_xlang.variants import load_variant


def plan_variants(cfg, vdir: Path):
    conditions = cfg.get("conditions", ["en-en", "en-x"])
    targets = cfg["languages"]["targets"]
    plan = []
    for cond in conditions:
        if cond == "en-en":
            plan.append(vdir / "en-en__en.json")
        else:
            plan += [vdir / f"{cond}__{lang}.json" for lang in targets]
    return [load_variant(p) for p in plan if p.exists()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--arm", choices=["generative", "xlmr", "esim"], default="generative")
    ap.add_argument("--model-tag")
    ap.add_argument("--model-hf")
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL"))
    ap.add_argument("--think", choices=["on", "off"])
    ap.add_argument("--ckpt")
    ap.add_argument("--results-dir")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    vdir = Path(cfg.get("paths", {}).get("data_variants", "data/variants"))
    results_root = args.results_dir or cfg.get("paths", {}).get("results", "results")
    dec = (cfg.get("eval") or {}).get("decoding", {})
    temp = dec.get("temperature", 0)
    variants = plan_variants(cfg, vdir)
    if not variants:
        raise SystemExit(f"no variants under {vdir} — run scripts.build_variants first")

    def items_of(v):
        return v.items[:args.limit] if args.limit else v.items

    if args.arm == "generative":
        if not args.base_url:
            raise SystemExit("generative arm needs --base-url or $LLM_BASE_URL (a served vLLM endpoint)")
        gen_cfg = (cfg["eval"].get("generative") or {})
        if args.model_hf:
            models = [{"tag": args.model_tag or args.model_hf, "hf": args.model_hf}]
        else:
            models = gen_cfg.get("models", [])
        thinks = [args.think] if args.think else (cfg["eval"].get("think_modes") or ["off"])
        for m in models:
            for think in thinks:
                npredict = dec.get("max_tokens_think_on", 8192) if think == "on" \
                    else dec.get("max_tokens_think_off", 32)
                for v in variants:
                    from csqa_xlang.variants import Variant
                    vv = v if not args.limit else Variant(v.condition, v.language, items_of(v))
                    if is_cached(results_root, m["tag"], vv, think) and not args.force:
                        print(f"  cached: {m['tag']} {v.condition}/{v.language} think={think}")
                        continue
                    preds = run_generative(items_of(v), model_hf=m["hf"], base_url=args.base_url,
                                           think=(think == "on"), num_predict=npredict,
                                           temperature=temp, concurrency=args.concurrency)
                    write_run(results_root, model_tag=m["tag"], model_snapshot=m["hf"],
                              variant=vv, predictions=preds, prompt_template=PROMPT_TEMPLATE,
                              think=think, decoding={"temperature": temp,
                                                     "max_tokens": npredict, "think": think})
                    acc = sum(p.correct for p in preds) / max(len(preds), 1)
                    print(f"  {m['tag']} {v.condition}/{v.language} think={think}: "
                          f"acc={acc:.3f} (n={len(preds)})", flush=True)

    elif args.arm == "xlmr":
        ckpt = args.ckpt or "checkpoints/xlmr-csqa"
        for v in variants:
            from csqa_xlang.variants import Variant
            vv = v if not args.limit else Variant(v.condition, v.language, items_of(v))
            preds = xlmr.predict(items_of(v), ckpt)
            write_run(results_root, model_tag="xlm-roberta-base", model_snapshot="finetuned-en",
                      variant=vv, predictions=preds, prompt_template="xlmr:[q]+[choice] MC-head",
                      think=None, decoding={"type": "mc-head", "argmax": True})
            print(f"  xlmr {v.condition}/{v.language}: "
                  f"acc={sum(p.correct for p in preds) / max(len(preds), 1):.3f}", flush=True)

    elif args.arm == "esim":
        ckpt = args.ckpt or "checkpoints/esim-csqa.pt"
        for v in variants:
            if v.condition != "en-en":
                continue  # English-only anchor
            from csqa_xlang.variants import Variant
            vv = v if not args.limit else Variant(v.condition, v.language, items_of(v))
            preds = esim.predict(items_of(v), ckpt)
            write_run(results_root, model_tag="esim", model_snapshot="glove-en",
                      variant=vv, predictions=preds, prompt_template="esim:bilstm-attention",
                      think=None, decoding={"type": "bilstm", "argmax": True})
            print(f"  esim {v.condition}/{v.language}: "
                  f"acc={sum(p.correct for p in preds) / max(len(preds), 1):.3f}", flush=True)


if __name__ == "__main__":
    main()
