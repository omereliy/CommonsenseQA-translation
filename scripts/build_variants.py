"""Stage 2: assemble condition variants → data/variants/<condition>__<lang>.json.

Builds the core en-en / en-x for every target. x-x / x-en are built only when
extensions.translate_questions is set and they appear in cfg['conditions']
(they need data/translated/questions_<lang>.json from stage 1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from csqa_xlang.config import load_config
from csqa_xlang.data import CSQAItem, load_csqa
from csqa_xlang.variants import build_variant, write_variant
from csqa_xlang.variants.conditions import QUESTION_TRANSLATED


def _load_items(path: Path) -> list[CSQAItem]:
    return [CSQAItem(**d) for d in json.loads(path.read_text(encoding="utf-8"))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    split = cfg["dataset"]["split"]
    targets = cfg["languages"]["targets"]
    conditions = cfg.get("conditions", ["en-en", "en-x"])
    tdir = Path(cfg.get("paths", {}).get("data_translated", "data/translated"))
    vdir = Path(cfg.get("paths", {}).get("data_variants", "data/variants"))

    en_items = load_csqa(split, limit=args.limit)
    n = 0
    for cond in conditions:
        if cond == "en-en":
            v = build_variant("en-en", "en", en_items)
            write_variant(v, vdir / "en-en__en.json"); n += 1
            continue
        for lang in targets:
            choice_x = _load_items(tdir / f"choices_{lang}.json")
            if args.limit:
                choice_x = choice_x[:args.limit]
            question_x = None
            if cond in QUESTION_TRANSLATED:
                qpath = tdir / f"questions_{lang}.json"
                if not qpath.exists():
                    print(f"  skip {cond}/{lang}: {qpath} missing "
                          f"(enable extensions.translate_questions in stage 1)")
                    continue
                question_x = _load_items(qpath)[:args.limit] if args.limit else _load_items(qpath)
            v = build_variant(cond, lang, en_items, choice_x=choice_x, question_x=question_x)
            write_variant(v, vdir / f"{cond}__{lang}.json"); n += 1
    print(f"wrote {n} variants -> {vdir}")


if __name__ == "__main__":
    main()
