"""Stage 1: load CSQA → translate answer choices → cache to data/translated/.

Writes one file per target language (choices translated, question English):
    data/translated/choices_<lang>.json   (list of CSQAItem dicts)
and, only when extensions.translate_questions is set, the symmetric
    data/translated/questions_<lang>.json  (question translated, choices English)
for the x-x / x-en conditions. English is the untouched base (loaded directly).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from csqa_xlang.config import load_config
from csqa_xlang.data import load_csqa
from csqa_xlang.translation import get_translator, translate_choices, translate_questions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    split = cfg["dataset"]["split"]
    targets = cfg["languages"]["targets"]
    out_dir = Path(cfg.get("paths", {}).get("data_translated", "data/translated"))
    out_dir.mkdir(parents=True, exist_ok=True)
    translate_questions_ext = bool((cfg.get("extensions") or {}).get("translate_questions", False))

    en_items = load_csqa(split, limit=args.limit)
    print(f"{len(en_items)} English items; targets={targets}")
    translator = get_translator(cfg.raw, cache_dir=out_dir / "_cache")

    def dump(items, path):
        Path(path).write_text(
            json.dumps([asdict(it) for it in items], ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"  wrote {path}")

    for lang in targets:
        dump(translate_choices(en_items, lang, translator), out_dir / f"choices_{lang}.json")
        if translate_questions_ext:
            dump(translate_questions(en_items, lang, translator), out_dir / f"questions_{lang}.json")


if __name__ == "__main__":
    main()
