"""Generate a parallel MT translation set + en-x variants for a free/local backend.

A second/third translation source alongside Google, to test whether the en->x
degradation replicates across MT systems (or is one system's artifact). For the
chosen ``--backend`` (nllb | opus) it writes, per target language:
  data/translated/<backend>/choices_<lang>.json   # backend-translated choices
  data/variants/en-x__<lang>-<backend>.json         # eval variant, language="<lang>-<backend>"

The en-en baseline is translation-source-independent and already built; the
<lang>-<backend> variants are compared against it by scripts.analyze exactly like
the Google en-x variants, so one model tag yields all sources side by side.

Usage:
    python -m scripts.build_mt --backend nllb
    python -m scripts.build_mt --backend opus
    python -m scripts.build_mt --backend opus --limit 50      # quick check
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from csqa_xlang.data import load_csqa
from csqa_xlang.translation import NLLBTranslator, OpusMTTranslator, translate_choices
from csqa_xlang.variants import build_variant, write_variant

LANGS = ["ru", "es", "he"]
# name -> factory(cache_dir) -> translator. nllb33 is the full NLLB-200-3.3B
# (fp16) with its own cache tag so it doesn't collide with the distilled 600M.
BACKENDS = {
    "nllb": lambda cd: NLLBTranslator(cache_dir=cd),
    "nllb33": lambda cd: NLLBTranslator(cache_dir=cd, model="facebook/nllb-200-3.3B",
                                        cache_tag="nllb33"),
    "opus": lambda cd: OpusMTTranslator(cache_dir=cd),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--translated-root", default="data/translated")
    ap.add_argument("--variants-dir", default="data/variants")
    args = ap.parse_args()

    en_items = load_csqa(args.split, limit=args.limit)
    tdir = Path(args.translated_root) / args.backend
    tdir.mkdir(parents=True, exist_ok=True)
    vdir = Path(args.variants_dir)
    translator = BACKENDS[args.backend](Path(args.translated_root) / "_cache")
    print(f"{len(en_items)} English items; {args.backend} -> {LANGS}")

    for lang in LANGS:
        tr = translate_choices(en_items, lang, translator)
        (tdir / f"choices_{lang}.json").write_text(
            json.dumps([asdict(it) for it in tr], ensure_ascii=False, indent=2),
            encoding="utf-8")
        v = build_variant("en-x", f"{lang}-{args.backend}", en_items, choice_x=tr)
        write_variant(v, vdir / f"en-x__{lang}-{args.backend}.json")
        print(f"  {lang}: choices_{lang}.json + en-x__{lang}-{args.backend}.json (n={len(tr)})")


if __name__ == "__main__":
    main()
