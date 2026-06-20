"""Build the MIXED-language answer variant.

Each item keeps its English question, but every answer choice (A-E) is
*independently* drawn — with probability 1/3 each — from one of the three target
languages {ru, he, es}. So within a single 5-way item the options can be a
mixture of Russian, Hebrew and Spanish. This isolates whether a model is
grounding on the concept (robust to per-option language) or on surface wording.

Output:
  data/mixed__mix.json         — a standard Variant file (condition="mixed",
                                 language="mix"); consumable by load_variant().
  data/mixed_assignment.json   — sidecar {id: {label: lang}} recording which
                                 language each option was drawn from, for the
                                 per-language breakdown in analyze_mixed.py.

Deterministic: the per-option language draw is seeded from cfg.seed (load-bearing
reproducibility convention, CLAUDE.md). Re-running reproduces the exact mixture.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from csqa_xlang.config import load_config
from csqa_xlang.data import CSQAItem
from csqa_xlang.variants import write_variant
from csqa_xlang.variants.conditions import Variant

LANGS = ["ru", "he", "es"]
HERE = Path(__file__).resolve().parent


def _load(path: Path) -> dict[str, CSQAItem]:
    items = [CSQAItem(**d) for d in json.loads(path.read_text(encoding="utf-8"))]
    return {it.id: it for it in items}


def main() -> None:
    cfg = load_config("configs/default.yaml")          # seeds RNGs from cfg.seed
    seed = cfg.seed
    tdir = Path(cfg.get("paths", {}).get("data_translated", "data/translated"))

    by_lang = {lang: _load(tdir / f"choices_{lang}.json") for lang in LANGS}
    ids = list(by_lang["ru"].keys())                   # canonical order/ids
    for lang in LANGS:
        assert set(by_lang[lang]) == set(ids), f"id mismatch in choices_{lang}.json"

    # A dedicated RNG so the draw is independent of any other library RNG use.
    rng = random.Random(seed)

    items: list[CSQAItem] = []
    assignment: dict[str, dict[str, str]] = {}
    lang_counts = {lang: 0 for lang in LANGS}
    for cid in ids:
        base = by_lang["ru"][cid]                       # English question + key
        choices, assign = {}, {}
        for label in sorted(base.choices):              # A..E, stable order
            lang = rng.choice(LANGS)                    # 1/3 each, independent
            choices[label] = by_lang[lang][cid].choices[label]
            assign[label] = lang
            lang_counts[lang] += 1
        items.append(CSQAItem(id=cid, question=base.question,
                              choices=choices, answer_key=base.answer_key))
        assignment[cid] = assign

    variant = Variant("mixed", "mix", items)
    vpath = write_variant(variant, HERE / "data" / "mixed__mix.json")
    (HERE / "data" / "mixed_assignment.json").write_text(
        json.dumps(assignment, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(lang_counts.values())
    print(f"seed={seed}  items={len(items)}  variant_hash={variant.variant_hash[:8]}")
    print("per-option language share (target 1/3 each):")
    for lang in LANGS:
        print(f"  {lang}: {lang_counts[lang]:5d}  ({lang_counts[lang]/total:.3f})")
    print(f"wrote {vpath}")
    print(f"wrote {HERE / 'data' / 'mixed_assignment.json'}")


if __name__ == "__main__":
    main()
