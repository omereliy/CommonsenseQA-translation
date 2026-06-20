"""Build a CONSENSUS choice-translation set by majority vote across MT systems.

For each answer choice we have three independent translations — Google, NLLB,
Opus. The consensus rule per choice:
  * if >=2 systems agree (on the normalized string), take that majority value;
  * otherwise (all three differ, or only Google is available) **Google breaks
    the tie** — it is the curated/strongest backend and the experiment's default.

Agreement is decided on the *normalized* string (NFC, casefold, niqqud/article
strip — the same normalizer as scripts.translation_agreement), but the emitted
surface form is a real translation: when a majority exists we prefer Google's
surface if Google is in the majority, else NLLB's, else Opus's.

Writes, per target language:
  data/translated/consensus/choices_<lang>.json     # merged choices
  data/variants/en-x__<lang>-consensus.json           # eval variant (language="<lang>-consensus")

Evaluate it like any other source, e.g.:
  python -m scripts.run_eval --arm xlmr  --ckpt checkpoints/xlmr-csqa-epoch6  --model-tag xlmr-ep6  --config configs/consensus.yaml
  python -m scripts.run_eval --arm mbert --ckpt checkpoints/mbert-csqa-epoch6 --model-tag mbert-ep6 --config configs/consensus.yaml
  python -m scripts.analyze

Usage:  python -m scripts.build_consensus
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from csqa_xlang.data import CSQAItem
from csqa_xlang.variants import build_variant, write_variant
from scripts.translation_agreement import normalize

LANGS = ["ru", "es", "he"]
# registry of available translation sources -> file template
SOURCE_PATHS = {
    "google": "data/translated/choices_{lang}.json",
    "nllb": "data/translated/nllb/choices_{lang}.json",
    "nllb33": "data/translated/nllb33/choices_{lang}.json",
    "opus": "data/translated/opus/choices_{lang}.json",
}
# default voters: Google + the STRONG NLLB-3.3B + Opus (one NLLB representative,
# so the vote isn't double-weighted toward NLLB). Google stays first = tie-break.
DEFAULT_SOURCES = ["google", "nllb33", "opus"]


def _load(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}


def _vote(cands: list[tuple[str, str]], lang: str) -> tuple[str, str]:
    """cands = [(source, surface), ...] in priority order (google first).

    Returns (chosen_surface, rule) where rule in {unanimous, majority, google-tiebreak}.
    """
    if len(cands) == 1:
        return cands[0][1], "google-tiebreak"
    norms = [(src, surf, normalize(surf, lang)) for src, surf in cands]
    counts: dict[str, int] = {}
    for _, _, nv in norms:
        counts[nv] = counts.get(nv, 0) + 1
    best_nv, best_c = max(counts.items(), key=lambda kv: kv[1])
    if best_c >= 2:
        # surface from the highest-priority source whose normalized == majority
        surf = next(surf for _, surf, nv in norms if nv == best_nv)
        return surf, ("unanimous" if best_c == len(norms) else "majority")
    # all distinct -> Google (first in priority order) breaks the tie
    return cands[0][1], "google-tiebreak"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES),
                    help="comma list, priority order; first source breaks ties (must include google)")
    ap.add_argument("--tag", default="consensus", help="output tag: data/translated/<tag>, en-x__<lang>-<tag>")
    ap.add_argument("--translated-root", default="data/translated")
    ap.add_argument("--variants-dir", default="data/variants")
    args = ap.parse_args()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    for s in sources:
        if s not in SOURCE_PATHS:
            raise SystemExit(f"unknown source {s!r}; pick from {list(SOURCE_PATHS)}")
    if sources[0] != "google":
        print(f"note: tie-break source is {sources[0]!r} (first in --sources), not google")

    out_root = Path(args.translated_root) / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    vdir = Path(args.variants_dir)
    print(f"consensus tag={args.tag!r} sources={sources}")

    for lang in LANGS:
        loaded = {name: _load(Path(SOURCE_PATHS[name].format(lang=lang)))
                  for name in sources if Path(SOURCE_PATHS[name].format(lang=lang)).exists()}
        anchor = sources[0]
        if anchor not in loaded:
            print(f"[{lang}] no {anchor} source — skipping")
            continue
        ids = list(loaded[anchor])
        rule_counts = {"unanimous": 0, "majority": 0, "tiebreak": 0}
        n_cells = 0
        items: list[CSQAItem] = []
        for cid in ids:
            base = loaded[anchor][cid]
            merged = dict(base["choices"])  # start from the anchor source
            for lab in base["choices"]:
                cands = []
                for name in sources:  # priority order; anchor first
                    row = loaded.get(name, {}).get(cid)
                    if row and lab in row["choices"]:
                        cands.append((name, row["choices"][lab]))
                surf, rule = _vote(cands, lang)
                merged[lab] = surf
                rule_counts["tiebreak" if rule == "google-tiebreak" else rule] += 1
                n_cells += 1
            items.append(CSQAItem(id=cid, question=base["question"],
                                  choices=merged, answer_key=base.get("answer_key")))

        # write merged choices + the eval variant
        (out_root / f"choices_{lang}.json").write_text(
            json.dumps([asdict(it) for it in items], ensure_ascii=False, indent=2),
            encoding="utf-8")
        v = build_variant("en-x", f"{lang}-{args.tag}", items, choice_x=items)
        write_variant(v, vdir / f"en-x__{lang}-{args.tag}.json")

        tot = n_cells or 1
        print(f"[{lang}] {len(items)} items, {n_cells} choices  "
              f"unanimous {rule_counts['unanimous']} ({rule_counts['unanimous']/tot:.1%}) | "
              f"majority {rule_counts['majority']} ({rule_counts['majority']/tot:.1%}) | "
              f"tiebreak {rule_counts['tiebreak']} ({rule_counts['tiebreak']/tot:.1%})")


if __name__ == "__main__":
    main()
