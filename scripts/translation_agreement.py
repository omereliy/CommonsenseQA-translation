"""Analyze the translations themselves: agreement across MT systems (no model).

For each target language, compares the answer-choice translations produced by the
wired sources (Google / NLLB / Opus-MT) on the SAME items. Low agreement on these
short ConceptNet concepts = MT is noisy (synonym collapse, sense errors) — the
confound the cross-lingual eval must be read against. Strings are normalized
(NFC, casefold, strip Hebrew niqqud + edge punctuation, collapse spaces, drop a
leading article) before comparison, so trivial surface differences don't count as
disagreement.

Reads:  data/translated/choices_<lang>.json            (Google; curated)
        data/translated/<backend>/choices_<lang>.json  (nllb, opus)
Writes: results/translation_agreement.csv

Usage:  python -m scripts.translation_agreement
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from itertools import combinations
from pathlib import Path

LANGS = ["ru", "es", "he"]
SOURCES = {
    "google": "data/translated/choices_{lang}.json",
    "nllb": "data/translated/nllb/choices_{lang}.json",
    "opus": "data/translated/opus/choices_{lang}.json",
}

_NIQQUD = re.compile(r"[֑-ׇ]")
_PUNCT_EDGE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)
_WS = re.compile(r"\s+")
_LATIN = re.compile(r"[A-Za-z]")
ARTICLES = {"es": {"el", "la", "los", "las", "un", "una", "unos", "unas"},
            "he": set(), "ru": set(), "en": {"a", "an", "the"}}


def normalize(text: str, lang: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text).casefold().strip()
    t = _WS.sub(" ", _PUNCT_EDGE.sub("", _NIQQUD.sub("", t)))
    toks = t.split(" ")
    if len(toks) > 1 and toks[0] in ARTICLES.get(lang, set()):
        toks = toks[1:]
    return " ".join(toks)


def _load(path: Path) -> dict[tuple[str, str], str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(r["id"], lab): txt for r in rows for lab, txt in r["choices"].items()}


def main() -> None:
    out_rows = []
    for lang in LANGS:
        sets = {}
        for name, tmpl in SOURCES.items():
            p = Path(tmpl.format(lang=lang))
            if p.exists():
                sets[name] = {k: normalize(v, lang) for k, v in _load(p).items()}
        if len(sets) < 2:
            print(f"[{lang}] need >=2 sources, found {list(sets)} — skipping")
            continue
        keys = set.intersection(*(set(s) for s in sets.values()))
        n = len(keys)
        print(f"\n[{lang}] {n} (id,label) cells across {list(sets)}")
        # pairwise exact-match agreement
        for a, b in combinations(sets, 2):
            agree = sum(sets[a][k] == sets[b][k] for k in keys)
            print(f"    {a:6} vs {b:6}: {agree/n:6.1%} exact-match")
            out_rows.append({"lang": lang, "comparison": f"{a}-{b}",
                             "n": n, "agreement": round(agree / n, 4)})
        # three-way unanimous
        if len(sets) >= 3:
            names = list(sets)
            unan = sum(len({sets[s][k] for s in names}) == 1 for k in keys)
            print(f"    all {len(names)} unanimous: {unan/n:6.1%}")
            out_rows.append({"lang": lang, "comparison": "all-unanimous",
                             "n": n, "agreement": round(unan / n, 4)})
        # Opus Hebrew Latin-token artifact rate (e.g. 'kgm', 'world')
        if lang == "he" and "opus" in sets:
            raw = _load(Path(SOURCES["opus"].format(lang=lang)))
            bad = sum(bool(_LATIN.search(v)) for v in raw.values())
            print(f"    opus he Latin-char artifacts: {bad}/{len(raw)} = {bad/len(raw):.1%}")
            out_rows.append({"lang": lang, "comparison": "opus-latin-artifact",
                             "n": len(raw), "agreement": round(bad / len(raw), 4)})

    if out_rows:
        out = Path("results/translation_agreement.csv")
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
