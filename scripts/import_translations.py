"""Import a teammate's answer-choice translations into our pipeline format.

Source: the `Do-LLMs-Choose-Concepts-or-English-Words` repo
(github.com/marikf98/Do-LLMs-Choose-Concepts-or-English-Words), which translates
the CSQA *validation* answer choices into he/es/ru with Google Translate plus a
reviewer QA pass. Its curated artifacts are vendored under
``data/translated/source_marikf98/`` (base_en.jsonl + cache/<lang>_merged.json +
qa/{manual_fixes_<lang>,excluded_ids}.csv).

This script mirrors that repo's ``04_build_datasets.py`` build logic — apply
``manual_fixes_<lang>.csv``, drop ``excluded_ids.csv``, strip Hebrew niqqud — and
emits, for each target language X, one parallel file in *our* CSQAItem schema:

    data/translated/choices_<X>.json   # list[{id, question, choices{A..E}, answer_key}]

i.e. the English question is preserved and only the choice *text* is translated,
labels/gold untouched. That is exactly what ``scripts/build_variants.py`` reads to
assemble the en-x condition (the en-en baseline still loads English from HF).

A provenance manifest (source commit, counts, fixes, exclusions) is written to
``data/translated/choices_import_manifest.json``.

Usage:
    python -m scripts.import_translations            # uses vendored source
    python -m scripts.import_translations --source <dir> --out data/translated
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# Short code -> the merged-cache / fixes filename stem used by the source repo.
LANGS = ["ru", "es", "he"]

# Hebrew vocalization marks: Google Translate adds niqqud to bare single words;
# the source repo strips them so the surface form matches normal written Hebrew.
_NIQQUD = re.compile(r"[֑-ׇ]")


def strip_niqqud(text: str) -> str:
    return _NIQQUD.sub("", text) if text else text


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_fixes(qa_dir: Path, code: str) -> dict[tuple[str, str], str]:
    """(id, label) -> reviewer-corrected text from manual_fixes_<code>.csv."""
    path = qa_dir / f"manual_fixes_{code}.csv"
    fixes: dict[tuple[str, str], str] = {}
    if not path.exists():
        return fixes
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            text = (row.get("fixed_text") or "").strip()
            if text:
                fixes[(row["id"].strip(), row["label"].strip())] = text
    return fixes


def read_exclusions(qa_dir: Path) -> dict[str, str]:
    """id -> reason from excluded_ids.csv (dropped from every language)."""
    path = qa_dir / "excluded_ids.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig") as f:
        return {row["id"].strip(): (row.get("reason") or "").strip()
                for row in csv.DictReader(f) if row.get("id", "").strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="data/translated/source_marikf98",
                    help="vendored teammate artifacts (base_en.jsonl, cache/, qa/)")
    ap.add_argument("--out", default="data/translated",
                    help="where to write choices_<lang>.json")
    ap.add_argument("--source-commit", default="86586d8f9f85dad968d15a5ced0b2247fd611a57",
                    help="provenance: the source repo commit the artifacts came from")
    args = ap.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir, qa_dir = src / "cache", src / "qa"

    base = load_jsonl(src / "base_en.jsonl")
    excluded = read_exclusions(qa_dir)
    kept = [r for r in base if r["id"] not in excluded]

    errors: list[str] = []
    manifest_langs: dict[str, dict] = {}

    for code in LANGS:
        merged = json.loads((cache_dir / f"{code}_merged.json").read_text(encoding="utf-8"))
        fixes = read_fixes(qa_dir, code)
        used_fixes = 0
        items_out = []
        for row in kept:
            choices: dict[str, str] = {}
            for label, en in zip(row["choices"]["label"], row["choices"]["text"]):
                fix = fixes.get((row["id"], label))
                text = fix if fix is not None else (merged.get(f"{row['id']}|{label}") or {}).get("text")
                if code == "he":
                    text = strip_niqqud(text)
                if not text or not text.strip():
                    errors.append(f"[{code}] {row['id']}|{label} empty translation for {en!r}")
                    text = ""
                if fix is not None:
                    used_fixes += 1
                choices[label] = text
            # our CSQAItem schema: question stays English; choices.text translated.
            items_out.append({
                "id": row["id"],
                "question": row["question"],
                "choices": choices,
                "answer_key": row["answerKey"],
            })

        path = out / f"choices_{code}.json"
        path.write_text(json.dumps(items_out, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_langs[code] = {"n": len(items_out), "manual_fixes": used_fixes,
                                "file": str(path)}
        print(f"[{code}] wrote {len(items_out)} items, {used_fixes} manual fix(es) -> {path}")

    # --- integrity gate: parallelism across languages (mirrors the source repo) ---
    counts = {c: m["n"] for c, m in manifest_langs.items()}
    if len(set(counts.values())) != 1:
        errors.append(f"row counts differ across languages: {counts}")

    if errors:
        print(f"\nINTEGRITY GATE FAILED — {len(errors)} problem(s):")
        for e in errors[:40]:
            print(" ", e)
        raise SystemExit(1)

    manifest = {
        "source_repo": "github.com/marikf98/Do-LLMs-Choose-Concepts-or-English-Words",
        "source_commit": args.source_commit,
        "split": "validation",
        "n_base": len(base),
        "n_excluded": len(excluded),
        "n_kept": len(kept),
        "languages": manifest_langs,
        "notes": "EN question preserved; only choice text translated. Hebrew niqqud "
                 "stripped. en-en baseline loads English from HF (ids match base_en).",
    }
    (out / "choices_import_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nN = {len(kept)} per language (excluded {len(excluded)} of {len(base)}); "
          f"integrity gate passed. Manifest -> {out / 'choices_import_manifest.json'}")


if __name__ == "__main__":
    main()
