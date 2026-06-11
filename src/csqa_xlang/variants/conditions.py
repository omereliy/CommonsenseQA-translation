"""Condition variants: a built variant is a fingerprinted list of CSQAItems.

Conditions (CLAUDE.md / methods.tex). The default pipeline builds the first two;
x-x / x-en need translated *questions* and are gated behind
`extensions.translate_questions` in the config (wired, off by default). `mixed`
and `noun` remain extension stubs.

| Condition | Question | Answers |
|-----------|----------|---------|
| en-en     | EN       | EN (baseline) |
| en-x      | EN       | X |
| x-x       | X        | X |
| x-en      | X        | EN |
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from csqa_xlang.data import CSQAItem
from csqa_xlang.manifest import content_hash

CONDITIONS = ["en-en", "en-x", "x-x", "x-en"]
CORE_CONDITIONS = ["en-en", "en-x"]          # implemented by default
QUESTION_TRANSLATED = {"x-x", "x-en"}        # require translated questions


@dataclass
class Variant:
    """One evaluation condition: items with question/choices in chosen languages."""

    condition: str
    language: str          # the X language; "en" for the en-en baseline
    items: list[CSQAItem]

    @property
    def variant_hash(self) -> str:
        """Stable fingerprint of the exact items (see manifest.content_hash)."""
        return content_hash([asdict(it) for it in self.items])

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "language": self.language,
            "variant_hash": self.variant_hash,
            "items": [asdict(it) for it in self.items],
        }


def write_variant(variant: Variant, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(variant.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_variant(path: str | Path) -> Variant:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    items = [CSQAItem(**it) for it in d["items"]]
    return Variant(condition=d["condition"], language=d["language"], items=items)
