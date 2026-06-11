"""Pluggable translation backend interface.

A backend turns a list of source strings into a list of target strings. Keep
this surface tiny so MT (Google/DeepL/NLLB) and LLM backends are interchangeable
(CLAUDE.md keeps the backend a team decision; the resolved default is Google).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Translator(Protocol):
    name: str

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        """Translate `texts` from `src` to `tgt`, order-preserving (1:1 with input)."""
        ...
