"""Answer-choice translation (resolved backend: Google Translate; see README.md).

The question stays English; only choice text is translated (labels/ordering/gold
preserved). Backend stays pluggable behind the `Translator` protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from csqa_xlang.translation.base import Translator
from csqa_xlang.translation.choices import translate_choices, translate_questions
from csqa_xlang.translation.google import GoogleTranslator

__all__ = ["Translator", "GoogleTranslator", "translate_choices",
           "translate_questions", "get_translator"]


def get_translator(cfg: dict[str, Any], cache_dir: str | Path) -> Translator:
    """Build the translation backend named by cfg['translation']['backend']."""
    backend = (cfg.get("translation") or {}).get("backend", "google")
    if backend == "google":
        return GoogleTranslator(cache_dir=cache_dir)
    raise ValueError(f"unknown translation backend {backend!r} (only 'google' is wired)")
