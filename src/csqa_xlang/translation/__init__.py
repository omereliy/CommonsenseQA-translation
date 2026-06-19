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
from csqa_xlang.translation.nllb import NLLBTranslator
from csqa_xlang.translation.opus import OpusMTTranslator

__all__ = ["Translator", "GoogleTranslator", "NLLBTranslator", "OpusMTTranslator",
           "translate_choices", "translate_questions", "get_translator"]


def get_translator(cfg: dict[str, Any], cache_dir: str | Path) -> Translator:
    """Build the translation backend named by cfg['translation']['backend']."""
    backend = (cfg.get("translation") or {}).get("backend", "google")
    if backend == "google":
        return GoogleTranslator(cache_dir=cache_dir)
    if backend == "nllb":
        return NLLBTranslator(cache_dir=cache_dir)
    if backend == "opus":
        return OpusMTTranslator(cache_dir=cache_dir)
    raise ValueError(f"unknown translation backend {backend!r} (wired: google, nllb, opus)")
