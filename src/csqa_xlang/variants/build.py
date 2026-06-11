"""Assemble condition variants from base English items + translations.

A variant is a pure, deterministic transform over loaded items — no model calls.
`en_items` are the English base; `choice_x` are items whose CHOICES are
translated into X (question still English); `question_x` are items whose
QUESTION is translated into X (choices still English). All three are parallel
lists (same order/ids), so conditions are built by recombining their fields.
"""

from __future__ import annotations

from dataclasses import replace

from csqa_xlang.data import CSQAItem
from csqa_xlang.variants.conditions import Variant


def build_variant(
    condition: str,
    lang: str,
    en_items: list[CSQAItem],
    choice_x: list[CSQAItem] | None = None,
    question_x: list[CSQAItem] | None = None,
) -> Variant:
    """Build one (condition, lang) Variant from the available translated pieces."""
    if condition == "en-en":
        return Variant("en-en", "en", list(en_items))

    if condition == "en-x":
        if choice_x is None:
            raise ValueError("en-x needs choice-translated items (choice_x)")
        return Variant("en-x", lang, list(choice_x))

    if condition == "x-x":
        if choice_x is None or question_x is None:
            raise ValueError("x-x needs both translated questions and choices "
                             "(enable extensions.translate_questions)")
        items = [replace(q, choices=c.choices) for q, c in zip(question_x, choice_x)]
        return Variant("x-x", lang, items)

    if condition == "x-en":
        if question_x is None:
            raise ValueError("x-en needs translated questions "
                             "(enable extensions.translate_questions)")
        # question in X, choices stay English (question_x already keeps EN choices)
        return Variant("x-en", lang, list(question_x))

    raise ValueError(f"condition {condition!r} not implemented (mixed/noun are extension stubs)")
