"""Translate the answer choices of CSQAItems, preserving labels, ordering, gold.

This is the load-bearing transform: the question stays English; only the choice
*text* is translated; A-E labels and `answer_key` are untouched, so the gold key
stays valid after translation. Choice texts are de-duplicated across the whole
split before translation (the backend also caches), keeping the paid step cheap.

`translate_questions` is the symmetric helper used ONLY by the x-x / x-en
extension conditions (translate_questions=true in the config); the default
choices-only pipeline never calls it.
"""

from __future__ import annotations

from dataclasses import replace

from csqa_xlang.data import CSQAItem
from csqa_xlang.translation.base import Translator


def translate_choices(items: list[CSQAItem], tgt: str, translator: Translator,
                      src: str = "en") -> list[CSQAItem]:
    """Return parallel CSQAItems with `choices` text translated into `tgt`."""
    uniq = list({txt for it in items for txt in it.choices.values()})
    table = dict(zip(uniq, translator.translate(uniq, src, tgt)))
    out = []
    for it in items:
        out.append(replace(it, choices={lab: table[txt] for lab, txt in it.choices.items()}))
    return out


def translate_questions(items: list[CSQAItem], tgt: str, translator: Translator,
                        src: str = "en") -> list[CSQAItem]:
    """Return parallel CSQAItems with the `question` translated (x-x / x-en only)."""
    questions = [it.question for it in items]
    translated = translator.translate(questions, src, tgt)
    return [replace(it, question=q) for it, q in zip(items, translated)]
