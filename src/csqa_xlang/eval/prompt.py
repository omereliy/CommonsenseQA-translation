"""Prompt construction + robust letter parsing (load-bearing: letter-based answers).

The prompt is deliberately code-switched: an English question, choices in the
target language labelled with Latin letters A-E, and an English instruction
asking for the letter. Instruction and labels are held CONSTANT across every
condition, so the only thing that varies is the choice language — and scoring
(a letter) never string-matches the translated answer text, which would confound
the flip metric (CLAUDE.md).

PROMPT_TEMPLATE is the stable string hashed into the run manifest; keep eval
prompts built from it so the template hash is meaningful across backends.
"""

from __future__ import annotations

import re

LABELS = ["A", "B", "C", "D", "E"]

INSTRUCTION = (
    "Answer with the letter (A, B, C, D, or E) of the single correct option. "
    "Respond with just the letter."
)
# Hashed into the manifest as prompt_template_hash.
PROMPT_TEMPLATE = "{question}\n\n{choices}\n\n" + INSTRUCTION


def build_prompt(question: str, choices: dict[str, str]) -> str:
    body = "\n".join(f"{lab}. {choices[lab]}" for lab in LABELS if lab in choices)
    return PROMPT_TEMPLATE.format(question=question.strip(), choices=body)


def build_messages(question: str, choices: dict[str, str]) -> list[dict]:
    return [{"role": "user", "content": build_prompt(question, choices)}]


_EXACT = re.compile(r"^\(?([A-Ea-e])\)?[.\):]?$")
_ANSWER_PAT = re.compile(r"(?:answer|option|choice|correct)\D{0,15}?\b([A-E])\b", re.IGNORECASE)
_LEADING = re.compile(r"^\s*\(?([A-E])\)?[.\):\s]")
_ISOLATED = re.compile(r"\b([A-E])\b")


def extract_letter(text: str | None, choices: dict[str, str] | None = None) -> str | None:
    """Return the chosen label A-E, or None if unparseable. Tiered: bare letter →
    'answer is X' → leading letter → first isolated letter → echoed-phrase fallback."""
    if not text:
        return None
    t = text.strip()
    for pat in (_EXACT, _ANSWER_PAT, _LEADING):
        m = pat.match(t) if pat is not _ANSWER_PAT else pat.search(t)
        if m:
            return m.group(1).upper()
    m = _ISOLATED.search(t)
    if m:
        return m.group(1)
    if choices:
        low = t.lower()
        hits = [lab for lab, txt in choices.items() if txt and txt.lower() in low]
        if len(hits) == 1:
            return hits[0]
    return None
