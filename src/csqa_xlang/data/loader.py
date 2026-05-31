"""Load CommonsenseQA into a normalized form.

Two load-bearing conventions live here (see CLAUDE.md):

1. Default split is ``validation`` (~1,221 items). The ``test`` split's answer keys
   are withheld for the leaderboard, so it cannot be scored locally — loading it
   for evaluation raises.
2. Choices are keyed by LETTER labels (A-E). Downstream, the model selects a letter,
   never the answer text — this keeps scoring language-invariant once choices are
   translated.
"""

from __future__ import annotations

from dataclasses import dataclass

# Canonical choice labels. CommonsenseQA uses A-E.
LABELS = ["A", "B", "C", "D", "E"]


@dataclass
class CSQAItem:
    """One normalized CommonsenseQA item.

    Attributes:
        id: the original question id.
        question: the English question stem.
        choices: ordered mapping label -> choice text, e.g. {"A": "river", ...}.
            Translation/variant steps produce parallel CSQAItems with translated
            ``choices`` values but identical labels and ordering.
        answer_key: the gold label (one of LABELS), or None when unavailable.
    """

    id: str
    question: str
    choices: dict[str, str]
    answer_key: str | None


def load_csqa(split: str = "validation", *, limit: int | None = None) -> list[CSQAItem]:
    """Load CommonsenseQA (``tau/commonsense_qa``) as a list of CSQAItem.

    Args:
        split: dataset split. Defaults to ``validation`` (the only scorable split
            with public labels). Passing ``test`` raises, because its labels are
            withheld — see the module docstring.
        limit: optional cap for quick smoke runs.
    """
    if split == "test":
        raise ValueError(
            "CommonsenseQA's `test` split has withheld answer keys and cannot be "
            "scored locally. Use split='validation' (the project default)."
        )

    from datasets import load_dataset

    ds = load_dataset("tau/commonsense_qa", split=split)
    items: list[CSQAItem] = []
    for row in ds:
        # HF schema: row["choices"] = {"label": [...], "text": [...]}
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        choices = {label: text for label, text in zip(labels, texts)}
        answer_key = row.get("answerKey") or None  # "" -> None for unlabeled splits
        items.append(
            CSQAItem(
                id=row["id"],
                question=row["question"],
                choices=choices,
                answer_key=answer_key,
            )
        )
        if limit is not None and len(items) >= limit:
            break
    return items
