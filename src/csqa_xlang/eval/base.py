"""Shared prediction record for every backend (generative + encoder arms)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Prediction:
    """One model decision on one item. `pred` is the chosen label A-E (or None)."""

    id: str
    gold: str | None
    pred: str | None
    correct: bool
    raw: str = ""              # raw model text (generative); "" for encoder arms
    thinking: str = ""         # reasoning trace when think=on
    prompt_tokens: int = 0
    completion_tokens: int = 0
    done_reason: str = ""
    truncated: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def score(pred: str | None, gold: str | None) -> bool:
    return bool(pred is not None and gold is not None and pred == gold)
