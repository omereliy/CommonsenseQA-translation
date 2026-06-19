"""mBERT instance of the encoder multiple-choice arm.

Fine-tunes ``bert-base-multilingual-cased`` (104 languages, incl. es/ru/he) on
ENGLISH CSQA; evaluating the en-x variants measures zero-shot cross-lingual
transfer — the multilingual-encoder counterpart to the XLM-R arm. mBERT uses
``token_type_ids`` (segment A=question, B=choice); ``encoder_mc`` passes them
through. All logic lives in ``encoder_mc``; this module just pins the base model.
"""

from __future__ import annotations

from pathlib import Path

from csqa_xlang.data import CSQAItem
from csqa_xlang.eval import encoder_mc
from csqa_xlang.eval.base import Prediction

MODEL = "bert-base-multilingual-cased"


def train(ckpt_dir: str | Path, *, train_items: list[CSQAItem] | None = None,
          eval_items: list[CSQAItem] | None = None, epochs: float = 3.0,
          batch_size: int = 16, lr: float = 1e-5,
          init_from: str | Path | None = None) -> None:
    encoder_mc.train(ckpt_dir, MODEL, train_items=train_items, eval_items=eval_items,
                     epochs=epochs, batch_size=batch_size, lr=lr, init_from=init_from)


def predict(items: list[CSQAItem], ckpt_dir: str | Path) -> list[Prediction]:
    return encoder_mc.predict(items, ckpt_dir, MODEL)
