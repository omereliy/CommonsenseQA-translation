"""Fine-tuned multilingual encoder baseline: XLM-R on CommonsenseQA.

`train` fine-tunes xlm-roberta-base on ENGLISH CSQA train (multiple-choice head).
`predict` scores a condition variant: input per choice is [question] + [choice],
argmax over the five → chosen label. Evaluating the en-x variants (English
question + translated choices) measures cross-lingual TRANSFER through XLM-R's
shared space. Report the en→he degradation pattern, not absolute accuracy vs the
zero-shot LLMs (this model is task-fine-tuned; they are not).
"""

from __future__ import annotations

from pathlib import Path

from csqa_xlang.data import CSQAItem, load_csqa
from csqa_xlang.eval.base import Prediction, score
from csqa_xlang.eval.prompt import LABELS

MODEL = "xlm-roberta-base"
MAXLEN = 96


def _encode(tokenizer, question: str, choices: list[str]):
    return tokenizer([question] * len(choices), choices, truncation=True,
                     max_length=MAXLEN, padding="max_length")


def train(ckpt_dir: str | Path, *, epochs: float = 3.0, batch_size: int = 16,
          lr: float = 1e-5) -> None:
    import torch
    from dataclasses import dataclass
    from transformers import (AutoModelForMultipleChoice, AutoTokenizer, Trainer,
                              TrainingArguments)

    ckpt_dir = Path(ckpt_dir)
    tok = AutoTokenizer.from_pretrained(MODEL)
    items = load_csqa("train")

    def feats(it: CSQAItem):
        choices = [it.choices[lab] for lab in LABELS]
        enc = _encode(tok, it.question, choices)
        enc["label"] = LABELS.index(it.answer_key)
        return enc

    data = [feats(it) for it in items]

    @dataclass
    class Collator:
        def __call__(self, features):
            labels = torch.tensor([f.pop("label") for f in features])
            batch = {k: torch.tensor([f[k] for f in features]) for k in features[0]}
            batch["labels"] = labels
            return batch

    model = AutoModelForMultipleChoice.from_pretrained(MODEL)
    targs = TrainingArguments(output_dir=str(ckpt_dir / "_trainer"),
                              per_device_train_batch_size=batch_size, learning_rate=lr,
                              num_train_epochs=epochs, logging_steps=50,
                              save_strategy="no", report_to=[])
    Trainer(model=model, args=targs, train_dataset=data, data_collator=Collator()).train()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tok.save_pretrained(str(ckpt_dir))
    print(f"saved fine-tuned XLM-R -> {ckpt_dir}")


def predict(items: list[CSQAItem], ckpt_dir: str | Path) -> list[Prediction]:
    import torch
    from transformers import AutoModelForMultipleChoice, AutoTokenizer

    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"{ckpt_dir} not found — run xlmr.train first")
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir))
    model = AutoModelForMultipleChoice.from_pretrained(str(ckpt_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    out: list[Prediction] = []
    for it in items:
        choices = [it.choices[lab] for lab in LABELS]
        enc = _encode(tok, it.question, choices)
        batch = {k: torch.tensor([v]).to(device) for k, v in enc.items()}
        with torch.no_grad():
            pred = LABELS[int(model(**batch).logits[0].argmax())]
        out.append(Prediction(id=it.id, gold=it.answer_key, pred=pred,
                              correct=score(pred, it.answer_key)))
    return out
