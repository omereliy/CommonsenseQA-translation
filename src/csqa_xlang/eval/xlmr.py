"""Fine-tuned multilingual encoder baseline: XLM-R on CommonsenseQA.

`train` fine-tunes xlm-roberta-base on ENGLISH CSQA train (multiple-choice head).
`predict` scores a condition variant: input per choice is [question] + [choice],
argmax over the five → chosen label. Evaluating the en-x variants (English
question + translated choices) measures cross-lingual TRANSFER through XLM-R's
shared space. Report the en→he degradation pattern, not absolute accuracy vs the
zero-shot LLMs (this model is task-fine-tuned; they are not).

We define the multiple-choice head ourselves (CLS → dropout → Linear(h, 1), one
shared score per [question]+[choice] pair, softmax over the five) instead of
`AutoModelForMultipleChoice`: transformers 5.12.x ships that class with the XLM-R
backbone built `add_pooling_layer=False` while its forward still reads the pooler
output, so it raises `IndexError` on every forward. The custom head is the same
architecture and is version-independent. Checkpoint = tokenizer files + a single
``mc_model.pt`` state-dict in ``ckpt_dir``.
"""

from __future__ import annotations

from pathlib import Path

from csqa_xlang.data import CSQAItem, load_csqa
from csqa_xlang.eval.base import Prediction, score
from csqa_xlang.eval.prompt import LABELS

MODEL = "xlm-roberta-base"
MAXLEN = 96
WEIGHTS = "mc_model.pt"


def _encode(tokenizer, question: str, choices: list[str]):
    return tokenizer([question] * len(choices), choices, truncation=True,
                     max_length=MAXLEN, padding="max_length")


def _build_model(*, pretrained: bool):
    """XLM-R encoder + a Linear(h, 1) multiple-choice head. Identical module layout
    whether we start from pretrained weights (train) or empty (load a checkpoint)."""
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(MODEL)
    encoder = AutoModel.from_pretrained(MODEL) if pretrained else AutoModel.from_config(cfg)

    class MultipleChoice(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.dropout = nn.Dropout(getattr(cfg, "hidden_dropout_prob", 0.1) or 0.1)
            self.classifier = nn.Linear(cfg.hidden_size, 1)

        def forward(self, input_ids=None, attention_mask=None, labels=None, **_):
            b, n, length = input_ids.shape
            flat_ids = input_ids.view(b * n, length)
            flat_mask = attention_mask.view(b * n, length) if attention_mask is not None else None
            hidden = self.encoder(input_ids=flat_ids, attention_mask=flat_mask).last_hidden_state
            cls = hidden[:, 0]                                   # <s> token per (item, choice)
            logits = self.classifier(self.dropout(cls)).view(b, n)
            loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
            return {"loss": loss, "logits": logits}

    return MultipleChoice()


def train(ckpt_dir: str | Path, *, train_items: list[CSQAItem] | None = None,
          eval_items: list[CSQAItem] | None = None, epochs: float = 3.0,
          batch_size: int = 16, lr: float = 1e-5,
          init_from: str | Path | None = None) -> None:
    """Full fine-tune (all params + MC head) on ENGLISH CSQA.

    ``train_items``/``eval_items`` default to the full HF ``train`` split with no
    held-out dev. Pass an 80-10-10 split (see ``training/xlmr_csqa``) to fine-tune
    on the train portion and keep the best-on-dev epoch.

    ``init_from`` warm-starts from an existing checkpoint dir (its ``mc_model.pt``)
    instead of pretrained xlm-roberta-base — i.e. continue training N more epochs
    from a prior run. The LR schedule restarts (fresh optimizer).
    """
    import numpy as np
    import torch
    from dataclasses import dataclass
    from transformers import (AutoTokenizer, Trainer, TrainerCallback,
                              TrainingArguments)

    ckpt_dir = Path(ckpt_dir)
    tok = AutoTokenizer.from_pretrained(MODEL)
    items = train_items if train_items is not None else load_csqa("train")

    def feats(it: CSQAItem):
        choices = [it.choices[lab] for lab in LABELS]
        enc = _encode(tok, it.question, choices)
        enc["label"] = LABELS.index(it.answer_key)
        return enc

    data = [feats(it) for it in items]
    eval_data = [feats(it) for it in eval_items] if eval_items else None
    has_eval = eval_data is not None

    @dataclass
    class Collator:
        def __call__(self, features):
            # Read, don't pop: the same feature dicts are reused every epoch, so
            # mutating them drops "label" after epoch 1 (KeyError on the next pass).
            labels = torch.tensor([f["label"] for f in features])
            keys = [k for k in features[0] if k != "label"]
            batch = {k: torch.tensor([f[k] for f in features]) for k in keys}
            batch["labels"] = labels
            return batch

    def accuracy(eval_pred):
        logits, labels = eval_pred
        preds = np.asarray(logits).argmax(axis=-1)
        return {"accuracy": float((preds == np.asarray(labels)).mean())}

    if init_from is not None:
        init_w = Path(init_from) / WEIGHTS
        model = _build_model(pretrained=False)
        model.load_state_dict(torch.load(init_w, map_location="cpu"))
        print(f"warm-start from {init_w} (+{epochs:g} more epochs)")
    else:
        model = _build_model(pretrained=True)

    # Keep the best-on-dev weights in memory (the buggy MC class would have made
    # Trainer's load_best_model_at_end the natural path; we do it by hand instead).
    class KeepBest(TrainerCallback):
        def __init__(self):
            self.best_acc = -1.0
            self.state = None

        def on_evaluate(self, args, state, control, metrics=None, **_):
            acc = (metrics or {}).get("eval_accuracy")
            if acc is not None and acc > self.best_acc:
                self.best_acc = acc
                self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    keep_best = KeepBest() if has_eval else None
    targs = TrainingArguments(
        output_dir=str(ckpt_dir / "_trainer"),
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        learning_rate=lr, num_train_epochs=epochs, logging_steps=50,
        eval_strategy="epoch" if has_eval else "no", save_strategy="no", report_to=[])
    Trainer(model=model, args=targs, train_dataset=data, eval_dataset=eval_data,
            data_collator=Collator(), compute_metrics=accuracy if has_eval else None,
            callbacks=[keep_best] if keep_best else None).train()

    if keep_best and keep_best.state is not None:
        model.load_state_dict(keep_best.state)
        print(f"restored best-on-dev epoch: accuracy={keep_best.best_acc:.4f}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / WEIGHTS)
    tok.save_pretrained(str(ckpt_dir))
    print(f"saved fine-tuned XLM-R -> {ckpt_dir}")


def predict(items: list[CSQAItem], ckpt_dir: str | Path) -> list[Prediction]:
    import torch
    from transformers import AutoTokenizer

    ckpt_dir = Path(ckpt_dir)
    weights = ckpt_dir / WEIGHTS
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found — run xlmr.train first")
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(pretrained=False)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()

    out: list[Prediction] = []
    for it in items:
        choices = [it.choices[lab] for lab in LABELS]
        enc = _encode(tok, it.question, choices)
        batch = {k: torch.tensor([v]).to(device) for k, v in enc.items()}
        with torch.no_grad():
            pred = LABELS[int(model(**batch)["logits"][0].argmax())]
        out.append(Prediction(id=it.id, gold=it.answer_key, pred=pred,
                              correct=score(pred, it.answer_key)))
    return out
