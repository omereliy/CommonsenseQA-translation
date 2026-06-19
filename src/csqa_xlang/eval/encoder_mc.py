"""Generic fine-tuned multilingual encoder + multiple-choice head.

Shared core for the XLM-R and mBERT arms (see ``xlmr.py`` / ``mbert.py``). An
``AutoModel`` encoder + a CLS → Dropout → ``Linear(h, 1)`` head: one score per
[question]+[choice] pair, softmax over the five choices, argmax → label. Fine-tune
on ENGLISH CSQA; evaluating the en-x variants measures zero-shot cross-lingual
TRANSFER through the encoder's shared multilingual space (both XLM-R and mBERT are
pretrained on es/ru/he; we never fine-tune on translated choices).

We define the head ourselves rather than ``AutoModelForMultipleChoice`` because
transformers 5.12 ships the XLM-R version broken (backbone built
``add_pooling_layer=False`` but its forward reads the pooler → IndexError). This
also lets us pass ``token_type_ids`` through for BERT-family encoders (mBERT uses
segment ids; XLM-R does not — its tokenizer omits them and forward gets None).

Checkpoint dir = ``mc_model.pt`` (state dict) + tokenizer files + ``arch.json``
({"base": <hf name>}) so ``predict`` self-describes which encoder to rebuild.
"""

from __future__ import annotations

import json
from pathlib import Path

from csqa_xlang.data import CSQAItem, load_csqa
from csqa_xlang.eval.base import Prediction, score
from csqa_xlang.eval.prompt import LABELS

MAXLEN = 96
WEIGHTS = "mc_model.pt"
ARCH = "arch.json"
DEFAULT_BASE = "xlm-roberta-base"


def _encode(tokenizer, question: str, choices: list[str]):
    return tokenizer([question] * len(choices), choices, truncation=True,
                     max_length=MAXLEN, padding="max_length")


def _build_model(base: str, *, pretrained: bool):
    """Encoder + Linear(h, 1) MC head. Same module layout whether starting from
    pretrained weights (train) or empty (load a checkpoint)."""
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(base)
    encoder = AutoModel.from_pretrained(base) if pretrained else AutoModel.from_config(cfg)

    class MultipleChoice(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.dropout = nn.Dropout(getattr(cfg, "hidden_dropout_prob", 0.1) or 0.1)
            self.classifier = nn.Linear(cfg.hidden_size, 1)

        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                    labels=None, **_):
            b, n, length = input_ids.shape
            kw = {"input_ids": input_ids.view(b * n, length)}
            if attention_mask is not None:
                kw["attention_mask"] = attention_mask.view(b * n, length)
            if token_type_ids is not None:                  # BERT-family; None for XLM-R
                kw["token_type_ids"] = token_type_ids.view(b * n, length)
            cls = self.encoder(**kw).last_hidden_state[:, 0]   # CLS / <s> per (item, choice)
            logits = self.classifier(self.dropout(cls)).view(b, n)
            loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
            return {"loss": loss, "logits": logits}

    return MultipleChoice()


def train(ckpt_dir: str | Path, base: str = DEFAULT_BASE, *,
          train_items: list[CSQAItem] | None = None,
          eval_items: list[CSQAItem] | None = None, epochs: float = 3.0,
          batch_size: int = 16, lr: float = 1e-5,
          init_from: str | Path | None = None) -> None:
    """Full fine-tune (all params + MC head) of ``base`` on ENGLISH CSQA.

    Pass an 80-10-10 split via ``train_items``/``eval_items`` to keep the best-on-dev
    epoch. ``init_from`` warm-starts from an existing checkpoint dir (continue N more
    epochs; the LR schedule restarts).
    """
    import numpy as np
    import torch
    from dataclasses import dataclass
    from transformers import (AutoTokenizer, Trainer, TrainerCallback,
                              TrainingArguments)

    ckpt_dir = Path(ckpt_dir)
    tok = AutoTokenizer.from_pretrained(base)
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
            # Read, don't pop: feature dicts are reused every epoch.
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
        model = _build_model(base, pretrained=False)
        model.load_state_dict(torch.load(Path(init_from) / WEIGHTS, map_location="cpu"))
        print(f"warm-start from {init_from} (+{epochs:g} more epochs)")
    else:
        model = _build_model(base, pretrained=True)

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
    (ckpt_dir / ARCH).write_text(json.dumps({"base": base}), encoding="utf-8")
    print(f"saved fine-tuned encoder ({base}) -> {ckpt_dir}")


def predict(items: list[CSQAItem], ckpt_dir: str | Path,
            base: str | None = None) -> list[Prediction]:
    import torch
    from transformers import AutoTokenizer

    ckpt_dir = Path(ckpt_dir)
    weights = ckpt_dir / WEIGHTS
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found — train first")
    if base is None:                                   # self-describe from the checkpoint
        arch = ckpt_dir / ARCH
        base = json.loads(arch.read_text())["base"] if arch.exists() else DEFAULT_BASE
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(base, pretrained=False)
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
