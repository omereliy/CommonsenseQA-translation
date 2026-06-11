"""ESIM (Chen et al. 2017) — the pre-transformer historical anchor.

Compact ESIM for 5-way MC: shared BiLSTM encodes question + each choice, soft
cross-attention aligns them, a composition BiLSTM + pooling yields a per-choice
vector scored by an MLP; softmax over the five.

ENGLISH-CONDITION ONLY, on purpose: ESIM reads static (GloVe) word embeddings
whose vocabulary is English, so Russian/Hebrew choices are near-all-OOV and would
score at chance. It anchors the "classic baseline" point on en-en; the runner
only ever calls it on the English variant. Set GLOVE_PATH to a glove.6B.300d.txt
file to use pretrained vectors; otherwise embeddings are random-init (weaker).
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

from csqa_xlang.data import CSQAItem, load_csqa
from csqa_xlang.eval.base import Prediction, score
from csqa_xlang.eval.prompt import LABELS

EMB_DIM, HID, MAXLEN = 300, 200, 40
_TOK = re.compile(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]")


def _tok(s: str) -> list[str]:
    return _TOK.findall(s.lower())


def _vocab(texts: list[str]) -> dict[str, int]:
    c = Counter(t for s in texts for t in _tok(s))
    v = {"<pad>": 0, "<unk>": 1}
    for w in c:
        v[w] = len(v)
    return v


def _enc(s: str, vocab: dict[str, int]) -> list[int]:
    ids = [vocab.get(t, 1) for t in _tok(s)][:MAXLEN]
    return ids + [0] * (MAXLEN - len(ids))


def _load_glove(path: str, vocab: dict[str, int]):
    import torch
    mat = torch.randn(len(vocab), EMB_DIM) * 0.1
    mat[0].zero_()
    found = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if parts[0] in vocab:
                mat[vocab[parts[0]]] = torch.tensor([float(x) for x in parts[1:]])
                found += 1
    print(f"  GloVe matched {found}/{len(vocab)}")
    return mat


def _model(vocab_size, emb_init=None):
    import torch
    import torch.nn as nn

    class ESIM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
            if emb_init is not None:
                self.emb.weight.data.copy_(emb_init)
            self.encode = nn.LSTM(EMB_DIM, HID, batch_first=True, bidirectional=True)
            self.project = nn.Sequential(nn.Linear(8 * HID, HID), nn.ReLU())
            self.compose = nn.LSTM(HID, HID, batch_first=True, bidirectional=True)
            self.cls = nn.Sequential(nn.Dropout(0.3), nn.Linear(8 * HID, HID), nn.Tanh(),
                                     nn.Dropout(0.3), nn.Linear(HID, 1))

        def _e(self, x):
            mask = (x != 0).float().unsqueeze(-1)
            h, _ = self.encode(self.emb(x))
            return h, mask

        def _pool(self, h, m):
            avg = (h * m).sum(1) / m.sum(1).clamp(min=1)
            mx = h.masked_fill(m == 0, -1e9).max(1).values
            return torch.cat([avg, mx], -1)

        def _pair(self, q, c):
            hq, mq = self._e(q)
            hc, mc = self._e(c)
            att = torch.bmm(hq, hc.transpose(1, 2))
            wq = torch.softmax(att.masked_fill(mc.transpose(1, 2) == 0, -1e9), 2)
            wc = torch.softmax(att.masked_fill(mq == 0, -1e9), 1).transpose(1, 2)
            aq, ac = torch.bmm(wq, hc), torch.bmm(wc, hq)
            mq_ = torch.cat([hq, aq, hq - aq, hq * aq], -1)
            mc_ = torch.cat([hc, ac, hc - ac, hc * ac], -1)
            vq, _ = self.compose(self.project(mq_))
            vc, _ = self.compose(self.project(mc_))
            return torch.cat([self._pool(vq, mq), self._pool(vc, mc)], -1)

        def forward(self, q, choices):
            B, K, L = choices.shape
            qx = q.unsqueeze(1).expand(B, K, L).reshape(B * K, L)
            return self.cls(self._pair(qx, choices.reshape(B * K, L))).view(B, K)

    return ESIM()


def train(ckpt_path: str | Path, *, epochs: int = 8, batch_size: int = 32,
          lr: float = 4e-4) -> None:
    import torch
    ckpt_path = Path(ckpt_path)
    items = load_csqa("train")
    vocab = _vocab([it.question for it in items] +
                   [t for it in items for t in it.choices.values()])
    glove = os.environ.get("GLOVE_PATH")
    emb = _load_glove(glove, vocab) if glove else (print("  WARNING: GLOVE_PATH unset — random-init") or None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _model(len(vocab), emb).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    rows = [(_enc(it.question, vocab), [_enc(it.choices[l], vocab) for l in LABELS],
             LABELS.index(it.answer_key)) for it in items]
    model.train()
    for ep in range(epochs):
        tot = cor = 0
        run = 0.0
        for i in range(0, len(rows), batch_size):
            b = rows[i:i + batch_size]
            q = torch.tensor([x[0] for x in b], device=device)
            c = torch.tensor([x[1] for x in b], device=device)
            y = torch.tensor([x[2] for x in b], device=device)
            logits = model(q, c)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * len(b); cor += (logits.argmax(1) == y).sum().item(); tot += len(b)
        print(f"  epoch {ep+1}/{epochs}: loss={run/tot:.4f} acc={cor/tot:.3f}", flush=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "vocab": vocab}, ckpt_path)
    print(f"saved ESIM -> {ckpt_path}")


def predict(items: list[CSQAItem], ckpt_path: str | Path) -> list[Prediction]:
    import torch
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found — run esim.train first")
    ck = torch.load(ckpt_path, map_location="cpu")
    vocab = ck["vocab"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _model(len(vocab)).to(device)
    model.load_state_dict(ck["state"]); model.eval()
    out: list[Prediction] = []
    for it in items:
        q = torch.tensor([_enc(it.question, vocab)], device=device)
        c = torch.tensor([[_enc(it.choices[l], vocab) for l in LABELS]], device=device)
        with torch.no_grad():
            pred = LABELS[int(model(q, c)[0].argmax())]
        out.append(Prediction(id=it.id, gold=it.answer_key, pred=pred,
                              correct=score(pred, it.answer_key)))
    return out
