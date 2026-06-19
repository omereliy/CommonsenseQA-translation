"""Opus-MT backend — open, local MT (Helsinki-NLP Marian models).

A third, fully-reproducible translation source alongside Google and NLLB. Opus-MT
ships one model per language pair (en->es, en->ru, en->he), so we lazy-load a
model per target language. Same `Translator` surface; deterministic beam search;
no API key/billing. Used to check the en->x degradation across a third MT system.

Cache: keyed by (text, tgt) under `data/translated/_cache/opus_<tgt>.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

# target code -> Helsinki-NLP Marian model (en -> X)
PAIR_MODEL = {
    "es": "Helsinki-NLP/opus-mt-en-es",
    "ru": "Helsinki-NLP/opus-mt-en-ru",
    "he": "Helsinki-NLP/opus-mt-en-he",
}
_BATCH = 32


class OpusMTTranslator:
    name = "opus"

    def __init__(self, cache_dir: str | Path, num_beams: int = 4, max_length: int = 64):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_beams = num_beams
        self.max_length = max_length
        self._device = None
        self._cache: dict[str, tuple] = {}    # tgt -> (tokenizer, model)

    def _ensure_model(self, tgt: str):
        if tgt in self._cache:
            return self._cache[tgt]
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        name = PAIR_MODEL[tgt]
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSeq2SeqLM.from_pretrained(name).to(self._device).eval()
        self._cache[tgt] = (tok, model)
        return self._cache[tgt]

    def _cache_path(self, tgt: str) -> Path:
        return self.cache_dir / f"opus_{tgt}.json"

    def _load_cache(self, tgt: str) -> dict[str, str]:
        p = self._cache_path(tgt)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def _mt(self, strings: list[str], tgt: str) -> list[str]:
        import torch
        tok, model = self._ensure_model(tgt)
        out: list[str] = []
        for i in range(0, len(strings), _BATCH):
            chunk = strings[i:i + _BATCH]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=self.max_length).to(self._device)
            with torch.no_grad():
                gen = model.generate(**enc, num_beams=self.num_beams, max_length=self.max_length)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return out

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        if src == tgt:
            return list(texts)
        if src != "en" or tgt not in PAIR_MODEL:
            raise ValueError(f"Opus-MT backend supports en->{sorted(PAIR_MODEL)}; got {src}->{tgt}")
        cache = self._load_cache(tgt)
        todo = [t for t in dict.fromkeys(texts) if t not in cache]
        if todo:
            cache.update(dict(zip(todo, self._mt(todo, tgt))))
            self._cache_path(tgt).write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        return [cache[t] for t in texts]
