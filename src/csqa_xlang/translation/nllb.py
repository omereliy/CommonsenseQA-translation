"""NLLB-200 backend — open, local MT (Meta's No Language Left Behind).

A second, fully-reproducible translation source to compare against Google: runs
locally (no API key/billing), deterministic beam search, covers he/es/ru. Used to
test whether the en→x degradation is a property of cross-lingual concept grounding
or an artifact of one MT system. Same `Translator` surface as the Google backend.

Cache: keyed by (text, tgt) under `data/translated/_cache/nllb_<tgt>.json`, so a
re-run (or a crash) doesn't re-translate. Short codes are mapped to FLORES-200.
"""

from __future__ import annotations

import json
from pathlib import Path

# short code -> FLORES-200 code used by NLLB
FLORES = {"en": "eng_Latn", "he": "heb_Hebr", "es": "spa_Latn", "ru": "rus_Cyrl"}
_BATCH = 32


class NLLBTranslator:
    name = "nllb"

    def __init__(self, cache_dir: str | Path, model: str = "facebook/nllb-200-distilled-600M",
                 num_beams: int = 4, max_length: int = 64):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model
        self.num_beams = num_beams
        self.max_length = max_length
        self._tok = None
        self._model = None
        self._device = None

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self._device).eval()

    def _cache_path(self, tgt: str) -> Path:
        return self.cache_dir / f"nllb_{tgt}.json"

    def _load_cache(self, tgt: str) -> dict[str, str]:
        p = self._cache_path(tgt)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def _mt(self, strings: list[str], src: str, tgt: str) -> list[str]:
        import torch
        self._ensure_model()
        self._tok.src_lang = FLORES[src]
        forced_bos = self._tok.convert_tokens_to_ids(FLORES[tgt])
        out: list[str] = []
        for i in range(0, len(strings), _BATCH):
            chunk = strings[i:i + _BATCH]
            enc = self._tok(chunk, return_tensors="pt", padding=True, truncation=True,
                            max_length=self.max_length).to(self._device)
            with torch.no_grad():
                gen = self._model.generate(**enc, forced_bos_token_id=forced_bos,
                                           num_beams=self.num_beams, max_length=self.max_length)
            out.extend(self._tok.batch_decode(gen, skip_special_tokens=True))
        return out

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        if src == tgt:
            return list(texts)
        if src not in FLORES or tgt not in FLORES:
            raise ValueError(f"NLLB backend supports {sorted(FLORES)}; got src={src} tgt={tgt}")
        cache = self._load_cache(tgt)
        todo = [t for t in dict.fromkeys(texts) if t not in cache]  # unique, uncached
        if todo:
            for out in (self._mt(todo, src, tgt),):
                cache.update(dict(zip(todo, out)))
            self._cache_path(tgt).write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        return [cache[t] for t in texts]
