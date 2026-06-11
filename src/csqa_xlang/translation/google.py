"""Google Cloud Translation (v2 REST) backend with an on-disk cache.

Resolved backend decision (CLAUDE.md): answer choices are translated with Google
Translate. The v2 REST endpoint is used directly so the only dependency is
`requests` (no google-cloud client / GCP project wiring needed) — auth is a
Cloud Translation **API key** in `GOOGLE_API_KEY`.

Cache: translations are keyed by (text, src, tgt) and persisted under
`data/translated/_cache/google_<tgt>.json`, so the paid step runs once and is
reproducible; a crash mid-run resumes without re-billing translated strings.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

ENDPOINT = "https://translation.googleapis.com/language/translate/v2"
_BATCH = 100  # API allows up to 128 strings/request


class GoogleTranslator:
    name = "google"

    def __init__(self, cache_dir: str | Path, api_key: str | None = None, sleep: float = 0.1):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("set GOOGLE_API_KEY (Cloud Translation API key) for the google backend")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep = sleep

    def _cache_path(self, tgt: str) -> Path:
        return self.cache_dir / f"google_{tgt}.json"

    def _load_cache(self, tgt: str) -> dict[str, str]:
        p = self._cache_path(tgt)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def _api_translate(self, strings: list[str], src: str, tgt: str) -> list[str]:
        resp = requests.post(
            ENDPOINT,
            params={"key": self.api_key},
            json={"q": strings, "source": src, "target": tgt, "format": "text"},
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Google Translate {resp.status_code}: {resp.text[:300]}")
        return [t["translatedText"] for t in resp.json()["data"]["translations"]]

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        if src == tgt:
            return list(texts)
        cache = self._load_cache(tgt)
        todo = [t for t in dict.fromkeys(texts) if t not in cache]  # unique, uncached
        cache_path = self._cache_path(tgt)
        for i in range(0, len(todo), _BATCH):
            chunk = todo[i:i + _BATCH]
            out = self._api_translate(chunk, src, tgt)
            cache.update(dict(zip(chunk, out)))
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            if self.sleep:
                time.sleep(self.sleep)
        return [cache[t] for t in texts]
