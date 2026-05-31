"""Run manifests and content hashing — the backbone of reproducibility.

Every eval run should write a manifest next to its raw outputs so that accuracy
and flip rates can be recomputed (and audited) without re-querying the model.
A manifest records *what produced these outputs*: model id + snapshot, the exact
prompt template, the dataset-variant content, and the decoding params.

This is intentionally small and dependency-free.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def content_hash(obj: Any, *, length: int = 12) -> str:
    """Stable short hash of any JSON-serializable object.

    Use it to fingerprint a prompt template, a dataset variant, or a config so two
    runs can be compared for "same inputs?" at a glance.
    """
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def build_manifest(
    *,
    model_id: str,
    model_snapshot: str,
    prompt_template: str,
    variant: Any,
    decoding: dict[str, Any],
    condition: str,
    language: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a manifest dict for one eval run.

    Args:
        model_id: e.g. "gpt-4o" or an HF repo id.
        model_snapshot: a pinned version/date/commit (e.g. "2024-08-06"). The point
            is that "gpt-4o" alone is not reproducible; the snapshot is.
        prompt_template: the literal template string (hashed into the manifest).
        variant: the dataset-variant object/records (hashed into the manifest).
        decoding: decoding params (must include temperature=0 for scored runs).
        condition: one of en-en / en-x / x-x / x-en / mixed / noun.
        language: the target language X, if applicable.

    NOTE: timestamps are deliberately NOT added here (keeps the manifest a pure
    function of inputs). Stamp wall-clock time at the call site if you want it.
    """
    manifest = {
        "model": {"id": model_id, "snapshot": model_snapshot},
        "prompt_template_hash": content_hash(prompt_template),
        "variant_hash": content_hash(variant),
        "decoding": decoding,
        "condition": condition,
        "language": language,
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    """Write a manifest as pretty JSON next to the run's raw outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
    return path
