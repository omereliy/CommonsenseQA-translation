"""Persist one (model, variant) run: raw outputs + a reproducibility manifest.

Layout (eval/README.md): results/<model>/<condition>__<lang>[__think-<m>]__<vhash>/
with outputs.jsonl + manifest.json. A completed run is a cache hit (outputs.jsonl
present) so accuracy/flip rates recompute without re-querying the model.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from csqa_xlang.eval.base import Prediction
from csqa_xlang.manifest import build_manifest, write_manifest
from csqa_xlang.variants.conditions import Variant


def run_dir(results_root: str | Path, model_tag: str, variant: Variant,
            think: str | None = None) -> Path:
    name = f"{variant.condition}__{variant.language}"
    if think is not None:
        name += f"__think-{think}"
    name += f"__{variant.variant_hash[:8]}"
    return Path(results_root) / model_tag.replace("/", "_").replace(":","-") / name


def is_cached(results_root, model_tag, variant, think=None) -> bool:
    return (run_dir(results_root, model_tag, variant, think) / "outputs.jsonl").exists()


def write_run(results_root: str | Path, *, model_tag: str, model_snapshot: str,
              variant: Variant, predictions: list[Prediction], decoding: dict[str, Any],
              prompt_template: str, think: str | None = None) -> Path:
    rd = run_dir(results_root, model_tag, variant, think)
    rd.mkdir(parents=True, exist_ok=True)
    with (rd / "outputs.jsonl").open("w", encoding="utf-8") as f:
        for p in predictions:
            row = p.to_dict()
            row.update(model=model_tag, condition=variant.condition,
                       lang=variant.language, think=think or "na")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len(predictions)
    acc = sum(p.correct for p in predictions) / n if n else 0.0
    manifest = build_manifest(
        model_id=model_tag, model_snapshot=model_snapshot,
        prompt_template=prompt_template,
        variant=[asdict(it) for it in variant.items],
        decoding=decoding, condition=variant.condition, language=variant.language,
        extra={"think": think or "na", "n": n, "accuracy": round(acc, 4),
               "unparsed": sum(p.pred is None for p in predictions)},
    )
    write_manifest(manifest, rd / "manifest.json")
    return rd
