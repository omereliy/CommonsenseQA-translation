"""Run models over condition variants; cache raw outputs + manifests (README.md).

Three arms, all returning `Prediction`s (chosen label A-E): generative (vLLM-served
Qwen lineup), xlmr (fine-tuned multilingual encoder), esim (English-only anchor).
The runner persists outputs.jsonl + a manifest per (model, variant); letter-based
selection + deterministic decoding are load-bearing conventions (CLAUDE.md).
"""

from __future__ import annotations

from csqa_xlang.eval import esim, xlmr
from csqa_xlang.eval.base import Prediction, score
from csqa_xlang.eval.prompt import (LABELS, PROMPT_TEMPLATE, build_messages,
                                    build_prompt, extract_letter)
from csqa_xlang.eval.runner import is_cached, run_dir, write_run
from csqa_xlang.eval.vllm_backend import run_generative

__all__ = ["Prediction", "score", "run_generative", "xlmr", "esim",
           "build_prompt", "build_messages", "extract_letter", "PROMPT_TEMPLATE",
           "LABELS", "run_dir", "is_cached", "write_run"]
