"""Generative arm: zero-shot Qwen lineup served by vLLM.

The model is served separately (cluster/serve_and_eval.sbatch); this hits the
OpenAI-compatible endpoint via the copied VLLMClient, builds the letter prompt,
and parses the chosen label. Deterministic decoding (temperature=0). think=on
routes a reasoning trace into Prediction.thinking.
"""

from __future__ import annotations

import asyncio

from csqa_xlang.data import CSQAItem
from csqa_xlang.eval.base import Prediction, score
from csqa_xlang.eval.prompt import build_messages, extract_letter
from csqa_xlang.eval.vllm_client import VLLMClient


async def _one(client, model_hf, item: CSQAItem, think: bool, num_predict: int,
               temperature: float, sem) -> Prediction:
    async with sem:
        resp = await client.chat(
            model=model_hf,
            messages=build_messages(item.question, item.choices),
            options={"temperature": temperature, "num_predict": num_predict},
            think=think,
        )
    msg = resp["message"]
    pred = extract_letter(msg.get("content", ""), item.choices)
    return Prediction(
        id=item.id, gold=item.answer_key, pred=pred, correct=score(pred, item.answer_key),
        raw=msg.get("content", ""), thinking=msg.get("thinking", ""),
        prompt_tokens=resp.get("prompt_eval_count", 0),
        completion_tokens=resp.get("eval_count", 0),
        done_reason=resp.get("done_reason", ""),
        truncated=resp.get("done_reason") == "length",
    )


async def _run_async(items, model_hf, base_url, think, num_predict, temperature,
                     concurrency) -> list[Prediction]:
    client = VLLMClient(base_url)
    sem = asyncio.Semaphore(concurrency)
    try:
        return await asyncio.gather(
            *[_one(client, model_hf, it, think, num_predict, temperature, sem) for it in items]
        )
    finally:
        await client.aclose()


def run_generative(items: list[CSQAItem], *, model_hf: str, base_url: str,
                   think: bool = False, num_predict: int = 32,
                   temperature: float = 0.0, concurrency: int = 8) -> list[Prediction]:
    """Evaluate one served generative model over `items`; returns Predictions."""
    return asyncio.run(_run_async(items, model_hf, base_url, think, num_predict,
                                  temperature, concurrency))
