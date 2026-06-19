"""Gemini arm: zero-shot Google Gemini 2.x via the OpenAI-compatible endpoint.

A managed-API generative arm alongside the vLLM-served Qwen lineup. It points the
same `openai` client the vLLM arm already uses at Google's OpenAI-compat endpoint
(https://generativelanguage.googleapis.com/v1beta/openai/), so the identical
letter prompt (`prompt.build_messages`), letter parsing (`extract_letter`), and
`Prediction` records flow through `write_run` unchanged — Gemini becomes a
config-driven arm, not a web-UI paste job. Choosing the OpenAI-compat layer over
the `google-genai` SDK keeps the dependency surface at `openai` (already wired).

Auth: `GEMINI_API_KEY` (a Google AI Studio key; distinct from `GOOGLE_API_KEY`,
which is the Cloud Translation key). Pass `api_key=` to override.

Determinism caveat (read scored Gemini runs as *approximately* deterministic):
Gemini does not guarantee bit-exact reproducibility even at temperature=0, and
gemini-2.5-pro is a *thinking* model whose reasoning budget cannot be forced to
zero (min ~128 tokens). So `think=off` here means **minimum** thinking budget,
not "no reasoning"; `think=on` lets the budget grow (dynamic) and surfaces the
trace into `Prediction.thinking`. The thinking_config + temperature land in the
manifest decoding block so the run records exactly what was asked of the model.
Because thinking tokens are billed as output and emitted before the answer, the
output budget (`num_predict`) must comfortably exceed the thinking budget or the
letter gets truncated — hence the larger think-off default than the vLLM arm.
"""

from __future__ import annotations

import asyncio
import os

from openai import APITimeoutError, AsyncOpenAI, RateLimitError

from csqa_xlang.data import CSQAItem
from csqa_xlang.eval.base import Prediction, score
from csqa_xlang.eval.prompt import build_messages, extract_letter

# Google's OpenAI-compatible base; the openai client appends /chat/completions.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Retry/backoff for the free + low tiers, which enforce tight per-minute limits
# and answer bursts with HTTP 429. Exponential backoff (honouring Retry-After
# when present) lets a sweep ride through rate-limit 429s. A persistent 429
# (free-tier quota exhausted / a model with no free quota, e.g. gemini-2.5-pro)
# still surfaces after the retries — that is a billing problem, not a transient.
_MAX_RETRIES = 6
_BACKOFF_BASE = 2.0  # seconds: 2, 4, 8, 16, 32, 64


def _thinking_config(think: bool) -> dict:
    """Gemini-specific thinking knob, in the OpenAI-compat passthrough shape.

    `thinking_budget`: 0 → minimum (pro clamps to its floor); -1 → dynamic.
    `include_thoughts` mirrors the vLLM arm's reasoning trace when think=on.
    """
    budget = -1 if think else 0
    return {"extra_body": {"google": {"thinking_config": {
        "thinking_budget": budget, "include_thoughts": think}}}}


def _retry_after_seconds(err: RateLimitError, attempt: int) -> float:
    """Honour a Retry-After header if the 429 carries one, else exponential backoff."""
    resp = getattr(err, "response", None)
    hdr = resp.headers.get("retry-after") if resp is not None else None
    if hdr:
        try:
            return float(hdr)
        except ValueError:
            pass
    return _BACKOFF_BASE * (2 ** attempt)


async def _one(client, model, item: CSQAItem, think: bool, num_predict: int,
               temperature: float, sem) -> Prediction:
    async with sem:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=build_messages(item.question, item.choices),
                    temperature=temperature,
                    max_tokens=num_predict,
                    extra_body=_thinking_config(think),
                )
                break
            except (RateLimitError, APITimeoutError) as e:
                if attempt == _MAX_RETRIES:
                    raise
                await asyncio.sleep(_retry_after_seconds(e, attempt)
                                    if isinstance(e, RateLimitError)
                                    else _BACKOFF_BASE * (2 ** attempt))
    choice = resp.choices[0]
    msg = choice.message
    content = msg.content or ""
    pred = extract_letter(content, item.choices)
    usage = getattr(resp, "usage", None)
    return Prediction(
        id=item.id, gold=item.answer_key, pred=pred,
        correct=score(pred, item.answer_key),
        raw=content,
        thinking=getattr(msg, "reasoning_content", None) or "",
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
        done_reason=choice.finish_reason or "",
        truncated=choice.finish_reason == "length",
    )


async def _run_async(items, model, api_key, base_url, think, num_predict,
                     temperature, concurrency) -> list[Prediction]:
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(concurrency)
    try:
        return await asyncio.gather(
            *[_one(client, model, it, think, num_predict, temperature, sem)
              for it in items]
        )
    finally:
        await client.close()


def run_gemini(items: list[CSQAItem], *, model: str = "gemini-2.5-pro",
               api_key: str | None = None, base_url: str | None = None,
               think: bool = False, num_predict: int = 512,
               temperature: float = 0.0, concurrency: int = 4) -> list[Prediction]:
    """Evaluate one Gemini model over `items` via the OpenAI-compat endpoint.

    Returns `Prediction`s (chosen label A-E). `api_key` defaults to
    $GEMINI_API_KEY; raises if neither is set. Concurrency defaults low to
    respect AI Studio rate limits.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "gemini arm needs an API key: set $GEMINI_API_KEY (AI Studio) "
            "or pass api_key=")
    return asyncio.run(_run_async(
        items, model, api_key, base_url or GEMINI_BASE_URL, think, num_predict,
        temperature, concurrency))
