# `eval/` — run models on the variants

**Responsibility:** for a given model and condition variant, prompt the model on each
item, parse its choice, and write **raw outputs + a manifest** to `results/`.

**Conventions (load-bearing — see CLAUDE.md):**

- **Letter-based selection.** Prompt the model to answer with a choice **label (A–E)**,
  not the answer text. Parse the letter robustly (the model may add prose). Never
  string-match the translated answer text — that would confound the flip metric.
- **Deterministic decoding.** `temperature=0`, fixed seed, small `max_tokens`.
- **Cache raw outputs.** Persist the full model response per item so accuracy/flip
  rates recompute without re-querying. Re-running a completed (model, variant) is a
  cache hit.
- **Write a manifest** beside the outputs via `csqa_xlang.manifest.build_manifest` /
  `write_manifest`: model id + snapshot, prompt-template hash, variant hash, decoding
  params. This is what makes a run reproducible and auditable.

**Design intent:**

- **Pluggable backend.** A small `predict(prompt) -> str` interface with `hf` (open
  multilingual models via `transformers`) and `api` (OpenAI/Anthropic)
  implementations. Which to use is a *pending decision* (CLAUDE.md).
- Keep prompting (template) separate from the backend so the template hash is stable
  across backends.

Suggested `results/` layout: `results/<model>/<condition>__<lang>__<variant_hash>/`
containing `outputs.jsonl` + `manifest.json`.
