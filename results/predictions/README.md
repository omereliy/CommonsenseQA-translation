# Per-item model predictions

Raw per-question answers for **every model under every answer-language condition** —
not just the aggregate accuracy/flip summaries in `results/summary.csv` and
`results/flips.csv`.

## `predictions_long.csv`

One row per (item × model × condition × language). Columns:

| column | meaning |
|---|---|
| `id` | CommonsenseQA item id (stable across conditions; join key for flips) |
| `model` | arm: `Qwen3.5:0.8B`, `Qwen3.5:4B`, `haiku-4.5-subagent`, `xlmr-ep3/ep6`, `mbert-ep3/ep6` |
| `think` | reasoning mode for the generative arms (`off`/`on`); `na` otherwise |
| `condition` | `en-en`, `en-x`, or `mixed` |
| `lang` | answer-choice language: `en`, `ru`, `es`, `he` (or `mix` / `*-nllb`, `*-opus`, `*-consensus` translation-source variants) |
| `gold` | gold choice letter (A–E) |
| `pred` | model's predicted letter (A–E) |
| `correct` | `True`/`False` |

The model's raw emitted text before letter-parsing (`raw`), reasoning traces, and
token counts are omitted here to keep the table one-row-per-line; they are kept in
full in the per-run `outputs.jsonl` files (below).

To recover a **flip** for a model between two languages, join `en-en` vs `en-x`
rows on `id` and compare `pred`. Selection is letter-based (A–E), never answer
text, so a changed `pred` is a genuine decision change, not a string-match artifact.

## Raw run outputs

The full per-run files live next to each run as
`results/<model>/<cond>__<lang>__<hash>/outputs.jsonl` (with a `manifest.json`
recording model snapshot, prompt-template hash, dataset-variant hash, and decoding
params). Those carry extra fields the consolidated CSV omits — reasoning traces
(`thinking`), token counts, and `done_reason`.

Regenerate the consolidated CSV from the raw outputs at any time; it is derived
from the `outputs.jsonl` files alone.
