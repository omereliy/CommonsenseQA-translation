# Haiku-4.5 subagent arm — exact prompt (frozen 2026-06-19)

Claude Haiku 4.5 answering CSQA via Claude-Code **subagents** (subscription-billed,
not the metered API). Items are batched (50/agent) and each agent returns an
`{id: letter}` map under a strict schema; answers are assembled into the standard
`outputs.jsonl` by `scripts/ingest_subagent.py`. The run manifest flags it
**non-deterministic** (`deterministic: false`, temperature not pinned) — an
exploratory LLM upper-anchor, not a bit-reproducible arm like the encoder/vLLM/Gemini ones.

## 1. The instruction each Haiku agent receives (held CONSTANT across all conditions)

The instruction does NOT mention the choice language, so it is identical for
en-en and every en-x condition — only the choice *text* in the batch file changes.
`{dir}` = `/tmp/csqa_batches/<condition>`, `{NNN}` = zero-padded batch index.

```
You are answering CommonsenseQA multiple-choice questions. Read the file {dir}/batch_{NNN}.json — a JSON array of items, each with "id", "question", and "choices" (a map of letters A-E to answer text). For EACH item choose the single best answer letter (A, B, C, D, or E) using commonsense reasoning. Return every item's id paired with your chosen letter. Do not skip any item; answer all of them.
```

## 2. How one item appears to the model (the batch JSON entry it reads)

**en-en** (English question, English choices):
```json
{
  "id": "1afa02df02c908a558b4036e80242fac",
  "question": "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?",
  "choices": {
    "A": "bank",
    "B": "library",
    "C": "department store",
    "D": "mall",
    "E": "new york"
  }
}
```

**en-x · he** (English question, Hebrew choices — same item, choices translated):
```json
{
  "id": "1afa02df02c908a558b4036e80242fac",
  "question": "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?",
  "choices": {
    "A": "בנק",
    "B": "ספריה",
    "C": "חנות כלבו",
    "D": "קניון",
    "E": "ניו יורק"
  }
}
```

The model sees the English question and the A–E choice labels; only the choice
*text* is translated. Selection is the **letter** (load-bearing convention #2),
so scoring never string-matches the translated answer text.

## 3. Canonical per-item prompt (for cross-reference)

The deterministic arms (encoders / vLLM / Gemini) send one item per call rendered
from `PROMPT_TEMPLATE` (hashed into their manifests). The subagent arm presents the
*same content* as a JSON batch instead, so accuracy/flip numbers are comparable, but
the surface formatting differs — recorded here for transparency.

`PROMPT_TEMPLATE`:
```
{question}

{choices}

Answer with the letter (A, B, C, D, or E) of the single correct option. Respond with just the letter.
```

Rendered (en-en, this item):
```
A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?

A. bank
B. library
C. department store
D. mall
E. new york

Answer with the letter (A, B, C, D, or E) of the single correct option. Respond with just the letter.
```

Rendered (en-x · he, this item):
```
A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?

A. בנק
B. ספריה
C. חנות כלבו
D. קניון
E. ניו יורק

Answer with the letter (A, B, C, D, or E) of the single correct option. Respond with just the letter.
```
