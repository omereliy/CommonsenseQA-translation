# Haiku-4.5 subagent arm — frozen 2026-06-19

Claude Haiku 4.5 answering CSQA via Claude-Code **subagents** (subscription-billed,
not the metered API). Items are batched (50/agent) and each agent returns an
`{id: letter}` map under a strict schema; answers are assembled into the standard
`outputs.jsonl` by `scripts/ingest_subagent.py`. The run manifest flags it
**non-deterministic** (`deterministic: false`, temperature not pinned) — an
exploratory LLM upper-anchor, not a bit-reproducible arm like the encoder/vLLM/Gemini ones.

Validation, n=1221 per condition. Cost: ~0.47M (en-en) + ~1.5M (en-x×3) ≈ **2M
Haiku tokens** on the Max plan, ~7 min wall-clock across two background workflows.

## Results

**Accuracy** (`accuracy.csv`):

| condition | acc | 95% CI | Δ vs en-en |
|---|---|---|---|
| en-en    | 0.854 | [0.833, 0.873] | — |
| en-x es  | 0.833 | [0.811, 0.853] | −2.1 |
| en-x he  | 0.817 | [0.795, 0.838] | −3.7 |
| en-x ru  | 0.812 | [0.790, 0.833] | −4.2 |

**Flip rate, en-en → en-x** — the core diagnostic (`flips.csv`):

| lang | flip | →gold | ←gold | net | McNemar p |
|---|---|---|---|---|---|
| ru | 0.128 | 47 | 98 | −51 | 0.00003 |
| he | 0.149 | 61 | 106 | −45 | 0.00066 |
| es | 0.131 | 60 | 86 | −26 | 0.039 |

**Reading.**
1. Translating *only the choices* flips **13–15%** of predictions, net *away from
   gold*, all McNemar-significant (p<0.05; ru/he p<0.001). So even a frontier LLM is
   **partly anchored to English wording**, not purely concept-grounded — the RQ effect.
2. But Haiku degrades **far less than the fine-tuned encoders**: en→he is **−3.7%**
   here vs **≈−30%** for XLM-R. The concept-grounding confound is real but
   **capability-dependent** — a strong model resists answer-language shifts much better.
3. **Hardest target language is model-dependent.** Encoders: en > ru ≈ es > **he**
   (Hebrew worst). Haiku: en > es > he ≈ **ru** (Russian lowest acc; Hebrew most churn).

`he` has 1 unparsed item (a duplicate `_1` id the agent mis-keyed) — scored wrong;
1/1221 is immaterial.

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
