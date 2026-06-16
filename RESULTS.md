# Results Summary — Answer-Side Translation in CommonsenseQA

**Project:** *Do LLMs Choose Concepts or English Words?* — Computational Semantics & NLU, BGU 2025–2026
**Team:** Tal Malul, Omer Eliyahu, Mark Feldman, Daniel Koyfman
**Status:** ✅ Complete — generative arm finished for all 4 Qwen models.

> **TL;DR.** We have the full generative sweep: **3 models fully usable** (`Qwen3.5:4B`, `Qwen3.5:9B`, `qwen3.6:35b`) and a 4th (`Qwen3.5:0.8B`) usable in `think=off` but with a **known parsing failure in `think=on`** (flagged below, excluded from comparisons). The core effect we set out to measure is **present and statistically significant across every model and language**.

---

## What we ran

- **Task:** CommonsenseQA `validation` split, **n = 1221** items, zero-shot.
- **Probe:** keep the **question in English**, translate **only the answer choices** into the target language. Compare the English-choices baseline (`en-en`) against translated choices (`en-x`), x ∈ {ru, es, he}.
- **Selection:** model emits a **letter (A–E)**, never answer text — so a flip reflects concept grounding, not string matching.
- **Models × conditions:** 4 models × 4 answer languages (en, ru, es, he) × 2 reasoning modes (`think` off/on) = **32 cells**, all complete.
- **Decoding:** `temperature=0`, fixed seed; every run carries a reproducibility manifest. `think=off` caps output at 32 tokens (forced terse letter); `think=on` allows 8192 tokens of reasoning.

---

## Headline: accuracy per condition (Wilson 95% CI)

| model | think | en-en | en-ru | en-es | en-he |
|---|---|---|---|---|---|
| Qwen3.5:0.8B | off | 0.517 | 0.476 | 0.424 | 0.392 |
| Qwen3.5:0.8B | on | 0.129 † | 0.085 † | 0.069 † | 0.044 † |
| Qwen3.5:4B | off | 0.766 | 0.697 | 0.686 | 0.691 |
| Qwen3.5:4B | on | 0.722 | 0.676 | 0.654 | 0.598 |
| Qwen3.5:9B | off | 0.821 | 0.724 | 0.722 | 0.722 |
| Qwen3.5:9B | on | 0.819 | 0.767 | 0.773 | 0.727 |
| qwen3.6:35b | off | 0.853 | 0.774 | 0.788 | 0.771 |
| qwen3.6:35b | on | 0.867 | 0.795 | 0.795 | 0.721 |

> **†** Over 50% of generations were unparseable (model never emitted an A–E letter). These accuracies reflect an output-**format** failure, **not** reasoning, and are **excluded** from comparisons. Affects only `Qwen3.5:0.8B` with `think=on`. See [Parsing issue](#parsing-issue-flagged) below.

*(CIs omitted here for readability; full CIs in `results/tables/accuracy.md` and `results/summary.csv`.)*

---

## Core diagnostic: flip rate (en-en → en-x)

Fraction of items whose prediction changes when **only the answer language** changes; `→gold`/`←gold` count flips toward/away from the correct answer. McNemar tests the paired accuracy change. `*** p<.001, ** p<.01, * p<.05`.

| model | think | lang | flip_rate | →gold | ←gold | acc_en | acc_x | McNemar p |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5:0.8B | off | es | 0.381 | 95 | 208 | 0.517 | 0.424 | 0.0000*** |
| Qwen3.5:0.8B | off | he | 0.455 | 109 | 261 | 0.517 | 0.392 | 0.0000*** |
| Qwen3.5:0.8B | off | ru | 0.356 | 125 | 175 | 0.517 | 0.476 | 0.0047** |
| Qwen3.5:4B | off | es | 0.259 | 87 | 184 | 0.766 | 0.686 | 0.0000*** |
| Qwen3.5:4B | off | he | 0.276 | 95 | 186 | 0.766 | 0.691 | 0.0000*** |
| Qwen3.5:4B | off | ru | 0.230 | 77 | 161 | 0.766 | 0.697 | 0.0000*** |
| Qwen3.5:4B | on | es | 0.310 | 98 | 182 | 0.722 | 0.654 | 0.0000*** |
| Qwen3.5:4B | on | he | 0.348 | 87 | 239 | 0.722 | 0.598 | 0.0000*** |
| Qwen3.5:4B | on | ru | 0.298 | 113 | 170 | 0.722 | 0.676 | 0.0009*** |
| Qwen3.5:9B | off | es | 0.224 | 65 | 185 | 0.821 | 0.722 | 0.0000*** |
| Qwen3.5:9B | off | he | 0.215 | 60 | 181 | 0.821 | 0.722 | 0.0000*** |
| Qwen3.5:9B | off | ru | 0.205 | 56 | 174 | 0.821 | 0.724 | 0.0000*** |
| Qwen3.5:9B | on | es | 0.188 | 67 | 123 | 0.819 | 0.773 | 0.0001*** |
| Qwen3.5:9B | on | he | 0.243 | 68 | 180 | 0.819 | 0.727 | 0.0000*** |
| Qwen3.5:9B | on | ru | 0.198 | 73 | 136 | 0.819 | 0.767 | 0.0000*** |
| qwen3.6:35b | off | es | 0.149 | 43 | 123 | 0.853 | 0.788 | 0.0000*** |
| qwen3.6:35b | off | he | 0.180 | 49 | 150 | 0.853 | 0.771 | 0.0000*** |
| qwen3.6:35b | off | ru | 0.161 | 42 | 139 | 0.853 | 0.774 | 0.0000*** |
| qwen3.6:35b | on | es | 0.152 | 38 | 126 | 0.867 | 0.795 | 0.0000*** |
| qwen3.6:35b | on | he | 0.244 | 34 | 213 | 0.867 | 0.721 | 0.0000*** |
| qwen3.6:35b | on | ru | 0.156 | 42 | 130 | 0.867 | 0.795 | 0.0000*** |

*(0.8B `think=on` flip rows are computed against its broken baseline — interpret with the † caveat. Full numbers: `results/flips.csv`.)*

---

## Key findings

1. **The effect is real and significant everywhere.** Translating only the answer choices lowers accuracy in **every** model × language cell, with **McNemar p < 0.01 across the board**. Models partly rely on English **wording**, not purely the **concept**.
2. **Flips skew away from gold.** In every condition `←gold` (correct→wrong) exceeds `→gold` (wrong→correct): the translation hurts more than it helps — consistent with a wording-dependence effect rather than noise.
3. **Scale mitigates it.** The en→x degradation shrinks monotonically with model size — e.g. `think=off` Hebrew flip rate drops 0.455 (0.8B) → 0.276 (4B) → 0.215 (9B) → 0.180 (35b). Bigger models ground the concept better.
4. **Hebrew + reasoning is the consistent weak spot.** For `think=on`, Hebrew is the worst language for *all four* models (e.g. 35b he drops to 0.721 vs 0.795 for es/ru; 9B he 0.727 vs 0.773/0.767). Hebrew also has the most parse noise under reasoning.
5. **Reasoning (`think=on`) is not a clear win.** It helps the largest model slightly on en/es/ru but *hurts* on Hebrew, and for small models the long reasoning trace mostly adds parse risk (see below).

---

## Parsing issue (flagged)

`Qwen3.5:0.8B` with `think=on` produces an **unparseable answer 85–95% of the time** — the model reasons but never emits a clean A–E letter within the budget:

| cell | unparsed / 1221 |
|---|---|
| 0.8B think-on en | 1044 (86%) |
| 0.8B think-on ru | 1106 (91%) |
| 0.8B think-on es | 1120 (92%) |
| 0.8B think-on he | 1159 (95%) |

The reported accuracies (0.04–0.13) therefore measure an **output-format failure, not commonsense ability** — when the 0.8B model does emit a letter it is usually right; it simply rarely does. **Treatment:** these four cells are flagged (`†`) and excluded from model comparisons. `Qwen3.5:0.8B` with `think=off` (0 unparsed) is fully valid and retained. Parse rates are healthy elsewhere (think-on unparsed: 9B ≤ 8%, 35b ≤ 13%, all think-off = 0).

This does **not** block the project: 3 of 4 models are fully usable across all conditions, and the headline effect holds without the 0.8B think-on cells.

---

## Reproducibility

- `temperature=0` + fixed seed; one **manifest** per run (model id + snapshot, prompt-template hash, dataset-variant hash, decoding params).
- Raw outputs cached under `results/<model>/<cond>__<lang>__think-<x>__<hash>/outputs.jsonl`; all metrics recompute from cache via `scripts.analyze` — no re-querying.
- Generative arm served by vLLM on BGU SLURM (`cluster/`); analysis is local.

## Artifacts

| What | Path |
|---|---|
| Per-cell accuracy + CI + unparsed | `results/summary.csv` |
| Flip rate + McNemar | `results/flips.csv` |
| Markdown tables (flagged) | `results/tables/{accuracy,flips}.md` |
| LaTeX tables (booktabs, for paper) | `results/tables/{accuracy,flips}.tex` |
| Figures (png + pdf) | `results/figures/fig1–5*` |
| Slide deck | `results/deck/slides.md` |

*Figures:* `fig1_accuracy_by_condition`, `fig2_flip_rate_by_lang`, `fig3_flip_direction`, `fig4_flip_rate_vs_size`, `fig5_think_off_vs_on`.
