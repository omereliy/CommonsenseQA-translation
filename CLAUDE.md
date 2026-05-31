# CLAUDE.md — Project Guide

> Draft. This file orients Claude Code (and the team) to the project. Keep it
> current as decisions get made — it is the single source of truth for shared
> conventions.

## Project

**Title:** Do LLMs Choose Concepts or English Words? Answer-Side Translation in CommonsenseQA

**Course:** Computational Semantics and Natural Language Understanding, 2025–2026 (BGU)

**Team:** Tal Malul, Omer Eliyahu, Mark Feldman, Daniel Koyfman

**Base paper:** Talmor et al., *CommonsenseQA: A Question Answering Challenge
Targeting Commonsense Knowledge*, NAACL 2019.

### Research question

Do language models choose the correct commonsense **concept**, or do they rely on
English answer **wording**?

We keep CommonsenseQA **questions in English** and translate **only the answer
choices** into target languages. This isolates cross-lingual concept grounding:
the model reads an English question but must map it to candidate concepts
expressed in another language. We measure prediction **flips** caused solely by
changing the answer language.

### Experimental conditions (Q-language → A-language)

| Condition | Question | Answers |
|-----------|----------|---------|
| `EN-EN`   | English  | English (baseline) |
| `EN-X`    | English  | Target language X |
| `X-X`     | X        | X |
| `X-EN`    | X        | English |

Plus two extensions: **mixed answer sets** (some choices in X₁, some in X₂) and
**partial question-noun translation** (translate select nouns in the question).

`X` ∈ candidate target languages: Hebrew, Arabic, and one high-resource language
(Spanish or French). *(Final set is a pending decision — see below.)*

## Load-bearing conventions (do not silently change these)

These three encode design decisions that protect the **Soundness** and
**Reproducibility** grades. Changing them invalidates cross-run comparisons.

1. **Evaluation split = `validation` (~1,221 items).** CommonsenseQA's `test`
   split has withheld `answerKey` labels (held out for the leaderboard), so it
   cannot be scored locally. All scored evaluation runs use the validation split.
   The data loader enforces this default.
2. **Answer selection is letter-based (A–E), never answer text.** The model is
   asked to emit a choice **label** (A–E), not the answer string. Emitting Hebrew
   or Arabic text would make string-matching a confound that masquerades as a
   concept-grounding effect. Parse the letter; map back to the choice.
3. **Deterministic decoding + manifests.** Every eval run uses `temperature=0`
   with a fixed `seed`, and writes a **run manifest** next to its raw outputs
   recording: model id + snapshot/date, prompt-template hash, dataset-variant
   hash, decoding params. Raw model outputs are cached so metrics (accuracy, flip
   rate) can be recomputed without re-querying the model.

## Metrics

- **Accuracy** per condition.
- **Flip rate**: fraction of items where the prediction changes between two
  conditions (e.g., `EN-EN` → `EN-X`) — the core diagnostic.
- **Statistical significance** on condition deltas (small differences need a test;
  see the writing recommendations — pick the test by metric type, e.g. McNemar
  for paired accuracy on the same items).

## Repository layout

```
.
├── CLAUDE.md            # this file
├── paper/               # LaTeX report (ACL-style); editable in Overleaf
│   ├── main.tex
│   ├── references.bib
│   └── sections/        # one .tex per section (template-agnostic)
├── src/csqa_xlang/      # experiment package: config · manifest · data ·
│                        #   translation · variants · eval · analysis
├── configs/             # YAML run configs (seed, temp, langs, paths)
├── scripts/             # CLI entry points
├── data/                # raw · translated · variants  (git-ignored payloads)
├── results/             # run outputs + manifests       (git-ignored payloads)
├── pyproject.toml · requirements.txt · .env.example
├── .venv/               # local virtual environment     (git-ignored)
└── .claude/             # Claude Code settings
```

The Python project lives at the repository root (`src/` layout; `pyproject.toml`).
Each major `src/csqa_xlang/*` subpackage has its own `README.md` describing
responsibilities. Read those before adding code there — the architecture is
intentionally described in prose so the four of us can shape it together rather
than inherit a speculative skeleton.

## Pipeline (intended flow)

`load CSQA (validation)` → `translate answer choices (+ optional question nouns)`
→ `build condition variants` → `evaluate model(s)` → `analyze (accuracy, flips,
significance)`.

## Conventions

- **Language:** Python ≥ 3.10, `src/` layout, package importable as `csqa_xlang`.
- **Config over flags:** runs are driven by a YAML in `configs/`; don't
  hardcode model names, languages, or paths in modules.
- **Determinism:** seed everything; `temperature=0` for eval.
- **Data is derived, never committed:** `data/**` and `results/**`
  payloads are git-ignored; the *recipe* (config + code + manifest) is what makes
  a run reproducible. Keep small canonical samples if a fixture is needed.
- **Report:** 6 pages, 11pt, ACL-style. Cite **conference/journal** versions (not
  arXiv) and verify entries against the ACL Anthology.

## Decisions pending (resolve deliberately as a team — don't inherit a stub default)

- [ ] **Target languages** — confirm the X set (Hebrew + Arabic + Spanish *or*
      French?).
- [ ] **Translation backend** — MT (e.g. NLLB / Google / DeepL) vs LLM-based
      translation. Affects quality, cost, and whether human validation is needed.
      The `translation/` module is pluggable so this isn't locked yet.
- [ ] **Evaluation target(s)** — API LLMs (OpenAI/Anthropic) vs open multilingual
      HF models (e.g. XLM-R / Aya / Qwen). The `eval/` module is pluggable.
- [ ] **Human validation** — whether/which subset to human-check for
      meaning-preserving translations.
- [ ] **Prompt format** — exact template; how choices are listed; few-shot vs
      zero-shot.

## Grading rubric (keep these in mind while building)

Reproducibility 15 · Presentation Quality 30 · Soundness 30 · Depth 15 ·
Related Work 10.
