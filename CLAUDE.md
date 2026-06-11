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

`X` ∈ **{Russian, Spanish, Hebrew}** (`ru`, `es`, `he`); English is the baseline.
Arabic from the initial proposal was dropped. We implement the **`en-en` / `en-x`
core** (English question, choices in X) first; `x-x` / `x-en` are wired but gated
behind `extensions.translate_questions` (they also translate the question), and
`mixed` / `noun` remain extension stubs.

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

## Decisions (resolved — reflected in `configs/default.yaml` + the package)

- [x] **Target languages** — English baseline + **Russian, Spanish, Hebrew**
      (`ru`, `es`, `he`). Arabic dropped.
- [x] **Translation backend** — **Google Cloud Translation** (v2 REST, API key in
      `GOOGLE_API_KEY`), choices only, cached under `data/translated/_cache/`.
      Still pluggable behind the `Translator` protocol in `translation/`.
- [x] **Evaluation target(s)** — **three arms**: (1) generative **Qwen lineup**
      (`Qwen3.5` 0.8/4/9B, `qwen3.6:35b`, `gemma4:26b-a4b`) served by **vLLM on
      BGU**, zero-shot, `think {off,on}`; (2) **`xlm-roberta-base`** fine-tuned on
      English CSQA → cross-lingual transfer; (3) **ESIM** as a pre-transformer,
      **English-only** anchor (GloVe vocab can't read ru/he). Report the en→he
      *degradation pattern*, not absolute accuracy across arms (XLM-R is
      fine-tuned, the Qwens zero-shot).
- [ ] **Human validation** — small Hebrew spot-check still recommended (short
      ConceptNet phrases are where MT slips). Open team task.
- [x] **Prompt format** — **zero-shot**, English instruction held constant, Latin
      `A–E` labels, choices listed one per line; model emits the **letter**. See
      `eval/prompt.py` (`PROMPT_TEMPLATE`, hashed into each manifest).

## Implementation map

`scripts/{translate,build_variants,run_eval,analyze}.py` (config-driven) drive the
`csqa_xlang` modules: `translation/` (Google) → `variants/` (build + fingerprint)
→ `eval/` (generative vLLM backend + `xlmr` + `esim`, writing `outputs.jsonl` +
manifest) → `analysis/` (accuracy + Wilson + **flip rate + McNemar**). The
generative arm runs on the cluster via `cluster/{submit,serve_and_eval.sbatch,defaults.sh}`
(serving layer reused from the sibling `pddl-copilot-experiments` harness).

## Grading rubric (keep these in mind while building)

Reproducibility 15 · Presentation Quality 30 · Soundness 30 · Depth 15 ·
Related Work 10.
