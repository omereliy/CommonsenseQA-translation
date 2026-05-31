# Do LLMs Choose Concepts or English Words?
### Answer-Side Translation in CommonsenseQA

Course project — Computational Semantics and Natural Language Understanding,
2025–2026, Ben-Gurion University.

**Team:** Tal Malul, Omer Eliyahu, Mark Feldman, Daniel Koyfman

## What this is

CommonsenseQA was designed in English. We keep the **questions in English** and
translate **only the answer choices** into other languages (Hebrew, Arabic, and a
high-resource language) to test whether models pick the right commonsense
**concept** or lean on English **wording**. We compare Q–A language conditions
(`EN-EN`, `EN-X`, `X-X`, `X-EN`) and measure prediction **flips** caused solely by
changing the answer language.

> Read [`CLAUDE.md`](CLAUDE.md) first — it defines the load-bearing conventions
> (validation split, letter-based answers, deterministic decoding + manifests).

## Repository layout

```
.
├── CLAUDE.md            # project guide + shared conventions
├── paper/              # the 6-page ACL-style report (LaTeX; editable in Overleaf)
├── src/csqa_xlang/     # the experiment package: config · manifest · data ·
│                       #   translation · variants · eval · analysis
├── configs/            # YAML run configs (seed, temperature, languages, paths)
├── scripts/            # thin CLI entry points
├── data/               # raw · translated · variants   (git-ignored payloads)
├── results/            # run outputs + manifests        (git-ignored payloads)
├── tests/
├── pyproject.toml · requirements.txt · .env.example
└── .venv/              # local virtual environment      (git-ignored)
```

Each `src/csqa_xlang/<module>/` has its own `README.md` describing its
responsibility. The architecture is described in prose on purpose — shape it as a
team rather than inheriting a speculative skeleton.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # makes `import csqa_xlang` work everywhere
cp .env.example .env               # then fill in any API keys you actually use
```

### First smoke test (run this on day one)

The whole pipeline rests on one external dependency: loading CommonsenseQA. The
loader assumes the `tau/commonsense_qa` schema (`choices.{label,text}`, `answerKey`)
and the ~1,221-item `validation` split — verified against documentation but **not
yet run against the live dataset**. Confirm it loads before building anything
downstream, so a schema surprise fails loudly now rather than after the translation
code is written against a wrong shape:

```bash
python -c "from csqa_xlang.data import load_csqa; xs = load_csqa('validation', limit=3); print(len(xs), xs[0])"
```

If `datasets` (3.x dropped script-based loaders) complains, you may need a specific
revision or `trust_remote_code=True` — fix it in `src/csqa_xlang/data/loader.py` and
note what worked.

## Pipeline (intended)

```
load CSQA (validation)
   └─> translate answer choices (+ optional question nouns)
        └─> build condition variants (EN-EN / EN-X / X-X / X-EN / mixed / noun)
             └─> evaluate model(s)        # temperature=0, seeded, raw outputs cached
                  └─> analyze             # accuracy, flips, significance, plots
```

Runs are config-driven: `python -m scripts.<entrypoint> --config configs/default.yaml`
(entry points are TBD — see [`scripts/README.md`](scripts/README.md)). Don't hardcode
models, languages, or paths in modules; put them in the config.

## Reproducibility (graded — 15 pts)

- `temperature=0` and a fixed `seed` for every eval run.
- Each run writes a **manifest** beside its raw outputs: model id + snapshot/date,
  prompt-template hash, dataset-variant hash, decoding params.
- Raw model outputs are cached so accuracy and flip rates recompute without
  re-querying the model.
- `data/**` and `results/**` payloads are git-ignored; the config + code + manifest
  are the reproducible recipe.

## The report

See [`paper/README.md`](paper/README.md) for the report structure and the Overleaf
setup (start from the official ACL template, drop in `paper/sections/` +
`paper/references.bib`).

## Source documents

The proposal and course guidelines (`Do LLMs Choose Concepts or English Words.pdf`,
`Project_assignment_submission_guidelines.pdf`,
`Project_writing_recommendations.pdf`) live at the repository root for reference.
