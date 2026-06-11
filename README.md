# Do LLMs Choose Concepts or English Words?
### Answer-Side Translation in CommonsenseQA

Course project — Computational Semantics and Natural Language Understanding,
2025–2026, Ben-Gurion University.

**Team:** Tal Malul, Omer Eliyahu, Mark Feldman, Daniel Koyfman

## What this is

CommonsenseQA was designed in English. We keep the **questions in English** and
translate **only the answer choices** into other languages (**Russian, Spanish,
Hebrew**) to test whether models pick the right commonsense **concept** or lean on
English **wording**. We compare Q–A language conditions (`EN-EN`, `EN-X`, with
`X-X`/`X-EN` as wired extensions) and measure prediction **flips** caused solely by
changing the answer language. Three model arms: the zero-shot **Qwen** lineup
(vLLM), a fine-tuned **XLM-R** encoder, and an **ESIM** historical anchor.

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
├── scripts/            # thin CLI entry points (translate · build_variants · run_eval · analyze)
├── cluster/            # BGU SLURM: serve vLLM per (model, think) cell + submit
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

## Pipeline

```
load CSQA (validation)
   └─> translate answer choices (Google)         scripts/translate.py
        └─> build condition variants (en-en/en-x) scripts/build_variants.py
             └─> evaluate arm(s)                   scripts/run_eval.py   (temperature=0, cached + manifest)
                  └─> analyze (acc, flips, McNemar) scripts/analyze.py
```

All stages are config-driven (edit `configs/default.yaml`; don't hardcode models,
languages, or paths in modules):

```bash
export GOOGLE_API_KEY=...                                   # Cloud Translation key
python -m scripts.translate       --config configs/default.yaml
python -m scripts.build_variants  --config configs/default.yaml

# Generative Qwen arm — on BGU (serves vLLM per (model, think) cell):
bash cluster/submit.sh --limit 50 Qwen3.5:4B               # smoke first
bash cluster/submit.sh                                     # full: all models x {off,on}

# Encoder arms — any single GPU:
python -m scripts.run_eval --arm xlmr  --config configs/default.yaml   # after: python -c "from csqa_xlang.eval import xlmr; xlmr.train('checkpoints/xlmr-csqa')"
python -m scripts.run_eval --arm esim  --config configs/default.yaml   # after esim.train('checkpoints/esim-csqa.pt')

python -m scripts.analyze         --config configs/default.yaml
```

Each completed `(model, variant)` run is a cache hit (skipped unless `--force`),
and writes `outputs.jsonl` + a reproducibility `manifest.json`.

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

The proposal and course guidelines live in [`documents/`](documents/):

- `Project_proposal.pdf` — the project proposal.
- `Project_assignment_submission_guidelines.pdf` — course assignment guidelines.
- `Project_writing_recommendations.pdf` — report writing recommendations.
