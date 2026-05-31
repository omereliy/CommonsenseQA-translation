# `scripts/` — CLI entry points

Thin command-line wrappers that load a config and call into `csqa_xlang`. Keep logic
in the package; scripts only parse args and orchestrate.

**Planned entry points (one stage each):**

- `translate.py`  — load CSQA → translate choices → cache to `data/translated/`.
- `build_variants.py` — assemble condition variants → `data/variants/`.
- `run_eval.py`   — evaluate a model on a variant → `results/` (+ manifest).
- `analyze.py`    — compute accuracy / flips / significance → tables & figures.

Intended usage (config-driven):

```bash
python -m scripts.run_eval --config configs/default.yaml
```

A top-level `run_all.py` (or a Makefile) chaining the four stages would help
reproducibility — add once the stages stabilize.
