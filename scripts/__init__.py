"""CLI entry points. Thin wrappers: parse args, load a config, call into csqa_xlang.

Run as modules from the repo root, e.g.:
    python -m scripts.translate      --config configs/default.yaml
    python -m scripts.build_variants --config configs/default.yaml
    python -m scripts.run_eval       --config configs/default.yaml --arm xlmr
    python -m scripts.analyze        --config configs/default.yaml
"""
