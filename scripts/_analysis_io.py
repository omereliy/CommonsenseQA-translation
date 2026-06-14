"""Shared helpers for the presentation scripts (plots / tables / deck).

These scripts sit DOWNSTREAM of ``scripts.analyze`` — they consume the two tidy
CSVs it writes (``summary.csv`` + ``flips.csv``) and never recompute metrics.
This module centralises (a) loading those CSVs robustly when the sweep is still
in flight (partial / missing files), and (b) a synthetic ``--demo`` fixture so
figures/tables/deck can be generated and eyeballed before real results land.

CSV schemas (from scripts/analyze.py → csqa_xlang.analysis.aggregate):
  summary.csv: model, think, condition, lang, n, accuracy, ci_low, ci_high, unparsed
  flips.csv:   model, think, lang, n, flip_rate, toward_gold, away_gold,
               acc_en, acc_x, mcnemar_p
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Canonical model order + a rough parameter-size (in B) for the scaling figure.
# Sizes aren't carried in the CSV, so we keep a small static map here. Anything
# not listed falls back to ordering after the known models, size = NaN.
MODEL_SIZE_B: dict[str, float] = {
    "Qwen3.5:0.8B": 0.8,
    "Qwen3.5:4B": 4.0,
    "Qwen3.5:9B": 9.0,
    "qwen3.6:35b": 35.0,
    "gemma4:26b-a4b": 26.0,
}
MODEL_ORDER = list(MODEL_SIZE_B.keys())
LANG_ORDER = ["ru", "es", "he"]
LANG_NAME = {"en": "English", "ru": "Russian", "es": "Spanish", "he": "Hebrew"}

SUMMARY_COLS = ["model", "think", "condition", "lang", "n",
                "accuracy", "ci_low", "ci_high", "unparsed"]
FLIPS_COLS = ["model", "think", "lang", "n", "flip_rate", "toward_gold",
              "away_gold", "acc_en", "acc_x", "mcnemar_p"]


def model_sort_key(model: str) -> tuple[int, str]:
    return (MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER), model)


def order_models(models) -> list[str]:
    return sorted(set(models), key=model_sort_key)


def load_csvs(results_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load summary + flips from a results dir. Missing files come back as empty
    frames with the right columns (so callers can degrade gracefully)."""
    root = Path(results_dir)
    summary = _read_or_empty(root / "summary.csv", SUMMARY_COLS)
    flips = _read_or_empty(root / "flips.csv", FLIPS_COLS)
    return summary, flips


def _read_or_empty(path: Path, cols: list[str]) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        # tolerate column drift — keep what we recognise, warn on the rest only implicitly
        return df
    return pd.DataFrame(columns=cols)


# --------------------------------------------------------------------------- #
# Synthetic fixture (--demo). Plausible numbers, NOT real results.
# --------------------------------------------------------------------------- #
def make_demo_csvs(out_dir: str | Path, n: int = 1221) -> tuple[Path, Path]:
    """Write a synthetic summary.csv + flips.csv covering the full sweep so the
    presentation scripts can be exercised end-to-end. Output is clearly synthetic
    (degradation patterns are hand-set, not measured)."""
    import random

    rng = random.Random(0)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Hand-set baseline accuracy per model (bigger = better) and a per-language
    # degradation that worsens for more distant scripts (he > ru > es drop).
    base_acc = {"Qwen3.5:0.8B": 0.55, "Qwen3.5:4B": 0.68, "Qwen3.5:9B": 0.74,
                "qwen3.6:35b": 0.82, "gemma4:26b-a4b": 0.79}
    lang_drop = {"es": 0.04, "ru": 0.08, "he": 0.13}   # accuracy lost vs en-en
    flip_base = {"es": 0.10, "ru": 0.16, "he": 0.22}   # headline flip rate

    summary_rows, flip_rows = [], []
    for model in MODEL_ORDER:
        for think in ("off", "on"):
            think_bonus = 0.03 if think == "on" else 0.0
            acc_en = min(0.95, base_acc[model] + think_bonus)
            unparsed_en = rng.randint(0, 3)
            summary_rows.append(_srow(model, think, "en-en", "en", n, acc_en, unparsed_en))
            for lang in LANG_ORDER:
                # think slightly cushions the cross-lingual drop
                acc_x = max(0.20, acc_en - lang_drop[lang] * (1.0 - 0.25 * (think == "on")))
                summary_rows.append(
                    _srow(model, think, "en-x", lang, n, acc_x, rng.randint(0, 6)))

                flip = flip_base[lang] * (1.0 - 0.20 * (think == "on"))
                # bigger drop in accuracy → more of the flips go away-from-gold
                away = int(round(flip * n * 0.62))
                toward = int(round(flip * n * 0.30))
                flips_total = away + toward + int(round(flip * n * 0.08))
                p = 0.5 ** (1 + (acc_en - acc_x) * 40)   # smaller p for bigger gap
                flip_rows.append({
                    "model": model, "think": think, "lang": lang, "n": n,
                    "flip_rate": round(flips_total / n, 4),
                    "toward_gold": toward, "away_gold": away,
                    "acc_en": round(acc_en, 4), "acc_x": round(acc_x, 4),
                    "mcnemar_p": round(min(1.0, p), 5),
                })

    spath, fpath = out / "summary.csv", out / "flips.csv"
    pd.DataFrame(summary_rows, columns=SUMMARY_COLS).to_csv(spath, index=False)
    pd.DataFrame(flip_rows, columns=FLIPS_COLS).to_csv(fpath, index=False)
    return spath, fpath


def _srow(model, think, cond, lang, n, acc, unparsed):
    # crude Wilson-ish band just for the demo (real CSV carries proper Wilson CIs)
    import math
    half = 1.96 * math.sqrt(max(acc * (1 - acc), 1e-6) / n)
    return {"model": model, "think": think, "condition": cond, "lang": lang,
            "n": n, "accuracy": round(acc, 4),
            "ci_low": round(max(0.0, acc - half), 4),
            "ci_high": round(min(1.0, acc + half), 4), "unparsed": unparsed}


DEMO_BANNER = ("SYNTHETIC DEMO DATA — numbers are hand-set placeholders, "
               "not measured results.")
