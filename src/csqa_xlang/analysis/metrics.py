"""Pure metric functions over result records (no model calls; easy to unit-test).

A *record* is a dict: {model, think, condition, lang, id, pred, gold, correct}.
Flip rate (between a baseline condition and another on the SAME items) is the
core diagnostic; significance on paired accuracy uses McNemar (CLAUDE.md).
"""

from __future__ import annotations

import math


def accuracy(rows: list[dict]) -> float:
    return sum(r["correct"] for r in rows) / len(rows) if rows else 0.0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def _by_id(rows: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in rows}


def flip_rate(base: list[dict], other: list[dict]) -> dict:
    """Fraction of shared items whose selected label changes base→other, with
    direction split (toward gold = becomes correct; away = becomes wrong)."""
    b, o = _by_id(base), _by_id(other)
    ids = b.keys() & o.keys()
    flips = toward = away = 0
    for i in ids:
        if b[i]["pred"] != o[i]["pred"]:
            flips += 1
            was, now = b[i]["correct"], o[i]["correct"]
            if now and not was:
                toward += 1
            elif was and not now:
                away += 1
    n = len(ids)
    return {"n": n, "flips": flips, "flip_rate": flips / n if n else 0.0,
            "toward_gold": toward, "away_gold": away}


def mcnemar(base: list[dict], other: list[dict]) -> dict:
    """Paired test on accuracy difference (same items). b = base-correct/other-wrong,
    c = base-wrong/other-correct. Returns the discordant counts and a p-value."""
    b_map, o_map = _by_id(base), _by_id(other)
    ids = b_map.keys() & o_map.keys()
    b = sum(b_map[i]["correct"] and not o_map[i]["correct"] for i in ids)
    c = sum(o_map[i]["correct"] and not b_map[i]["correct"] for i in ids)
    try:
        from statsmodels.stats.contingency_tables import mcnemar as _mc
        p = float(_mc([[0, b], [c, 0]], exact=(b + c) < 25).pvalue)
    except Exception:
        # exact two-sided binomial(min(b,c); b+c, 0.5) fallback
        n, k = b + c, min(b, c)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)) if n else 1.0
    return {"discordant_b": b, "discordant_c": c, "p_value": p}
