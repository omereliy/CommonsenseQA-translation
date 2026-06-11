"""Metrics and analysis: accuracy, flip rate (core diagnostic), McNemar. README.md.

Pure functions over cached result records (never re-queries models).
"""

from __future__ import annotations

from csqa_xlang.analysis.aggregate import (aggregate, flip_table, load_records,
                                           summarize)
from csqa_xlang.analysis.metrics import accuracy, flip_rate, mcnemar, wilson

__all__ = ["accuracy", "wilson", "flip_rate", "mcnemar",
           "load_records", "summarize", "flip_table", "aggregate"]
