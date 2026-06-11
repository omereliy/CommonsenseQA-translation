"""Build Q-A language condition variants (en-en / en-x / x-x / x-en).

A variant is a deterministic, fingerprinted transform over loaded items +
translations (no model calls). See README.md and conditions.py.
"""

from __future__ import annotations

from csqa_xlang.variants.build import build_variant
from csqa_xlang.variants.conditions import (CONDITIONS, CORE_CONDITIONS,
                                            QUESTION_TRANSLATED, Variant,
                                            load_variant, write_variant)

__all__ = ["Variant", "CONDITIONS", "CORE_CONDITIONS", "QUESTION_TRANSLATED",
           "build_variant", "write_variant", "load_variant"]
