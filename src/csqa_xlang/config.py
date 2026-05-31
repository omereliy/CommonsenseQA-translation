"""Config loading and global seeding.

Runs are config-driven (see ``configs/default.yaml``). Loading a config also
seeds the RNGs so runs are reproducible — a load-bearing convention (CLAUDE.md).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Thin wrapper over the parsed YAML.

    Kept as a dict-backed object on purpose: the schema is still being shaped by
    the team. Access nested values with ``cfg["eval"]["model"]`` or ``cfg.get(...)``.
    """

    raw: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))


def load_config(path: str | Path, *, seed: bool = True) -> Config:
    """Load a YAML config and (by default) seed all RNGs from ``cfg.seed``."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config(raw=raw, path=path)
    if seed:
        seed_everything(cfg.seed)
    return cfg


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and (if installed) PyTorch for deterministic runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
