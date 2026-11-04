"""EveryMind 988 training insights pipeline.

Shared settings come from ``config/schema.yaml``. ``MIN_N`` is read from
``analysis.min_n`` there and is never hardcoded in Python.
"""
from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT: Path = Path(__file__).resolve().parent.parent
SCHEMA_PATH: Path = ROOT / "config" / "schema.yaml"


@lru_cache(maxsize=None)
def _read_schema(path: str) -> dict[str, Any]:
    """Read and cache the YAML schema at ``path``."""
    with open(path, encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    if not isinstance(schema, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return schema


def load_schema(path: str | Path = SCHEMA_PATH) -> dict[str, Any]:
    """Return a fresh copy of the schema so callers cannot mutate the cache."""
    return copy.deepcopy(_read_schema(str(path)))


def min_n_from(schema: dict[str, Any]) -> int:
    """Extract and validate ``analysis.min_n`` from a schema dict."""
    try:
        value = schema["analysis"]["min_n"]
    except (KeyError, TypeError):
        raise KeyError("config/schema.yaml is missing analysis.min_n") from None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"analysis.min_n must be a positive integer, got {value!r}")
    return value


MIN_N: int = min_n_from(load_schema())

__all__ = ["ROOT", "SCHEMA_PATH", "MIN_N", "load_schema", "min_n_from"]
