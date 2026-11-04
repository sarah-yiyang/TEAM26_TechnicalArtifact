"""Shared pytest fixtures. No network, no full export needed."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from pipeline import ROOT, load_schema

SAMPLE_PATH: Path = ROOT / "data" / "sample_30rows.xlsx"
SAMPLE_SPEC_PATH: Path = ROOT / "tests" / "fixtures" / "sample_spec.yaml"


@pytest.fixture(scope="session")
def schema() -> dict[str, Any]:
    """The project schema (a fresh copy)."""
    return load_schema()


@pytest.fixture(scope="session")
def sample_path() -> Path:
    """Path to the 30-row test sample."""
    return SAMPLE_PATH


@pytest.fixture(scope="session")
def sample_spec() -> dict[str, Any]:
    """Recipe + expected case counts for the sample."""
    return yaml.safe_load(SAMPLE_SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def sample_raw(sample_path: Path) -> pd.DataFrame:
    """The sample exactly as exported: raw headers, text values."""
    return pd.read_excel(sample_path, dtype=object)
