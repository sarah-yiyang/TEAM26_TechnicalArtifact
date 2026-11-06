"""Hand-checked numbers from the full export in Excel. Skipped if the file is absent
(it is not committed)."""
from __future__ import annotations

import pytest

from pipeline import ROOT
from pipeline.cleaning import clean
from pipeline.ingest import load_files
from pipeline.mapping import map_columns
from pipeline.metrics import compute

FULL = ROOT / "BuildFest_EveryMind_Survey_Data_FULL.xlsx"

pytestmark = pytest.mark.skipif(not FULL.exists(), reason="full export not present")


@pytest.fixture(scope="module")
def metrics(schema):
    mapped = map_columns(load_files([FULL], schema)[0], schema)[0]
    return compute(clean(mapped, schema)[0], schema)


def test_rows(metrics) -> None:
    assert metrics["rows"]["value"] == 1299


def test_nps_n(metrics) -> None:
    assert metrics["nps"]["n"] == 1266


def test_post_when_agree(metrics) -> None:
    assert metrics["agree_pct.post_when"]["value"] == pytest.approx(0.9230, abs=5e-5)


def test_pre_when_agree(metrics) -> None:
    assert metrics["agree_pct.pre_when"]["value"] == pytest.approx(0.5566, abs=5e-5)
