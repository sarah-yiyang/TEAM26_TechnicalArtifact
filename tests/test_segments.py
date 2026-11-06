"""M3 tests for pipeline/segments.py and metrics.row_values."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pipeline.cleaning import clean
from pipeline.ingest import load_files
from pipeline.mapping import map_columns
from pipeline.metrics import ROW_METRICS, compute, drop_flagged, row_values
from pipeline.quality import add_flags
from pipeline.segments import NOT_RECORDED, compare


@pytest.fixture
def sample(sample_path, schema) -> pd.DataFrame:
    mapped = map_columns(load_files([sample_path], schema)[0], schema)[0]
    return add_flags(clean(mapped, schema)[0], schema)


@pytest.fixture
def low_n(schema) -> dict[str, Any]:
    s = copy.deepcopy(schema)
    s["analysis"]["min_n"] = 1
    return s


def _rows(**cols: list[Any]) -> pd.DataFrame:
    df = pd.DataFrame(cols)
    for c in df.columns:
        if c.startswith(("pre_", "post_", "presenter_")) or c in ("well_organized", "nps",
                                                                 "likely_call"):
            df[c] = df[c].astype("Int64")
    return df


def _sessions(spec: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """Sessions -> (rows, rows agreeing on post_when)."""
    sid, post = [], []
    for name, (n, agree) in spec.items():
        sid += [name] * n
        post += [5] * agree + [2] * (n - agree)
    return _rows(session_id=sid, post_when=post)


# ----- row_values ----------------------------------------------------------

def test_row_values_mean_equals_compute(sample, low_n) -> None:
    rv, m = row_values(sample, low_n), compute(sample, low_n)
    for key in ROW_METRICS:
        assert rv[key].mean() == pytest.approx(m[key]["value"])
        assert rv[key].count() == m[key]["n"]


def test_row_values_blank_stays_blank(low_n) -> None:
    rv = row_values(_rows(post_when=[5, None, 2], nps=[10, 7, None]), low_n)
    assert list(rv["agree_pct.post_when"].isna()) == [False, True, False]
    assert list(rv["nps"][:2]) == [100.0, 0.0] and pd.isna(rv["nps"][2])


def test_row_values_skips_missing_questions(low_n) -> None:
    rv = row_values(_rows(post_when=[5]), low_n)
    assert list(rv.columns) == ["agree_pct.post_when"]


def test_drop_flagged(sample) -> None:
    assert len(drop_flagged(sample)) == 26


# ----- compare -------------------------------------------------------------

def test_group_n_adds_up_and_not_recorded(low_n) -> None:
    df = _rows(role_primary=["Student", "Student", None], post_when=[5, 2, 5])
    t = compare(df, "role_primary", schema=low_n)
    assert t["n"].sum() == 3
    assert list(t["group"]) == ["Student", NOT_RECORDED]
    assert t.loc[0, "value"] == pytest.approx(0.5)


def test_suppression_at_min_n(schema) -> None:
    df = _rows(language=["A"] * 19 + ["B"] * 20, post_when=[5] * 39)
    t = compare(df, "language", schema=schema).set_index("group")
    assert t.loc["A", "suppressed"] and pd.isna(t.loc["A", "value"])
    assert t.loc["A", "n"] == 19
    assert not t.loc["B", "suppressed"] and t.loc["B", "value"] == 1.0


def test_blank_answers_not_counted(low_n) -> None:
    df = _rows(language=["A"] * 3, post_when=[5, None, 2])
    t = compare(df, "language", schema=low_n)
    assert t.loc[0, "n"] == 3 and t.loc[0, "metric_n"] == 2
    assert t.loc[0, "value"] == pytest.approx(0.5)


def test_group_value_matches_compute(sample, low_n) -> None:
    t = compare(sample, "role_primary", "nps", low_n).set_index("group")
    students = sample[sample["role_primary"] == "Student"]
    assert t.loc["Student", "value"] == pytest.approx(compute(students, low_n)["nps"]["value"])


def test_out_of_area_column(sample, low_n) -> None:
    t = compare(sample, "county", schema=low_n).set_index("group")
    assert t.loc["Fairfax (VA)", "out_of_area"]
    assert not t.loc["Montgomery", "out_of_area"]


def test_bad_by_and_metric_raise(sample) -> None:
    with pytest.raises(ValueError):
        compare(sample, "response_id")
    with pytest.raises(ValueError):
        compare(sample, "county", metric="made_up")


def test_missing_column_gives_empty_table(low_n) -> None:
    assert compare(_rows(post_when=[5]), "county", schema=low_n).empty
    assert compare(_rows(county=["X"]), "county", schema=low_n).empty   # no post_when


def test_input_untouched(sample, schema) -> None:
    before = sample.copy()
    compare(sample, "session_id", schema=schema)
    pd.testing.assert_frame_equal(sample, before)


# ----- session funnel ------------------------------------------------------

def test_funnel_limits_match_hand_calc(schema) -> None:
    df = _sessions({"s1": (100, 90), "s2": (25, 20)})
    t = compare(df, "session_id", schema=schema).set_index("group")
    p = 110 / 125
    sd = np.sqrt(p * (1 - p))
    assert t.loc["s1", "centre"] == pytest.approx(p)
    assert t.loc["s1", "lower95"] == pytest.approx(p - 1.96 * sd / 10)
    assert t.loc["s2", "lower998"] == pytest.approx(p - 3.09 * sd / 5)
    # bigger session -> narrower limits
    assert t.loc["s1", "lower998"] > t.loc["s2", "lower998"]


def test_funnel_flags_low_session(schema) -> None:
    spec = {f"s{i}": (40, 36) for i in range(8)}
    spec["bad"] = (40, 8)
    t = compare(_sessions(spec), "session_id", schema=schema).set_index("group")
    assert t.loc["bad", "outlier"]
    assert not t.drop("bad")["outlier"].any()


def test_funnel_suppressed_never_outlier(schema) -> None:
    spec = {f"s{i}": (40, 38) for i in range(5)}
    spec["tiny"] = (5, 0)
    t = compare(_sessions(spec), "session_id", schema=schema).set_index("group")
    assert t.loc["tiny", "suppressed"] and not t.loc["tiny", "outlier"]


def test_non_session_has_no_funnel(sample, schema) -> None:
    assert "outlier" not in compare(sample, "role_primary", schema=schema).columns
