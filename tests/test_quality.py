"""M2 tests for pipeline/quality.py: sample counts plus small hand-built edge cases."""
from __future__ import annotations

import copy
from typing import Any

import pandas as pd
import pytest

from pipeline.cleaning import clean
from pipeline.ingest import load_files
from pipeline.mapping import map_columns
from pipeline.quality import add_flags

LIKERT = ["pre_identify", "pre_describe", "pre_when", "post_identify", "post_describe",
          "post_when", "presenter_engaged", "presenter_knew", "well_organized"]
PRESENTER = ["presenter_engaged", "presenter_knew", "well_organized"]
POST = ["post_identify", "post_describe", "post_when"]

# Response ids of sample rows 19 (speeder), 20 (straight-liner), 21 (reversed).
SPEEDER_ID, STRAIGHT_ID, REVERSED_ID = 860, 14, 7


@pytest.fixture
def cleaned(sample_path, schema) -> pd.DataFrame:
    """Sample after ingest + mapping + cleaning."""
    mapped = map_columns(load_files([sample_path], schema)[0], schema)[0]
    return clean(mapped, schema)[0]


@pytest.fixture
def flagged(cleaned, schema) -> pd.DataFrame:
    return add_flags(cleaned, schema)


def _rows(**cols: list[Any]) -> pd.DataFrame:
    """Small cleaned-style table; Likert/0-10 as Int64, duration as float."""
    df = pd.DataFrame(cols)
    for c in df.columns:
        if c in LIKERT or c == "nps":
            df[c] = df[c].astype("Int64")
        elif c == "duration_sec":
            df[c] = df[c].astype(float)
    return df


def _base(n: int = 1, **over: list[Any]) -> pd.DataFrame:
    """n ordinary, unflagged rows, with column overrides."""
    cols: dict[str, list[Any]] = {"duration_sec": [60.0] * n, "nps": [9] * n}
    for i, f in enumerate(LIKERT):
        cols[f] = [3 + i % 2] * n          # alternating 3/4: not straight, not reversed
    cols.update(over)
    return _rows(**cols)


def _ids(df: pd.DataFrame, col: str) -> list[Any]:
    return list(df.loc[df[col].fillna(False), "response_id"])


# ----- sample file -------------------------------------------------------

def test_sample_flag_counts(flagged) -> None:
    assert _ids(flagged, "flag_speeder") == [SPEEDER_ID]
    assert _ids(flagged, "flag_straightline") == [STRAIGHT_ID]
    assert _ids(flagged, "flag_reversed") == [REVERSED_ID]
    assert flagged["flag_any"].sum() == 3


def test_flag_dtypes(flagged) -> None:
    for c in ("flag_speeder", "flag_straightline", "flag_reversed"):
        assert flagged[c].dtype == "boolean"
    assert flagged["flag_any"].dtype == bool


def test_no_rows_dropped_and_input_untouched(cleaned, schema) -> None:
    before = cleaned.copy()
    out = add_flags(cleaned, schema)
    pd.testing.assert_frame_equal(cleaned, before)
    assert len(out) == len(cleaned)
    pd.testing.assert_frame_equal(out[cleaned.columns], cleaned)


def test_default_schema_matches_explicit(cleaned, schema) -> None:
    pd.testing.assert_frame_equal(add_flags(cleaned), add_flags(cleaned, schema))


# ----- speeder -----------------------------------------------------------

def test_speeder_boundary_and_blank(schema) -> None:
    df = _base(3, duration_sec=[19.9, 20.0, None])
    out = add_flags(df, schema)["flag_speeder"]
    assert out[0] == True  # noqa: E712
    assert out[1] == False  # noqa: E712  (exactly 20 s is not a speeder)
    assert pd.isna(out[2])


def test_speeder_threshold_from_schema(schema) -> None:
    df = _base(1, duration_sec=[25.0])
    assert add_flags(df, schema)["flag_speeder"][0] == False  # noqa: E712
    loose = copy.deepcopy(schema)
    loose["quality"]["speeder_seconds"] = 30
    assert add_flags(df, loose)["flag_speeder"][0] == True  # noqa: E712


def test_no_duration_column_only_speeder_unknown(schema) -> None:
    df = _base(1).drop(columns="duration_sec")
    out = add_flags(df, schema)
    assert out["flag_speeder"].isna().all()
    assert out["flag_straightline"][0] == False  # noqa: E712
    assert out["flag_reversed"][0] == False  # noqa: E712


# ----- straight-liner ----------------------------------------------------

def test_straightline_all_nine_identical(schema) -> None:
    df = _base(1, **{f: [5] for f in LIKERT})
    assert add_flags(df, schema)["flag_straightline"][0] == True  # noqa: E712


def test_straightline_blank_presenter_identical_rest_unknown(schema) -> None:
    over = {f: [5] for f in LIKERT}
    over.update({f: [None] for f in PRESENTER})
    out = add_flags(_base(1, **over), schema)
    assert pd.isna(out["flag_straightline"][0])
    assert out["flag_any"][0] == False  # noqa: E712  (unknown is not flagged)


def test_straightline_blank_presenter_differing_rest_false(schema) -> None:
    over = {f: [None] for f in PRESENTER}
    over["pre_when"] = [2]
    over["post_when"] = [5]
    assert add_flags(_base(1, **over), schema)["flag_straightline"][0] == False  # noqa: E712


def test_straightline_all_blank_unknown(schema) -> None:
    out = add_flags(_base(1, **{f: [None] for f in LIKERT}), schema)
    assert pd.isna(out["flag_straightline"][0])


# ----- reversed ----------------------------------------------------------

def test_reversed_true_and_ruled_out_by_nps(schema) -> None:
    over = {f: [1, 1] for f in POST}
    over["nps"] = [8, 7]
    out = add_flags(_base(2, **over), schema)["flag_reversed"]
    assert out[0] == True  # noqa: E712
    assert out[1] == False  # noqa: E712


def test_reversed_blank_nps_unknown(schema) -> None:
    over = {f: [1] for f in POST}
    over["nps"] = [None]
    assert pd.isna(add_flags(_base(1, **over), schema)["flag_reversed"][0])


def test_reversed_one_post_blank_unknown(schema) -> None:
    over = {"post_identify": [1], "post_describe": [None], "post_when": [1], "nps": [9]}
    assert pd.isna(add_flags(_base(1, **over), schema)["flag_reversed"][0])


def test_reversed_blank_nps_but_post_high_false(schema) -> None:
    out = add_flags(_base(1, nps=[None]), schema)["flag_reversed"]
    assert out[0] == False  # noqa: E712  (post answers already rule it out)


def test_no_nps_column_only_reversed_unknown(schema) -> None:
    out = add_flags(_base(1).drop(columns="nps"), schema)
    assert out["flag_reversed"].isna().all()
    assert out["flag_speeder"][0] == False  # noqa: E712
    assert out["flag_straightline"][0] == False  # noqa: E712


# ----- flag_any ----------------------------------------------------------

def test_flag_any_true_if_any_flag_true(schema) -> None:
    df = _base(2, duration_sec=[10.0, None], nps=[None, None])
    out = add_flags(df, schema)
    assert list(out["flag_any"]) == [True, False]


# ----- flag_summary ------------------------------------------------------

def test_flag_summary_counts(flagged) -> None:
    from pipeline.quality import flag_summary
    s = flag_summary(flagged)
    assert s["flag_any"] == {"flagged": 3, "unknown": 0, "n": 29}
    assert s["flag_speeder"]["flagged"] == 1


def test_flag_summary_counts_unknown(schema) -> None:
    from pipeline.quality import flag_summary
    s = flag_summary(add_flags(_base(2, nps=[None, None], **{f: [1, 1] for f in POST}), schema))
    assert s["flag_reversed"] == {"flagged": 0, "unknown": 2, "n": 2}
