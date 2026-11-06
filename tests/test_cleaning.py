"""M1 tests for pipeline/cleaning.py, run on the sample after ingest + mapping."""
from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from pipeline.cleaning import clean
from pipeline.ingest import load_files
from pipeline.mapping import map_columns


@pytest.fixture
def mapped(sample_path, schema) -> pd.DataFrame:
    """Sample after ingest + mapping (standard names, raw values)."""
    return map_columns(load_files([sample_path])[0], schema)[0]


@pytest.fixture
def cleaned(mapped, schema) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    return clean(mapped, schema)


def _row(df: pd.DataFrame, rid: Any) -> pd.Series:
    """Single row by response id (ids reused in the sample are avoided)."""
    hits = df[df["response_id"].astype(str) == str(rid)]
    assert len(hits) == 1
    return hits.iloc[0]


def _fix(log: list[dict[str, Any]], change: str) -> dict[str, Any]:
    return next(e for e in log if e["change"] == change)


# ----- likert ----------------------------------------------------------

def test_assumption_counts_logged(cleaned) -> None:
    _, log, _ = cleaned
    satisfied = _fix(log, "'Satisfied' -> 4")
    assert (satisfied["cells"], satisfied["rows"], satisfied["assumption"]) == (4, 2, True)
    very = _fix(log, "'Very agree' -> 5")
    assert (very["cells"], very["assumption"]) == (3, True)


def test_likert_values_translated(cleaned) -> None:
    df, _, _ = cleaned
    likert = ["pre_identify", "pre_describe", "pre_when", "post_identify", "post_describe",
              "post_when", "presenter_engaged", "presenter_knew", "well_organized"]
    for col in likert:
        assert str(df[col].dtype) == "Int64"
        assert df[col].dropna().between(1, 5).all()
    assert _row(df, 26)["well_organized"] == 3          # "Neither agree or disagree"
    assert _row(df, 14)["presenter_engaged"] == 5       # "Strongly agree\xa0" straight-liner


def test_unknown_likert_reported_and_blank(cleaned) -> None:
    df, _, unknown = cleaned
    assert unknown == [
        {"field": "post_describe", "value": "Somewhat agree", "rows": 1, "response_ids": [126]}
    ]
    assert pd.isna(_row(df, 126)["post_describe"])


def test_blank_presenter_block_stays_blank(cleaned) -> None:
    df, _, _ = cleaned
    block = df[["presenter_engaged", "presenter_knew", "well_organized"]]
    assert block.isna().all(axis=1).sum() == 5


# ----- 0-10 scales -----------------------------------------------------

def test_scales_are_ints_and_blanks_stay_blank(cleaned) -> None:
    df, _, _ = cleaned
    assert str(df["nps"].dtype) == "Int64" and str(df["likely_call"].dtype) == "Int64"
    assert pd.isna(_row(df, 147)["nps"]) and _row(df, 147)["likely_call"] == 10
    assert df["likely_call"].isna().sum() >= 1
    assert (df["likely_call"].dropna() != 0).all()  # blanks never become 0


def test_out_of_range_scale_reported(mapped, schema) -> None:
    bad = mapped.copy()
    bad.loc[0, "nps"] = 11
    bad.loc[1, "nps"] = "ten"
    bad.loc[2, "nps"] = 7.5
    df, _, unknown = clean(bad, schema)
    values = {u["value"] for u in unknown if u["field"] == "nps"}
    assert values == {"11", "ten", "7.5"}
    assert df.loc[[0, 1, 2], "nps"].isna().all()


# ----- roles -----------------------------------------------------------

@pytest.mark.parametrize("rid, primary, student", [
    (210, "Student", True),        # Caregiver/parent;Student
    (21, "School staff", False),   # School staff member;Caregiver/parent;
    (35, "Caregiver", False),      # Other;Caregiver/parent;
    (31, "Community", False),      # trailing ';'
])
def test_role_primary(cleaned, rid, primary, student) -> None:
    df, _, _ = cleaned
    row = _row(df, rid)
    assert row["role_primary"] == primary and bool(row["is_student"]) is student


def test_role_raw_kept_as_exported(cleaned, mapped) -> None:
    df, _, _ = cleaned
    pd.testing.assert_series_equal(df["role_raw"], mapped["role_raw"].reset_index(drop=True))


def test_unknown_role_reported(mapped, schema) -> None:
    odd = mapped.copy()
    odd.loc[0, "role_raw"] = "Student;Volunteer;"
    df, _, unknown = clean(odd, schema)
    assert any(u["field"] == "role_raw" and u["value"] == "Volunteer" for u in unknown)
    assert df.loc[0, "role_primary"] == "Student"


# ----- county / language -----------------------------------------------

@pytest.mark.parametrize("rid, county", [
    (127, "Montgomery"),        # "Montgomery\xa0"
    (31, "Montgomery"),         # "Montgomery County with NAMI MC"
    (21, "Fairfax (VA)"),       # "Fairfax "
    (23, "Fairfax (VA)"),       # "Fairfax county "
    (195, "Queen Anne's"),      # "Queen Anne's County "
    (59, "Other / unclear"),    # "Harris"
])
def test_county_variants(cleaned, rid, county) -> None:
    df, _, _ = cleaned
    assert _row(df, rid)["county"] == county


def test_county_merges_logged(cleaned) -> None:
    _, log, _ = cleaned
    assert _fix(log, "'montgomery county with nami mc' -> Montgomery")["rows"] == 1


def test_language(cleaned) -> None:
    df, _, _ = cleaned
    assert (df["language"] == "Spanish").sum() == 2
    assert set(df["language"]) == {"English", "Spanish"}


# ----- dates and derived columns ---------------------------------------

def test_duration_and_speeder(cleaned) -> None:
    df, _, _ = cleaned
    assert _row(df, 860)["duration_sec"] == 18


def test_month_and_session_id(cleaned) -> None:
    df, _, _ = cleaned
    row = _row(df, 126)
    assert row["month"] == "2026-04"
    assert row["session_id"] == "2026-04-08_Prince George's"


def test_missing_training_date(cleaned) -> None:
    df, _, _ = cleaned
    row = _row(df, 62)
    assert pd.isna(row["training_date"])
    assert row["month"] is None and row["session_id"] is None


def test_negative_duration_blank_and_logged(mapped, schema) -> None:
    bad = mapped.copy()
    bad.loc[0, "end_time"] = "2000-01-01 00:00:00"
    df, log, _ = clean(bad, schema)
    assert pd.isna(df.loc[0, "duration_sec"])
    assert _fix(log, "end before start -> blank")["rows"] == 1


def test_unparseable_date_reported(mapped, schema) -> None:
    bad = mapped.copy()
    bad.loc[0, "training_date"] = "last Tuesday"
    df, _, unknown = clean(bad, schema)
    hit = [u for u in unknown if u["field"] == "training_date"]
    assert len(hit) == 1 and hit[0]["value"] == "last Tuesday" and hit[0]["rows"] == 1
    assert pd.isna(df.loc[0, "training_date"])


# ----- text ------------------------------------------------------------

def test_comments_kept(cleaned) -> None:
    df, _, _ = cleaned
    improve = df["comment_improve"].dropna()
    assert improve.str.lower().eq("nothing").sum() == 1
    assert improve.str.contains(r"\?").any()
    assert not df["comment_valuable"].dropna().str.contains("\xa0").any()


# ----- missing questions / purity --------------------------------------

def test_missing_county_skips_session_id(mapped, schema) -> None:
    df, _, _ = clean(mapped.drop(columns=["county"]), schema)
    assert "county" not in df.columns and "session_id" not in df.columns
    assert "month" in df.columns


def test_missing_likely_call_runs(mapped, schema) -> None:
    df, _, _ = clean(mapped.drop(columns=["likely_call"]), schema)
    assert "likely_call" not in df.columns


def test_input_not_modified(mapped, schema) -> None:
    before = mapped.copy()
    clean(mapped, schema)
    pd.testing.assert_frame_equal(mapped, before)
