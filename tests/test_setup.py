"""M0 tests: settings load correctly and the sample covers every test case."""
from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import pipeline
from pipeline import MIN_N, load_schema, min_n_from

STANDARD_MAPPED_FIELDS = [
    "response_id", "start_time", "end_time", "language", "role_raw",
    "training_date", "county",
    "pre_identify", "pre_describe", "pre_when",
    "post_identify", "post_describe", "post_when",
    "presenter_engaged", "presenter_knew", "well_organized",
    "nps", "likely_call", "comment_valuable", "comment_improve",
]


def _clean(value: Any) -> Any:
    """Light text cleanup for counting cells (lowercase, trim, nbsp -> space)."""
    if pd.isna(value):
        return value
    return str(value).replace("\xa0", " ").strip().lower()


# ----- settings -------------------------------------------------------

def test_min_n_comes_from_schema(schema: dict[str, Any]) -> None:
    assert MIN_N == schema["analysis"]["min_n"]
    assert isinstance(MIN_N, int)


def test_min_n_follows_whatever_yaml_says() -> None:
    assert min_n_from({"analysis": {"min_n": 7}}) == 7


@pytest.mark.parametrize("bad", [{}, {"analysis": {}}, {"analysis": {"min_n": 0}},
                                 {"analysis": {"min_n": "20"}}, {"analysis": {"min_n": True}}])
def test_min_n_rejects_missing_or_bad_values(bad: dict[str, Any]) -> None:
    with pytest.raises((KeyError, ValueError)):
        min_n_from(bad)


def test_load_schema_returns_independent_copies() -> None:
    first = load_schema()
    first["analysis"]["min_n"] = 999
    assert load_schema()["analysis"]["min_n"] == pipeline.MIN_N


def test_schema_defines_every_standard_field(schema: dict[str, Any]) -> None:
    missing = [f for f in STANDARD_MAPPED_FIELDS if f not in schema["fields"]]
    assert missing == []


# ----- sample coverage --------------------------------------------------

def test_sample_shape(sample_raw: pd.DataFrame, sample_spec: dict[str, Any]) -> None:
    exp = sample_spec["expected"]
    assert sample_raw.shape == (exp["rows"], exp["columns"])
    assert len(sample_spec["columns"]) == exp["columns"]


def test_sample_headers_keep_nbsp(sample_raw: pd.DataFrame) -> None:
    assert any("\xa0" in str(c) for c in sample_raw.columns)


def test_sample_covers_every_case(sample_raw: pd.DataFrame, sample_spec: dict[str, Any],
                                  schema: dict[str, Any]) -> None:
    exp, pos = sample_spec["expected"], sample_spec["columns"]
    col = lambda field: sample_raw.iloc[:, pos[field]]  # noqa: E731
    likert_fields = schema["quality"]["straightline_fields"]
    likert = sample_raw.iloc[:, [pos[f] for f in likert_fields]]

    # duplicates: exact (id + start_time) vs. reused id only
    id_time = sample_raw.iloc[:, [pos["response_id"], pos["start_time"]]].astype(str)
    assert id_time.duplicated().sum() == exp["exact_duplicate_rows"]
    starts_per_id = id_time.groupby(id_time.columns[0])[id_time.columns[1]].nunique()
    assert (starts_per_id > 1).sum() == exp["reused_id_rows"]

    # answer wording variants and unknown value
    counts = likert.map(_clean).stack().value_counts()
    for value, n in exp["likert_value_counts"].items():
        assert counts.get(value, 0) == n, value

    # non-breaking spaces in values
    nbsp = lambda df: df.map(lambda v: isinstance(v, str) and "\xa0" in v).to_numpy().sum()  # noqa: E731
    assert nbsp(likert) >= exp["min_likert_cells_with_nbsp"]
    assert nbsp(col("county").to_frame()) >= exp["min_county_cells_with_nbsp"]

    # multi-role (a separator between two roles, not just trailing)
    sep = schema["fields"]["role_raw"]["separator"]
    multi = col("role_raw").astype(str).str.rstrip(sep).str.contains(sep, regex=False)
    assert multi.sum() == exp["multi_role_rows"]

    assert col("training_date").isna().sum() == exp["missing_training_date"]
    assert col("likely_call").isna().sum() >= exp["blank_likely_call_min"]

    # speeder
    seconds = (pd.to_datetime(col("end_time")) - pd.to_datetime(col("start_time"))).dt.total_seconds()
    assert (seconds < schema["quality"]["speeder_seconds"]).sum() == exp["speeders"]

    # blank presenter block
    presenter = sample_raw.iloc[:, [pos["presenter_engaged"], pos["presenter_knew"], pos["well_organized"]]]
    assert presenter.isna().all(axis=1).sum() == exp["blank_presenter_block"]

    # comments
    improve = col("comment_improve").map(_clean)
    assert (improve == "nothing").sum() == exp["nothing_comments"]
    assert improve.astype(str).str.contains("?", regex=False).sum() >= exp["question_comments_min"]

    # Spanish (mapped through schema language_values)
    spanish = [k for k, v in schema["language_values"].items() if v == "Spanish"]
    lang = col("language").map(_clean).str.replace("‎", "", regex=False)
    assert lang.isin(spanish).sum() == exp["spanish_rows"]
