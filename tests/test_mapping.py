"""M1 tests for pipeline/mapping.py. No network."""
from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from pipeline.ingest import SOURCE_COL, load_files
from pipeline.mapping import map_columns

STANDARD = [
    "response_id", "start_time", "end_time", "language", "role_raw",
    "training_date", "county",
    "post_identify", "post_describe", "post_when",
    "pre_identify", "pre_describe", "pre_when",
    "presenter_engaged", "presenter_knew", "well_organized",
    "nps", "likely_call", "comment_valuable", "comment_improve",
]


@pytest.fixture
def loaded(sample_path) -> pd.DataFrame:
    """Sample after ingest (raw headers + source_file)."""
    return load_files([sample_path])[0]


def _raw_col(df: pd.DataFrame, spec: dict[str, Any], field: str) -> str:
    return df.columns[spec["columns"][field]]


def _by_field(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {m["field"]: m for m in report["matched"]}


def test_sample_all_fields_matched_by_keyword(loaded, schema) -> None:
    df, report = map_columns(loaded, schema)
    assert sorted(_by_field(report)) == sorted(STANDARD)
    assert {m["method"] for m in report["matched"]} == {"keyword"}
    assert report["missing"] == []
    assert report["unrecognized"] == []
    assert report["missing_required"] == []
    assert list(df.columns) == STANDARD + [SOURCE_COL]


def test_before_and_after_not_mixed(loaded, schema, sample_spec) -> None:
    _, report = map_columns(loaded, schema)
    matched = _by_field(report)
    for item in ("identify", "describe", "when"):
        assert matched[f"pre_{item}"]["column"] == _raw_col(loaded, sample_spec, f"pre_{item}")
        assert matched[f"post_{item}"]["column"] == _raw_col(loaded, sample_spec, f"post_{item}")


def test_messy_header_spacing_still_matches(loaded, schema, sample_spec) -> None:
    col = _raw_col(loaded, sample_spec, "well_organized")
    messy = loaded.rename(columns={col: "  " + col.replace(" ", "\xa0 ", 2) + "‎ "})
    _, report = map_columns(messy, schema)
    assert _by_field(report)["well_organized"]["method"] == "keyword"


def test_reworded_question_matches_by_fuzzy(loaded, schema, sample_spec) -> None:
    col = _raw_col(loaded, sample_spec, "post_when")
    reworded = col.replace("I know when to use 988", "I know when I should use 988")
    df, report = map_columns(loaded.rename(columns={col: reworded}), schema)
    hit = _by_field(report)["post_when"]
    assert hit["method"] == "fuzzy" and hit["column"] == reworded
    assert hit["score"] >= schema["matching"]["fuzzy_threshold"]
    assert "post_when" in df.columns


def test_fuzzy_never_crosses_before_after(loaded, schema, sample_spec) -> None:
    # Looks like the AFTER example, but says "think back": none_of must block it.
    col = _raw_col(loaded, sample_spec, "post_when")
    tricky = col.replace("I know when to use 988", "I know when I should use 988 (think back)")
    _, report = map_columns(loaded.rename(columns={col: tricky}), schema)
    assert "post_when" not in _by_field(report)
    assert {"column": tricky, "reason": "no field matches"} in report["unrecognized"]


def test_removed_optional_question_is_missing(loaded, schema, sample_spec) -> None:
    df, report = map_columns(loaded.drop(columns=[_raw_col(loaded, sample_spec, "likely_call")]), schema)
    assert {"field": "likely_call", "required": False} in report["missing"]
    assert report["missing_required"] == []
    assert "likely_call" not in df.columns


def test_removed_required_question_reported(loaded, schema, sample_spec) -> None:
    _, report = map_columns(loaded.drop(columns=[_raw_col(loaded, sample_spec, "response_id")]), schema)
    assert report["missing_required"] == ["response_id"]


def test_new_question_unrecognized_and_kept(loaded, schema) -> None:
    extra = loaded.assign(**{"Would you attend another training?": "Yes"})
    df, report = map_columns(extra, schema)
    assert report["unrecognized"] == [
        {"column": "Would you attend another training?", "reason": "no field matches"}
    ]
    assert "Would you attend another training?" in df.columns


def test_two_columns_for_one_field_both_unrecognized(loaded, schema, sample_spec) -> None:
    col = _raw_col(loaded, sample_spec, "comment_improve")
    twin = loaded.assign(**{"What could improve the training?": None})
    df, report = map_columns(twin, schema)
    assert "comment_improve" not in _by_field(report)
    reasons = {u["column"]: u["reason"] for u in report["unrecognized"]}
    assert reasons[col] == reasons["What could improve the training?"] == "several columns match comment_improve"
    assert {"field": "comment_improve", "required": False} in report["missing"]
    assert col in df.columns  # kept under raw name


def test_column_matching_two_fields_unrecognized(loaded, schema) -> None:
    both = loaded.assign(**{"How likely are you to recommend us or call 988?": None})
    _, report = map_columns(both, schema)
    reasons = {u["column"]: u["reason"] for u in report["unrecognized"]}
    assert reasons["How likely are you to recommend us or call 988?"].startswith("matches several fields")


def test_input_not_modified(loaded, schema) -> None:
    before = loaded.copy()
    map_columns(loaded, schema)
    pd.testing.assert_frame_equal(loaded, before)
