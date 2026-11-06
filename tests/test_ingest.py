"""M1 tests for pipeline/ingest.py. No network; extra files go in tmp_path."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pipeline.ingest import SOURCE_COL, find_key_columns, load_files


def _col(sample_raw: pd.DataFrame, spec: dict[str, Any], field: str) -> str:
    """Raw header of a standard field, by its position in the sample."""
    return sample_raw.columns[spec["columns"][field]]


def _write_xlsx(df: pd.DataFrame, path: Path) -> Path:
    df.to_excel(path, index=False)
    return path


# ----- sample file ----------------------------------------------------

def test_sample_counts(sample_path: Path) -> None:
    df, stats = load_files([sample_path])
    assert stats["rows_in"] == 30
    assert stats["duplicates_removed"] == 1
    assert stats["rows_out"] == 29 == len(df)
    assert stats["files"] == {"sample_30rows.xlsx": 30}
    assert stats["blank_rows_removed"] == 0
    assert stats["blank_key_rows"] == 0
    assert stats["dedupe_skipped"] is False


def test_reused_id_kept_true_duplicate_removed(
    sample_path: Path, sample_raw: pd.DataFrame, sample_spec: dict[str, Any]
) -> None:
    df, _ = load_files([sample_path])
    ids = df[_col(sample_raw, sample_spec, "response_id")].astype(str)
    assert (ids == "3").sum() == 2      # same Id, different start time
    assert (ids == "114").sum() == 1    # exact duplicate collapsed


def test_source_file_column(sample_path: Path) -> None:
    df, _ = load_files([sample_path])
    assert (df[SOURCE_COL] == "sample_30rows.xlsx").all()


def test_raw_values_untouched(
    sample_path: Path, sample_raw: pd.DataFrame, sample_spec: dict[str, Any]
) -> None:
    df, _ = load_files([sample_path])
    assert list(df.columns) == list(sample_raw.columns) + [SOURCE_COL]
    county = df[_col(sample_raw, sample_spec, "county")].dropna().astype(str)
    assert county.str.contains("\xa0").any()
    knew = df[_col(sample_raw, sample_spec, "presenter_knew")].dropna().astype(str)
    assert knew.str.strip().eq("Satisfied").any()
    # "none" comment must survive (not turned into a missing value)
    improve = df[_col(sample_raw, sample_spec, "comment_improve")].dropna().astype(str)
    assert improve.str.strip().str.lower().eq("none").any()


def test_input_file_not_modified(sample_path: Path) -> None:
    before = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    load_files([sample_path])
    assert hashlib.sha256(sample_path.read_bytes()).hexdigest() == before


# ----- several files / file kinds -------------------------------------

def test_same_file_twice_dedupes_across_files(sample_path: Path, tmp_path: Path) -> None:
    copy = tmp_path / "copy.xlsx"
    copy.write_bytes(sample_path.read_bytes())
    df, stats = load_files([sample_path, copy])
    assert stats["rows_in"] == 60
    assert stats["duplicates_removed"] == 31   # 30 cross-file + 1 within-file
    assert stats["rows_out"] == 29 == len(df)
    assert stats["files"] == {"sample_30rows.xlsx": 30, "copy.xlsx": 30}
    assert (df[SOURCE_COL] == "sample_30rows.xlsx").sum() == 29  # first copy kept


def test_uploaded_file_object_matches_path(sample_path: Path) -> None:
    upload = io.BytesIO(sample_path.read_bytes())
    upload.name = "uploads/sample_30rows.xlsx"
    df_up, stats_up = load_files([upload])
    df_path, stats_path = load_files([sample_path])
    assert stats_up == stats_path
    pd.testing.assert_frame_equal(df_up, df_path)


def test_csv_matches_xlsx(sample_raw: pd.DataFrame, tmp_path: Path) -> None:
    csv = tmp_path / "sample.csv"
    sample_raw.to_csv(csv, index=False)
    _, stats = load_files([csv])
    assert (stats["rows_in"], stats["duplicates_removed"], stats["rows_out"]) == (30, 1, 29)


def test_xlsx_datetime_vs_csv_text_start_time_match(
    sample_raw: pd.DataFrame, sample_spec: dict[str, Any], tmp_path: Path
) -> None:
    start = _col(sample_raw, sample_spec, "start_time")
    as_dates = sample_raw.copy()
    as_dates[start] = pd.to_datetime(as_dates[start])
    xlsx = _write_xlsx(as_dates, tmp_path / "dates.xlsx")
    csv = tmp_path / "text.csv"
    sample_raw.to_csv(csv, index=False)
    _, stats = load_files([xlsx, csv])
    assert stats["rows_in"] == 60
    assert stats["rows_out"] == 29


def test_header_spelling_variants_align(
    sample_raw: pd.DataFrame, sample_spec: dict[str, Any], tmp_path: Path
) -> None:
    county = _col(sample_raw, sample_spec, "county")
    a = _write_xlsx(sample_raw.iloc[:10], tmp_path / "a.xlsx")
    b_df = sample_raw.iloc[10:20].rename(columns={county: county + "\xa0 "})
    b = _write_xlsx(b_df, tmp_path / "b.xlsx")
    df, stats = load_files([a, b])
    assert county in df.columns
    assert county + "\xa0 " not in df.columns
    assert df[county].notna().sum() == sample_raw[county].iloc[:20].notna().sum()
    assert stats["rows_out"] == 20


def test_extra_column_in_one_file_kept(sample_raw: pd.DataFrame, tmp_path: Path) -> None:
    a = _write_xlsx(sample_raw.iloc[:5], tmp_path / "a.xlsx")
    b_df = sample_raw.iloc[5:10].copy()
    b_df["Brand new question"] = "yes"
    b = _write_xlsx(b_df, tmp_path / "b.xlsx")
    df, _ = load_files([a, b])
    assert "Brand new question" in df.columns
    assert df["Brand new question"].isna().sum() == 5


# ----- blanks -------------------------------------------------------------

def test_blank_key_rows_kept_and_counted(
    sample_raw: pd.DataFrame, sample_spec: dict[str, Any], tmp_path: Path
) -> None:
    rid = _col(sample_raw, sample_spec, "response_id")
    start = _col(sample_raw, sample_spec, "start_time")
    df_in = pd.concat([sample_raw.iloc[[0, 0]], sample_raw.iloc[[1, 1]]], ignore_index=True)
    df_in.loc[[0, 1], rid] = None      # identical rows but no Id
    df_in.loc[[2, 3], start] = None    # identical rows but no start time
    path = _write_xlsx(df_in, tmp_path / "blank_keys.xlsx")
    df, stats = load_files([path])
    assert stats["blank_key_rows"] == 4
    assert stats["duplicates_removed"] == 0
    assert len(df) == 4


def test_fully_blank_rows_removed(sample_raw: pd.DataFrame, tmp_path: Path) -> None:
    blank = pd.DataFrame([[None] * sample_raw.shape[1]] * 3, columns=sample_raw.columns)
    blank.iloc[1, 3] = "   "  # whitespace only still counts as blank
    # blanks go in the middle: Excel does not store trailing empty rows
    mixed = pd.concat([sample_raw.iloc[:15], blank, sample_raw.iloc[15:]], ignore_index=True)
    path = _write_xlsx(mixed, tmp_path / "t.xlsx")
    _, stats = load_files([path])
    assert stats["blank_rows_removed"] == 3
    assert stats["rows_out"] == 29


def test_partly_blank_row_kept(sample_raw: pd.DataFrame, tmp_path: Path) -> None:
    row = pd.DataFrame([[None] * sample_raw.shape[1]], columns=sample_raw.columns)
    row.iloc[0, 7] = "Agree"
    path = _write_xlsx(pd.concat([sample_raw, row], ignore_index=True), tmp_path / "p.xlsx")
    _, stats = load_files([path])
    assert stats["blank_rows_removed"] == 0
    assert stats["rows_out"] == 30


# ----- key columns ------------------------------------------------------

def test_find_key_columns_on_sample(
    sample_raw: pd.DataFrame, sample_spec: dict[str, Any], schema: dict[str, Any]
) -> None:
    found, problems = find_key_columns(sample_raw.columns, schema)
    assert problems == []
    assert found == {
        "response_id": _col(sample_raw, sample_spec, "response_id"),
        "start_time": _col(sample_raw, sample_spec, "start_time"),
    }


def test_missing_key_column_skips_dedupe(
    sample_raw: pd.DataFrame, sample_spec: dict[str, Any], tmp_path: Path
) -> None:
    rid = _col(sample_raw, sample_spec, "response_id")
    path = _write_xlsx(sample_raw.drop(columns=[rid]), tmp_path / "no_id.xlsx")
    df, stats = load_files([path])
    assert stats["dedupe_skipped"] is True
    assert "response_id" in stats["dedupe_note"]
    assert stats["duplicates_removed"] == 0
    assert len(df) == 30


# ----- errors -------------------------------------------------------------

def test_no_files_raises() -> None:
    with pytest.raises(ValueError, match="No files"):
        load_files([])


def test_unsupported_type_raises(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(ValueError, match="notes.txt.*not a supported"):
        load_files([path])


def test_corrupt_xlsx_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.xlsx"
    path.write_bytes(b"this is not a workbook")
    with pytest.raises(ValueError, match="Could not read 'broken.xlsx'"):
        load_files([path])


def test_header_only_file_raises(sample_raw: pd.DataFrame, tmp_path: Path) -> None:
    path = _write_xlsx(sample_raw.iloc[:0], tmp_path / "empty.xlsx")
    with pytest.raises(ValueError, match="empty.xlsx.*no responses"):
        load_files([path])
