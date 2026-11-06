"""Load one or more survey exports into a single raw table and drop duplicates.

Values are kept exactly as exported (everything is read as text); cleaning
happens later. A duplicate is a row with the same response id AND start time
as an earlier row (keys from ``dedupe_keys`` in ``config/schema.yaml``). ID
alone is not enough because a new Microsoft Form restarts IDs at 1.

Columns still have raw names here, so the two key columns are located with a
small keyword lookup (``all_of`` / ``any_of`` / ``none_of`` / ``exact`` from the
schema, no fuzzy matching). Full column mapping is done in ``mapping.py``.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from pipeline import load_schema

SOURCE_COL: str = "source_file"
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".xlsx", ".csv")

# Only truly empty cells count as missing; text such as "None" or "NA" is a
# real answer and must survive ingest.
_READ_OPTS: dict[str, Any] = {"dtype": object, "keep_default_na": False, "na_values": [""]}


# ----- helpers --------------------------------------------------------

def normalize_text(value: Any, norm: dict[str, Any]) -> str:
    """Apply the schema's ``normalization`` rules to one header or value."""
    text = str(value)
    for char in norm.get("remove_chars", []) or []:
        text = text.replace(char, "")
    for old, new in (norm.get("replace_chars", {}) or {}).items():
        text = text.replace(old, new)
    if norm.get("collapse_whitespace", False):
        text = re.sub(r"\s+", " ", text)
    if norm.get("strip", False):
        text = text.strip()
    if norm.get("lowercase", False):
        text = text.lower()
    return text


def _file_name_and_bytes(file: Any) -> tuple[str, bytes]:
    """Return (base file name, raw bytes) for a path or an uploaded file object."""
    if isinstance(file, (str, Path)):
        path = Path(file)
        try:
            return path.name, path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Could not open '{path.name}': {exc}") from exc
    name = getattr(file, "name", None)
    if not name:
        raise ValueError("Uploaded file has no name, so its type cannot be told.")
    if hasattr(file, "getvalue"):
        data = file.getvalue()
    else:
        if hasattr(file, "seek"):
            file.seek(0)
        data = file.read()
    return Path(str(name)).name, bytes(data)


def _read_one(name: str, data: bytes) -> pd.DataFrame:
    """Read one export (first sheet of .xlsx, or .csv) with all values as text."""
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"'{name}' is not a supported file type. Please upload .xlsx or .csv."
        )
    try:
        if ext == ".xlsx":
            df = pd.read_excel(io.BytesIO(data), sheet_name=0, **_READ_OPTS)
        else:
            try:
                df = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", **_READ_OPTS)
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(data), encoding="cp1252", **_READ_OPTS)
    except Exception as exc:  # pandas/openpyxl raise many different types
        raise ValueError(f"Could not read '{name}': {exc}") from exc
    if df.shape[1] == 0:
        raise ValueError(f"'{name}' has no columns.")
    return df


def _is_blank(value: Any) -> bool:
    """True for empty cells and whitespace-only text."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.replace("\xa0", " ").strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _blank_row_mask(df: pd.DataFrame) -> pd.Series:
    """Rows where every column (ignoring ``source_file``) is blank."""
    cols = [c for c in df.columns if c != SOURCE_COL]
    return df[cols].apply(lambda col: col.map(_is_blank)).all(axis=1)


def _rule_matches(header_norm: str, rule: dict[str, Any], norm: dict[str, Any]) -> bool:
    """Keyword-rule check for one normalized header (no fuzzy matching)."""
    all_of = [normalize_text(p, norm) for p in rule.get("all_of", []) or []]
    any_of = [normalize_text(p, norm) for p in rule.get("any_of", []) or []]
    none_of = [normalize_text(p, norm) for p in rule.get("none_of", []) or []]
    if not all_of:
        return False
    if rule.get("exact", False):
        if header_norm != all_of[0]:
            return False
    elif not all(p in header_norm for p in all_of):
        return False
    if any_of and not any(p in header_norm for p in any_of):
        return False
    if any(p in header_norm for p in none_of):
        return False
    return True


def find_key_columns(
    columns: Iterable[Any], schema: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Locate the raw column for each dedupe key.

    Returns ``(found, problems)``: ``found`` maps key field -> raw column name;
    ``problems`` lists keys that were missing or matched more than one column.
    """
    norm = schema.get("normalization", {}) or {}
    fields = schema.get("fields", {}) or {}
    cols = list(columns)
    found: dict[str, Any] = {}
    problems: list[str] = []
    for key in schema.get("dedupe_keys", []) or []:
        rule = fields.get(key)
        if rule is None:
            problems.append(f"{key}: not defined in schema")
            continue
        hits = [c for c in cols if _rule_matches(normalize_text(c, norm), rule, norm)]
        if len(hits) == 1:
            found[key] = hits[0]
        elif not hits:
            problems.append(f"{key}: no matching column")
        else:
            problems.append(f"{key}: several matching columns {hits}")
    return found, problems


def _key_value(value: Any, kind: str | None) -> Any:
    """Comparable form of one key cell, or None if blank. Does not alter the data."""
    if _is_blank(value):
        return None
    if kind in ("datetime", "date"):
        ts = pd.to_datetime(value, errors="coerce")
        if not pd.isna(ts):
            return ("ts", pd.Timestamp(ts))
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).replace("\xa0", " ").strip()
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".")[0]
    return ("text", text)


# ----- main entry point -----------------------------------------------

def load_files(
    files: Iterable[Any], schema: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read every export, stack them, and remove duplicate responses.

    ``files`` are paths or uploaded file objects (with ``.name`` and ``.read()``).
    Returns ``(df, stats)`` where ``stats`` has ``rows_in``, ``duplicates_removed``,
    ``rows_out``, ``files`` (rows per file), ``blank_rows_removed``,
    ``blank_key_rows``, ``dedupe_skipped`` and ``dedupe_note``.
    """
    schema = load_schema() if schema is None else schema
    norm = schema.get("normalization", {}) or {}
    file_list = list(files or [])
    if not file_list:
        raise ValueError("No files were given. Please upload at least one survey export.")

    frames: list[pd.DataFrame] = []
    per_file: dict[str, int] = {}
    canonical: dict[str, Any] = {}  # normalized header -> first spelling seen
    for file in file_list:
        name, data = _file_name_and_bytes(file)
        df = _read_one(name, data)
        if df.empty:
            raise ValueError(f"'{name}' has a header row but no responses.")
        rename = {}
        for col in df.columns:
            key = normalize_text(col, norm)
            canonical.setdefault(key, col)
            rename[col] = canonical[key]
        df = df.rename(columns=rename)
        df[SOURCE_COL] = name
        per_file[name] = per_file.get(name, 0) + len(df)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.astype(object).where(combined.notna(), None)
    rows_in = len(combined)

    blank = _blank_row_mask(combined)
    blank_rows_removed = int(blank.sum())
    combined = combined.loc[~blank].reset_index(drop=True)

    found, problems = find_key_columns(
        [c for c in combined.columns if c != SOURCE_COL], schema
    )
    dedupe_skipped = bool(problems)
    duplicates_removed = 0
    blank_key_rows = 0
    note = ""
    if dedupe_skipped:
        note = "Duplicate check skipped: " + "; ".join(problems)
    else:
        fields = schema.get("fields", {}) or {}
        keys = pd.DataFrame({
            k: combined[col].map(lambda v, t=fields[k].get("type"): _key_value(v, t))
            for k, col in found.items()
        })
        has_blank = keys.isna().any(axis=1)
        blank_key_rows = int(has_blank.sum())
        dup = keys.duplicated(keep="first") & ~has_blank
        duplicates_removed = int(dup.sum())
        combined = combined.loc[~dup].reset_index(drop=True)

    stats: dict[str, Any] = {
        "rows_in": rows_in,
        "duplicates_removed": duplicates_removed,
        "rows_out": len(combined),
        "files": per_file,
        "blank_rows_removed": blank_rows_removed,
        "blank_key_rows": blank_key_rows,
        "dedupe_skipped": dedupe_skipped,
        "dedupe_note": note,
    }
    return combined, stats


__all__ = ["load_files", "find_key_columns", "normalize_text", "SOURCE_COL"]
