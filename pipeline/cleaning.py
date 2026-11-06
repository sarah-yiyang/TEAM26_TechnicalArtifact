"""Turn mapped survey answers into clean, typed standard columns.

Value translations come from ``config/schema.yaml``. Any answer not listed
there is set blank in the cleaned column and reported in ``unknown_values``;
it is never guessed. Judgment-call translations (``ambiguous_values``) and
other notable fixes are counted in ``fix_log``.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from pipeline.ingest import normalize_text

HIDDEN_CHARS_FIX: str = "removed hidden characters / doubled spaces"
# Category fields where merging spelling variants is worth logging
# (e.g. "Fairfax county " -> "Fairfax (VA)"). Language labels are not.
VARIANT_LOG_FIELDS: tuple[str, ...] = ("county",)


def _blank(value: Any) -> bool:
    """True for None/NaN and whitespace-only text."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.replace("\xa0", " ").strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


class _Log:
    """Collects fix-log entries and unknown values while cleaning."""

    def __init__(self, ids: pd.Series) -> None:
        self.ids = ids
        self.fixes: dict[tuple[str, str, bool], set[int]] = defaultdict(set)
        self.fix_cells: dict[tuple[str, str, bool], int] = defaultdict(int)
        self.unknown: dict[tuple[str, str], list[int]] = defaultdict(list)

    def fix(self, field: str, change: str, row: int, assumption: bool = False) -> None:
        key = (field, change, assumption)
        self.fixes[key].add(row)
        self.fix_cells[key] += 1

    def unknown_value(self, field: str, value: Any, row: int) -> None:
        self.unknown[(field, str(value))].append(row)

    def fix_log(self) -> list[dict[str, Any]]:
        """One entry per fix. Assumptions are merged across fields, so
        "'Satisfied' -> 4" is a single line with total cells and rows."""
        merged: dict[tuple[str, str, bool], dict[str, Any]] = {}
        for (f, c, a), rows in self.fixes.items():
            key = ("", c, a) if a else (f, c, a)
            entry = merged.setdefault(key, {"fields": [], "rows": set(), "cells": 0})
            entry["fields"].append(f)
            entry["rows"] |= rows
            entry["cells"] += self.fix_cells[(f, c, a)]
        out = [
            {"field": ", ".join(e["fields"]), "change": c, "cells": e["cells"],
             "rows": len(e["rows"]), "assumption": a}
            for (_, c, a), e in merged.items()
        ]
        return sorted(out, key=lambda e: (not e["assumption"], e["field"], e["change"]))

    def unknown_values(self) -> list[dict[str, Any]]:
        out = []
        for (f, v), rows in self.unknown.items():
            ids = [self.ids.iloc[r] for r in rows] if self.ids is not None else rows
            out.append({"field": f, "value": v, "rows": len(rows),
                        "response_ids": [_plain(i) for i in ids]})
        return sorted(out, key=lambda e: (e["field"], -e["rows"], e["value"]))


def _plain(value: Any) -> Any:
    """numpy scalars -> plain Python, for tidy reports."""
    return value.item() if isinstance(value, np.generic) else value


def _hidden_chars(raw: Any, norm: dict[str, Any]) -> bool:
    """True when normalization removed/replaced something beyond trim + case."""
    if not isinstance(raw, str):
        return False
    soft = {"lowercase": True, "strip": True}
    return normalize_text(raw, norm) != normalize_text(raw, soft)


# ----- per-type cleaners ----------------------------------------------

def _translate(
    series: pd.Series, field: str, mapping: dict[str, Any], schema: dict[str, Any], log: _Log
) -> pd.Series:
    """Map normalized text through ``mapping``; unknowns -> None + reported."""
    norm = schema.get("normalization", {}) or {}
    lookup = {normalize_text(k, norm): v for k, v in mapping.items()}
    ambiguous = {normalize_text(v, norm) for v in schema.get("ambiguous_values", []) or []}
    out: list[Any] = []
    for i, raw in enumerate(series):
        if _blank(raw):
            out.append(None)
            continue
        key = normalize_text(raw, norm)
        if _hidden_chars(raw, norm):
            log.fix(field, HIDDEN_CHARS_FIX, i)
        if key not in lookup:
            log.unknown_value(field, raw, i)
            out.append(None)
            continue
        value = lookup[key]
        if key in ambiguous:
            log.fix(field, f"'{str(raw).strip()}' -> {value}", i, assumption=True)
        out.append(value)
    return pd.Series(out, index=series.index, dtype=object)


def _variant_log(
    series: pd.Series, cleaned: pd.Series, field: str, schema: dict[str, Any], log: _Log
) -> None:
    """Log category spellings merged into a different canonical label."""
    norm = schema.get("normalization", {}) or {}
    for i, (raw, value) in enumerate(zip(series, cleaned)):
        if value is None or _blank(raw):
            continue
        if normalize_text(raw, norm) != normalize_text(value, norm):
            log.fix(field, f"'{normalize_text(raw, norm)}' -> {value}", i)


def _scale(series: pd.Series, field: str, schema: dict[str, Any], log: _Log) -> pd.Series:
    """Whole numbers inside ``analysis.scale_0_10_valid_range``; else unknown."""
    lo, hi = (schema.get("analysis", {}) or {}).get("scale_0_10_valid_range", [0, 10])
    out: list[Any] = []
    for i, raw in enumerate(series):
        if _blank(raw):
            out.append(pd.NA)
            continue
        num = pd.to_numeric(str(raw).replace("\xa0", " ").strip(), errors="coerce")
        if pd.isna(num) or float(num) != int(num) or not lo <= int(num) <= hi:
            log.unknown_value(field, raw, i)
            out.append(pd.NA)
        else:
            out.append(int(num))
    return pd.Series(out, index=series.index, dtype="Int64")


def _datetime(series: pd.Series, field: str, log: _Log, date_only: bool) -> pd.Series:
    """Parse dates/times; unparseable values -> NaT + reported."""
    out: list[Any] = []
    for i, raw in enumerate(series):
        if _blank(raw):
            out.append(pd.NaT)
            continue
        ts = pd.to_datetime(str(raw).strip() if isinstance(raw, str) else raw, errors="coerce")
        if pd.isna(ts):
            log.unknown_value(field, raw, i)
            out.append(pd.NaT)
        else:
            out.append(ts.normalize() if date_only else ts)
    return pd.Series(pd.to_datetime(out), index=series.index)


def _text(series: pd.Series, field: str, schema: dict[str, Any], log: _Log) -> pd.Series:
    """Remove hidden characters and trim; keep the writer's capitals."""
    norm = dict(schema.get("normalization", {}) or {})
    norm["lowercase"] = False
    out: list[Any] = []
    for i, raw in enumerate(series):
        if _blank(raw):
            out.append(None)
            continue
        text = normalize_text(raw, norm)
        if _hidden_chars(raw, schema.get("normalization", {}) or {}):
            log.fix(field, HIDDEN_CHARS_FIX, i)
        out.append(text)
    return pd.Series(out, index=series.index, dtype=object)


def _roles(
    series: pd.Series, rule: dict[str, Any], schema: dict[str, Any], log: _Log
) -> tuple[pd.Series, pd.Series]:
    """Return (role_primary label, is_student) from a multi-select answer."""
    norm = schema.get("normalization", {}) or {}
    sep = rule.get("separator", ";")
    lookup = {normalize_text(k, norm): v for k, v in (schema.get("role_values", {}) or {}).items()}
    priority = schema.get("role_priority", []) or []
    labels = schema.get("role_labels", {}) or {}
    primary: list[Any] = []
    student: list[Any] = []
    for i, raw in enumerate(series):
        if _blank(raw):
            primary.append(None)
            student.append(pd.NA)
            continue
        codes = set()
        for part in str(raw).split(sep):
            if _blank(part):
                continue
            key = normalize_text(part, norm)
            if key in lookup:
                codes.add(lookup[key])
            else:
                log.unknown_value("role_raw", part.strip(), i)
        top = next((c for c in priority if c in codes), None)
        primary.append(labels.get(top, top) if top else None)
        student.append("student" in codes if codes else pd.NA)
    return (pd.Series(primary, index=series.index, dtype=object),
            pd.Series(student, index=series.index, dtype="boolean"))


# ----- main entry point -----------------------------------------------

def clean(
    df: pd.DataFrame, schema: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Clean every mapped standard column and add derived columns.

    Returns ``(df, fix_log, unknown_values)``. Fields that were not mapped are
    skipped, and derived columns that need them are not created. Adds
    ``duration_sec``, ``role_primary``, ``is_student``, ``month`` (YYYY-MM from
    ``training_date``) and ``session_id`` (``YYYY-MM-DD_County``).
    """
    out = df.copy().reset_index(drop=True)
    fields: dict[str, dict[str, Any]] = schema.get("fields", {}) or {}
    ids = out["response_id"] if "response_id" in out.columns else None
    log = _Log(ids)

    for field, rule in fields.items():
        if field not in out.columns:
            continue
        col = out[field]
        kind = rule.get("type")
        if kind in ("likert", "category"):
            mapping = schema.get(rule.get("translate", ""), {}) or {}
            cleaned = _translate(col, field, mapping, schema, log)
            if kind == "category":
                if field in VARIANT_LOG_FIELDS:
                    _variant_log(col, cleaned, field, schema, log)
                out[field] = cleaned
            else:
                out[field] = pd.to_numeric(cleaned, errors="coerce").astype("Int64")
        elif kind == "scale_0_10":
            out[field] = _scale(col, field, schema, log)
        elif kind == "datetime":
            out[field] = _datetime(col, field, log, date_only=False)
        elif kind == "date":
            out[field] = _datetime(col, field, log, date_only=True)
        elif kind == "text":
            out[field] = _text(col, field, schema, log)
        elif kind == "multiselect" and field == "role_raw":
            out["role_primary"], out["is_student"] = _roles(col, rule, schema, log)
        # id: kept exactly as exported

    if {"start_time", "end_time"} <= set(out.columns):
        dur = (out["end_time"] - out["start_time"]).dt.total_seconds()
        negative = dur < 0
        for i in np.flatnonzero(negative.to_numpy()):
            log.fix("duration_sec", "end before start -> blank", int(i))
        out["duration_sec"] = dur.mask(negative)

    if "training_date" in out.columns:
        dates = out["training_date"]
        out["month"] = dates.dt.strftime("%Y-%m").where(dates.notna(), None)
        if "county" in out.columns:
            ok = dates.notna() & out["county"].notna()
            sid = dates.dt.strftime("%Y-%m-%d") + "_" + out["county"].astype(str)
            out["session_id"] = sid.where(ok, None)

    return out, log.fix_log(), log.unknown_values()


__all__ = ["clean"]
