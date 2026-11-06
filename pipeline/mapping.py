"""Match raw survey column names to the standard field names in the schema.

Keyword rules come first (``all_of`` / ``any_of`` / ``none_of`` / ``exact``).
Columns no rule matches get one fuzzy try against each field's ``example``;
fuzzy candidates must still pass ``any_of`` / ``none_of``. Anything ambiguous
or unmatched is reported as "unrecognized" and kept under its raw name.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from pipeline.ingest import SOURCE_COL, normalize_text

PASS_THROUGH: tuple[str, ...] = (SOURCE_COL,)


def _phrases(rule: dict[str, Any], key: str, norm: dict[str, Any]) -> list[str]:
    """Normalized phrases listed under ``rule[key]``."""
    return [normalize_text(p, norm) for p in rule.get(key, []) or []]


def _passes_filters(header: str, rule: dict[str, Any], norm: dict[str, Any]) -> bool:
    """``any_of`` / ``none_of`` checks (also applied to fuzzy candidates)."""
    any_of = _phrases(rule, "any_of", norm)
    none_of = _phrases(rule, "none_of", norm)
    if any_of and not any(p in header for p in any_of):
        return False
    return not any(p in header for p in none_of)


def keyword_match(header: str, rule: dict[str, Any], norm: dict[str, Any]) -> bool:
    """True when a normalized header passes every keyword rule of a field."""
    all_of = _phrases(rule, "all_of", norm)
    if not all_of:
        return False
    if rule.get("exact", False):
        if header != all_of[0]:
            return False
    elif not all(p in header for p in all_of):
        return False
    return _passes_filters(header, rule, norm)


def map_columns(
    raw: pd.DataFrame, schema: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rename recognized columns to standard names and report the rest.

    Returns ``(df, report)``. ``report`` has ``matched`` (field, column, method,
    score), ``missing`` (field, required), ``unrecognized`` (column, reason) and
    ``missing_required`` (field names). Missing fields are not added as columns.
    """
    df = raw.copy()
    norm = schema.get("normalization", {}) or {}
    fields: dict[str, dict[str, Any]] = schema.get("fields", {}) or {}
    threshold = float((schema.get("matching", {}) or {}).get("fuzzy_threshold", 85))

    columns = [c for c in df.columns if c not in PASS_THROUGH]
    headers = {c: normalize_text(c, norm) for c in columns}

    # --- 1. keyword rules -------------------------------------------------
    col_hits: dict[Any, list[str]] = {
        c: [f for f, rule in fields.items() if keyword_match(h, rule, norm)]
        for c, h in headers.items()
    }
    unrecognized: dict[Any, str] = {}
    field_cols: dict[str, list[tuple[Any, str, float | None]]] = {f: [] for f in fields}
    no_rule: list[Any] = []
    for col, hits in col_hits.items():
        if len(hits) == 1:
            field_cols[hits[0]].append((col, "keyword", None))
        elif len(hits) > 1:
            unrecognized[col] = f"matches several fields: {', '.join(hits)}"
        else:
            no_rule.append(col)

    # --- 2. fuzzy fallback for columns no rule matched --------------------
    for col in no_rule:
        header = headers[col]
        scores = []
        for f, rule in fields.items():
            if field_cols[f] or "example" not in rule:
                continue  # field already taken by a keyword match
            if not _passes_filters(header, rule, norm):
                continue
            score = fuzz.token_sort_ratio(header, normalize_text(rule["example"], norm))
            if score >= threshold:
                scores.append((score, f))
        if not scores:
            unrecognized[col] = "no field matches"
            continue
        scores.sort(reverse=True)
        if len(scores) > 1 and scores[0][0] == scores[1][0]:
            unrecognized[col] = f"equally close to {scores[0][1]} and {scores[1][1]}"
            continue
        score, f = scores[0]
        field_cols[f].append((col, "fuzzy", round(float(score), 1)))

    # --- 3. one column per field ------------------------------------------
    matched: list[dict[str, Any]] = []
    rename: dict[Any, str] = {}
    for f, hits in field_cols.items():
        if len(hits) == 1:
            col, method, score = hits[0]
            rename[col] = f
            matched.append({"field": f, "column": col, "method": method, "score": score})
        elif len(hits) > 1:
            for col, _, _ in hits:
                unrecognized[col] = f"several columns match {f}"

    taken = set(rename.values())
    missing = [
        {"field": f, "required": bool(rule.get("required", False))}
        for f, rule in fields.items()
        if f not in taken
    ]
    df = df.rename(columns=rename)
    report = {
        "matched": matched,
        "missing": missing,
        "unrecognized": [
            {"column": c, "reason": unrecognized[c]} for c in columns if c in unrecognized
        ],
        "missing_required": [m["field"] for m in missing if m["required"]],
    }
    return df, report


__all__ = ["map_columns", "keyword_match"]
