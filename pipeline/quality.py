"""Data-quality flags. Flags never delete rows; the caller decides what to filter.

Thresholds come from ``quality:`` in ``config/schema.yaml``. Each flag is
True, False, or unknown (``pd.NA``): a blank answer is never read as "fine".
A row is False only when the answers it does have already rule the flag out.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline import load_schema

FLAG_COLUMNS: tuple[str, ...] = (
    "flag_speeder", "flag_straightline", "flag_reversed", "flag_any",
)


def _unknown(index: pd.Index) -> pd.Series:
    """A flag column where every row is unknown."""
    return pd.Series(pd.NA, index=index, dtype="boolean")


def _speeder(df: pd.DataFrame, seconds: float) -> pd.Series:
    """True when the survey took under ``seconds``; unknown if no duration."""
    if "duration_sec" not in df.columns:
        return _unknown(df.index)
    dur = pd.to_numeric(df["duration_sec"], errors="coerce")
    return pd.Series(dur < seconds, index=df.index, dtype="boolean").mask(dur.isna())


def _straightline(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    """True when every listed item is answered and all answers are identical.

    False as soon as two answered items differ. Unknown when the answered
    items agree but some are blank. Uses the listed items that exist (>= 2).
    """
    present = [f for f in fields if f in df.columns]
    if len(present) < 2:
        return _unknown(df.index)
    vals = df[present].apply(pd.to_numeric, errors="coerce")
    distinct = vals.nunique(axis=1)
    complete = vals.notna().all(axis=1)
    out = _unknown(df.index)
    out[distinct >= 2] = False
    out[complete & (distinct == 1)] = True
    return out


def _reversed(df: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
    """True when every post item <= the cut-off and nps >= the cut-off.

    False when any known answer rules it out; unknown otherwise.
    """
    post = [f for f in rule.get("post_fields", []) if f in df.columns]
    if not post or "nps" not in df.columns:
        return _unknown(df.index)
    low = rule["post_fields_all_at_most"]
    vals = df[post].apply(pd.to_numeric, errors="coerce")
    nps = pd.to_numeric(df["nps"], errors="coerce")
    ruled_out = (vals > low).any(axis=1) | (nps < rule["nps_at_least"])
    matches = (vals <= low).all(axis=1) & vals.notna().all(axis=1) & (nps >= rule["nps_at_least"])
    out = _unknown(df.index)
    out[ruled_out] = False
    out[matches & ~ruled_out] = True
    return out


def add_flags(df: pd.DataFrame, schema: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return a copy of ``df`` with quality flag columns added.

    ``flag_speeder``, ``flag_straightline`` and ``flag_reversed`` are nullable
    booleans (unknown when the answers needed are blank or not collected).
    ``flag_any`` is a plain bool: True when any flag is True.
    """
    schema = load_schema() if schema is None else schema
    rules = schema["quality"]
    out = df.copy()
    out["flag_speeder"] = _speeder(out, rules["speeder_seconds"])
    out["flag_straightline"] = _straightline(out, list(rules["straightline_fields"]))
    out["flag_reversed"] = _reversed(out, rules["reversed_scale"])
    any_flag = out["flag_speeder"] | out["flag_straightline"] | out["flag_reversed"]
    out["flag_any"] = any_flag.fillna(False).astype(bool)
    return out


def flag_summary(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Counts per flag column: ``{flag: {flagged, unknown, n}}`` (after ``add_flags``)."""
    out: dict[str, dict[str, int]] = {}
    for c in FLAG_COLUMNS:
        if c in df.columns:
            col = df[c].astype("boolean")
            out[c] = {"flagged": int(col.fillna(False).sum()),
                      "unknown": int(col.isna().sum()), "n": len(df)}
    return out


__all__ = ["add_flags", "flag_summary", "FLAG_COLUMNS"]
