"""Compare one metric across groups (role, county, language, month, session).

Every group is listed with its n; groups with fewer than ``MIN_N`` answers are
suppressed (value blank). Rows with no group value form a "Not recorded" group,
so nothing is dropped. Sessions also get funnel limits and an ``outlier`` flag.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pipeline import load_schema, min_n_from
from pipeline.metrics import ROW_METRICS, row_values

NOT_RECORDED: str = "Not recorded"
Z95: float = 1.96
Z998: float = 3.09                  # outlier = outside these (99.8%) limits
FUNNEL_BY: str = "session_id"

COLUMNS: list[str] = ["group", "n", "metric_n", "value", "suppressed"]
FUNNEL_COLUMNS: list[str] = ["centre", "lower95", "upper95", "lower998", "upper998", "outlier"]


def _funnel(table: pd.DataFrame, scores: pd.Series, kind: str) -> pd.DataFrame:
    """Add funnel limits: centre +/- z * spread / sqrt(n), centre = overall value."""
    centre = float(scores.mean())
    spread = float(scores.std(ddof=0))
    n = table["metric_n"].astype(float).where(table["metric_n"] > 0)
    out = table.copy()
    out["centre"] = centre
    for name, z in (("95", Z95), ("998", Z998)):
        lo, hi = centre - z * spread / np.sqrt(n), centre + z * spread / np.sqrt(n)
        if kind == "share":
            lo, hi = lo.clip(lower=0.0), hi.clip(upper=1.0)
        out[f"lower{name}"], out[f"upper{name}"] = lo, hi
    outside = (out["value"] < out["lower998"]) | (out["value"] > out["upper998"])
    out["outlier"] = outside.fillna(False).astype(bool) & ~out["suppressed"]
    return out


def compare(
    df: pd.DataFrame,
    by: str,
    metric: str = "agree_pct.post_when",
    schema: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return one row per group: ``group, n, metric_n, value, suppressed``.

    ``n`` = rows in the group, ``metric_n`` = rows that answered. Counties also
    get ``out_of_area``; sessions get funnel limits and ``outlier``. Returns an
    empty table when ``by`` or the metric's questions were not collected.
    """
    schema = load_schema() if schema is None else schema
    a = schema.get("analysis", {}) or {}
    options = a.get("group_by_options", []) or []
    if by not in options:
        raise ValueError(f"Cannot compare by {by!r}; options are {options}")
    if metric not in ROW_METRICS:
        raise ValueError(f"Unknown metric {metric!r}; options are {list(ROW_METRICS)}")

    columns = COLUMNS + (FUNNEL_COLUMNS if by == FUNNEL_BY else [])
    scores_all = row_values(df, schema)
    if by not in df.columns or metric not in scores_all.columns:
        return pd.DataFrame(columns=columns)

    min_n = min_n_from(schema)
    groups = df[by].astype(object).where(df[by].notna(), NOT_RECORDED).astype(str)
    scores = scores_all[metric]
    g = scores.groupby(groups)
    table = pd.DataFrame({"n": groups.value_counts(), "metric_n": g.count(), "value": g.mean()})
    table = table.rename_axis("group").reset_index()
    table["n"] = table["n"].astype(int)
    table["metric_n"] = table["metric_n"].fillna(0).astype(int)
    table["suppressed"] = table["metric_n"] < min_n
    table["value"] = table["value"].astype(float).where(~table["suppressed"])

    if by == "county":
        away = set(schema.get("out_of_area_counties", []) or [])
        table["out_of_area"] = table["group"].isin(away)
    if by == FUNNEL_BY:
        table = _funnel(table, scores.dropna(), ROW_METRICS[metric])

    # Months and sessions in date order; others largest first. "Not recorded" last.
    last = table["group"].eq(NOT_RECORDED)
    if by in ("month", FUNNEL_BY):
        table = table.assign(_last=last).sort_values(["_last", "group"])
    else:
        table = table.assign(_last=last).sort_values(["_last", "n"], ascending=[True, False])
    return table.drop(columns="_last").reset_index(drop=True)


__all__ = ["compare", "NOT_RECORDED"]
