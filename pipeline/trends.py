"""Monthly trends, adjusted for audience mix, plus data-health alerts.

Mix adjustment (indirect standardization): each response gets the usual score
of its group (e.g. its role) across all months. A month's *expected* score is
the average of those usual scores for the people who attended that month.
``adjusted = overall + (raw - expected)``. This works even when a month has
only one role, which is common in this data.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline import load_schema, min_n_from
from pipeline.metrics import ROW_METRICS, row_values

# ----- alert thresholds ---------------------------------------------------
BLANK_ALERT_MIN: float = 0.25        # month's blank share must be at least this...
BLANK_ALERT_JUMP: float = 0.20       # ...and this much above the other months
COMMENT_DROP: float = 0.15           # latest month's comment rate this far below earlier months

MONTHLY_COLUMNS: list[str] = ["month", "rows", "n", "raw", "expected", "adjusted", "suppressed"]


def monthly(
    df: pd.DataFrame,
    metric: str,
    adjust_by: str = "role_primary",
    schema: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return ``month, rows, n, raw, expected, adjusted, suppressed``, one row per month.

    ``n`` = answers used. Months with n < MIN_N are suppressed (values blank).
    Groups of ``adjust_by`` with fewer than MIN_N answers overall, or blank,
    get the overall score (they neither raise nor lower the expectation).
    ``adjusted`` is blank for every month if ``adjust_by`` was not collected.
    """
    schema = load_schema() if schema is None else schema
    if metric not in ROW_METRICS:
        raise ValueError(f"Unknown metric {metric!r}; options are {list(ROW_METRICS)}")
    scores_all = row_values(df, schema)
    if "month" not in df.columns or metric not in scores_all.columns:
        return pd.DataFrame(columns=MONTHLY_COLUMNS)

    min_n = min_n_from(schema)
    has_month = df["month"].notna()
    d = pd.DataFrame({"month": df["month"], "y": scores_all[metric]})[has_month]
    answered = d["y"].notna()
    overall = float(d.loc[answered, "y"].mean()) if answered.any() else float("nan")

    if adjust_by in df.columns:
        grp = df.loc[has_month, adjust_by]
        stats = d[answered].groupby(grp[answered])["y"].agg(["mean", "count"])
        usual = stats["mean"].where(stats["count"] >= min_n)
        d["usual"] = grp.map(usual).astype(float).fillna(overall)
    else:
        d["usual"] = float("nan")

    used = d[answered]
    out = pd.DataFrame({
        "rows": d.groupby("month").size(),
        "n": used.groupby("month").size(),
        "raw": used.groupby("month")["y"].mean(),
        "expected": used.groupby("month")["usual"].mean(),
    }).rename_axis("month").reset_index().sort_values("month")
    out["n"] = out["n"].fillna(0).astype(int)
    out["adjusted"] = overall + out["raw"] - out["expected"]
    if ROW_METRICS[metric] == "share":
        out["adjusted"] = out["adjusted"].clip(0.0, 1.0)
    out["suppressed"] = out["n"] < min_n
    for c in ("raw", "expected", "adjusted"):
        out[c] = out[c].astype(float).where(~out["suppressed"])
    return out[MONTHLY_COLUMNS].reset_index(drop=True)


# ----- alerts ---------------------------------------------------------------

def _question_fields(schema: dict[str, Any]) -> list[str]:
    kinds = {"likert", "scale_0_10"}
    return [f for f, r in (schema.get("fields", {}) or {}).items() if r.get("type") in kinds]


def _blank_alerts(df: pd.DataFrame, schema: dict[str, Any], min_n: int) -> list[dict[str, Any]]:
    """Questions left blank far more often in some months than in others."""
    alerts = []
    months = df["month"]
    sizes = months.value_counts()
    big = sorted(sizes[sizes >= min_n].index)
    for field in _question_fields(schema):
        if field not in df.columns:
            continue
        blank = df[field].isna()
        hits = []
        for m in big:
            in_m = months == m
            rest = months.notna() & ~in_m
            share, other = float(blank[in_m].mean()), float(blank[rest].mean()) if rest.any() else 0.0
            if share >= BLANK_ALERT_MIN and share - other >= BLANK_ALERT_JUMP:
                hits.append({"month": str(m), "blank_pct": share, "rows": int(in_m.sum()),
                             "other_months_blank_pct": other})
        if hits:
            names = ", ".join(h["month"] for h in hits)
            alerts.append({
                "id": "E", "kind": "blank_answers", "field": field, "months": hits,
                "n": sum(h["rows"] for h in hits),
                "text": f"{field} was blank (skipped or not asked) much more often in {names}. "
                        "Scores for it in those months rest on fewer answers; blanks are not "
                        "counted as disagree.",
            })
    return alerts


def _comment_alerts(df: pd.DataFrame, schema: dict[str, Any], min_n: int) -> list[dict[str, Any]]:
    """Latest month's comment rate well below the months before it."""
    alerts = []
    sizes = df["month"].value_counts()
    big = sorted(sizes[sizes >= min_n].index)
    if len(big) < 2:
        return alerts
    latest = big[-1]
    in_latest = df["month"] == latest
    earlier = df["month"].isin(big[:-1])
    for field, rule in (schema.get("fields", {}) or {}).items():
        if rule.get("type") != "text" or field not in df.columns:
            continue
        wrote = df[field].notna()
        now, before = float(wrote[in_latest].mean()), float(wrote[earlier].mean())
        if before - now >= COMMENT_DROP:
            alerts.append({
                "id": "G", "kind": "fewer_comments", "field": field, "month": str(latest),
                "rate": now, "rows": int(in_latest.sum()),
                "earlier_rate": before, "earlier_rows": int(earlier.sum()),
                "n": int(in_latest.sum()),
                "text": f"Fewer people wrote a {field} comment in {latest}: {now:.0%} of "
                        f"{int(in_latest.sum())} vs {before:.0%} of {int(earlier.sum())} "
                        "in earlier months.",
            })
    return alerts


def alerts(df: pd.DataFrame, schema: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Data-health alerts: unusually blank questions (check E) and falling
    comment rates (check G). Only months with at least MIN_N rows are judged."""
    schema = load_schema() if schema is None else schema
    if "month" not in df.columns:
        return []
    min_n = min_n_from(schema)
    return _blank_alerts(df, schema, min_n) + _comment_alerts(df, schema, min_n)


__all__ = ["monthly", "alerts"]
