"""Headline metrics. Every result is ``{value, n}``; blanks never count as disagree.

Percentages are fractions (0-1) of non-missing answers only. A question that
was not collected gives ``value=None, status="not collected"``; fewer than
``MIN_N`` answers gives ``value=None, status="too few responses"``.
Before/after numbers are a *self-reported retrospective change*, not impact.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from pipeline import load_schema, min_n_from

LIKERT_MAX: int = 5                 # ceiling = "before" answer at the top of the scale
BOOTSTRAP_RESAMPLES: int = 1000
BOOTSTRAP_SEED: int = 42
NOT_COLLECTED: str = "not collected"
TOO_FEW: str = "too few responses"


# ----- result helpers -------------------------------------------------

def _missing() -> dict[str, Any]:
    return {"value": None, "n": 0, "status": NOT_COLLECTED}


def _result(value: Any, n: int, min_n: int, **extra: Any) -> dict[str, Any]:
    """``{value, n}``; value hidden when n < min_n or value is undefined."""
    n = int(n)
    if n < min_n:
        return {"value": None, "n": n, "status": TOO_FEW}
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {"value": None, "n": n, "status": "cannot be calculated"}
    if isinstance(value, np.generic):
        value = value.item()
    return {"value": value, "n": n, **extra}


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Column as float with blanks dropped."""
    return pd.to_numeric(df[col], errors="coerce").astype(float).dropna()


def _both(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Rows where both columns are answered, as floats."""
    return df[[a, b]].apply(pd.to_numeric, errors="coerce").astype(float).dropna()


# ----- field lists from the schema -------------------------------------

def _fields(schema: dict[str, Any], kind: str) -> list[str]:
    return [f for f, r in (schema.get("fields", {}) or {}).items() if r.get("type") == kind]


def _pairs(schema: dict[str, Any]) -> dict[str, tuple[str, str]]:
    raw = (schema.get("analysis", {}) or {}).get("knowledge_pairs", {}) or {}
    return {k: (v[0], v[1]) for k, v in raw.items()}


# ----- metric groups -----------------------------------------------------

def _agree(df: pd.DataFrame, items: list[str], cut: int, min_n: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in items:
        if c not in df.columns:
            out[f"agree_pct.{c}"] = _missing()
            continue
        v = _num(df, c)
        out[f"agree_pct.{c}"] = _result(float(v.ge(cut).mean()) if len(v) else None, len(v), min_n)
    return out


def _wilcoxon_p(diff: pd.Series) -> float | None:
    """Two-sided Wilcoxon signed-rank p (zero differences dropped)."""
    if not diff.ne(0).any():
        return None
    return float(stats.wilcoxon(diff).pvalue)


def _gain_table(d: pd.DataFrame, pre: str, post: str, min_n: int) -> dict[int, Any]:
    """Mean change for each starting (before) level."""
    table: dict[int, Any] = {}
    for level, grp in d.groupby(pre):
        change = grp[post] - grp[pre]
        table[int(level)] = _result(float(change.mean()), len(grp), min_n)
    return table


def _pre_post(df: pd.DataFrame, schema: dict[str, Any], min_n: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, (pre, post) in _pairs(schema).items():
        names = ["mean_change", "wilcoxon_p", "declined_pct", "ceiling_pct", "gain_by_baseline"]
        if pre not in df.columns or post not in df.columns:
            out.update({f"{m}.{k}": _missing() for m in names})
            continue
        d = _both(df, pre, post)
        diff = d[post] - d[pre]
        n = len(d)
        out[f"mean_change.{k}"] = _result(float(diff.mean()) if n else None, n, min_n)
        out[f"wilcoxon_p.{k}"] = _result(_wilcoxon_p(diff) if n else None, n, min_n)
        out[f"declined_pct.{k}"] = _result(float(diff.lt(0).mean()) if n else None, n, min_n)
        pre_all = _num(df, pre)
        out[f"ceiling_pct.{k}"] = _result(
            float(pre_all.eq(LIKERT_MAX).mean()) if len(pre_all) else None, len(pre_all), min_n)
        out[f"gain_by_baseline.{k}"] = _result(_gain_table(d, pre, post, min_n), n, min_n)
    return out


def _nps_score(values: np.ndarray, promoter: int, detractor: int) -> np.ndarray:
    """NPS (-100..100) along the last axis."""
    return 100 * ((values >= promoter).mean(axis=-1) - (values <= detractor).mean(axis=-1))


def _nps(df: pd.DataFrame, a: dict[str, Any], min_n: int) -> dict[str, Any]:
    if "nps" not in df.columns:
        return {"nps": _missing()}
    v = _num(df, "nps").to_numpy()
    if len(v) < min_n:
        return {"nps": _result(None, len(v), min_n)}
    prom, det = a["nps_promoter_min"], a["nps_detractor_max"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = v[rng.integers(0, len(v), (BOOTSTRAP_RESAMPLES, len(v)))]
    lo, hi = np.percentile(_nps_score(boot, prom, det), [2.5, 97.5])
    return {"nps": _result(float(_nps_score(v, prom, det)), len(v), min_n,
                           ci95=[float(lo), float(hi)],
                           promoters_pct=float((v >= prom).mean()),
                           detractors_pct=float((v <= det).mean()))}


def _likely_call(df: pd.DataFrame, high: int, min_n: int) -> dict[str, Any]:
    keys = ["likely_call.mean", "likely_call.pct_high", "likely_call.coverage"]
    if "likely_call" not in df.columns:
        return {k: _missing() for k in keys}
    v = _num(df, "likely_call")
    n = len(v)
    return {
        "likely_call.mean": _result(float(v.mean()) if n else None, n, min_n),
        "likely_call.pct_high": _result(float(v.ge(high).mean()) if n else None, n, min_n),
        # coverage: share of ALL rows that answered (n = all rows)
        "likely_call.coverage": _result(n / len(df) if len(df) else None, len(df), 1,
                                        answered=n),
    }


def _know_call(df: pd.DataFrame, schema: dict[str, Any], a: dict[str, Any],
               min_n: int) -> dict[str, Any]:
    """Know when to use 988 (post_when agree) vs very likely to call (9-10)."""
    know = _pairs(schema).get("when", (None, "post_when"))[1]
    if know not in df.columns or "likely_call" not in df.columns:
        return {"know_call_gap": _missing(), "know_call_2x2": _missing()}
    d = _both(df, know, "likely_call")
    knows = d[know].ge(a["agree_threshold"])
    calls = d["likely_call"].ge(a["likely_call_high"])
    n = len(d)
    cells = {
        "knows_and_calls": int((knows & calls).sum()),
        "knows_not_calls": int((knows & ~calls).sum()),
        "not_knows_calls": int((~knows & calls).sum()),
        "not_knows_not_calls": int((~knows & ~calls).sum()),
    }
    return {
        "know_call_gap": _result(cells["knows_not_calls"] / n if n else None, n, min_n),
        "know_call_2x2": _result(cells, n, min_n),
    }


def _drivers(df: pd.DataFrame, items: list[str], min_n: int) -> dict[str, Any]:
    """Spearman correlation of each item with likely_call."""
    out: dict[str, Any] = {}
    for c in items:
        key = f"intent_driver.{c}"
        if c not in df.columns or "likely_call" not in df.columns:
            out[key] = _missing()
            continue
        d = _both(df, c, "likely_call")
        rho = d.corr("spearman").iloc[0, 1] if len(d) > 1 else None
        out[key] = _result(rho, len(d), min_n)
    return out


def _delivery(df: pd.DataFrame, items: list[str], cut: int, min_n: int) -> dict[str, Any]:
    """Share agreeing with ALL delivery items, among those who answered all."""
    if not items or any(c not in df.columns for c in items):
        return {"delivery_agree_pct": _missing()}
    d = df[items].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(d)
    return {"delivery_agree_pct": _result(
        float(d.ge(cut).all(axis=1).mean()) if n else None, n, min_n)}


def _comment_rates(df: pd.DataFrame, fields: list[str], min_n: int) -> dict[str, Any]:
    """Share of rows with any written comment, by month (blank = did not write)."""
    out: dict[str, Any] = {}
    for c in fields:
        key = f"comment_rate_by_month.{c}"
        if c not in df.columns or "month" not in df.columns:
            out[key] = _missing()
            continue
        wrote = df[c].notna()
        months = df["month"]
        table = {
            str(m): _result(float(wrote[months == m].mean()), int((months == m).sum()), min_n)
            for m in sorted(months.dropna().unique())
        }
        out[key] = _result(table, int(months.notna().sum()), 1)
    return out


# ----- main entry point -----------------------------------------------

def compute(
    df: pd.DataFrame, schema: dict[str, Any] | None = None, exclude_flagged: bool = False
) -> dict[str, dict[str, Any]]:
    """Return every headline metric as ``{name: {value, n, ...}}``.

    ``exclude_flagged=True`` drops rows with ``flag_any`` (run
    ``quality.add_flags`` first). The input is not modified.
    """
    schema = load_schema() if schema is None else schema
    a = schema.get("analysis", {}) or {}
    min_n = min_n_from(schema)
    data = df.copy()
    if exclude_flagged:
        if "flag_any" not in data.columns:
            raise ValueError("exclude_flagged needs flag_any; run quality.add_flags first")
        data = data[~data["flag_any"].astype(bool)]

    likert = _fields(schema, "likert")
    pairs = _pairs(schema)
    pre_items = {p for p, _ in pairs.values()}
    knowledge = {x for pair in pairs.values() for x in pair}
    delivery = [c for c in likert if c not in knowledge]
    driver_items = [c for c in likert if c not in pre_items]
    cut = a["agree_threshold"]

    out: dict[str, dict[str, Any]] = {"rows": {"value": len(data), "n": len(data)}}
    out.update(_agree(data, likert, cut, min_n))
    out.update(_pre_post(data, schema, min_n))
    out.update(_nps(data, a, min_n))
    out.update(_likely_call(data, a["likely_call_high"], min_n))
    out.update(_know_call(data, schema, a, min_n))
    out.update(_drivers(data, driver_items, min_n))
    out.update(_delivery(data, delivery, cut, min_n))
    out.update(_comment_rates(data, _fields(schema, "text"), min_n))
    return out


__all__ = ["compute", "NOT_COLLECTED", "TOO_FEW"]
