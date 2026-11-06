"""Insight checks A-J: yes/no questions about the data that become cards.

Each check returns a result (finding / no_finding / too_few / not_collected /
needs_themes) and named ``evidence``. Wording lives in
``config/decisions.yaml``; ``load_decisions`` is the only file read here and is
called once by the caller. ``run_checks`` is pure.

Inputs to ``run_checks``:
- ``metrics``: ``{"headline": compute(...), "all": compute(...),
  "unflagged": compute(..., exclude_flagged=True)}``
- ``segments``: ``{"session_id": compare(df, "session_id", metric), "metric": metric}``
- ``trends``: ``{"monthly": monthly(df, metric), "metric": metric, "alerts": alerts(df)}``
- ``themes``: ``None`` until M5, else ``{"low_raters": {"cutoff", "n", "silent"},
  "comments": [{"topic", "kind"}, ...]}``
"""
from __future__ import annotations

import copy
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from pipeline import MIN_N, ROOT, load_schema
from pipeline.metrics import ROW_METRICS
from pipeline.trends import BLANK_ALERT_MIN, COMMENT_DROP

DECISIONS_PATH: Path = ROOT / "config" / "decisions.yaml"
CHECK_METRIC: str = "agree_pct.post_when"   # measure used for checks A and H

# ----- thresholds (a check "finds something" at or beyond these) -----------
A_MIX_GAP_SHARE: float = 0.05     # raw vs adjusted change differ by >= 5 points
A_MIX_GAP_MEAN: float = 0.25      # same, for average-type measures
B_GAP_MIN: float = 0.20           # know-but-wouldn't-call share
C_MARGIN: float = 0.10            # delivery link stronger than knowledge link by this
C_TARGET_LIFT: float = 0.10       # success target: delivery agreement up 10 points
D_CEILING_MIN: float = 0.30       # share already at top score before training
F_SILENT_MIN: float = 0.50        # share of low raters with no comment
I_SHIFT: dict[str, float] = {     # headline change when flagged rows are left out
    "agree_pct.post_when": 0.03,
    "likely_call.pct_high": 0.03,
    "delivery_agree_pct": 0.03,
    "nps": 5.0,
}
I_FLAGGED_TARGET: float = 0.05    # success target: flagged share below 5%
J_QUESTION_MIN: float = 0.10      # share of comments that are open questions
J_TARGET_FRACTION: float = 0.5    # success target: halve the top topic
CONF_STRONG: int = 100
CONF_MODERATE: int = 50

LEVERS: tuple[str, ...] = (
    "curriculum_content", "presenter_coaching", "audience_targeting", "language_access",
    "scheduling_format", "survey_design", "funder_reporting",
)
CHECK_IDS: tuple[str, ...] = tuple("ABCDEFGHIJ")
CHART_KEYS: dict[str, str] = {
    "A": "trend_adjusted", "B": "know_call_2x2", "C": "intent_drivers",
    "D": "gain_by_baseline", "E": "data_health", "F": "keep_fix_add",
    "G": "trend_comments", "H": "session_funnel", "I": "headline", "J": "keep_fix_add",
}
NOT_ASSIGNED: str = "Not assigned"
FINDING, NO_FINDING, TOO_FEW, NOT_COLLECTED, NEEDS_THEMES = (
    "finding", "no_finding", "too_few", "not_collected", "needs_themes")

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


# ----- config ---------------------------------------------------------------

def load_decisions(path: str | Path = DECISIONS_PATH) -> dict[str, Any]:
    """Read decisions.yaml. The only file I/O in this module."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def validate_decisions(decisions: dict[str, Any]) -> list[str]:
    """Return a list of problems in the decision map (empty = fine)."""
    errors: list[str] = []
    checks = decisions.get("checks")
    if not isinstance(checks, dict):
        return ["decisions.yaml has no 'checks' section"]
    for cid in CHECK_IDS:
        if cid not in checks:
            errors.append(f"{cid}: no entry in decisions.yaml")
    for cid, entry in checks.items():
        if cid not in CHECK_IDS:
            errors.append(f"{cid}: not a known check ID")
            continue
        errors.extend(f"{cid}: {e}" for e in _entry_errors(entry))
    return errors


def _entry_errors(entry: Any) -> list[str]:
    if not isinstance(entry, dict):
        return ["entry is not a mapping"]
    errs = []
    if entry.get("lever") not in LEVERS:
        errs.append(f"lever {entry.get('lever')!r} is not one of {', '.join(LEVERS)}")
    sec = entry.get("secondary_lever")
    if sec is not None and sec not in LEVERS:
        errs.append(f"secondary_lever {sec!r} is not one of {', '.join(LEVERS)}")
    for key in ("decision", "owner_role"):
        if not str(entry.get(key) or "").strip():
            errs.append(f"{key} is missing")
    return errs


# ----- formatting and templates --------------------------------------------

def _pct(x: float) -> str:
    return f"{x:.0%}"


def _points(x: float) -> str:
    """Change in a share, as signed percentage points."""
    return f"{x * 100:+.0f} points"


def _fmt_value(x: float, kind: str) -> str:
    return _pct(x) if kind == "share" else f"{x:.2f}"


def _fmt_change(x: float, kind: str) -> str:
    return _points(x) if kind == "share" else f"{x:+.2f}"


def numbers_in(text: str) -> set[str]:
    """Every number written in ``text`` (signs and % ignored). Shared with summarize."""
    return set(_NUMBER.findall(text or ""))


def check_numbers(text: str, evidence: dict[str, Any], allowed: list[str] | None = None) -> list[str]:
    """Numbers in ``text`` that are not in ``evidence`` values or ``allowed``."""
    ok = set(allowed or [])
    for v in evidence.values():
        ok |= numbers_in(str(v))
    return sorted(numbers_in(text) - ok)


def render(template: Any, evidence: dict[str, Any],
           allowed: list[str] | None = None) -> tuple[str | None, list[str]]:
    """Fill ``{name}`` placeholders from evidence only.

    Returns ``(text, problems)``; text is ``None`` if any placeholder is
    missing or a number in the result is not in evidence.
    """
    if not isinstance(template, str) or not template.strip():
        return None, ["template missing"]
    names = {f for _, f, _, _ in string.Formatter().parse(template) if f is not None}
    missing = sorted(names - set(evidence))
    if missing:
        return None, [f"placeholder not in evidence: {', '.join(missing)}"]
    text = " ".join(template.format(**evidence).split())
    extra = check_numbers(text, evidence, allowed)
    if extra:
        return None, [f"number not in evidence: {', '.join(extra)}"]
    return text, []


def _confidence(n: int) -> str:
    return "strong" if n >= CONF_STRONG else "moderate" if n >= CONF_MODERATE else "thin"


def _priority(confidence: str, ratio: float) -> str:
    """ratio = how far past the threshold (1 = just at it)."""
    if confidence == "thin":
        return "low"
    return "high" if ratio >= 2 else "medium"


def _label(labels: dict[str, Any], group: str, key: str) -> str:
    return str(((labels or {}).get(group) or {}).get(key) or key)


# ----- the checks ------------------------------------------------------------
# Each returns {"result", "evidence", "n", "ratio"}.

def _status(m: dict[str, Any] | None) -> str | None:
    """too_few / not_collected for a metric with no value, else None."""
    if not m or m.get("value") is None:
        return TOO_FEW if m and m.get("n", 0) > 0 else NOT_COLLECTED
    return None


def _out(result: str, evidence: dict[str, Any] | None = None, n: int = 0,
         ratio: float = 1.0) -> dict[str, Any]:
    return {"result": result, "evidence": evidence or {}, "n": int(n), "ratio": ratio}


def _check_a(ctx: dict[str, Any]) -> dict[str, Any]:
    t = ctx["trends"].get("monthly")
    metric = ctx["trends"].get("metric", "agree_pct.post_when")
    if t is None or t.empty or t["adjusted"].isna().all() and t["raw"].isna().all():
        return _out(NOT_COLLECTED if t is None or t.empty else TOO_FEW)
    ok = t[~t["suppressed"]]
    if len(ok) < 2:
        return _out(TOO_FEW, n=int(t["n"].sum()))
    if ok["adjusted"].isna().all():
        return _out(NOT_COLLECTED)
    first, last = ok.iloc[0], ok.iloc[-1]
    kind = ROW_METRICS.get(metric, "share")
    raw_c, adj_c = float(last["raw"] - first["raw"]), float(last["adjusted"] - first["adjusted"])
    gap = A_MIX_GAP_SHARE if kind == "share" else A_MIX_GAP_MEAN
    n = int(first["n"] + last["n"])
    ev = {"metric_label": _label(ctx["labels"], "metrics", metric),
          "first_month": str(first["month"]), "last_month": str(last["month"]),
          "raw_change": _fmt_change(raw_c, kind), "adjusted_change": _fmt_change(adj_c, kind),
          "n_first": int(first["n"]), "n_last": int(last["n"]), "n": n,
          "target_gap": _points(gap).lstrip("+") if kind == "share" else f"{gap:.2f}"}
    diff = abs(raw_c - adj_c)
    return _out(FINDING if diff >= gap else NO_FINDING, ev, n, diff / gap)


def _check_b(ctx: dict[str, Any]) -> dict[str, Any]:
    gap, cells = ctx["m"].get("know_call_gap"), ctx["m"].get("know_call_2x2")
    if (s := _status(gap)) or (s := _status(cells)):
        return _out(s, n=(gap or {}).get("n", 0))
    a = ctx["analysis"]
    n = int(gap["n"])
    ev = {"gap_pct": _pct(gap["value"]), "gap_n": int(cells["value"]["knows_not_calls"]), "n": n,
          "call_cutoff": int(a["likely_call_high"]) - 1,
          "call_max": int(a["scale_0_10_valid_range"][1]),
          "target_pct": _pct(B_GAP_MIN), "min_n": MIN_N}
    found = gap["value"] >= B_GAP_MIN
    return _out(FINDING if found else NO_FINDING, ev, n, gap["value"] / B_GAP_MIN)


def _check_c(ctx: dict[str, Any]) -> dict[str, Any]:
    m, schema = ctx["m"], ctx["schema"]
    pairs = (schema.get("analysis", {}) or {}).get("knowledge_pairs", {}) or {}
    knowledge = {v[1] for v in pairs.values()}
    pre = {v[0] for v in pairs.values()}
    likert = [f for f, r in (schema.get("fields", {}) or {}).items() if r.get("type") == "likert"]
    delivery = [f for f in likert if f not in knowledge and f not in pre]
    drivers = {f: m.get(f"intent_driver.{f}") for f in sorted(knowledge) + delivery}
    if any(_status(d) == NOT_COLLECTED for d in drivers.values()) or not delivery or not knowledge:
        return _out(NOT_COLLECTED)
    if any(_status(d) for d in drivers.values()) or _status(m.get("delivery_agree_pct")):
        return _out(TOO_FEW, n=min(int((d or {}).get("n", 0)) for d in drivers.values()))
    k_link = max(drivers[f]["value"] for f in knowledge)
    top = max(delivery, key=lambda f: drivers[f]["value"])
    d_link = drivers[top]["value"]
    n = min(int(d["n"]) for d in drivers.values())
    agree = float(m["delivery_agree_pct"]["value"])
    ev = {"delivery_link": f"{d_link:.2f}", "knowledge_link": f"{k_link:.2f}", "n": n,
          "top_delivery_item": _label(ctx["labels"], "fields", top),
          "delivery_agree_pct": _pct(agree), "target_pct": _pct(min(1.0, agree + C_TARGET_LIFT))}
    margin = d_link - k_link
    return _out(FINDING if margin >= C_MARGIN else NO_FINDING, ev, n, max(margin, 0) / C_MARGIN)


def _check_d(ctx: dict[str, Any]) -> dict[str, Any]:
    pairs = (ctx["schema"].get("analysis", {}) or {}).get("knowledge_pairs", {}) or {}
    items = {k: ctx["m"].get(f"ceiling_pct.{k}") for k in pairs}
    usable = {k: v for k, v in items.items() if not _status(v)}
    if not usable:
        any_n = any(v and v.get("n", 0) for v in items.values())
        return _out(TOO_FEW if any_n else NOT_COLLECTED)
    k = max(usable, key=lambda x: usable[x]["value"])
    v = usable[k]
    ev = {"ceiling_pct": _pct(v["value"]), "n": int(v["n"]),
          "item_label": _label(ctx["labels"], "fields", pairs[k][0]),
          "target_pct": _pct(D_CEILING_MIN)}
    found = v["value"] >= D_CEILING_MIN
    return _out(FINDING if found else NO_FINDING, ev, int(v["n"]), v["value"] / D_CEILING_MIN)


def _judged_months(ctx: dict[str, Any]) -> pd.DataFrame:
    t = ctx["trends"].get("monthly")
    if t is None or t.empty:
        return pd.DataFrame(columns=["month", "rows"])
    return t[t["rows"] >= MIN_N]


def _check_e(ctx: dict[str, Any]) -> dict[str, Any]:
    months = _judged_months(ctx)
    if months.empty:
        return _out(TOO_FEW if ctx["trends"].get("monthly") is not None else NOT_COLLECTED)
    hits = [(a, h) for a in ctx["trends"].get("alerts", []) if a.get("id") == "E"
            for h in a.get("months", [])]
    total = int(months["rows"].sum())
    if not hits:
        return _out(NO_FINDING, {"n": total}, total)
    a, h = max(hits, key=lambda x: x[1]["blank_pct"])
    ev = {"month": h["month"], "blank_pct": _pct(h["blank_pct"]), "n": int(h["rows"]),
          "question_label": _label(ctx["labels"], "fields", a["field"]),
          "target_pct": _pct(BLANK_ALERT_MIN)}
    return _out(FINDING, ev, int(h["rows"]), h["blank_pct"] / BLANK_ALERT_MIN)


def _check_g(ctx: dict[str, Any]) -> dict[str, Any]:
    months = _judged_months(ctx)
    if len(months) < 2:
        return _out(TOO_FEW if ctx["trends"].get("monthly") is not None else NOT_COLLECTED)
    hits = [a for a in ctx["trends"].get("alerts", []) if a.get("id") == "G"]
    total = int(months["rows"].sum())
    if not hits:
        return _out(NO_FINDING, {"n": total}, total)
    a = max(hits, key=lambda x: x["earlier_rate"] - x["rate"])
    drop = a["earlier_rate"] - a["rate"]
    ev = {"field_label": _label(ctx["labels"], "fields", a["field"]),
          "late_month": a["month"], "late_rate": _pct(a["rate"]), "n_late": int(a["rows"]),
          "early_rate": _pct(a["earlier_rate"]), "n_early": int(a["earlier_rows"]),
          "target_pct": _pct(max(0.0, a["earlier_rate"] - COMMENT_DROP))}
    return _out(FINDING, ev, int(a["rows"]), drop / COMMENT_DROP)


def _check_h(ctx: dict[str, Any]) -> dict[str, Any]:
    t = ctx["segments"].get("session_id")
    metric = ctx["segments"].get("metric", "agree_pct.post_when")
    if t is None or t.empty or "outlier" not in t.columns:
        return _out(NOT_COLLECTED)
    shown = t[~t["suppressed"]]
    if shown.empty:
        return _out(TOO_FEW, n=int(t["metric_n"].max()))
    out = shown[shown["outlier"]]
    if out.empty:
        return _out(NO_FINDING, {"min_n": MIN_N}, int(shown["metric_n"].sum()))
    over = (out["value"] - out["upper998"]).clip(lower=0) + (out["lower998"] - out["value"]).clip(lower=0)
    r = out.loc[over.idxmax()]
    kind = ROW_METRICS.get(metric, "share")
    side = "above" if r["value"] > r["upper998"] else "below"
    group = str(r["group"])
    ev = {"session_label": group, "n": int(r["metric_n"]),
          "session_value": _fmt_value(float(r["value"]), kind),
          "centre": _fmt_value(float(r["centre"]), kind),
          "metric_label": _label(ctx["labels"], "metrics", metric),
          "limit_side": _label(ctx["labels"], "words", side),
          "county": group.split("_", 1)[1] if "_" in group else group}
    return _out(FINDING, ev, int(r["metric_n"]), 1.0)


def _check_i(ctx: dict[str, Any]) -> dict[str, Any]:
    full, kept = ctx["metrics"].get("all"), ctx["metrics"].get("unflagged")
    if not full or not kept:
        return _out(NOT_COLLECTED)
    rows, kept_rows = int(full["rows"]["value"]), int(kept["rows"]["value"])
    if rows < MIN_N:
        return _out(TOO_FEW, n=rows)
    best, ratio = None, 0.0
    for key, limit in I_SHIFT.items():
        a, b = full.get(key), kept.get(key)
        if _status(a) or _status(b):
            continue
        r = abs(b["value"] - a["value"]) / limit
        if best is None or r > ratio:
            best, ratio = key, r
    flagged = rows - kept_rows
    base = {"flagged_n": flagged, "n": rows}
    if best is None:
        return _out(TOO_FEW, n=rows)
    if ratio < 1:
        return _out(NO_FINDING, base, rows, ratio)
    kind = "share" if ROW_METRICS.get(best) == "share" else "mean"
    fmt = _pct if kind == "share" else (lambda x: f"{x:.0f}")
    ev = {**base, "flagged_pct": _pct(flagged / rows),
          "metric_label": _label(ctx["labels"], "metrics", best),
          "with_flags": fmt(full[best]["value"]), "without_flags": fmt(kept[best]["value"]),
          "target_pct": _pct(I_FLAGGED_TARGET)}
    return _out(FINDING, ev, rows, ratio)


def _check_f(ctx: dict[str, Any]) -> dict[str, Any]:
    themes = ctx["themes"]
    if themes is None:
        return _out(NEEDS_THEMES)
    low = themes.get("low_raters") or {}
    n, silent = int(low.get("n", 0)), int(low.get("silent", 0))
    if n < MIN_N:
        return _out(TOO_FEW, n=n)
    share = silent / n
    ev = {"silent_pct": _pct(share), "low_cutoff": int(low["cutoff"]), "low_n": n,
          "target_pct": _pct(F_SILENT_MIN)}
    return _out(FINDING if share >= F_SILENT_MIN else NO_FINDING, ev, n, share / F_SILENT_MIN)


def _check_j(ctx: dict[str, Any]) -> dict[str, Any]:
    themes = ctx["themes"]
    if themes is None:
        return _out(NEEDS_THEMES)
    items = [c for c in themes.get("comments", []) if c.get("kind") != "empty"]
    if len(items) < MIN_N:
        return _out(TOO_FEW, n=len(items))
    questions = [c for c in items if c.get("kind") == "open_question"]
    ev: dict[str, Any] = {"question_n": len(questions), "comment_n": len(items)}
    share = len(questions) / len(items)
    if share < J_QUESTION_MIN or not questions:
        return _out(NO_FINDING, ev, len(items), share / J_QUESTION_MIN)
    topic, count = Counter(c.get("topic", "Other") for c in questions).most_common(1)[0]
    ev.update({"top_topic": topic, "top_topic_n": count,
               "target_n": max(1, int(count * J_TARGET_FRACTION))})
    return _out(FINDING, ev, len(items), share / J_QUESTION_MIN)


CHECKS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "A": _check_a, "B": _check_b, "C": _check_c, "D": _check_d, "E": _check_e,
    "F": _check_f, "G": _check_g, "H": _check_h, "I": _check_i, "J": _check_j,
}


# ----- cards ------------------------------------------------------------------

def _fill(entry: dict[str, Any], section: str, key: str, ev: dict[str, Any],
          allowed: list[str], gaps: list[str]) -> str | None:
    text, problems = render(((entry.get(section) or {}).get(key)), ev, allowed)
    gaps.extend(f"{section}.{key}: {p}" for p in problems)
    return text


def _evidence_text(ev: dict[str, Any]) -> str:
    """Fallback card text when no wording is configured: the evidence itself."""
    return "; ".join(f"{k}: {v}" for k, v in ev.items())


def _build(cid: str, res: dict[str, Any], entry: dict[str, Any] | None,
           allowed: list[str]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    gaps: list[str] = []
    if entry is None:
        gaps.append("no entry in decisions.yaml")
        entry = {}
    else:
        gaps.extend(_entry_errors(entry))
    ev, n = res["evidence"], res["n"]
    question = str(entry.get("question") or f"Check {cid}")
    run = {"id": cid, "question": question, "result": res["result"], "text": None,
           "n": n, "evidence": ev, "config_gaps": gaps}
    if res["result"] == NO_FINDING:
        run["text"] = _fill(entry, "no_finding", "text", ev, allowed, gaps) if entry else None
        return None, run
    if res["result"] != FINDING:
        return None, run

    lever = entry.get("lever") if entry.get("lever") in LEVERS else None
    sec = entry.get("secondary_lever")
    owner = str(entry.get("owner_role") or "").strip() or NOT_ASSIGNED
    decision = " ".join(str(entry.get("decision") or "").split()) or NOT_ASSIGNED
    text = _fill(entry, "finding", "text", ev, allowed, gaps) if entry else None
    conf = _confidence(n)
    card = {
        "id": cid, "title": question, "text": text or _evidence_text(ev), "evidence": ev, "n": n,
        "confidence": conf, "priority": _priority(conf, res["ratio"]), "chart_key": CHART_KEYS[cid],
        "decision": decision, "lever": lever,
        "secondary_lever": sec if sec in LEVERS else None,
        "owner_role": owner, "confirmed": bool(entry.get("confirmed", False)) if entry else False,
        "action": _fill(entry, "finding", "action", ev, allowed, gaps) if entry else None,
        "success_measure": _fill(entry, "finding", "success_measure", ev, allowed, gaps)
        if entry else None,
    }
    run["text"] = card["text"]
    return card, run


def run_checks(
    metrics: dict[str, Any],
    segments: dict[str, Any],
    trends: dict[str, Any],
    themes: dict[str, Any] | None,
    decisions: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run checks A-J. Returns ``(cards, checks_run)``; cards sorted by priority.

    Pure: inputs are copied and never modified; no file I/O.
    """
    schema = load_schema() if schema is None else copy.deepcopy(schema)
    metrics, segments, trends = copy.deepcopy(metrics), copy.deepcopy(segments), copy.deepcopy(trends)
    themes, decisions = copy.deepcopy(themes), copy.deepcopy(decisions)
    ctx = {"metrics": metrics, "m": metrics.get("headline") or metrics.get("all") or {},
           "segments": segments or {}, "trends": trends or {}, "themes": themes,
           "schema": schema, "analysis": schema.get("analysis", {}) or {},
           "labels": decisions.get("labels") or {}}
    entries = decisions.get("checks") or {}
    allowed = [str(x) for x in decisions.get("allowed_literals") or []]
    cards, runs = [], []
    for cid, check in CHECKS.items():
        card, run = _build(cid, check(ctx), entries.get(cid), allowed)
        runs.append(run)
        if card:
            cards.append(card)
    order = {"high": 0, "medium": 1, "low": 2}
    cards.sort(key=lambda c: (order[c["priority"]], c["id"]))
    return cards, runs


__all__ = ["run_checks", "load_decisions", "validate_decisions", "render", "check_numbers",
           "numbers_in", "LEVERS", "CHECK_IDS", "NOT_ASSIGNED", "DECISIONS_PATH", "CHECK_METRIC"]
