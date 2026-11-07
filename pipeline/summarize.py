"""Program Brief (M6): one page that answers "is the training working, for whom,
what do we do next, and how will we know", built only from pipeline results.

Rule-based, no AI. All wording lives in ``config/decisions.yaml`` under
``brief:``; placeholders are filled only from evidence (``recommend.render``).
Every number written into the brief is recorded, so ``to_text`` can prove that
each number in the copy-ready text came from the data or the config.

- ``build_brief(...)`` -> dict of display-ready blocks for one audience
- ``to_text(brief)`` -> ``(markdown, numbers_ok, mismatches)``

Pure: inputs are copied, no file I/O, no Streamlit, no network.
"""
from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from pipeline import MIN_N
from pipeline.metrics import ROW_METRICS
from pipeline.recommend import CHECK_METRIC, numbers_in, render
from pipeline.segments import NOT_RECORDED

# ----- thresholds -----------------------------------------------------------
TOP_PRIORITIES: int = 3            # cards shown as top priorities; the rest go to the watch list
TREND_MIN: dict[str, float] = {    # latest month must differ from earlier months by this
    "share": 0.05, "score": 10.0, "mean": 0.25}
REACH_GAP: float = 0.05            # a group this far from overall is called out
COVERAGE_MIN: float = 0.50         # below this share answering "likely to call" -> measurement gap
QUOTE_MAX: int = 160               # characters of a quote before it is cut
TOPICS_PER_COLUMN: int = 2
TOO_FEW_LISTED: int = 5

# ----- metric keys (not wording) -------------------------------------------
KNOWLEDGE_BEFORE, KNOWLEDGE_AFTER = "agree_pct.pre_when", "agree_pct.post_when"
INTENT, INTENT_COVERAGE = "likely_call.pct_high", "likely_call.coverage"
MEASUREMENT_LEVER: str = "survey_design"
OTHER_TOPIC: str = "Other"

AUDIENCES: tuple[str, ...] = ("internal", "funder")
BLOCKS: dict[str, tuple[str, ...]] = {
    "internal": ("bottom_line", "scorecard", "priorities", "watch", "reach", "comments",
                 "measurement", "limits"),
    "funder": ("bottom_line", "scorecard", "reach", "limits"),
}
_CONF = {"strong": 0, "moderate": 1, "thin": 2}
_PRIO = {"high": 0, "medium": 1, "low": 2}


# ----- formatting -------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{x:.0%}"


def _fmt(x: float, kind: str) -> str:
    """Value as shown: shares as %, scores signed whole numbers, means 2 decimals."""
    if kind == "share":
        return _pct(x)
    return f"{x:+.0f}" if kind == "score" else f"{x:.2f}"


def _kind(key: str, tile_format: str | None = None) -> str:
    if tile_format:
        return str(tile_format)
    return "share" if ROW_METRICS.get(key, "share") == "share" else "mean"


class _Writer:
    """Fills templates and remembers every number written, for ``to_text``."""

    def __init__(self, cfg: dict[str, Any], allowed: list[str]) -> None:
        self.cfg, self.allowed = cfg, allowed
        self.numbers: set[str] = set(allowed)
        self.problems: list[str] = []

    def static(self, text: Any) -> str:
        """Config wording used as is (headings, labels)."""
        s = " ".join(str(text or "").split())
        self.numbers |= numbers_in(s)
        return s

    def data(self, text: Any) -> str:
        """Text that is itself data (a card text already checked, a quote)."""
        s = " ".join(str(text or "").split())
        self.numbers |= numbers_in(s)
        return s

    def fill(self, path: str, ev: dict[str, Any]) -> str | None:
        """Render ``brief.<path>`` with ``ev``; None (and a problem) if it can't."""
        template: Any = self.cfg
        for part in path.split("."):
            template = template.get(part) if isinstance(template, dict) else None
        text, problems = render(template, ev, self.allowed)
        if problems:
            self.problems.extend(f"brief.{path}: {p}" for p in problems)
            return None
        self.numbers |= numbers_in(text)
        return text


# ----- blocks -----------------------------------------------------------------

def _status_text(m: dict[str, Any] | None, w: _Writer) -> str:
    tile = w.cfg.get("tile") or {}
    if m and m.get("n", 0) > 0:
        return w.static(tile.get("too_few"))
    return w.static(tile.get("not_collected"))


def _trend(table: pd.DataFrame | None, kind: str,
           w: _Writer) -> tuple[str | None, str, str | None]:
    """(direction, text, month) comparing the latest judged month with the months before it."""
    tile = w.cfg.get("tile") or {}
    if table is None or len(table) == 0:
        return None, w.static(tile.get("trend_none")), None
    ok = table[~table["suppressed"].astype(bool)].dropna(subset=["raw"])
    if len(ok) < 2:
        return None, w.static(tile.get("trend_none")), None
    last, earlier = ok.iloc[-1], ok.iloc[:-1]
    before = float((earlier["raw"] * earlier["n"]).sum() / earlier["n"].sum())
    diff = float(last["raw"]) - before
    step = TREND_MIN.get(kind, TREND_MIN["mean"])
    if kind == "share":
        change = w.fill("tile.points", {"x": f"{abs(diff) * 100:.0f}"}) or ""
    else:
        change = f"{abs(diff):.0f}" if kind == "score" else f"{abs(diff):.2f}"
    ev = {"change": change, "last_month": str(last["month"])}
    if abs(diff) < step:
        return "steady", w.fill("tile.trend_steady", ev) or "", ev["last_month"]
    direction = "up" if diff > 0 else "down"
    return direction, w.fill(f"tile.trend_{direction}", ev) or "", ev["last_month"]


def _scorecard(metrics: dict[str, Any], trends: dict[str, pd.DataFrame],
               targets: dict[str, Any], w: _Writer) -> list[dict[str, Any]]:
    tiles = []
    for spec in w.cfg.get("scorecard") or []:
        after_key = spec.get("after")
        kind = _kind(after_key, spec.get("format"))
        m = metrics.get(after_key) or {}
        tile: dict[str, Any] = {"label": w.static(spec.get("label")), "metric": after_key,
                                "value": None, "n": int(m.get("n") or 0), "notes": [],
                                "n_text": None, "trend": None, "trend_text": None,
                                "trend_month": None}
        if m.get("value") is None:
            tile["value_text"] = _status_text(m, w)
            tiles.append(tile)
            continue
        tile["value"] = float(m["value"])
        tile["value_text"] = w.data(_fmt(tile["value"], kind))
        tile["n_text"] = w.fill("n_label", {"n": tile["n"]})
        b = metrics.get(spec.get("before") or "") or {}
        if b.get("value") is not None:
            note = w.fill("tile.before_after", {"before": _fmt(b["value"], kind),
                                                "after": tile["value_text"]})
            tile["notes"].append(note)
        c = metrics.get(spec.get("coverage") or "") or {}
        if c.get("value") is not None:
            tile["notes"].append(w.fill("tile.coverage", {"coverage": _pct(c["value"])}))
        if m.get("ci95"):
            lo, hi = m["ci95"]
            tile["notes"].append(w.fill("tile.score_range", {"low": _fmt(lo, kind),
                                                             "high": _fmt(hi, kind)}))
        target = targets.get(after_key)
        if target is not None:
            met = tile["value"] >= float(target)
            tile["notes"].append(w.fill("tile.target_met" if met else "tile.target_missed",
                                        {"target": _fmt(float(target), kind)}))
        if spec.get("trend"):
            tile["trend"], tile["trend_text"], tile["trend_month"] = _trend(
                trends.get(spec["trend"]), kind, w)
        tile["notes"] = [x for x in tile["notes"] if x]
        tiles.append(tile)
    return tiles


def _card_view(c: dict[str, Any], w: _Writer) -> dict[str, Any]:
    """The parts of an insight card the brief shows (text already checked by recommend)."""
    w.numbers |= {str(c.get("n"))}
    for v in (c.get("evidence") or {}).values():
        w.numbers |= numbers_in(str(v))
    n_label = w.fill("n_label", {"n": c.get("n")}) or ""
    return {
        "id": c["id"], "title": w.data(c.get("title")), "text": w.data(c.get("text")),
        "n": c.get("n"), "confidence": c.get("confidence"), "priority": c.get("priority"),
        "lever": c.get("lever"), "owner_role": w.data(c.get("owner_role")),
        "evidence_text": w.fill("card.evidence", {"n_label": n_label,
                                                  "confidence": c.get("confidence")}),
        "action": w.data(c.get("action")) or None,
        "success_measure": w.data(c.get("success_measure")) or None,
    }


def _rank(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(cards, key=lambda c: (_PRIO.get(c.get("priority"), 3),
                                        _CONF.get(c.get("confidence"), 3),
                                        -int(c.get("n") or 0), c["id"]))


def _split(cards: list[dict[str, Any]]) -> tuple[list, list, list]:
    """(top priorities, watch list, measurement cards)."""
    measure = [c for c in cards if c.get("lever") == MEASUREMENT_LEVER]
    rest = _rank([c for c in cards if c.get("lever") != MEASUREMENT_LEVER])
    solid = [c for c in rest if c.get("confidence") != "thin"]
    top = solid[:TOP_PRIORITIES]
    watch = [c for c in rest if c not in top]
    return top, watch, _rank(measure)


def _declines(tiles: list[dict[str, Any]], w: _Writer) -> str | None:
    """One sentence naming the goals that fell in the latest month, if any."""
    down = [t for t in tiles if t.get("trend") == "down"]
    if not down:
        return None
    month = down[0]["trend_month"]
    names = [f'"{t["label"]}"' for t in down]
    items = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return w.fill("bottom_line.declines", {"month": month, "items": items})


def _bottom_line(metrics: dict[str, Any], top: list[dict[str, Any]], watch_n: int,
                 tiles: list[dict[str, Any]], labels: dict[str, Any], w: _Writer) -> list[str]:
    out = []
    b, a = metrics.get(KNOWLEDGE_BEFORE) or {}, metrics.get(KNOWLEDGE_AFTER) or {}
    if b.get("value") is not None and a.get("value") is not None:
        label = (labels.get("fields") or {}).get("post_when", "know when to use 988")
        out.append(w.fill("bottom_line.knowledge", {
            "label": label, "before": _pct(b["value"]), "after": _pct(a["value"]),
            "n": min(int(b["n"]), int(a["n"]))}))
    i, cov = metrics.get(INTENT) or {}, metrics.get(INTENT_COVERAGE) or {}
    if i.get("value") is not None:
        low = cov.get("value") is not None and cov["value"] < COVERAGE_MIN
        out.append(w.fill("bottom_line.intent_low_coverage" if low else "bottom_line.intent", {
            "value": _pct(i["value"]), "n": i["n"],
            "coverage": _pct(cov["value"]) if cov.get("value") is not None else ""}))
    if top:
        c = top[0]
        out.append(w.fill("bottom_line.gap", {"text": c["text"], **_ev_numbers(c)}))
        if c.get("action"):
            out.append(w.fill("bottom_line.action", {"owner": c["owner_role"],
                                                     "action": c["action"], **_ev_numbers(c)}))
    elif watch_n:
        out.append(w.fill("bottom_line.none_strong", {"watch_n": watch_n}))
    else:
        out.append(w.static((w.cfg.get("bottom_line") or {}).get("none")))
    out.append(_declines(tiles, w))
    return [x for x in out if x]


def _ev_numbers(c: dict[str, Any]) -> dict[str, Any]:
    """Card evidence passed alongside its text so ``render`` accepts its numbers."""
    return {f"_ev_{k}": v for k, v in {**(c.get("evidence") or {}), "n": c.get("n")}.items()}


def _reach(reach: dict[str, pd.DataFrame], metrics: dict[str, Any],
           w: _Writer) -> list[dict[str, Any]]:
    cfg = w.cfg.get("reach") or {}
    overall_m = metrics.get(CHECK_METRIC) or {}
    overall = overall_m.get("value")
    out = []
    for dim in cfg.get("dims") or []:
        label = w.static(dim.get("label"))
        table = reach.get(dim.get("by"))
        if table is None or len(table) == 0:
            out.append({"label": label,
                        "lines": [w.fill("reach.not_collected", {"label": label})]})
            continue
        total = int(table["n"].sum())
        known = table[table["group"] != NOT_RECORDED]
        lines = []
        if len(known):
            big = known.sort_values("n", ascending=False).iloc[0]
            lines.append(w.fill("reach.largest", {"group": big["group"], "share": _pct(big["n"] / total),
                                                  "n": int(big["n"]), "total": total}))
        judged = known[~known["suppressed"].astype(bool)].dropna(subset=["value"])
        if overall is not None and len(judged):
            far = judged.assign(_d=(judged["value"] - overall).abs()).sort_values(
                ["_d", "n"], ascending=[False, False]).iloc[0]
            if far["_d"] >= REACH_GAP:
                lines.append(w.fill("reach.gap", {
                    "group": far["group"],
                    "value": _pct(far["value"]), "overall": _pct(overall), "n": int(far["metric_n"])}))
            else:
                lines.append(w.fill("reach.no_gap", {"overall": _pct(overall),
                                                     "threshold": f"{REACH_GAP * 100:.0f} points"}))
        small = known[known["suppressed"].astype(bool)].sort_values("n", ascending=False)
        if len(small):
            names = [f"{g} ({n})" for g, n in zip(small["group"], small["n"])]
            extra = len(names) - TOO_FEW_LISTED
            groups = ", ".join(names[:TOO_FEW_LISTED]) + (f" and {extra} more" if extra > 0 else "")
            lines.append(w.fill("reach.too_few", {"groups": groups}))
        missing = table[table["group"] == NOT_RECORDED]
        if len(missing) and int(missing["n"].iloc[0]) > 0:
            lines.append(w.fill("reach.not_recorded", {"n": int(missing["n"].iloc[0]),
                                                       "label": label.lower()}))
        out.append({"label": label, "lines": [x for x in lines if x]})
    return out


def _quote(text: str) -> str:
    s = " ".join(str(text).split())
    if len(s) <= QUOTE_MAX:
        return s
    return s[:QUOTE_MAX].rsplit(" ", 1)[0].rstrip(",.;:") + "…"


def _comments(board: dict[str, list[dict[str, Any]]] | None, w: _Writer) -> dict[str, Any]:
    cfg = w.cfg.get("comments") or {}
    items = [x for col in (board or {}).values() for x in col]
    total = sum(int(x["n"]) for x in items)
    if not total:
        return {"coverage": None, "columns": [], "none": w.static(cfg.get("none"))}
    sorted_n = sum(int(x["n"]) for x in items if x["topic"] != OTHER_TOPIC)
    cols = []
    for name, topics in (board or {}).items():
        shown = [t for t in topics if t["topic"] != OTHER_TOPIC][:TOPICS_PER_COLUMN]
        lines = []
        for t in shown:
            quote = _quote(t["comments"][0]["comment"]) if t.get("comments") else ""
            w.data(quote)
            lines.append({"topic": t["topic"], "n": int(t["n"]), "quote": quote,
                          "text": w.fill("comments.topic", {"topic": t["topic"], "n": int(t["n"]),
                                                            "quote": quote})})
        cols.append({"name": name,
                     "heading": w.static((cfg.get("columns") or {}).get(name, name)),
                     "topics": lines,
                     "none": None if lines else w.static(cfg.get("column_none"))})
    return {"coverage": w.fill("comments.coverage", {
                "sorted_n": sorted_n, "total_n": total, "coverage": _pct(sorted_n / total)}),
            "columns": cols, "none": None}


def _measurement(metrics: dict[str, Any], cards: list[dict[str, Any]],
                 w: _Writer) -> list[dict[str, Any]]:
    out = []
    cov = metrics.get(INTENT_COVERAGE) or {}
    if cov.get("value") is not None and cov["value"] < COVERAGE_MIN:
        rows = int(cov.get("n") or 0)
        answered = round(cov["value"] * rows)
        out.append({"id": "coverage",
                    "text": w.fill("measurement.coverage", {"coverage": _pct(cov["value"]),
                                                            "n": answered, "rows": rows}),
                    "action": w.static((w.cfg.get("measurement") or {}).get("coverage_action")),
                    "evidence_text": None})
    for c in cards:
        v = _card_view(c, w)
        out.append({"id": v["id"], "text": v["text"], "action": v["action"],
                    "evidence_text": v["evidence_text"]})
    return out


def _intro(metrics: dict[str, Any], trends: dict[str, pd.DataFrame],
           excluded_flagged: bool, w: _Writer) -> str | None:
    rows = (metrics.get("rows") or {}).get("value")
    states = w.cfg.get("flag_states") or {}
    state = states.get("excluded" if excluded_flagged else "included", "")
    months = sorted({str(m) for t in trends.values() if t is not None and len(t)
                     for m in t["month"].dropna()})
    if months:
        return w.fill("intro", {"rows": rows, "first_month": months[0],
                                "last_month": months[-1], "flag_state": state})
    return w.fill("intro_no_months", {"rows": rows, "flag_state": state})


# ----- public -------------------------------------------------------------------

def build_brief(
    metrics: dict[str, Any],
    cards: list[dict[str, Any]],
    reach: dict[str, pd.DataFrame] | None,
    trends: dict[str, pd.DataFrame] | None,
    board: dict[str, list[dict[str, Any]]] | None,
    decisions: dict[str, Any],
    audience: str = "internal",
    excluded_flagged: bool = False,
) -> dict[str, Any]:
    """Build the Program Brief for ``audience`` ("internal" or "funder").

    ``metrics``: ``metrics.compute`` output. ``cards``: ``recommend.run_checks``
    cards. ``reach``: ``{by: segments.compare(df, by, CHECK_METRIC)}``.
    ``trends``: ``{metric: trends.monthly(df, metric)}``. ``board``:
    ``themes.board`` output. Returns display-ready strings; ``problems`` lists
    config wording that could not be filled.
    """
    if audience not in AUDIENCES:
        raise ValueError(f"audience must be one of {AUDIENCES}, got {audience!r}")
    decisions = copy.deepcopy(decisions or {})
    cfg = decisions.get("brief")
    if not isinstance(cfg, dict):
        raise KeyError("config/decisions.yaml is missing the 'brief:' block")
    metrics, cards = copy.deepcopy(metrics or {}), copy.deepcopy(cards or [])
    reach = {k: v.copy() for k, v in (reach or {}).items() if v is not None}
    trends = {k: v.copy() for k, v in (trends or {}).items() if v is not None}
    board = copy.deepcopy(board)

    w = _Writer(cfg, [str(x) for x in decisions.get("allowed_literals") or []])
    if audience == "funder":
        hide_lev = set(cfg.get("funder_hide_levers") or [])
        hide_ids = {str(x) for x in cfg.get("funder_hide_checks") or []}
        cards = [c for c in cards if c.get("lever") not in hide_lev and str(c["id"]) not in hide_ids]
    top, watch, measure = _split(cards)
    top_v, watch_v = [_card_view(c, w) for c in top], [_card_view(c, w) for c in watch]

    blocks = BLOCKS[audience]
    brief: dict[str, Any] = {
        "audience": audience,
        "audience_label": w.static((cfg.get("audiences") or {}).get(audience, audience)),
        "title": w.static(cfg.get("title")),
        "intro": _intro(metrics, trends, excluded_flagged, w),
        "blocks": list(blocks),
        "headings": {b: w.static((cfg.get("headings") or {}).get(b, b)) for b in blocks},
        "blurbs": {b: (w.fill(f"blurbs.{b}", {"metric_label": (cfg.get("reach") or {}).get(
                       "metric_label", CHECK_METRIC)}) if (cfg.get("blurbs") or {}).get(b) else "")
                   for b in blocks},
        "labels": {k: w.static(v) for k, v in (cfg.get("card") or {}).items()
                   if k.endswith("_label")},
        "scorecard": _scorecard(metrics, trends, cfg.get("targets") or {}, w),
        "priorities": top_v if "priorities" in blocks else [],
        "watch": watch_v if "watch" in blocks else [],
        "reach": _reach(reach, metrics, w) if "reach" in blocks else [],
        "comments": _comments(board, w) if "comments" in blocks else None,
        "measurement": _measurement(metrics, measure, w) if "measurement" in blocks else [],
        "measurement_none": w.static((cfg.get("measurement") or {}).get("none")),
        "limits": [x for x in (_limit(t, w) for t in cfg.get("limits") or []) if x],
    }
    brief["bottom_line"] = _bottom_line(metrics, top_v, len(brief["watch"]), brief["scorecard"],
                                        decisions.get("labels") or {}, w)
    brief["numbers"] = sorted(w.numbers)
    brief["problems"] = list(dict.fromkeys(w.problems))
    return brief


def _limit(template: Any, w: _Writer) -> str | None:
    text, problems = render(template, {"min_n": MIN_N}, w.allowed)
    if problems:
        w.problems.extend(f"brief.limits: {p}" for p in problems)
        return None
    w.numbers |= numbers_in(text)
    return text


def _card_md(c: dict[str, Any], labels: dict[str, str]) -> list[str]:
    lines = [f"### {c['title']}", "", f"{c['text']} ({c['evidence_text']})", ""]
    for key, label in (("action", "action_label"), ("owner_role", "owner_label"),
                       ("success_measure", "success_label")):
        if c.get(key):
            lines.append(f"- {labels.get(label, '')} {c[key]}")
    return lines


def to_text(brief: dict[str, Any]) -> tuple[str, bool, list[str]]:
    """Markdown for copying or download. Returns ``(text, numbers_ok, mismatches)``;
    ``mismatches`` lists any number in the text that did not come from the data."""
    h, labels = brief["headings"], brief.get("labels") or {}
    out = [f"# {brief['title']}", ""]
    if brief.get("intro"):
        out += [brief["intro"], ""]
    for block in brief["blocks"]:
        body: list[str] = []
        if block == "bottom_line":
            body = [" ".join(brief["bottom_line"])]
        elif block == "scorecard":
            for t in brief["scorecard"]:
                bits = [t["value_text"]] + ([t["n_text"]] if t.get("n_text") else [])
                bits += t["notes"] + ([t["trend_text"]] if t.get("trend_text") else [])
                body.append(f"- {t['label']}: " + "; ".join(bits))
        elif block in ("priorities", "watch"):
            for c in brief[block]:
                body += _card_md(c, labels) + [""]
            body = body[:-1]
        elif block == "reach":
            for r in brief["reach"]:
                body += [f"- {r['label']}: " + " ".join(r["lines"])]
        elif block == "comments":
            cm = brief["comments"] or {}
            if cm.get("none"):
                body = [cm["none"]]
            else:
                body = [cm.get("coverage") or "", ""]
                for col in cm.get("columns", []):
                    body.append(f"- {col['heading']}")
                    body += [f"  - {t['text']}" for t in col["topics"]] or [f"  - {col['none']}"]
        elif block == "measurement":
            items = brief["measurement"]
            body = [f"- {m['text']} {m.get('action') or ''}".rstrip() for m in items] \
                or [brief["measurement_none"]]
        elif block == "limits":
            body = [f"- {x}" for x in brief["limits"]]
        body = [b for b in body if b is not None]
        if not body:
            continue
        out += [f"## {h[block]}", ""] + body + [""]
    text = "\n".join(out).rstrip() + "\n"
    stray = sorted(numbers_in(text) - set(brief.get("numbers") or []))
    return text, not stray, stray


__all__ = ["build_brief", "to_text", "AUDIENCES", "BLOCKS"]
