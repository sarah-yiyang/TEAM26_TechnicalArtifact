"""Tests for pipeline.summarize (M6 Program Brief). Rule-based; no network, no AI."""
from __future__ import annotations

import ast
import copy
from typing import Any

import pandas as pd
import pytest

from pipeline import MIN_N, ROOT
from pipeline.cleaning import clean
from pipeline.ingest import load_files
from pipeline.mapping import map_columns
from pipeline.metrics import compute, drop_flagged
from pipeline.quality import add_flags
from pipeline.recommend import CHECK_METRIC, load_decisions, run_checks
from pipeline.segments import compare
from pipeline.summarize import QUOTE_MAX, build_brief, to_text
from pipeline.themes import board, check_input, classify_df
from pipeline.trends import alerts, monthly


# ----- helpers ------------------------------------------------------------------

@pytest.fixture(scope="module")
def decisions() -> dict[str, Any]:
    return load_decisions()


def card(cid: str, priority: str = "high", confidence: str = "strong", n: int = 150,
         lever: str = "curriculum_content", **extra: Any) -> dict[str, Any]:
    c = {"id": cid, "title": f"Question {cid}?",
         "text": f"{n} people show gap {cid} at 34%.",
         "evidence": {"gap_pct": "34%", "threshold": "20%", "min_n": MIN_N}, "n": n,
         "confidence": confidence, "priority": priority, "chart_key": "x",
         "decision": "Decide?", "lever": lever, "owner_role": "Curriculum",
         "action": "Consider a short call demo.", "success_measure": "Gap falls below 20%."}
    c.update(extra)
    return c


def m(value: float | None, n: int, **extra: Any) -> dict[str, Any]:
    return {"value": value, "n": n, **extra}


METRICS: dict[str, Any] = {
    "rows": m(400, 400),
    "agree_pct.pre_identify": m(0.70, 400), "agree_pct.post_identify": m(0.90, 400),
    "agree_pct.pre_describe": m(0.40, 400), "agree_pct.post_describe": m(0.88, 400),
    "agree_pct.pre_when": m(0.55, 400), "agree_pct.post_when": m(0.92, 400),
    "likely_call.pct_high": m(0.47, 300), "likely_call.coverage": m(0.75, 400),
    "nps": m(12.0, 380, ci95=[7.6, 16.7]),
    "delivery_agree_pct": m(0.85, 390),
}


def month_table(raws: list[float], ns: list[int] | None = None) -> pd.DataFrame:
    ns = ns or [100] * len(raws)
    return pd.DataFrame({"month": [f"2026-{i + 1:02d}" for i in range(len(raws))],
                         "rows": ns, "n": ns, "raw": raws, "expected": raws, "adjusted": raws,
                         "suppressed": [x < MIN_N for x in ns]})


def group_table(rows: list[tuple[str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"group": g, "n": n, "metric_n": n,
                          "value": None if n < MIN_N else v, "suppressed": n < MIN_N}
                         for g, n, v in rows])


def brief(cards: list[dict[str, Any]] | None = None, metrics: dict[str, Any] | None = None,
          decisions: dict[str, Any] | None = None, **kw: Any) -> dict[str, Any]:
    return build_brief(metrics or METRICS, cards or [], kw.pop("reach", {}),
                       kw.pop("trends", {}), kw.pop("board", None),
                       decisions if decisions is not None else load_decisions(), **kw)


# ----- bottom line ------------------------------------------------------------------

def test_bottom_line_leads_with_knowledge_then_top_card() -> None:
    b = brief([card("C", "medium"), card("B", "high")])
    lines = b["bottom_line"]
    assert "55%" in lines[0] and "92%" in lines[0]
    assert "47%" in lines[1] and "answered" not in lines[1]       # coverage 75%: no warning
    assert lines[2].startswith("Biggest gap:") and "gap B" in lines[2]
    assert "Curriculum" in lines[3] and "call demo" in lines[3]


def test_bottom_line_flags_low_coverage_on_intent() -> None:
    metrics = {**METRICS, "likely_call.coverage": m(0.23, 1299)}
    line = brief(metrics=metrics)["bottom_line"][1]
    assert "only 23% of respondents answered" in line


def test_bottom_line_with_no_cards_says_nothing_to_act_on(decisions) -> None:
    b = brief([])
    assert decisions["brief"]["bottom_line"]["none"] in b["bottom_line"]


def test_bottom_line_with_only_thin_cards_points_to_watch_list() -> None:
    b = brief([card("D", "low", "thin", 29)])
    assert any("Also worth a look" in x for x in b["bottom_line"])
    assert not any(x.startswith("Biggest gap") for x in b["bottom_line"])


def test_bottom_line_names_goals_that_fell() -> None:
    trends = {"likely_call.pct_high": month_table([0.50, 0.50, 0.40]),
              "agree_pct.post_when": month_table([0.92, 0.92, 0.93])}
    line = brief(trends=trends)["bottom_line"][-1]
    assert line.startswith("Watch: in 2026-03") and "Very likely to call 988" in line
    assert "Know when" not in line


# ----- priorities, watch list, measurement ------------------------------------------

def test_priorities_ranked_by_priority_confidence_then_n() -> None:
    cards = [card("A", "medium", "strong", 500), card("B", "high", "moderate", 60),
             card("C", "high", "strong", 120), card("J", "high", "strong", 900)]
    b = brief(cards)
    assert [c["id"] for c in b["priorities"]] == ["J", "C", "B"]
    assert [c["id"] for c in b["watch"]] == ["A"]                 # beyond the top three


def test_thin_cards_go_to_watch_and_survey_design_to_measurement() -> None:
    b = brief([card("B"), card("H", "low", "thin", 26, lever="presenter_coaching"),
               card("G", "high", "strong", 274, lever="survey_design")])
    assert [c["id"] for c in b["priorities"]] == ["B"]
    assert [c["id"] for c in b["watch"]] == ["H"]
    assert [x["id"] for x in b["measurement"]] == ["G"]


def test_low_coverage_of_likely_call_is_a_measurement_gap() -> None:
    metrics = {**METRICS, "likely_call.coverage": m(0.23, 1299)}
    item = brief(metrics=metrics)["measurement"][0]
    assert item["id"] == "coverage" and "23%" in item["text"] and "299 of 1299" in item["text"]
    assert brief()["measurement"] == []                         # 75% answered: no gap


# ----- scorecard -------------------------------------------------------------------

def test_scorecard_shows_before_after_n_and_range() -> None:
    tiles = {t["metric"]: t for t in brief()["scorecard"]}
    when = tiles["agree_pct.post_when"]
    assert when["value_text"] == "92%" and when["n_text"] == "n = 400"
    assert "55% before, 92% after" in when["notes"]
    assert tiles["nps"]["value_text"] == "+12"
    assert any("+8 to +17" in x for x in tiles["nps"]["notes"])


def test_scorecard_missing_and_too_few(decisions) -> None:
    metrics = {k: v for k, v in METRICS.items() if k != "delivery_agree_pct"}
    metrics["likely_call.pct_high"] = m(None, 9, status="too few responses")
    tiles = {t["metric"]: t for t in brief(metrics=metrics)["scorecard"]}
    tile_cfg = decisions["brief"]["tile"]
    assert tiles["delivery_agree_pct"]["value_text"] == tile_cfg["not_collected"]
    assert tiles["likely_call.pct_high"]["value_text"] == tile_cfg["too_few"]
    assert tiles["likely_call.pct_high"]["n_text"] is None


@pytest.mark.parametrize("raws,direction", [
    ([0.80, 0.80, 0.90], "up"), ([0.90, 0.90, 0.80], "down"), ([0.90, 0.90, 0.92], "steady")])
def test_scorecard_trend_direction(raws: list[float], direction: str) -> None:
    tiles = {t["metric"]: t for t in brief(trends={"agree_pct.post_when": month_table(raws)})["scorecard"]}
    assert tiles["agree_pct.post_when"]["trend"] == direction
    assert "2026-03" in tiles["agree_pct.post_when"]["trend_text"]


def test_scorecard_trend_needs_two_judged_months() -> None:
    table = month_table([0.80, 0.90], ns=[5, 100])               # first month suppressed
    tile = {t["metric"]: t for t in brief(trends={"agree_pct.post_when": table})["scorecard"]}
    assert tile["agree_pct.post_when"]["trend"] is None


def test_scorecard_target_met_or_not(decisions) -> None:
    d = copy.deepcopy(decisions)
    d["brief"]["targets"] = {"agree_pct.post_when": 0.90, "likely_call.pct_high": 0.60}
    tiles = {t["metric"]: t for t in brief(decisions=d)["scorecard"]}
    assert "target 90%: met" in tiles["agree_pct.post_when"]["notes"]
    assert "target 60%: not yet met" in tiles["likely_call.pct_high"]["notes"]


# ----- reach ---------------------------------------------------------------------------

def test_reach_lists_small_groups_and_calls_out_a_gap() -> None:
    reach = {"role_primary": group_table([("Student", 300, 0.95), ("Caregiver", 80, 0.80),
                                          ("Community", 12, 0.5), ("Not recorded", 8, 0.9)]),
             "language": group_table([("English", 381, 0.92), ("Spanish", 19, 0.9)])}
    r = {x["label"]: x["lines"] for x in brief(reach=reach)["reach"]}
    role = " ".join(r["Role"])
    assert "Student, 75% of respondents (300 of 400)" in role
    assert "Caregiver is at 80% vs 92% overall (n = 80)" in role
    assert "Too few responses to judge: Community (12)" in role
    assert "8 responses have no role recorded" in role
    assert "Spanish (19)" in " ".join(r["Language"])
    assert "No group differs" in " ".join(r["Language"])
    assert "not collected" in " ".join(r["County"])


# ----- comments --------------------------------------------------------------------------

def test_comments_skip_other_and_cut_long_quotes() -> None:
    long = "word " * 80
    bd = {"Keep": [{"topic": "Other", "n": 50, "comments": [{"comment": "x", "month": ""}]},
                   {"topic": "911 vs 988", "n": 10, "comments": [{"comment": long, "month": ""}]}],
          "Fix": [], "Add": []}
    cm = brief(board=bd)["comments"]
    assert "10 of 60 comments (17%)" in cm["coverage"]
    keep = cm["columns"][0]
    assert [t["topic"] for t in keep["topics"]] == ["911 vs 988"]
    assert len(keep["topics"][0]["quote"]) <= QUOTE_MAX + 1 and keep["topics"][0]["quote"].endswith("…")
    assert cm["columns"][1]["none"]


# ----- audiences -------------------------------------------------------------------------

def test_funder_version_hides_sessions_and_presenter_items() -> None:
    session = card("H", "high", "strong", 200, lever="presenter_coaching",
                   text="Session 2026-06-01_Prince George's scored 69%.")
    b = brief([session, card("C", lever="presenter_coaching"), card("B")], audience="funder")
    assert b["blocks"] == ["bottom_line", "scorecard", "reach", "limits"]
    text, ok, _ = to_text(b)
    assert ok and "Prince George" not in text and "Question C" not in text
    assert "gap B" in " ".join(b["bottom_line"])
    assert not b["priorities"] and not b["watch"] and not b["measurement"]


def test_unknown_audience_raises() -> None:
    with pytest.raises(ValueError):
        brief(audience="board")


# ----- number check and config -------------------------------------------------------

def test_every_number_in_text_is_traced() -> None:
    b = brief([card("B"), card("D", "low", "thin", 29)],
              trends={"nps": month_table([10.0, 30.0, 5.0])})
    text, ok, stray = to_text(b)
    assert ok and stray == []
    b["bottom_line"].append("Calls rose 37 points.")              # a number nobody computed
    _, ok, stray = to_text(b)
    assert not ok and stray == ["37"]


def test_missing_brief_block_is_a_clear_error(decisions) -> None:
    d = {k: v for k, v in decisions.items() if k != "brief"}
    with pytest.raises(KeyError, match="brief"):
        brief(decisions=d)


def test_bad_template_is_reported_not_crashed(decisions) -> None:
    d = copy.deepcopy(decisions)
    d["brief"]["bottom_line"]["knowledge"] = "Rose to {nonsense}."
    b = brief(decisions=d)
    assert any("bottom_line.knowledge" in p for p in b["problems"])
    assert not any("Rose to" in x for x in b["bottom_line"])


def test_inputs_not_modified() -> None:
    cards, metrics = [card("B")], copy.deepcopy(METRICS)
    trends = {"nps": month_table([10.0, 20.0])}
    before = (copy.deepcopy(cards), copy.deepcopy(metrics), trends["nps"].copy())
    brief(cards, metrics, trends=trends)
    assert cards == before[0] and metrics == before[1]
    pd.testing.assert_frame_equal(trends["nps"], before[2])


def test_module_uses_no_ai_or_network() -> None:
    tree = ast.parse((ROOT / "pipeline" / "summarize.py").read_text(encoding="utf-8"))
    names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
             for a in getattr(n, "names", [])}
    mods = {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not ({"groq", "openai", "anthropic", "requests", "httpx"} & (names | mods))


# ----- end to end on the sample ----------------------------------------------------------

@pytest.fixture(scope="module")
def sample_inputs(decisions) -> dict[str, Any]:
    from pipeline import load_schema
    s = load_schema()
    df, _ = load_files([ROOT / "data" / "sample_30rows.xlsx"])
    df, _ = map_columns(df, s)
    df, _, _ = clean(df, s)
    df = add_flags(df)
    metrics = compute(df)
    cl = classify_df(df, s)
    cards, _ = run_checks(
        {"headline": metrics, "all": metrics, "unflagged": compute(drop_flagged(df))},
        {"session_id": compare(df, "session_id", CHECK_METRIC), "metric": CHECK_METRIC},
        {"monthly": monthly(df, CHECK_METRIC), "metric": CHECK_METRIC, "alerts": alerts(df)},
        check_input(cl, df, s), decisions)
    cfg = decisions["brief"]
    return {"metrics": metrics, "cards": cards,
            "reach": {d["by"]: compare(df, d["by"], CHECK_METRIC) for d in cfg["reach"]["dims"]},
            "trends": {t["trend"]: monthly(df, t["trend"]) for t in cfg["scorecard"] if t.get("trend")},
            "board": board(cl, s)}


@pytest.mark.parametrize("audience", ["internal", "funder"])
def test_sample_brief_end_to_end(sample_inputs, decisions, audience: str) -> None:
    b = build_brief(**sample_inputs, decisions=decisions, audience=audience)
    assert b["problems"] == []
    text, ok, stray = to_text(b)
    assert ok, stray
    assert text.startswith("# " + decisions["brief"]["title"])
    assert "Based on 29 survey responses" in text
    for block in b["blocks"]:
        if block not in ("priorities", "watch") or b[block]:       # empty lists are left out
            assert decisions["brief"]["headings"][block] in text
