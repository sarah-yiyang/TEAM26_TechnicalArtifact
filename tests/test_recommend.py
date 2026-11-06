"""Tests for pipeline.recommend (insight checks A-J). No network."""
from __future__ import annotations

import builtins
import copy
import re
from typing import Any

import pandas as pd
import pytest

from pipeline import MIN_N
from pipeline.cleaning import clean
from pipeline.ingest import load_files
from pipeline.mapping import map_columns
from pipeline.metrics import compute
from pipeline.quality import add_flags
from pipeline.recommend import (
    CHECK_IDS, CHECK_METRIC, LEVERS, NOT_ASSIGNED, _confidence, check_numbers,
    load_decisions, run_checks, validate_decisions,
)
from pipeline.segments import compare
from pipeline.trends import alerts, monthly

CARD_FIELDS = {"id", "title", "text", "evidence", "n", "confidence", "priority", "chart_key",
               "decision", "lever", "owner_role", "action", "success_measure"}
KNOWLEDGE = ["post_identify", "post_describe", "post_when"]
DELIVERY = ["presenter_engaged", "presenter_knew", "well_organized"]


# ----- builders: small hand-made inputs, nothing found by default -----------

def r(value: Any, n: int) -> dict[str, Any]:
    return {"value": value, "n": n}


def base_metrics(rows: int = 200) -> dict[str, Any]:
    m = {
        "rows": r(rows, rows),
        "know_call_gap": r(0.10, 150),
        "know_call_2x2": r({"knows_and_calls": 100, "knows_not_calls": 15,
                            "not_knows_calls": 20, "not_knows_not_calls": 15}, 150),
        "delivery_agree_pct": r(0.70, 150),
        "agree_pct.post_when": r(0.90, rows),
        "likely_call.pct_high": r(0.50, 150),
        "nps": r(40.0, 190),
    }
    for k in ("identify", "describe", "when"):
        m[f"ceiling_pct.{k}"] = r(0.10, rows)
    for f in KNOWLEDGE:
        m[f"intent_driver.{f}"] = r(0.40, 150)
    for f in DELIVERY:
        m[f"intent_driver.{f}"] = r(0.35, 150)
    return m


def month_table(raw: list[float], adjusted: list[float], n: int = 60) -> pd.DataFrame:
    months = [f"2026-0{i + 1}" for i in range(len(raw))]
    return pd.DataFrame({"month": months, "rows": n, "n": n, "raw": raw, "expected": raw,
                         "adjusted": adjusted, "suppressed": n < MIN_N})


def session_table(outlier: bool = False) -> pd.DataFrame:
    rows = []
    for i, (value, n) in enumerate([(0.90, 40), (0.88, 50), (0.40 if outlier else 0.87, 45)]):
        rows.append({"group": f"2026-03-1{i}_Howard", "n": n, "metric_n": n, "value": value,
                     "suppressed": False, "centre": 0.88, "lower95": 0.80, "upper95": 0.96,
                     "lower998": 0.75, "upper998": 1.0, "outlier": value < 0.75})
    return pd.DataFrame(rows)


def inputs(**over: Any) -> dict[str, Any]:
    """Default run_checks inputs where no check finds anything."""
    m = base_metrics()
    unflagged = copy.deepcopy(m)
    unflagged["rows"] = r(190, 190)
    args = {
        "metrics": {"headline": m, "all": m, "unflagged": unflagged},
        "segments": {"session_id": session_table(), "metric": CHECK_METRIC},
        "trends": {"monthly": month_table([0.80, 0.70], [0.80, 0.70]), "metric": CHECK_METRIC,
                   "alerts": []},
        "themes": None,
    }
    args.update(over)
    return args


@pytest.fixture(scope="module")
def decisions() -> dict[str, Any]:
    return load_decisions()


def run(decisions: dict[str, Any], schema: dict[str, Any], **over: Any):
    a = inputs(**over)
    return run_checks(a["metrics"], a["segments"], a["trends"], a["themes"], decisions, schema)


def result_of(runs: list[dict[str, Any]], cid: str) -> dict[str, Any]:
    return next(x for x in runs if x["id"] == cid)


def card_of(cards: list[dict[str, Any]], cid: str) -> dict[str, Any] | None:
    return next((c for c in cards if c["id"] == cid), None)


# finding inputs per check
def finding_inputs(cid: str) -> dict[str, Any]:
    a = inputs()
    m = a["metrics"]["headline"]
    if cid == "A":
        a["trends"]["monthly"] = month_table([0.84, 0.66], [0.75, 0.75])
    elif cid == "B":
        m["know_call_gap"] = r(0.34, 150)
        m["know_call_2x2"]["value"]["knows_not_calls"] = 51
    elif cid == "C":
        m["intent_driver.presenter_engaged"] = r(0.62, 150)
    elif cid == "D":
        m["ceiling_pct.when"] = r(0.45, 200)
    elif cid == "E":
        a["trends"]["alerts"] = [{"id": "E", "kind": "blank_answers", "field": "presenter_knew",
                                  "months": [{"month": "2026-02", "blank_pct": 0.47, "rows": 60,
                                              "other_months_blank_pct": 0.1}], "n": 60}]
    elif cid == "G":
        a["trends"]["alerts"] = [{"id": "G", "kind": "fewer_comments", "field": "comment_improve",
                                  "month": "2026-02", "rate": 0.20, "rows": 60,
                                  "earlier_rate": 0.58, "earlier_rows": 60, "n": 60}]
    elif cid == "H":
        a["segments"]["session_id"] = session_table(outlier=True)
    elif cid == "I":
        u = a["metrics"]["unflagged"]
        u["agree_pct.post_when"] = r(0.80, 190)
    elif cid in ("F", "J"):
        a["themes"] = themes_stub(found=True)
    return a


def themes_stub(found: bool) -> dict[str, Any]:
    comments = [{"topic": "Positive/no change", "kind": "praise"}] * 40
    if found:
        comments += [{"topic": "What happens when you call", "kind": "open_question"}] * 8
    return {"low_raters": {"cutoff": 6, "n": 30, "silent": 24 if found else 5},
            "comments": comments}


def run_with(decisions, schema, a):
    return run_checks(a["metrics"], a["segments"], a["trends"], a["themes"], decisions, schema)


# ----- each check: one finding, one no-finding -------------------------------

@pytest.mark.parametrize("cid", list(CHECK_IDS))
def test_check_finds(cid: str, decisions, schema) -> None:
    cards, runs = run_with(decisions, schema, finding_inputs(cid))
    assert result_of(runs, cid)["result"] == "finding"
    assert card_of(cards, cid) is not None


@pytest.mark.parametrize("cid", list(CHECK_IDS))
def test_check_no_finding(cid: str, decisions, schema) -> None:
    over = {"themes": themes_stub(found=False)} if cid in ("F", "J") else {}
    cards, runs = run(decisions, schema, **over)
    res = result_of(runs, cid)
    assert res["result"] == "no_finding"
    assert card_of(cards, cid) is None
    assert res["text"] and "{" not in res["text"]
    assert res["config_gaps"] == []


def test_f_and_j_need_themes_until_m5(decisions, schema) -> None:
    cards, runs = run(decisions, schema)
    for cid in ("F", "J"):
        assert result_of(runs, cid)["result"] == "needs_themes"
        assert card_of(cards, cid) is None


# ----- decision map -----------------------------------------------------------

def test_shipped_decisions_are_valid(decisions) -> None:
    assert validate_decisions(decisions) == []


@pytest.mark.parametrize("cid", list(CHECK_IDS))
def test_card_filled_from_decisions(cid: str, decisions, schema) -> None:
    cards, runs = run_with(decisions, schema, finding_inputs(cid))
    card, entry = card_of(cards, cid), decisions["checks"][cid]
    assert CARD_FIELDS <= set(card)
    assert card["lever"] == entry["lever"] and card["lever"] in LEVERS
    assert card["owner_role"] == entry["owner_role"]
    assert card["decision"] == " ".join(entry["decision"].split())
    assert card["confirmed"] is False
    for key in ("text", "action", "success_measure"):
        assert card[key] and "{" not in card[key], key
    assert result_of(runs, cid)["config_gaps"] == []


def test_missing_entry_is_not_assigned(decisions, schema) -> None:
    d = copy.deepcopy(decisions)
    del d["checks"]["B"]
    assert "B: no entry in decisions.yaml" in validate_decisions(d)
    cards, runs = run_with(d, schema, finding_inputs("B"))
    card = card_of(cards, "B")
    assert card["owner_role"] == NOT_ASSIGNED and card["decision"] == NOT_ASSIGNED
    assert card["lever"] is None and card["action"] is None and card["success_measure"] is None
    assert "34%" in card["text"]                         # evidence still shown
    assert "no entry in decisions.yaml" in result_of(runs, "B")["config_gaps"]


@pytest.mark.parametrize("key", ["lever", "secondary_lever"])
def test_invalid_lever_is_reported(key: str, decisions, schema) -> None:
    d = copy.deepcopy(decisions)
    d["checks"]["B"][key] = "marketing"
    assert any(e.startswith(f"B: {key} 'marketing'") for e in validate_decisions(d))
    cards, runs = run_with(d, schema, finding_inputs("B"))
    card = card_of(cards, "B")
    assert card[key] is None
    assert any("'marketing'" in g for g in result_of(runs, "B")["config_gaps"])


def test_unknown_placeholder_falls_back(decisions, schema) -> None:
    d = copy.deepcopy(decisions)
    d["checks"]["B"]["finding"]["action"] = "Consider {made_up}."
    cards, runs = run_with(d, schema, finding_inputs("B"))
    assert card_of(cards, "B")["action"] is None
    assert any("made_up" in g for g in result_of(runs, "B")["config_gaps"])


# ----- numbers ------------------------------------------------------------------

@pytest.mark.parametrize("cid", list(CHECK_IDS))
def test_no_number_outside_evidence(cid: str, decisions, schema) -> None:
    cards, _ = run_with(decisions, schema, finding_inputs(cid))
    card = card_of(cards, cid)
    for key in ("text", "action", "success_measure"):
        assert check_numbers(card[key], card["evidence"], decisions["allowed_literals"]) == []


def test_planted_number_is_caught(decisions, schema) -> None:
    d = copy.deepcopy(decisions)
    d["checks"]["B"]["finding"]["action"] = "Consider it: 50% of people agree."
    cards, runs = run_with(d, schema, finding_inputs("B"))
    assert card_of(cards, "B")["action"] is None
    assert any("50" in g for g in result_of(runs, "B")["config_gaps"])


# ----- purity, suppression, wording ---------------------------------------------

def test_run_checks_does_no_file_io(decisions, schema, monkeypatch) -> None:
    a = finding_inputs("B")

    def no_open(*args: Any, **kw: Any) -> None:
        raise AssertionError("run_checks opened a file")

    monkeypatch.setattr(builtins, "open", no_open)
    cards, _ = run_with(decisions, schema, a)
    assert card_of(cards, "B")


def test_inputs_not_modified(decisions, schema) -> None:
    a = finding_inputs("H")
    before, dec_before = copy.deepcopy(a), copy.deepcopy(decisions)
    run_with(decisions, schema, a)
    assert a["metrics"] == before["metrics"] and decisions == dec_before
    assert a["segments"]["session_id"].equals(before["segments"]["session_id"])
    assert a["trends"]["monthly"].equals(before["trends"]["monthly"])


def test_too_few_gives_no_card(decisions, schema) -> None:
    a = finding_inputs("B")
    a["metrics"]["headline"]["know_call_gap"] = {"value": None, "n": 9, "status": "too few"}
    cards, runs = run_with(decisions, schema, a)
    assert result_of(runs, "B")["result"] == "too_few" and card_of(cards, "B") is None


def test_not_collected(decisions, schema) -> None:
    a = finding_inputs("B")
    a["metrics"]["headline"]["know_call_gap"] = {"value": None, "n": 0, "status": "not collected"}
    _, runs = run_with(decisions, schema, a)
    assert result_of(runs, "B")["result"] == "not_collected"


@pytest.mark.parametrize("n,level", [(99, "moderate"), (100, "strong"), (49, "thin"), (50, "moderate")])
def test_confidence_boundaries(n: int, level: str) -> None:
    assert _confidence(n) == level


def test_session_card_reads_as_review(decisions, schema) -> None:
    cards, _ = run_with(decisions, schema, finding_inputs("H"))
    card = card_of(cards, "H")
    words = " ".join([card["text"], card["action"], card["success_measure"], card["decision"]])
    for bad in ("poor", "fail", "underperform", "worst", "weak", "bad"):
        assert not re.search(rf"\b{bad}", words, re.I), bad


def test_actions_are_options_not_directives(decisions) -> None:
    for cid, entry in decisions["checks"].items():
        action = entry["finding"]["action"]
        assert not re.search(r"\b(must|should|need to)\b", action, re.I), cid


def test_cards_sorted_by_priority(decisions, schema) -> None:
    a = finding_inputs("B")
    a["trends"]["monthly"] = month_table([0.84, 0.66], [0.75, 0.75])
    cards, _ = run_with(decisions, schema, a)
    order = {"high": 0, "medium": 1, "low": 2}
    assert [order[c["priority"]] for c in cards] == sorted(order[c["priority"]] for c in cards)


# ----- end to end on the 30-row sample ------------------------------------------

def test_sample_runs_end_to_end(sample_path, schema, decisions) -> None:
    raw, _ = load_files([sample_path], schema)
    mapped, _ = map_columns(raw, schema)
    df, _, _ = clean(mapped, schema)
    df = add_flags(df, schema)
    m_all, m_kept = compute(df, schema), compute(df, schema, exclude_flagged=True)
    cards, runs = run_checks(
        {"headline": m_all, "all": m_all, "unflagged": m_kept},
        {"session_id": compare(df, "session_id", CHECK_METRIC, schema), "metric": CHECK_METRIC},
        {"monthly": monthly(df, CHECK_METRIC, schema=schema), "metric": CHECK_METRIC,
         "alerts": alerts(df, schema)},
        None, decisions, schema)
    assert [x["id"] for x in runs] == list(CHECK_IDS)
    assert all(x["config_gaps"] == [] for x in runs)
    by = {x["id"]: x["result"] for x in runs}
    assert by["F"] == by["J"] == "needs_themes"
    # 29 rows: anything split by month, session or likely_call is too small
    for cid in ("A", "B", "C", "E", "G", "H"):
        assert by[cid] in ("too_few", "not_collected"), cid
    for card in cards:
        assert card["n"] >= MIN_N
