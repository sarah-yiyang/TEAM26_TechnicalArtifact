"""Tests for pipeline.themes (keyword rules, M5a). No network."""
from __future__ import annotations

import builtins
import copy
import socket
from typing import Any

import pandas as pd
import pytest

from pipeline.cleaning import clean
from pipeline.ingest import load_files
from pipeline.mapping import map_columns
from pipeline.recommend import load_decisions, run_checks
from pipeline.themes import board, check_input, classify, classify_df
from tests.test_recommend import inputs, result_of

V, I = "comment_valuable", "comment_improve"


@pytest.fixture(scope="module")
def sample_df(sample_path, schema) -> pd.DataFrame:
    raw, _ = load_files([sample_path], schema)
    mapped, _ = map_columns(raw, schema)
    df, _, _ = clean(mapped, schema)
    return df


@pytest.fixture(scope="module")
def sample_classified(sample_df, schema) -> pd.DataFrame:
    return classify_df(sample_df, schema)


def label(classified: pd.DataFrame, field: str, text: str) -> tuple[str, str]:
    row = classified[(classified["field"] == field) & (classified["text"] == text)].iloc[0]
    return row["kind"], row["topic"]


# ----- sample comments -------------------------------------------------------

@pytest.mark.parametrize("field,text,kind,topic", [
    (I, "nothing", "empty", "Other"),
    (I, "none", "empty", "Other"),
    (I, "what if someone doesn't want help?", "open_question", "What happens when you call"),
    (I, "How the calls usually go", "open_question", "What happens when you call"),
    (I, "Is there a way to call 988 without leaving the history on your phone if you have "
        "strict parents?", "open_question", "Confidentiality/anonymity"),
    (I, "How many calls they get a year versus the amount that they should get?",
     "open_question", "More detail/content"),
    (I, "It was awesome!", "praise", "Positive/no change"),
    (I, "Todo está muy bien", "praise", "Positive/no change"),
    (I, "I thought it was well organized", "praise", "Positive/no change"),
    (I, "Tech issues detracted a bit from the preentation. Specifically, we couldn't see "
        "the video.", "suggestion", "Logistics/format"),
    (I, "Time management.", "suggestion", "Logistics/format"),
    (I, "By adding students also", "suggestion", "Other"),
    (V, "How to identify sign", "praise", "Other"),
    (V, "When you can/should call 988", "praise", "Other"),
    (V, "When to use 988 and the difference between 988 and 911", "praise", "911 vs 988"),
    (V, "I didn't know that you could text 988", "praise", "What happens when you call"),
])
def test_sample_comment_labels(sample_classified, field, text, kind, topic) -> None:
    assert label(sample_classified, field, text) == (kind, topic)


def test_blank_is_empty_other(sample_classified) -> None:
    blank = sample_classified[sample_classified["text"].isna()]
    assert len(blank) > 0
    assert set(blank["kind"]) == {"empty"} and set(blank["topic"]) == {"Other"}


def test_one_row_per_response_and_field(sample_df, sample_classified) -> None:
    assert len(sample_classified) == 2 * len(sample_df)
    assert set(sample_classified["field"]) == {V, I}


# ----- rule behaviour --------------------------------------------------------

def test_accents_and_case_ignored(schema) -> None:
    [out] = classify([(I, "Más en ESPAÑOL por favor")], schema)
    assert out["topic"] == "Language access"


def test_question_mark_counts_in_valuable_field(schema) -> None:
    [out] = classify([(V, "Who answers the phone?")], schema)
    assert out["kind"] == "open_question"


def test_question_word_start_only_in_improve_field(schema) -> None:
    v, i = classify([(V, "What happens next"), (I, "What happens next")], schema)
    assert v["kind"] == "praise" and i["kind"] == "open_question"


def test_punctuation_only_is_empty(schema) -> None:
    assert classify([(I, " - ")], schema)[0]["kind"] == "empty"


def test_rules_come_from_schema(schema) -> None:
    s = copy.deepcopy(schema)
    s["themes"]["topics"] = {"Careers at 988": ["awesome"]}
    assert classify([(I, "It was awesome!")], s)[0]["topic"] == "Careers at 988"


def test_missing_themes_block_is_reported(schema) -> None:
    s = copy.deepcopy(schema)
    del s["themes"]
    with pytest.raises(KeyError, match="themes"):
        classify([(I, "hi")], s)


def test_classify_is_pure_and_offline(schema, monkeypatch) -> None:
    def blocked(*a: Any, **k: Any) -> None:
        raise AssertionError("no file or network access allowed")
    monkeypatch.setattr(builtins, "open", blocked)
    monkeypatch.setattr(socket, "socket", blocked)
    comments = [(I, "what happens when I call?"), (V, None)]
    before = copy.deepcopy(comments)
    out = classify(comments, schema, cache_path="unused.json", use_llm=True)
    assert comments == before
    assert all(o["method"] == "keywords" for o in out)


def test_topics_and_kinds_are_from_fixed_lists(sample_classified, schema) -> None:
    allowed = set(schema["themes"]["topics"]) | {"Other"}
    assert set(sample_classified["topic"]) <= allowed
    assert set(sample_classified["kind"]) <= {"suggestion", "open_question", "praise", "empty"}


# ----- board -----------------------------------------------------------------

def test_board_counts_add_up(sample_classified, schema) -> None:
    b = board(sample_classified, schema)
    assert list(b) == ["Keep", "Fix", "Add"]
    total = sum(x["n"] for col in b.values() for x in col)
    assert total == int((sample_classified["kind"] != "empty").sum())
    assert {"topic": "What happens when you call", "n": 2} in b["Add"]
    for col in b.values():
        assert [x["n"] for x in col] == sorted((x["n"] for x in col), reverse=True)


# ----- check input + checks F and J ------------------------------------------

def frame(found: bool) -> pd.DataFrame:
    """40 people: 30 low raters (NPS 3), 10 high raters (NPS 9)."""
    nps = [3] * 30 + [9] * 10
    if found:   # 20 low raters silent; 10 open questions out of 30 comments
        improve = [None] * 20 + ["what happens after I call?"] * 10 + ["More examples"] * 10
        valuable = [None] * 30 + ["When to call 988"] * 10
    else:       # everyone comments; no questions
        improve = ["Great training"] * 40
        valuable = [None] * 40
    return pd.DataFrame({"nps": nps, V: valuable, I: improve})


def test_check_input_counts(schema) -> None:
    df = frame(found=True)
    themes = check_input(classify_df(df, schema), df, schema)
    assert themes["low_raters"] == {"cutoff": 6, "n": 30, "silent": 20}
    assert len(themes["comments"]) == 80


def test_check_input_none_without_comment_fields(schema) -> None:
    df = pd.DataFrame({"nps": [3, 9]})
    assert check_input(classify_df(df, schema), df, schema) is None


def test_check_input_without_nps(schema) -> None:
    df = frame(found=True).drop(columns="nps")
    assert check_input(classify_df(df, schema), df, schema)["low_raters"]["n"] == 0


@pytest.mark.parametrize("found,expected", [(True, "finding"), (False, "no_finding")])
def test_checks_f_and_j_from_themes(schema, found, expected) -> None:
    df = frame(found)
    themes = check_input(classify_df(df, schema), df, schema)
    a = inputs(themes=themes)
    _, runs = run_checks(a["metrics"], a["segments"], a["trends"], a["themes"],
                         load_decisions(), schema)
    assert result_of(runs, "F")["result"] == expected
    assert result_of(runs, "J")["result"] == expected
    if found:
        assert result_of(runs, "J")["evidence"]["top_topic"] == "What happens when you call"


def test_sample_is_too_few_for_f_and_j(sample_df, sample_classified, schema) -> None:
    themes = check_input(sample_classified, sample_df, schema)
    a = inputs(themes=themes)
    _, runs = run_checks(a["metrics"], a["segments"], a["trends"], a["themes"],
                         load_decisions(), schema)
    assert result_of(runs, "F")["result"] == "too_few"   # 5 low raters
    assert result_of(runs, "J")["result"] in ("too_few", "finding", "no_finding")
