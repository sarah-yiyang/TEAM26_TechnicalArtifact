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
from pipeline.themes import board, check_input, classify, classify_df, comment_table
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
    out = classify(comments, schema)
    assert comments == before
    assert all(set(o) == {"topic", "kind"} for o in out)


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
    add = {x["topic"]: x for x in b["Add"]}
    assert add["What happens when you call"]["n"] == 2
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


# ----- redaction --------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("write to jo.smith@school.edu", "write to [EMAIL]"),
    ("(301) 555-1234 or +1 240.555.9876", "[PHONE] or [PHONE]"),
    ("Ms. Lopez and Dr Kim were great", "[NAME] and [NAME] were great"),
    ("thank you Maria!", "thank you [NAME]!"),
    ("me llamo Ana", "me llamo [NAME]"),
    ("call 988, not 911", "call 988, not 911"),
    ("Thank you EveryMind", "Thank you EveryMind"),
    ("thank you so much", "thank you so much"),
])
def test_redact(text, expected) -> None:
    from pipeline.themes import redact
    assert redact(text) == expected


# ----- board lists and the all-comments table ---------------------------------

def test_comment_table_has_every_non_empty_comment_once(sample_classified, schema) -> None:
    table = comment_table(sample_classified, schema)
    said = sample_classified[sample_classified["kind"] != "empty"]
    assert len(table) == len(said) == 32
    assert list(table.columns) == ["column", "topic", "comment", "month"]
    assert sorted(table["comment"]) == sorted(" ".join(str(x).split()) for x in said["text"])
    kind_of = {"Keep": "praise", "Fix": "suggestion", "Add": "open_question"}
    merged = table.merge(said.assign(comment=said["text"].map(lambda x: " ".join(str(x).split()))),
                         on="comment")
    assert (merged["column"].map(kind_of) == merged["kind"]).all()
    assert (merged["topic_x"] == merged["topic_y"]).all()


def test_board_lists_every_comment_in_every_column(sample_classified, schema) -> None:
    b = board(sample_classified, schema)
    for items in b.values():
        for x in items:
            assert x["n"] == len(x["comments"]) > 0
    assert sum(x["n"] for items in b.values() for x in items) == 32
    keep = {x["topic"]: x for x in b["Keep"]}
    assert {"comment": "It was awesome!", "month": "2025-10"} in keep["Positive/no change"]["comments"]


def test_sample_topic_list_newest_first(sample_classified, schema) -> None:
    add = {x["topic"]: x for x in board(sample_classified, schema)["Add"]}
    assert add["More detail/content"]["comments"] == [
        {"comment": "How many calls they get a year versus the amount that they should get?",
         "month": "2026-10"},
        {"comment": "how many people use it", "month": "2026-05"}]


def dated(texts: list[str], times: list[Any]) -> pd.DataFrame:
    return pd.DataFrame({I: texts, "start_time": pd.to_datetime(times)})


def test_order_newest_first_ties_file_order_blank_last(schema) -> None:
    texts = [f"how many people call in month {i}?" for i in range(5)]
    df = dated(texts, ["2026-01-01", "2026-05-01", None, "2026-03-01", "2026-05-01"])
    table = comment_table(classify_df(df, schema), schema)
    assert list(table["comment"]) == [texts[1], texts[4], texts[3], texts[0], texts[2]]
    assert list(table["month"]) == ["2026-05", "2026-05", "2026-03", "2026-01", ""]
    assert comment_table(classify_df(df, schema), schema).equals(table)


def test_comments_redacted_but_not_shortened(schema) -> None:
    long_q = "how many people call " + "and then what " * 30 + "?"
    df = dated(["Can Ms. Lopez email me at a@b.org or 301-555-1234?", long_q],
               ["2026-02-01", "2026-01-01"])
    table = comment_table(classify_df(df, schema), schema)
    assert "Can [NAME] email me at [EMAIL] or [PHONE]?" in set(table["comment"])
    assert " ".join(long_q.split()) in set(table["comment"])


def test_empty_answers_left_out(schema) -> None:
    df = dated(["nothing", "none", None, "how many calls?"], ["2026-01-01"] * 4)
    assert list(comment_table(classify_df(df, schema), schema)["comment"]) == ["how many calls?"]


def test_works_without_start_time_column(schema) -> None:
    df = pd.DataFrame({I: ["how many calls?", "what happens next?"]})
    table = comment_table(classify_df(df, schema), schema)
    assert list(table["comment"]) == ["how many calls?", "what happens next?"]
    assert set(table["month"]) == {""}
