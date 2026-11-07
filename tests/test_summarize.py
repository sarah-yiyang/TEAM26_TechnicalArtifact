"""Tests for pipeline.summarize (M6). The LLM is always mocked; no network."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any

import pytest

from pipeline import ROOT
from pipeline.cleaning import clean
from pipeline.ingest import load_files
from pipeline.mapping import map_columns
from pipeline.metrics import compute
from pipeline.quality import add_flags
from pipeline.recommend import CHECK_METRIC, load_decisions, run_checks
from pipeline.segments import compare
from pipeline.summarize import (
    cache_key, make_client, method_note, numbers_ok, sentences, write_summary,
)
from pipeline.trends import alerts, monthly


def card(cid: str, pct: str = "34%", n: int = 150, **extra: Any) -> dict[str, Any]:
    c = {"id": cid, "title": f"Question {cid}",
         "text": f"{pct} of people who know when to use 988 would not call.",
         "evidence": {"gap_pct": pct, "threshold": "20%", "min_n": 20}, "n": n,
         "confidence": "strong", "priority": "high", "chart_key": "x",
         "decision": "Add a practice-calling segment?", "lever": "curriculum_content",
         "owner_role": "Curriculum", "action": "Consider a short call demo.",
         "success_measure": "Gap falls."}
    c.update(extra)
    return c


class FakeClient:
    """Mimics groq.Groq().chat.completions.create; records every call."""

    def __init__(self, reply: Any = None, raises: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.reply, self.raises = reply, raises
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def sent_ids(self, i: int = -1) -> list[str]:
        return [p["id"] for p in json.loads(self.calls[i]["messages"][1]["content"])]

    def _create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if self.raises:
            raise self.raises
        if callable(self.reply):
            content = self.reply(json.loads(kw["messages"][1]["content"]))
        else:
            content = self.reply
        if not isinstance(content, str):
            content = json.dumps(content)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def echo_ai(payload: list[dict[str, str]]) -> dict[str, str]:
    """A well-behaved model: reuses only the numbers it was given."""
    return {p["id"]: "AI: " + p["text"] for p in payload}


def test_no_client_rule_text(schema) -> None:
    cards = [card("B"), card("D", "41%", 80, confidence="moderate")]
    text, ok, mm = write_summary(cards, schema, rows=1299)
    assert ok and mm == []
    assert "1299" in text and "n = 150" in text and "n = 80" in text
    assert cards[0]["text"] in text and "Consider a short call demo." in text
    assert method_note(cards, None, mm, schema).startswith("Rule text")


def test_ai_used_with_settings_from_schema(schema) -> None:
    fc = FakeClient(echo_ai)
    text, ok, mm = write_summary([card("B")], schema, client=fc)
    assert ok and mm == [] and "AI: 34% of people" in text
    kw = fc.calls[0]
    ai = schema["ai"]
    assert kw["model"] == ai["model"] and kw["temperature"] == 0
    assert kw["response_format"] == {"type": "json_object"}
    assert kw["max_tokens"] == ai["max_tokens_per_card"]
    assert kw["extra_body"] == {"reasoning_effort": ai["reasoning_effort"]}
    assert "AI wording" in method_note([card("B")], fc, mm, schema)


def test_only_small_payload_sent(schema) -> None:
    fc = FakeClient(echo_ai)
    write_summary([card("B")], schema, client=fc)
    sent = json.loads(fc.calls[0]["messages"][1]["content"])[0]
    assert set(sent) == {"id", "title", "text", "action"}


def test_cache_reused_and_only_changed_card_sent(schema) -> None:
    fc, cache = FakeClient(echo_ai), {}
    cards = [card("B"), card("D", "41%")]
    first = write_summary(cards, schema, client=fc, cache=cache)
    assert len(fc.calls) == 1 and len(cache) == 2
    assert write_summary(cards, schema, client=fc, cache=cache) == first
    assert len(fc.calls) == 1                      # nothing sent again
    changed = [card("B"), card("D", "45%")]
    write_summary(changed, schema, client=fc, cache=cache)
    assert len(fc.calls) == 2 and fc.sent_ids() == ["D"]


def test_uncached_cards_batched_in_one_call(schema) -> None:
    fc = FakeClient(echo_ai)
    cards = [card("B"), card("C", "12%"), card("D", "41%")]
    write_summary(cards, schema, client=fc)
    assert len(fc.calls) == 1 and fc.sent_ids() == ["B", "C", "D"]
    assert fc.calls[0]["max_tokens"] == 3 * schema["ai"]["max_tokens_per_card"]


def test_invented_number_falls_back(schema) -> None:
    fc, cache = FakeClient({"B": "Half of people, 50%, would not call."}), {}
    c = card("B")
    text, ok, mm = write_summary([c], schema, client=fc, cache=cache)
    assert not ok and cache == {}
    assert mm == [{"id": "B", "reason": "number not in evidence", "numbers": ["50"]}]
    assert c["text"] in text and "50%" not in text
    assert "0 of 1" in method_note([c], fc, mm, schema)


def test_service_names_allowed(schema) -> None:
    fc = FakeClient({"B": "34% know when to use 988 instead of 911 but would not call."})
    text, ok, mm = write_summary([card("B")], schema, client=fc, allowed=["988", "911"])
    assert ok and mm == [] and "instead of 911" in text


@pytest.mark.parametrize("fc", [FakeClient(raises=RuntimeError("down")),
                                FakeClient("not json"), FakeClient({"Z": "wrong id"})])
def test_ai_failure_rule_text_not_cached(schema, fc) -> None:
    cache, c = {}, card("B")
    text, ok, mm = write_summary([c], schema, client=fc, cache=cache)
    assert ok and cache == {} and c["text"] in text
    assert len(mm) == 1 and mm[0]["id"] == "B" and mm[0]["numbers"] == []


def test_redacted_before_sending(schema) -> None:
    fc = FakeClient(echo_ai)
    c = card("J", action="Email jane.doe@example.org or call 410-555-1234.")
    write_summary([c], schema, client=fc)
    sent = fc.calls[0]["messages"][1]["content"]
    assert "jane.doe" not in sent and "555-1234" not in sent and "[EMAIL]" in sent


def test_no_cards_no_call(schema) -> None:
    fc = FakeClient(echo_ai)
    text, ok, mm = write_summary([], schema, client=fc)
    assert ok and mm == [] and fc.calls == []
    assert schema["ai"]["summary"]["none_text"] in text
    assert "no findings" in method_note([], fc, mm, schema)


def test_inputs_not_modified(schema) -> None:
    cards = [card("B")]
    before = copy.deepcopy(cards)
    write_summary(cards, schema, client=FakeClient(echo_ai))
    assert cards == before


def test_cache_key_tracks_content_and_model(schema) -> None:
    other = copy.deepcopy(schema)
    other["ai"]["model"] = "another-model"
    assert cache_key(card("B"), schema) == cache_key(card("B", success_measure="x"), schema)
    assert cache_key(card("B"), schema) != cache_key(card("B", "35%"), schema)
    assert cache_key(card("B"), schema) != cache_key(card("B"), other)


def test_no_key_no_client() -> None:
    assert make_client(None) is None and make_client("  ") is None


def test_no_model_or_prompt_in_python(schema) -> None:
    src = (ROOT / "pipeline" / "summarize.py").read_text(encoding="utf-8")
    assert schema["ai"]["model"] not in src
    assert "qwen" not in src.lower() and "gpt-oss" not in src
    assert schema["ai"]["system_prompt"].split()[0] + " each" not in src


def test_sample_end_to_end(sample_path, schema) -> None:
    decisions = load_decisions()
    raw, _ = load_files([sample_path], schema)
    mapped, _ = map_columns(raw, schema)
    df, _, _ = clean(mapped, schema)
    df = add_flags(df, schema)
    m_all, m_kept = compute(df, schema), compute(df, schema, exclude_flagged=True)
    cards, _ = run_checks(
        {"headline": m_all, "all": m_all, "unflagged": m_kept},
        {"session_id": compare(df, "session_id", CHECK_METRIC, schema), "metric": CHECK_METRIC},
        {"monthly": monthly(df, CHECK_METRIC, schema=schema), "metric": CHECK_METRIC,
         "alerts": alerts(df, schema)},
        None, decisions, schema)
    text, ok, mm = write_summary(cards, schema, rows=m_all["rows"]["value"],
                                 allowed=decisions.get("allowed_literals"))
    assert ok and mm == [] and text.startswith(schema["ai"]["summary"]["title"])
    fc = FakeClient(echo_ai)
    _, ok2, mm2 = write_summary(cards, schema, client=fc,
                                allowed=decisions.get("allowed_literals"))
    assert ok2 and mm2 == [] and len(fc.calls) == (1 if cards else 0)


def test_sentences_then_subset_without_new_call(schema) -> None:
    fc, cache = FakeClient(echo_ai), {}
    cards = [card("B"), card("D", "41%")]
    sents, mm = sentences(cards, schema, client=fc, cache=cache)
    assert set(sents) == {"B", "D"} and mm == [] and numbers_ok(mm)
    text, ok, mm2 = write_summary(cards[1:], schema, reworded=sents)
    assert len(fc.calls) == 1 and ok and mm2 == []
    assert "AI: 41%" in text and "34%" not in text


def test_reworded_missing_card_keeps_rule_text(schema) -> None:
    c = card("B")
    text, _, _ = write_summary([c], schema, client=FakeClient(echo_ai), reworded={})
    assert c["text"] in text
