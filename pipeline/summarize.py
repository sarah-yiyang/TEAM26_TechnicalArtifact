"""Copy-ready summary of the insight cards (M6).

Each finding card's rule text (from ``recommend.run_checks``) can be reworded
into one plain sentence by an LLM (Groq). To keep token use low:

- only finding cards are sent, each as id + title + text + action, redacted;
- every card already in ``cache`` is reused, never sent again;
- all uncached cards go out in ONE call, with a per-card output cap.

The AI may not add numbers. A sentence with any number not in that card's
evidence, rule text or ``allowed`` falls back to the rule text and sets
``numbers_ok=False``. No client (no key) -> rule text for every card.

All wording (prompt, headings) and the model name live in
``config/schema.yaml`` under ``ai:``. Pure: no file I/O, no Streamlit; the
only side effect is adding entries to the caller's ``cache`` dict.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from pipeline import load_schema
from pipeline.recommend import check_numbers
from pipeline.themes import redact

PAYLOAD_FIELDS: tuple[str, ...] = ("title", "text", "action")


# ----- client ---------------------------------------------------------------

def make_client(api_key: str | None) -> Any | None:
    """Groq client for ``api_key``, or ``None`` when there is no key."""
    if not api_key or not str(api_key).strip():
        return None
    from groq import Groq  # imported lazily so tests never need the network
    return Groq(api_key=str(api_key).strip())


# ----- helpers --------------------------------------------------------------

def _ai(schema: dict[str, Any]) -> dict[str, Any]:
    ai = schema.get("ai")
    if not isinstance(ai, dict) or not ai.get("model") or not ai.get("system_prompt"):
        raise KeyError("config/schema.yaml is missing ai.model / ai.system_prompt")
    return ai


def _payload(card: dict[str, Any]) -> dict[str, str]:
    """The small, redacted part of a card that is sent to the model."""
    out = {"id": str(card["id"])}
    for f in PAYLOAD_FIELDS:
        v = card.get(f)
        if isinstance(v, str) and v.strip():
            out[f] = redact(" ".join(v.split()))
    return out


def cache_key(card: dict[str, Any], schema: dict[str, Any]) -> str:
    """Content hash of what would be sent: same card + same prompt/model -> same key."""
    ai = _ai(schema)
    blob = json.dumps({"model": ai["model"], "prompt": ai["system_prompt"],
                       "card": _payload(card)}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _ask(client: Any, cards: list[dict[str, Any]], ai: dict[str, Any]) -> dict[str, Any]:
    """One call for all ``cards``; returns the parsed ``{id: sentence}`` JSON."""
    # extra_body: works with every groq SDK version (0.15 lacks reasoning_effort)
    extra = ({"extra_body": {"reasoning_effort": ai["reasoning_effort"]}}
             if ai.get("reasoning_effort") else {})
    resp = client.chat.completions.create(
        **extra,
        model=ai["model"],
        temperature=ai.get("temperature", 0),
        max_tokens=int(ai.get("max_tokens_per_card", 60)) * len(cards),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": " ".join(str(ai["system_prompt"]).split())},
            {"role": "user", "content": json.dumps([_payload(c) for c in cards],
                                                   ensure_ascii=False, separators=(",", ":"))},
        ],
    )
    data = json.loads(resp.choices[0].message.content or "")
    if not isinstance(data, dict):
        raise ValueError("reply is not a JSON object")
    return data


def _vet(card: dict[str, Any], sentence: Any,
         allowed: list[str]) -> tuple[str | None, dict[str, Any] | None]:
    """Accept ``sentence`` or return a mismatch explaining why not."""
    if not isinstance(sentence, str) or not sentence.strip():
        return None, {"id": card["id"], "reason": "no AI sentence returned", "numbers": []}
    sentence = " ".join(sentence.split())
    known = {**(card.get("evidence") or {}), "_text": card.get("text") or "", "_n": card.get("n")}
    extra = check_numbers(sentence, known, allowed)
    if extra:
        return None, {"id": card["id"], "reason": "number not in evidence", "numbers": extra}
    return sentence, None


def _reword(cards: list[dict[str, Any]], schema: dict[str, Any], client: Any | None,
            cache: dict[str, str], allowed: list[str]
            ) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """``{card id: AI sentence}`` for cards the AI wording was accepted for."""
    if client is None or not cards:
        return {}, []
    ai = _ai(schema)
    keys = {c["id"]: cache_key(c, schema) for c in cards}
    todo = [c for c in cards if keys[c["id"]] not in cache]
    mismatches: list[dict[str, Any]] = []
    if todo:
        try:
            reply = _ask(client, todo, ai)
        except Exception as exc:  # network, auth, bad JSON: rule text, not cached
            reply = {}
            mismatches += [{"id": c["id"], "reason": f"AI call failed ({type(exc).__name__})",
                            "numbers": []} for c in todo]
        else:
            for c in todo:
                sentence, bad = _vet(c, reply.get(str(c["id"])), allowed)
                if bad:
                    mismatches.append(bad)
                else:
                    cache[keys[c["id"]]] = sentence
    return {cid: cache[k] for cid, k in keys.items() if k in cache}, mismatches


# ----- public ---------------------------------------------------------------

def sentences(
    cards: list[dict[str, Any]],
    schema: dict[str, Any] | None = None,
    client: Any | None = None,
    cache: dict[str, str] | None = None,
    allowed: list[str] | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """``({card id: accepted AI sentence}, mismatches)``; cards missing from the dict
    keep their rule text. Pass the dict to ``write_summary(reworded=...)`` to build
    text for any subset of cards without another AI call.
    """
    schema = load_schema() if schema is None else copy.deepcopy(schema)
    return _reword(copy.deepcopy(cards or []), schema, client,
                   {} if cache is None else cache, [str(a) for a in allowed or []])


def numbers_ok(mismatches: list[dict[str, Any]]) -> bool:
    """False when any AI sentence was rejected for a number not in evidence."""
    return not any(m["reason"] == "number not in evidence" for m in mismatches)


def write_summary(
    cards: list[dict[str, Any]],
    schema: dict[str, Any] | None = None,
    client: Any | None = None,
    cache: dict[str, str] | None = None,
    rows: int | None = None,
    allowed: list[str] | None = None,
    reworded: dict[str, str] | None = None,
) -> tuple[str, bool, list[dict[str, Any]]]:
    """Plain-text summary of finding ``cards``. Returns ``(text, numbers_ok, mismatches)``.

    ``client``: Groq-like client (``None`` -> rule text). ``cache``: dict kept by the
    caller across runs; reworded cards are stored there and reused. ``rows``: total
    responses for the header. ``allowed``: extra numbers allowed (``allowed_literals``).
    ``reworded``: sentences from ``sentences()``; when given, no AI call is made.
    """
    schema = load_schema() if schema is None else copy.deepcopy(schema)
    cards = copy.deepcopy(cards or [])
    s = _ai(schema).get("summary") or {}
    if reworded is not None:
        ai_text, mismatches = dict(reworded), []
    else:
        ai_text, mismatches = sentences(cards, schema, client, cache, allowed)

    lines = [str(s.get("title", ""))]
    if rows is not None:
        lines.append(str(s.get("rows_line", "{rows}")).format(rows=rows))
    lines.append("")
    if not cards:
        lines.append(str(s.get("none_text", "")))
    else:
        lines.append(str(s.get("findings_heading", "")))
        for i, c in enumerate(cards, 1):
            n_label = str(s.get("n_label", "n = {n}")).format(n=c.get("n"),
                                                              confidence=c.get("confidence"))
            lines += ["", f"{i}. {c.get('title')}",
                      f"   {ai_text.get(c['id'], c.get('text') or '')} ({n_label})"]
            if c.get("decision"):
                lines.append(f"   {s.get('decision_label', '')} {c['decision']}")
            if c.get("action"):
                lines.append(f"   {s.get('action_label', '')} {c['action']}")
    return "\n".join(lines).strip() + "\n", numbers_ok(mismatches), mismatches


def method_note(cards: list[dict[str, Any]], client: Any | None,
                mismatches: list[dict[str, Any]], schema: dict[str, Any] | None = None) -> str:
    """One line saying how the summary was worded (shown under it in the app)."""
    schema = load_schema() if schema is None else schema
    ai = _ai(schema)
    s = ai.get("summary") or {}
    if not cards:
        return str(s.get("method_rule", "{why}")).format(why=s.get("why_no_cards", ""))
    if client is None:
        return str(s.get("method_rule", "{why}")).format(why=s.get("why_no_key", ""))
    fell_back = {m["id"] for m in mismatches}
    n_ai = sum(c["id"] not in fell_back for c in cards)
    return str(s.get("method_ai", "")).format(model=ai["model"], ai=n_ai, total=len(cards))
