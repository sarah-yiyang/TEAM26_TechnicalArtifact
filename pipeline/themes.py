"""Sort open-ended comments into a topic and a kind (M5a: keyword rules only).

All phrases come from the ``themes`` block of ``config/schema.yaml``; this
module holds no survey wording. Functions are pure: no file I/O, no network.

- ``classify``: list of ``(field, text)`` -> ``[{topic, kind, method}]``
- ``classify_df``: one row per comment cell, for the board and checks F / J
- ``check_input``: the ``themes`` argument ``recommend.run_checks`` expects
- ``board``: Keep / Fix / Add topic counts
"""
from __future__ import annotations

import re
import string
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.ingest import normalize_text

KINDS: tuple[str, ...] = ("suggestion", "open_question", "praise", "empty")
METHOD_KEYWORDS: str = "keywords"
_PUNCT = re.compile(f"[{re.escape(string.punctuation)}¿¡]")


def _rules(schema: dict[str, Any]) -> dict[str, Any]:
    rules = schema.get("themes")
    if not isinstance(rules, dict):
        raise KeyError("config/schema.yaml is missing the 'themes' block")
    return rules


def _norm(text: Any, schema: dict[str, Any]) -> str:
    """Lowercase, trim, collapse spaces, drop hidden characters and accents."""
    norm = dict(schema.get("normalization", {}) or {})
    norm.update(lowercase=True, strip=True, collapse_whitespace=True)
    text = normalize_text(text, norm)
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _has(text: str, phrases: list[str], schema: dict[str, Any]) -> bool:
    """True when any phrase appears at the start of a word in ``text``."""
    for phrase in phrases:
        p = _norm(phrase, schema)
        if not p:
            continue
        lead = r"(?<!\w)" if p[0].isalnum() else ""
        if re.search(lead + re.escape(p), text):
            return True
    return False


def _starts_question(text: str, words: list[str], schema: dict[str, Any]) -> bool:
    """Starts with a question word that is not followed by "to"."""
    for word in words:
        w = _norm(word, schema)
        if text == w:
            return True
        if text.startswith(w + " "):
            nxt = text[len(w) + 1:].split(" ", 1)[0]
            return _PUNCT.sub("", nxt) != "to"
    return False


def _is_blank(text: Any) -> bool:
    if text is None:
        return True
    try:
        return bool(pd.isna(text))
    except (TypeError, ValueError):
        return False


def _kind(field: str, text: str, rules: dict[str, Any], schema: dict[str, Any]) -> str:
    k = rules.get("kinds", {}) or {}
    bare = " ".join(_PUNCT.sub(" ", text).split())
    empty = {_norm(w, schema) for w in k.get("empty", []) or []}
    if not bare or bare in empty:
        return "empty"
    if _has(text, k.get("question_contains", []) or [], schema):
        return "open_question"
    starts = k.get("question_starts", {}) or {}
    if field in (starts.get("fields") or []) and _starts_question(
            text, starts.get("words", []) or [], schema):
        return "open_question"
    if field in (k.get("overrides_fields") or []):
        if _has(text, k.get("suggestion_contains", []) or [], schema):
            return "suggestion"
        if _has(text, k.get("praise_contains", []) or [], schema):
            return "praise"
    kind = (k.get("default_by_field", {}) or {}).get(field, "suggestion")
    if kind not in KINDS:
        raise ValueError(f"themes.kinds.default_by_field.{field}: unknown kind {kind!r}")
    return kind


def _topic(text: str, rules: dict[str, Any], schema: dict[str, Any]) -> str:
    for name, phrases in (rules.get("topics", {}) or {}).items():
        if _has(text, phrases or [], schema):
            return str(name)
    return str(rules.get("other_topic", "Other"))


def classify(
    comments: list[tuple[str, Any]],
    schema: dict[str, Any],
    cache_path: str | Path | None = None,
    use_llm: bool = False,
) -> list[dict[str, str]]:
    """Give each ``(field, text)`` a topic and kind using the schema's keyword rules.

    ``cache_path`` and ``use_llm`` are accepted for the later AI step (M5b) and
    are ignored for now: no network, ``method`` is always "keywords".
    """
    rules = _rules(schema)
    other = str(rules.get("other_topic", "Other"))
    out: list[dict[str, str]] = []
    for field, raw in comments:
        text = "" if _is_blank(raw) else _norm(raw, schema)
        kind = _kind(field, text, rules, schema)
        topic = other if kind == "empty" else _topic(text, rules, schema)
        out.append({"topic": topic, "kind": kind, "method": METHOD_KEYWORDS})
    return out


def comment_fields(df: pd.DataFrame, schema: dict[str, Any]) -> list[str]:
    """Comment fields from the schema that are present in ``df``."""
    return [f for f in _rules(schema).get("comment_fields", []) or [] if f in df.columns]


def classify_df(df: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    """One row per (response, comment field): ``row, field, text, topic, kind, method``.

    ``row`` is the index label in ``df``. Blank answers are kept as kind "empty".
    """
    fields = comment_fields(df, schema)
    pairs = [(idx, f, df.at[idx, f]) for f in fields for idx in df.index]
    labels = classify([(f, t) for _, f, t in pairs], schema)
    rows = [{"row": idx, "field": f, "text": None if _is_blank(t) else t, **lab}
            for (idx, f, t), lab in zip(pairs, labels)]
    return pd.DataFrame(rows, columns=["row", "field", "text", "topic", "kind", "method"])


def check_input(
    classified: pd.DataFrame, df: pd.DataFrame, schema: dict[str, Any]
) -> dict[str, Any] | None:
    """Build the ``themes`` input for checks F and J.

    Low rater = NPS at or below ``analysis.nps_detractor_max``; silent = every
    comment answer from that person is empty. Returns ``None`` when the file has
    no comment questions.
    """
    if not comment_fields(df, schema):
        return None
    cutoff = int(schema["analysis"]["nps_detractor_max"])
    low_n = silent = 0
    if "nps" in df.columns:
        nps = pd.to_numeric(df["nps"], errors="coerce")
        low_rows = set(nps[nps <= cutoff].index)
        low_n = len(low_rows)
        said = set(classified.loc[classified["kind"] != "empty", "row"])
        silent = len(low_rows - said)
    return {"low_raters": {"cutoff": cutoff, "n": low_n, "silent": silent},
            "comments": classified[["topic", "kind"]].to_dict("records")}


def board(classified: pd.DataFrame, schema: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Keep / Fix / Add columns: ``[{topic, n}]`` sorted by n, then topic."""
    out: dict[str, list[dict[str, Any]]] = {}
    for column, kind in (_rules(schema).get("board", {}) or {}).items():
        sub = classified[classified["kind"] == kind]
        counts = sub["topic"].value_counts()
        items = [{"topic": str(t), "n": int(n)} for t, n in counts.items()]
        out[str(column)] = sorted(items, key=lambda x: (-x["n"], x["topic"]))
    return out


__all__ = ["classify", "classify_df", "check_input", "board", "comment_fields",
           "KINDS", "METHOD_KEYWORDS"]
