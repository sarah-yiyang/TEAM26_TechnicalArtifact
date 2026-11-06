"""Smoke test for app.py: runs headless with the sample, no network."""
from __future__ import annotations

from streamlit.testing.v1 import AppTest

from pipeline import ROOT

APP = str(ROOT / "app.py")


def test_app_starts_without_data() -> None:
    at = AppTest.from_file(APP).run(timeout=30)
    assert not at.exception
    assert any("Upload a survey export" in i.value for i in at.info)
    assert len(at.metric) == 0


def test_app_shows_sample_counts() -> None:
    at = AppTest.from_file(APP)
    at.session_state["use_sample"] = True
    at.run(timeout=30)
    assert not at.exception
    assert not at.error
    values = {m.label: m.value for m in at.metric}
    assert {k: values[k] for k in ("Rows read", "Duplicates removed",
                                   "Blank rows removed", "Rows kept")} == {
        "Rows read": "30",
        "Duplicates removed": "1",
        "Blank rows removed": "0",
        "Rows kept": "29",
    }


def test_app_shows_mapping_and_cleaning() -> None:
    at = AppTest.from_file(APP)
    at.session_state["use_sample"] = True
    at.run(timeout=30)
    assert not at.exception
    headers = [h.value for h in at.subheader]
    assert "Column matching" in headers and "Fixes and unknown values" in headers
    values = {m.label: m.value for m in at.metric}
    assert values["Questions matched"] == "20"
    assert values["Questions missing"] == "0"
    assert values["Columns not recognized"] == "0"
    # the hand-edited "Somewhat agree" is flagged for review
    assert any("not recognized" in w.value for w in at.warning)


def test_app_shows_m2_headline_and_toggle() -> None:
    at = AppTest.from_file(APP)
    at.session_state["use_sample"] = True
    at.run(timeout=30)
    assert not at.exception
    headers = [h.value for h in at.subheader]
    for h in ("Rushed or doubtful answers", "Headline numbers",
              "Know when to use 988 vs. likely to call", "Change by starting level"):
        assert h in headers
    values = {m.label: m.value for m in at.metric}
    assert values["Know when to use 988"] == "97%"          # 28 of 29 on the sample
    # sample has only 9 likely_call answers -> suppressed, not a number
    assert values["Very likely to call 988"].startswith("too few responses")
    assert any("Not shown" in i.value for i in at.info)     # 2x2 suppressed too
    assert "Leave out the 3 flagged" in at.toggle[0].label

    at.toggle[0].set_value(True).run(timeout=30)
    assert not at.exception
    assert any("n = 26" in c.value for c in at.caption)
