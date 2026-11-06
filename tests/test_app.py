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
