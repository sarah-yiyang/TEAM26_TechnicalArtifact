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
    labels = [e.label for e in at.expander]
    for name in ("Files", "Preview", "Rows loaded", "Column matching", "Fixes and unknown values"):
        assert any(lbl.startswith(name) for lbl in labels)
    values = {m.label: m.value for m in at.metric}
    assert values["Questions matched"] == "20"
    assert values["Questions missing"] == "0"
    assert values["Columns not recognized"] == "0"
    # the hand-edited "Somewhat agree" is flagged for review
    assert any("not recognized" in w.value for w in at.warning)


def test_app_shows_m2_headline_and_toggle() -> None:
    at = AppTest.from_file(APP)
    at.session_state["use_sample"] = True
    at.session_state["section"] = 2
    at.run(timeout=30)
    assert not at.exception
    labels = [e.label for e in at.expander]
    for h in ("Rushed or doubtful answers", "Headline numbers", "Before and after",
              "Know when to use 988 vs. likely to call", "What changed for each person"):
        assert any(lbl.startswith(h) for lbl in labels)
    values = {m.label: m.value for m in at.metric}
    assert values["Know when to use 988"] == "97%"          # 28 of 29 on the sample
    # sample has only 9 likely_call answers -> suppressed, not a number
    assert values["Very likely to call 988"].startswith("too few responses")
    assert any("Not shown" in i.value for i in at.info)     # 2x2 suppressed too
    assert "Leave out the 3 flagged" in at.toggle[0].label

    at.toggle[0].set_value(True).run(timeout=30)
    assert not at.exception
    assert any("n = 26" in c.value for c in at.caption)


def test_app_shows_m3_sections() -> None:
    at = AppTest.from_file(APP)
    at.session_state["use_sample"] = True
    at.session_state["section"] = 3
    at.run(timeout=30)
    assert not at.exception
    labels = [e.label for e in at.expander]
    for h in ("Compare groups", "Sessions compared with the average",
              "Monthly trend, adjusted for who attended", "Data health"):
        assert any(lbl.startswith(h) for lbl in labels)
    # sample is tiny: every session is hidden rather than scored
    assert any("no session has enough answers" in i.value for i in at.info)

    for by in ("role_primary", "county", "language", "month", "session_id"):
        at.selectbox(key="m3_by").set_value(by).run(timeout=30)
        assert not at.exception
    for metric in ("nps", "likely_call.pct_high", "delivery_agree_pct", "mean_change.when"):
        at.selectbox(key="m3_metric").set_value(metric).run(timeout=30)
        assert not at.exception
    assert any("Flagged responses are included" in c.value for c in at.caption)


def test_app_shows_m4_insights() -> None:
    at = AppTest.from_file(APP)
    at.session_state["use_sample"] = True
    at.session_state["section"] = 4
    at.run(timeout=30)
    assert not at.exception
    assert "What to act on" in [h.value for h in at.header]
    assert any(e.label == "Checks run" for e in at.expander)
    text = " ".join(m.value for m in at.markdown)
    assert "Runs once comments are sorted into themes" not in text   # F and J run from M5
    assert "Only 9 people answered both" in text                 # B: says what fell short
    assert "Coming in milestone 4" not in text
    assert not any("decisions.yaml needs attention" in w.value for w in at.warning)


def _sample(section: int | None = None) -> AppTest:
    at = AppTest.from_file(APP)
    at.session_state["use_sample"] = True
    if section is not None:
        at.session_state["section"] = section
    return at.run(timeout=30)


def test_app_nav_hidden_without_data() -> None:
    at = AppTest.from_file(APP).run(timeout=30)
    assert not at.exception
    assert len(at.radio) == 0


def test_app_nav_lists_every_section_and_starts_on_data_check() -> None:
    at = _sample()
    assert not at.exception
    assert at.radio(key="section").options == ["1  Data check", "2  Headline results",
                                               "3  Comparisons and trends", "4  What to act on",
                                               "5  Comment board",
                                               "6  Summary  · coming soon"]
    assert [h.value for h in at.header] == ["Data check"]


def test_app_shows_only_the_picked_section() -> None:
    at = _sample()
    for n, title in ((2, "Headline results"), (3, "Comparisons and trends"),
                     (4, "What to act on"), (1, "Data check")):
        at.radio(key="section").set_value(n).run(timeout=30)
        assert not at.exception
        assert [h.value for h in at.header] == [title]


def test_app_unbuilt_section_shows_placeholder() -> None:
    at = _sample(6)
    assert not at.exception
    assert [h.value for h in at.header] == ["Summary"]
    assert any("Coming in milestone 6" in m.value for m in at.markdown)


def test_app_shows_comment_board() -> None:
    at = _sample(5)
    assert not at.exception
    assert [h.value for h in at.header] == ["Comment board"]
    subs = [s.value for s in at.subheader]
    assert subs == ["Keep · n = 22", "Fix · n = 5", "Add · n = 5"]
    assert any("keyword rules" in c.value for c in at.caption)
    assert len(at.dataframe) == 3


def test_app_flag_toggle_carries_to_other_sections() -> None:
    at = _sample(2)
    at.toggle(key="exclude_flagged").set_value(True).run(timeout=30)
    at.radio(key="section").set_value(4).run(timeout=30)
    assert not at.exception
    assert any("Flagged responses are left out" in c.value for c in at.caption)
    at.radio(key="section").set_value(2).run(timeout=30)
    assert at.toggle(key="exclude_flagged").value is True
