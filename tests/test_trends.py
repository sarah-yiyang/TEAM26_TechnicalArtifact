"""M3 tests for pipeline/trends.py: mix adjustment and data-health alerts."""
from __future__ import annotations

import copy
from typing import Any

import pandas as pd
import pytest

from pipeline.cleaning import clean
from pipeline.ingest import load_files
from pipeline.mapping import map_columns
from pipeline.quality import add_flags
from pipeline.trends import alerts, monthly


@pytest.fixture
def sample(sample_path, schema) -> pd.DataFrame:
    mapped = map_columns(load_files([sample_path], schema)[0], schema)[0]
    return add_flags(clean(mapped, schema)[0], schema)


@pytest.fixture
def low_n(schema) -> dict[str, Any]:
    s = copy.deepcopy(schema)
    s["analysis"]["min_n"] = 1
    return s


def _people(month: str, role: str, n: int, agree: int) -> pd.DataFrame:
    return pd.DataFrame({"month": [month] * n, "role_primary": [role] * n,
                         "post_when": [5] * agree + [2] * (n - agree)})


def _memo_example() -> pd.DataFrame:
    """The memo example: same per-role rates, only the mix changes."""
    df = pd.concat([
        _people("2026-09", "Student", 80, 72), _people("2026-09", "School staff", 20, 12),
        _people("2026-10", "Student", 20, 18), _people("2026-10", "School staff", 80, 48),
    ], ignore_index=True)
    df["post_when"] = df["post_when"].astype("Int64")
    return df


# ----- monthly -------------------------------------------------------------

def test_memo_example_mix_only_change_is_flat(schema) -> None:
    t = monthly(_memo_example(), "agree_pct.post_when", schema=schema).set_index("month")
    assert t.loc["2026-09", "raw"] == pytest.approx(0.84)
    assert t.loc["2026-10", "raw"] == pytest.approx(0.66)
    assert t.loc["2026-09", "adjusted"] == pytest.approx(0.75)
    assert t.loc["2026-10", "adjusted"] == pytest.approx(0.75)


def test_real_change_survives_adjustment(schema) -> None:
    df = _memo_example()
    # staff in October do worse than usual: 40 of 80 instead of 48
    oct_staff = df.index[(df.month == "2026-10") & (df.role_primary == "School staff")]
    df.loc[oct_staff[:8], "post_when"] = 2
    t = monthly(df, "agree_pct.post_when", schema=schema).set_index("month")
    assert t.loc["2026-10", "adjusted"] < t.loc["2026-09", "adjusted"]


def test_single_role_months_still_adjusted(schema) -> None:
    df = pd.concat([_people("2026-09", "Student", 40, 36),
                    _people("2026-10", "School staff", 40, 24)], ignore_index=True)
    df["post_when"] = df["post_when"].astype("Int64")
    t = monthly(df, "agree_pct.post_when", schema=schema)
    assert t["adjusted"].notna().all()
    # gap fully explained by role: both months adjust to the overall 60/80
    assert list(t["adjusted"]) == pytest.approx([0.75, 0.75])


def test_small_month_suppressed(schema) -> None:
    df = pd.concat([_people("2026-09", "Student", 30, 27),
                    _people("2026-10", "Student", 5, 5)], ignore_index=True)
    df["post_when"] = df["post_when"].astype("Int64")
    t = monthly(df, "agree_pct.post_when", schema=schema).set_index("month")
    assert t.loc["2026-10", "suppressed"] and pd.isna(t.loc["2026-10", "raw"])
    assert t.loc["2026-10", "rows"] == 5


def test_small_role_gets_overall_rate(schema) -> None:
    # 3 caregivers (< MIN_N) all disagree; they must not set their own expectation
    df = pd.concat([_people("2026-09", "Student", 40, 30),
                    _people("2026-09", "Caregiver", 3, 0)], ignore_index=True)
    df["post_when"] = df["post_when"].astype("Int64")
    t = monthly(df, "agree_pct.post_when", schema=schema)
    overall = 30 / 43
    # students keep their own usual rate (0.75); caregivers get the overall rate
    assert t.loc[0, "expected"] == pytest.approx((40 * 0.75 + 3 * overall) / 43)


def test_no_role_column_adjusted_blank(schema) -> None:
    df = _memo_example().drop(columns="role_primary")
    t = monthly(df, "agree_pct.post_when", schema=schema)
    assert t["raw"].notna().all() and t["adjusted"].isna().all()


def test_not_collected_and_bad_metric(schema) -> None:
    assert monthly(_memo_example().drop(columns="month"), "agree_pct.post_when",
                   schema=schema).empty
    assert monthly(_memo_example(), "nps", schema=schema).empty
    with pytest.raises(ValueError):
        monthly(_memo_example(), "made_up", schema=schema)


def test_sample_runs_and_untouched(sample, low_n) -> None:
    before = sample.copy()
    t = monthly(sample, "nps", schema=low_n)
    assert t["n"].sum() == sample.loc[sample.month.notna(), "nps"].notna().sum()
    pd.testing.assert_frame_equal(sample, before)


# ----- alerts --------------------------------------------------------------

def _months(spec: dict[str, tuple[int, int]], field: str = "presenter_engaged") -> pd.DataFrame:
    """month -> (rows, blank rows) for one question."""
    month, val = [], []
    for m, (n, blank) in spec.items():
        month += [m] * n
        val += [None] * blank + [4] * (n - blank)
    return pd.DataFrame({"month": month, field: pd.array(val, dtype="Int64")})


def test_blank_alert_found(schema) -> None:
    df = _months({"2026-08": (30, 0), "2026-09": (30, 1), "2026-10": (30, 15)})
    found = [a for a in alerts(df, schema) if a["kind"] == "blank_answers"]
    assert len(found) == 1 and found[0]["field"] == "presenter_engaged"
    assert [m["month"] for m in found[0]["months"]] == ["2026-10"]
    assert found[0]["months"][0]["blank_pct"] == pytest.approx(0.5)


def test_blank_alert_not_found(schema) -> None:
    df = _months({"2026-08": (30, 3), "2026-09": (30, 4), "2026-10": (30, 5)})
    assert not [a for a in alerts(df, schema) if a["kind"] == "blank_answers"]


def test_blank_alert_ignores_small_months(schema) -> None:
    df = _months({"2026-08": (30, 0), "2026-09": (30, 0), "2026-10": (5, 5)})
    assert not alerts(df, schema)


def _comments(spec: dict[str, tuple[int, int]]) -> pd.DataFrame:
    month, text = [], []
    for m, (n, wrote) in spec.items():
        month += [m] * n
        text += ["idea"] * wrote + [None] * (n - wrote)
    return pd.DataFrame({"month": month, "comment_improve": text})


def test_comment_alert_found(schema) -> None:
    df = _comments({"2026-08": (40, 30), "2026-09": (40, 28), "2026-10": (40, 8)})
    found = [a for a in alerts(df, schema) if a["kind"] == "fewer_comments"]
    assert len(found) == 1 and found[0]["month"] == "2026-10"
    assert found[0]["rate"] == pytest.approx(0.2)
    assert found[0]["earlier_rate"] == pytest.approx(58 / 80)


def test_comment_alert_not_found(schema) -> None:
    df = _comments({"2026-08": (40, 30), "2026-09": (40, 28), "2026-10": (40, 26)})
    assert not [a for a in alerts(df, schema) if a["kind"] == "fewer_comments"]


def test_alerts_no_month_column(schema) -> None:
    assert alerts(pd.DataFrame({"post_when": [5]}), schema) == []
