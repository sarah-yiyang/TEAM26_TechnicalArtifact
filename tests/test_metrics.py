"""M2 tests for pipeline/metrics.py: hand-built tables, blank traps, sample file."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pipeline.cleaning import clean
from pipeline.ingest import load_files
from pipeline.mapping import map_columns
from pipeline.metrics import NOT_COLLECTED, TOO_FEW, compute
from pipeline.quality import add_flags

LIKERT = ["pre_identify", "pre_describe", "pre_when", "post_identify", "post_describe",
          "post_when", "presenter_engaged", "presenter_knew", "well_organized"]


@pytest.fixture
def low_n(schema) -> dict[str, Any]:
    """Schema with min_n = 1 so tiny hand-built tables are not suppressed."""
    s = copy.deepcopy(schema)
    s["analysis"]["min_n"] = 1
    return s


@pytest.fixture
def sample(sample_path, schema) -> pd.DataFrame:
    """Sample after ingest + mapping + cleaning + flags."""
    mapped = map_columns(load_files([sample_path], schema)[0], schema)[0]
    return add_flags(clean(mapped, schema)[0], schema)


def _table(n: int, **cols: list[Any]) -> pd.DataFrame:
    """n rows; every Likert = 3 and nps/likely_call = 7 unless overridden."""
    base: dict[str, list[Any]] = {c: [3] * n for c in LIKERT}
    base.update({"nps": [7] * n, "likely_call": [7] * n, "month": ["2026-01"] * n,
                 "comment_valuable": [None] * n, "comment_improve": [None] * n})
    base.update(cols)
    df = pd.DataFrame(base)
    for c in LIKERT + ["nps", "likely_call"]:
        if c in df.columns:
            df[c] = df[c].astype("Int64")
    return df


def _v(m: dict[str, Any], key: str) -> Any:
    return m[key]["value"]


# ----- blanks are never disagree ----------------------------------------

def test_agree_pct_ignores_blanks(low_n) -> None:
    m = compute(_table(3, post_when=[5, None, 2]), low_n)
    assert _v(m, "agree_pct.post_when") == pytest.approx(0.5)
    assert m["agree_pct.post_when"]["n"] == 2


def test_nps_blanks_not_detractors(low_n) -> None:
    m = compute(_table(4, nps=[10, 10, 0, None]), low_n)
    assert _v(m, "nps") == pytest.approx(100 * (2 / 3 - 1 / 3))
    assert m["nps"]["n"] == 3


def test_know_call_gap_ignores_blank_likely_call(low_n) -> None:
    # 2 answered: one knows+calls (5, 10), one knows-not-calls (5, 6); 2 blanks
    df = _table(4, post_when=[5, 5, 5, 5], likely_call=[10, 6, None, None])
    m = compute(df, low_n)
    assert _v(m, "know_call_gap") == pytest.approx(0.5)
    assert m["know_call_gap"]["n"] == 2
    assert _v(m, "know_call_2x2") == {"knows_and_calls": 1, "knows_not_calls": 1,
                                      "not_knows_calls": 0, "not_knows_not_calls": 0}


def test_likely_call_pct_and_coverage(low_n) -> None:
    m = compute(_table(4, likely_call=[9, 10, 5, None]), low_n)
    assert _v(m, "likely_call.pct_high") == pytest.approx(2 / 3)
    assert _v(m, "likely_call.mean") == pytest.approx(8.0)
    assert _v(m, "likely_call.coverage") == pytest.approx(0.75)
    assert m["likely_call.coverage"]["answered"] == 3


def test_delivery_needs_all_three_answered(low_n) -> None:
    df = _table(3, presenter_engaged=[5, 5, None], presenter_knew=[5, 2, 5],
                well_organized=[4, 5, 5])
    m = compute(df, low_n)
    assert _v(m, "delivery_agree_pct") == pytest.approx(0.5)
    assert m["delivery_agree_pct"]["n"] == 2


# ----- before / after ----------------------------------------------------

def test_pre_post_change_declined_ceiling(low_n) -> None:
    df = _table(4, pre_when=[1, 5, 4, 5], post_when=[5, 5, 3, None])
    m = compute(df, low_n)
    assert _v(m, "mean_change.when") == pytest.approx((4 + 0 - 1) / 3)
    assert m["mean_change.when"]["n"] == 3
    assert _v(m, "declined_pct.when") == pytest.approx(1 / 3)
    assert _v(m, "ceiling_pct.when") == pytest.approx(0.5)   # pre = 5 in 2 of 4
    gain = _v(m, "gain_by_baseline.when")
    assert gain[1]["value"] == 4 and gain[5] == {"value": 0.0, "n": 1}


def test_wilcoxon_none_when_no_change(low_n) -> None:
    m = compute(_table(5), low_n)
    assert _v(m, "wilcoxon_p.when") is None


def test_wilcoxon_small_when_all_improve(low_n) -> None:
    m = compute(_table(25, pre_when=[2] * 25, post_when=[5] * 25), low_n)
    assert _v(m, "wilcoxon_p.when") < 0.001


# ----- NPS bootstrap -----------------------------------------------------

def test_nps_ci_reproducible_and_contains_score(low_n) -> None:
    rng = np.random.default_rng(0)
    df = _table(200, nps=list(rng.integers(0, 11, 200)))
    a, b = compute(df, low_n)["nps"], compute(df, low_n)["nps"]
    assert a["ci95"] == b["ci95"]
    assert a["ci95"][0] <= a["value"] <= a["ci95"][1]


# ----- intent drivers ----------------------------------------------------

def test_intent_driver_spearman(low_n) -> None:
    df = _table(5, presenter_engaged=[1, 2, 3, 4, 5], likely_call=[2, 4, 6, 8, 10])
    m = compute(df, low_n)
    assert _v(m, "intent_driver.presenter_engaged") == pytest.approx(1.0)
    assert "intent_driver.pre_when" not in m           # "before" items are not drivers


def test_intent_driver_constant_cannot_be_calculated(low_n) -> None:
    m = compute(_table(5), low_n)
    assert m["intent_driver.post_when"]["value"] is None


# ----- comments ----------------------------------------------------------

def test_comment_rate_by_month(low_n) -> None:
    df = _table(4, month=["2026-01", "2026-01", "2026-02", None],
                comment_improve=["more videos", None, "nothing", "x"])
    table = _v(compute(df, low_n), "comment_rate_by_month.comment_improve")
    assert table["2026-01"] == {"value": 0.5, "n": 2}
    assert table["2026-02"] == {"value": 1.0, "n": 1}
    assert set(table) == {"2026-01", "2026-02"}          # no-month row not grouped


# ----- not collected / too few / flags ------------------------------------

def test_missing_likely_call_not_collected_rest_works(low_n) -> None:
    m = compute(_table(3).drop(columns="likely_call"), low_n)
    for key in ("likely_call.mean", "likely_call.coverage", "know_call_gap",
                "intent_driver.post_when"):
        assert m[key] == {"value": None, "n": 0, "status": NOT_COLLECTED}
    assert _v(m, "agree_pct.post_when") == 0.0
    assert m["nps"]["value"] is not None


def test_too_few_responses(schema) -> None:
    m = compute(_table(5), schema)                      # real MIN_N = 20
    assert m["agree_pct.post_when"] == {"value": None, "n": 5, "status": TOO_FEW}
    assert m["nps"]["status"] == TOO_FEW


def test_exclude_flagged(low_n) -> None:
    df = _table(3, post_when=[5, 5, 1])
    df["flag_any"] = [False, False, True]
    m = compute(df, low_n, exclude_flagged=True)
    assert m["rows"]["value"] == 2
    assert _v(m, "agree_pct.post_when") == 1.0


def test_exclude_flagged_without_flags_errors(low_n) -> None:
    with pytest.raises(ValueError, match="add_flags"):
        compute(_table(2), low_n, exclude_flagged=True)


# ----- sample file -------------------------------------------------------

def test_sample_matches_independent_pandas(sample, schema) -> None:
    m = compute(sample, schema)
    df = sample
    assert m["rows"]["value"] == 29
    assert _v(m, "agree_pct.post_describe") == pytest.approx(
        df.post_describe.dropna().ge(4).mean())          # unknown "Somewhat agree" left out
    assert m["agree_pct.post_describe"]["n"] == 28
    assert m["agree_pct.presenter_engaged"]["n"] == 24   # 5 blank presenter blocks
    assert _v(m, "mean_change.when") == pytest.approx(
        (df.post_when - df.pre_when).dropna().mean())
    v = df.nps.dropna()
    assert _v(m, "nps") == pytest.approx(100 * (v.ge(9).mean() - v.le(6).mean()))
    assert m["nps"]["n"] == 27
    assert m["likely_call.mean"]["status"] == TOO_FEW     # only 9 answers
    assert m["likely_call.coverage"]["answered"] == 9


def test_sample_input_untouched(sample, schema) -> None:
    before = sample.copy()
    compute(sample, schema, exclude_flagged=True)
    pd.testing.assert_frame_equal(sample, before)


def test_sample_exclude_flagged_removes_three(sample, schema) -> None:
    assert compute(sample, schema, exclude_flagged=True)["rows"]["value"] == 26
