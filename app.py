"""EveryMind 988 Training Insights: Streamlit front end (display only).

All calculations live in ``pipeline/``. This file only calls pipeline
functions and shows what they return.
"""
from __future__ import annotations

import io
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline import ROOT, load_schema
from pipeline.cleaning import clean
from pipeline.ingest import load_files
from pipeline.mapping import map_columns
from pipeline.metrics import compute, drop_flagged
from pipeline.quality import add_flags, flag_summary
from pipeline.segments import NOT_RECORDED, compare
from pipeline.trends import alerts, monthly

SAMPLE_PATH = ROOT / "data" / "sample_30rows.xlsx"
PREVIEW_ROWS = 20
BAR_COLOR = "#2a78d6"
ACCENT_COLOR = "#eb6834"
MUTED_COLOR = "rgba(128,128,128,0.6)"
METRIC_LABELS: dict[str, tuple[str, str]] = {
    "agree_pct.post_when": ("Know when to use 988 (agree)", "pct"),
    "mean_change.when": ("Self-reported change, 'know when to use 988' (1–5)", "change"),
    "nps": ("Would recommend (NPS)", "num"),
    "likely_call.pct_high": ("Very likely to call 988 (9–10)", "pct"),
    "delivery_agree_pct": ("Liked the delivery (all 3 items)", "pct"),
}
GROUP_LABELS = {"role_primary": "Role", "county": "County", "language": "Language",
                "month": "Month", "session_id": "Session (date + county)"}
FIELD_LABELS = {
    "pre_identify": "Could identify challenges (before)", "pre_describe": "Could describe 988 (before)",
    "pre_when": "Knew when to use 988 (before)", "post_identify": "Can identify challenges (after)",
    "post_describe": "Can describe 988 (after)", "post_when": "Know when to use 988 (after)",
    "presenter_engaged": "Presenter engaged participants", "presenter_knew": "Presenter knew the subject",
    "well_organized": "Training well organized", "nps": "Would recommend",
    "likely_call": "Likely to call 988", "comment_valuable": "'Most valuable' comment",
    "comment_improve": "'How to improve' comment",
}
KNOWLEDGE_LABELS = {"identify": "Can identify mental health challenges",
                    "describe": "Can describe 988 and how it works",
                    "when": "Knows when to use 988"}

LATER_MILESTONES: list[tuple[int, str, str]] = [
    (4, "Insight cards", "Automatic checks on the data, with the evidence behind each finding."),
    (5, "Comment board", "Comments sorted into Keep / Fix / Add themes."),
    (6, "Summary", "A copy-ready written summary of the findings."),
]


# ----- loading (cached on file contents) ------------------------------

def _run_pipeline(files: list[Any]) -> dict[str, Any]:
    """Call ingest -> mapping -> cleaning and collect their outputs.

    Cleaning is skipped when a required column is missing.
    """
    schema = load_schema()
    raw, stats = load_files(files, schema)
    mapped, mapping = map_columns(raw, schema)
    result: dict[str, Any] = {"raw": raw, "stats": stats, "mapping": mapping,
                              "clean": None, "fix_log": [], "unknown": []}
    if not mapping["missing_required"]:
        cleaned, result["fix_log"], result["unknown"] = clean(mapped, schema)
        result["clean"] = add_flags(cleaned, schema)
        result["flags"] = flag_summary(result["clean"])
        result["metrics"] = {
            False: compute(result["clean"], schema),
            True: compute(result["clean"], schema, exclude_flagged=True),
        }
    return result


@st.cache_data(show_spinner="Reading files...")
def _load_uploads(payload: tuple[tuple[str, bytes], ...]) -> dict[str, Any]:
    """Run the pipeline on uploaded files, given as (name, bytes) pairs."""
    files = []
    for name, data in payload:
        buf = io.BytesIO(data)
        buf.name = name
        files.append(buf)
    return _run_pipeline(files)


@st.cache_data(show_spinner="Reading sample...")
def _load_sample(path: str) -> dict[str, Any]:
    """Run the pipeline on the bundled 30-row sample."""
    return _run_pipeline([path])


# ----- display helpers ------------------------------------------------

def _placeholder(milestone: int, title: str, text: str, part: str | None = None) -> None:
    """Grey 'coming soon' box for features not built yet."""
    when = f"later in milestone {milestone} ({part})" if part else f"in milestone {milestone}"
    st.markdown(
        f"""<div style="background:#f0f2f6;color:#555;border-radius:8px;
        padding:12px 16px;margin-bottom:10px">
        <b>{title}</b> &nbsp;·&nbsp; <i>Coming {when}</i><br>{text}</div>""",
        unsafe_allow_html=True,
    )


def _show_counts(stats: dict[str, Any]) -> None:
    """Row-count tiles with plain-English captions."""
    st.subheader("Rows loaded")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows read", stats["rows_in"])
    c1.caption("All rows in the uploaded files.")
    c2.metric("Duplicates removed", stats["duplicates_removed"])
    c2.caption("Same response ID and start time as an earlier row.")
    c3.metric("Blank rows removed", stats["blank_rows_removed"])
    c3.caption("Rows with no answers at all.")
    c4.metric("Rows kept", stats["rows_out"])
    c4.caption("Rows used from here on.")


def _show_files(stats: dict[str, Any]) -> None:
    """Rows read from each file."""
    st.subheader("Files")
    files = pd.DataFrame(
        [{"File": name, "Rows read": n} for name, n in stats["files"].items()]
    )
    st.dataframe(files, hide_index=True, width="stretch")


def _show_health(stats: dict[str, Any]) -> None:
    """Data-health notes, only when they apply."""
    if stats["dedupe_skipped"]:
        st.warning(stats["dedupe_note"] + ". All rows were kept.")
    if stats["blank_key_rows"]:
        st.info(
            f"{stats['blank_key_rows']} of {stats['rows_out']} rows had no ID or start "
            "time, so they were kept and not checked for duplicates."
        )


def _show_mapping(mapping: dict[str, Any]) -> None:
    """Which survey questions were found, missing, or not recognized."""
    st.subheader("Column matching")
    matched, missing, unrec = mapping["matched"], mapping["missing"], mapping["unrecognized"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Questions matched", len(matched))
    c1.caption(f"of {len(matched) + len(missing)} expected questions.")
    c2.metric("Questions missing", len(missing))
    c2.caption("Expected but not found. Their results show as \"not collected\".")
    c3.metric("Columns not recognized", len(unrec))
    c3.caption("Kept, but not used until the settings file is updated.")

    if mapping["missing_required"]:
        st.error(
            "These required columns were not found, so the data cannot be used: "
            + ", ".join(mapping["missing_required"])
            + ". Check that this is a 988 training survey export."
        )

    fuzzy = [m for m in matched if m["method"] == "fuzzy"]
    if fuzzy:
        st.warning(
            f"{len(fuzzy)} question(s) were matched by close wording, not exact keywords. "
            "Please check them in the table below."
        )
    with st.expander(f"Matched questions ({len(matched)})"):
        table = pd.DataFrame([
            {"Standard name": m["field"], "Column in file": m["column"],
             "How matched": "close wording" if m["method"] == "fuzzy" else "keywords",
             "Similarity": m["score"]}
            for m in matched
        ])
        st.dataframe(table, hide_index=True, width="stretch")
    if missing:
        st.markdown("**Missing:** " + ", ".join(
            m["field"] + (" (required)" if m["required"] else "") for m in missing))
    if unrec:
        st.markdown("**Not recognized:**")
        st.dataframe(
            pd.DataFrame([{"Column in file": u["column"], "Why": u["reason"]} for u in unrec]),
            hide_index=True, width="stretch",
        )


def _show_cleaning(fix_log: list[dict[str, Any]], unknown: list[dict[str, Any]]) -> None:
    """Fixes made to answers, and answers that were not recognized."""
    st.subheader("Fixes and unknown values")
    st.markdown("**Fixes made**")
    if fix_log:
        st.dataframe(
            pd.DataFrame([
                {"Question(s)": e["field"], "Change": e["change"], "Answers": e["cells"],
                 "Respondents": e["rows"], "Assumption": "⚠ yes" if e["assumption"] else ""}
                for e in fix_log
            ]),
            hide_index=True, width="stretch",
        )
        st.caption("⚠ Assumption = a judgment call set in the settings file; review if unsure.")
    else:
        st.success("✓ No fixes were needed.")

    st.markdown("**Unknown values**")
    if unknown:
        st.warning(
            "These answers were not recognized, so they were left blank in the results. "
            "Add them to the settings file if they should count."
        )
        st.dataframe(
            pd.DataFrame([
                {"Question": u["field"], "Answer": u["value"], "Rows": u["rows"],
                 "Response IDs": ", ".join(str(i) for i in u["response_ids"])}
                for u in unknown
            ]),
            hide_index=True, width="stretch",
        )
    else:
        st.success("✓ No unknown values found.")


# ----- M2: flags and headline numbers ---------------------------------

def _fmt(m: dict[str, Any], kind: str = "pct") -> str:
    """Format a metric value, or say why it is not shown."""
    if m.get("value") is None:
        status = m.get("status", "not available")
        return status if status != "too few responses" else f"too few responses (n={m['n']})"
    v = m["value"]
    if kind == "pct":
        return f"{v:.0%}"
    if kind == "change":
        return f"{v:+.2f}"
    if kind == "p":
        return "p < 0.001" if v < 0.001 else f"p = {v:.3f}"
    return f"{v:.1f}"


def _n(m: dict[str, Any]) -> str:
    return f"n = {m['n']:,}"


def _show_flags(flags: dict[str, dict[str, int]]) -> bool:
    """Quality-flag counts and the leave-out toggle. Returns the toggle value."""
    st.subheader("Rushed or doubtful answers")
    any_ = flags["flag_any"]
    secs = load_schema()["quality"]["speeder_seconds"]
    rows = [(f"Finished in under {secs} seconds", "flag_speeder"),
            ("Same answer to all 9 rating questions", "flag_straightline"),
            ("Lowest rating on every 'after' item but would recommend (8+)", "flag_reversed")]
    st.dataframe(
        pd.DataFrame([{"Check": label, "Flagged": flags[c]["flagged"],
                       "Could not check (blank answers)": flags[c]["unknown"]}
                      for label, c in rows]),
        hide_index=True, width="stretch",
    )
    exclude = st.toggle(
        f"Leave out the {any_['flagged']} flagged responses (of {any_['n']:,})",
        key="exclude_flagged",
    )
    st.caption("Flagged rows are never deleted. \"Could not check\" rows are kept.")
    return exclude


def _show_headline(m: dict[str, dict[str, Any]]) -> None:
    """Headline tiles; every number shows its n."""
    st.subheader("Headline numbers")
    c1, c2, c3, c4, c5 = st.columns(5)
    post, pre = m["agree_pct.post_when"], m["agree_pct.pre_when"]
    c1.metric("Know when to use 988", _fmt(post))
    c1.caption(f"Agree or strongly agree, after training · {_n(post)}. "
               f"Before (recalled): {_fmt(pre)}.")
    ch = m["mean_change.when"]
    c2.metric("Self-reported change", _fmt(ch, "change"))
    c2.caption(f"'Know when to use 988', 1–5 scale, after minus recalled before · "
               f"{_n(ch)} · {_fmt(m['wilcoxon_p.when'], 'p')}. Retrospective, not proof of impact.")
    nps = m["nps"]
    c3.metric("Would recommend (NPS)", _fmt(nps, "num"))
    ci = (f"95% range {nps['ci95'][0]:.1f} to {nps['ci95'][1]:.1f}"
          if nps.get("ci95") else "")
    c3.caption(f"−100 to +100 · {_n(nps)}. {ci}")
    call, cov = m["likely_call.pct_high"], m["likely_call.coverage"]
    c4.metric("Very likely to call 988", _fmt(call))
    answered = (f" Answered by {cov['answered']:,} of {cov['n']:,} ({_fmt(cov)})."
                if cov.get("value") is not None else "")
    c4.caption(f"Rated 9–10 out of 10 · {_n(call)}.{answered}")
    dl = m["delivery_agree_pct"]
    c5.metric("Liked the delivery", _fmt(dl))
    c5.caption(f"Agreed with all 3 presenter / organization items · {_n(dl)}.")


def _show_knowledge(m: dict[str, dict[str, Any]]) -> None:
    """Before/after table for the three knowledge items."""
    st.markdown("**Before and after, by knowledge item** (self-reported retrospective change)")
    rows = []
    for k, label in KNOWLEDGE_LABELS.items():
        rows.append({
            "Item": label,
            "Agree before": _fmt(m[f"agree_pct.pre_{k}"]),
            "Agree after": _fmt(m[f"agree_pct.post_{k}"]),
            "Average change (1–5)": _fmt(m[f"mean_change.{k}"], "change"),
            "Went down": _fmt(m[f"declined_pct.{k}"]),
            "Already at top before": _fmt(m[f"ceiling_pct.{k}"]),
            "n": m[f"mean_change.{k}"]["n"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _show_know_call(m: dict[str, dict[str, Any]]) -> None:
    """2x2 of knowing when to use 988 vs being very likely to call."""
    st.subheader("Know when to use 988 vs. likely to call")
    grid, gap = m["know_call_2x2"], m["know_call_gap"]
    if grid.get("value") is None:
        st.info(f"Not shown: {_fmt(grid)}.")
        return
    cells = grid["value"]
    st.caption(f"People who answered both questions · {_n(grid)}. "
               "'Very likely' = 9–10 out of 10.")
    head, c1, c2 = st.columns([1.2, 1, 1])
    c1.markdown("**Very likely to call**")
    c2.markdown("**Not very likely (0–8)**")
    head, c1, c2 = st.columns([1.2, 1, 1])
    head.markdown("**Knows when to use 988**")
    c1.metric("Knows · very likely", cells["knows_and_calls"], label_visibility="collapsed")
    c2.metric("Knows · not very likely", cells["knows_not_calls"],
              label_visibility="collapsed")
    head, c1, c2 = st.columns([1.2, 1, 1])
    head.markdown("**Not sure when to use 988**")
    c1.metric("Not sure · very likely", cells["not_knows_calls"],
              label_visibility="collapsed")
    c2.metric("Not sure · not very likely", cells["not_knows_not_calls"],
              label_visibility="collapsed")
    st.markdown(f"**{_fmt(gap)}** know when to use 988 but are not very likely to call "
                f"({cells['knows_not_calls']} of {gap['n']}).")


def _show_gain(m: dict[str, dict[str, Any]]) -> None:
    """Average change by starting (recalled before) level."""
    st.subheader("Change by starting level")
    k = st.selectbox("Knowledge item", list(KNOWLEDGE_LABELS),
                     format_func=KNOWLEDGE_LABELS.get, key="gain_item")
    table = m[f"gain_by_baseline.{k}"]
    if table.get("value") is None:
        st.info(f"Not shown: {_fmt(table)}.")
        return
    levels = sorted(table["value"])
    cells = [table["value"][lv] for lv in levels]
    x = [f"Started at {lv}" + (" (top)" if lv == max(levels) else "")
         + f"<br>n = {c['n']}" + ("<br>too few" if c["value"] is None else "")
         for lv, c in zip(levels, cells)]
    y = [c["value"] for c in cells]
    fig = go.Figure(go.Bar(
        x=x, y=y, marker_color=BAR_COLOR, marker_cornerradius=4,
        customdata=[c["n"] for c in cells],
        hovertemplate="%{x}<br>Average change: %{y:+.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10), bargap=0.45, showlegend=False,
        yaxis=dict(title="Average change (1–5 scale)", zeroline=True, zerolinewidth=1,
                   gridcolor="rgba(128,128,128,0.15)"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, width="stretch", key="gain_chart")
    top = m[f"ceiling_pct.{k}"]
    st.caption(f"Total {_n(table)}. People who started high have little room to go up; "
               f"{_fmt(top)} were already at the top before ({_n(top)}). "
               "Self-reported retrospective change.")


# ----- M3: comparisons, sessions, trend, data health ------------------

def _val(v: Any, kind: str) -> str:
    """Format a plain value from a segments/trends table."""
    if v is None or pd.isna(v):
        return "too few responses"
    return _fmt({"value": float(v)}, kind)


def _show_compare(df: pd.DataFrame, metric: str) -> None:
    """Table of one measure by the chosen group."""
    st.subheader("Compare groups")
    by = st.selectbox("Compare by", list(GROUP_LABELS), format_func=GROUP_LABELS.get,
                      key="m3_by")
    label, kind = METRIC_LABELS[metric]
    t = compare(df, by, metric)
    if t.empty:
        st.info("Not collected: this file has no answers for this measure or grouping.")
        return
    rows = []
    for r in t.itertuples():
        notes = []
        if r.suppressed:
            notes.append("too few responses")
        if getattr(r, "out_of_area", False):
            notes.append("outside Maryland")
        if getattr(r, "outlier", False):
            notes.append("outside session limits")
        rows.append({GROUP_LABELS[by]: r.group, label: _val(r.value, kind),
                     "Answered": r.metric_n, "Rows": r.n, "Note": ", ".join(notes)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Groups with fewer than {load_schema()['analysis']['min_n']} answers are not "
               f"scored. \"{NOT_RECORDED}\" = rows with this field blank; they are kept.")


def _show_funnel(df: pd.DataFrame, metric: str) -> None:
    """Session funnel: each session against limits that widen for small sessions."""
    st.subheader("Sessions compared with the average")
    label, kind = METRIC_LABELS[metric]
    t = compare(df, "session_id", metric)
    shown = t[~t["suppressed"]].sort_values("metric_n") if not t.empty else t
    if shown.empty:
        st.info("Not shown: no session has enough answers.")
        return
    fig = go.Figure()
    for col, name, dash in (("lower998", "99.8% limits", "solid"),
                            ("upper998", None, "solid"),
                            ("lower95", "95% limits", "dot"), ("upper95", None, "dot")):
        fig.add_trace(go.Scatter(
            x=shown["metric_n"], y=shown[col], mode="lines", name=name or "",
            showlegend=name is not None, legendgroup=dash, hoverinfo="skip",
            line=dict(color=MUTED_COLOR, width=1.5, dash=dash)))
    fig.add_trace(go.Scatter(
        x=[shown["metric_n"].min(), shown["metric_n"].max()], y=[t["centre"].iloc[0]] * 2,
        mode="lines", name="All sessions", hoverinfo="skip",
        line=dict(color=MUTED_COLOR, width=1, dash="dash")))
    for is_out, name, color, symbol in ((False, "Session", BAR_COLOR, "circle"),
                                        (True, "Outside 99.8% limits", ACCENT_COLOR, "diamond")):
        pts = shown[shown["outlier"] == is_out]
        if pts.empty:
            continue
        fig.add_trace(go.Scatter(
            x=pts["metric_n"], y=pts["value"], mode="markers", name=name,
            customdata=pts["group"],
            marker=dict(color=color, size=11 if is_out else 9, symbol=symbol,
                        line=dict(color="white", width=2)),
            hovertemplate="%{customdata}<br>n = %{x}<br>%{y:.2f}<extra></extra>"))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=-0.2),
        xaxis=dict(title="Answers in session (n)", showgrid=False),
        yaxis=dict(title=label, gridcolor="rgba(128,128,128,0.15)",
                   tickformat=".0%" if kind == "pct" else None))
    st.plotly_chart(fig, width="stretch", key="funnel_chart")
    hidden = int(t["suppressed"].sum())
    outliers = list(shown.loc[shown["outlier"], "group"])
    st.caption(f"{len(shown)} sessions shown; {hidden} with too few answers are hidden. "
               "Limits are wider for small sessions, so a session is only flagged when it is "
               "unlikely to be chance. Sessions are not adjusted for audience: a session of "
               "adults can look different from one of students for that reason alone.")
    if outliers:
        st.warning("Outside the 99.8% limits: " + ", ".join(outliers))
    else:
        st.success("✓ Checked: no session is outside the 99.8% limits.")


def _show_trend(df: pd.DataFrame, metric: str) -> None:
    """Monthly trend, raw and adjusted for role mix."""
    st.subheader("Monthly trend, adjusted for who attended")
    label, kind = METRIC_LABELS[metric]
    t = monthly(df, metric)
    if t.empty:
        st.info("Not collected: no training month or no answers for this measure.")
        return
    hover = [f"{m}<br>n = {n}" + (" (too few)" if s else "")
             for m, n, s in zip(t["month"], t["n"], t["suppressed"])]
    fig = go.Figure()
    for col, name, color, dash in (("raw", "As reported", ACCENT_COLOR, "dot"),
                                   ("adjusted", "Adjusted for role mix", BAR_COLOR, "solid")):
        fig.add_trace(go.Scatter(
            x=t["month"], y=t[col], mode="lines+markers", name=name, connectgaps=False,
            customdata=hover, line=dict(color=color, width=2, dash=dash),
            marker=dict(size=8, line=dict(color="white", width=2)),
            hovertemplate="%{customdata}<br>" + name + ": %{y:.2f}<extra></extra>"))
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        xaxis=dict(type="category", showgrid=False),
        yaxis=dict(title=label, gridcolor="rgba(128,128,128,0.15)",
                   tickformat=".0%" if kind == "pct" else None))
    st.plotly_chart(fig, width="stretch", key="trend_chart")
    small = list(t.loc[t["suppressed"], "month"])
    st.caption("Adjusted = what the month would score if its audience had the usual mix of "
               "roles. If only the 'as reported' line moves, the change came from who attended."
               + (f" Gaps: {', '.join(small)} had too few answers." if small else ""))


def _show_alerts(df: pd.DataFrame) -> None:
    """Data-health alerts (blank answers by month, falling comment rate)."""
    st.subheader("Data health")
    found = alerts(df)
    kinds = {a["kind"] for a in found}
    for a in found:
        name = FIELD_LABELS.get(a["field"], a["field"])
        if a["kind"] == "blank_answers":
            parts = [f"{m['month']}: {m['blank_pct']:.0%} of {m['rows']}" for m in a["months"]]
            other = a["months"][0]["other_months_blank_pct"]
            st.warning(f"{name} was blank (skipped or not asked) much more often in some "
                       f"months: {'; '.join(parts)} (other months: about {other:.0%}). "
                       "Scores for those months rest on fewer answers. Blanks are not counted "
                       "as disagree.")
        else:
            st.warning(f"Fewer people wrote a {name} in {a['month']}: {a['rate']:.0%} of "
                       f"{a['rows']} vs {a['earlier_rate']:.0%} of {a['earlier_rows']} in "
                       "earlier months.")
    if "blank_answers" not in kinds:
        st.success("✓ Checked: no question is unusually blank in any month.")
    if "fewer_comments" not in kinds:
        st.success("✓ Checked: comment rate has not dropped in the latest month.")


# ----- page -----------------------------------------------------------

def main() -> None:
    """Build the page."""
    st.set_page_config(page_title="988 Training Insights", layout="wide")

    with st.sidebar:
        st.title("988 Training Insights")
        uploads = st.file_uploader(
            "Survey exports (.xlsx or .csv)", type=["xlsx", "csv"], accept_multiple_files=True
        )
        use_sample = st.checkbox("Use sample data", key="use_sample")
        st.caption("Files are processed in this session only and are never saved.")

    st.header("Survey data check")
    st.write("Upload one or more post-training survey exports. Several files are combined into one.")

    if use_sample and uploads:
        st.info("Showing the sample data. Untick **Use sample data** to see your uploads.")

    try:
        if use_sample:
            result = _load_sample(str(SAMPLE_PATH))
        elif uploads:
            result = _load_uploads(tuple((f.name, f.getvalue()) for f in uploads))
        else:
            st.info("Upload a survey export to begin, or tick **Use sample data**.")
            return
    except ValueError as exc:
        st.error(str(exc))
        return

    stats, raw, cleaned = result["stats"], result["raw"], result["clean"]
    _show_counts(stats)
    _show_files(stats)
    _show_health(stats)
    _show_mapping(result["mapping"])
    if cleaned is None:
        return
    _show_cleaning(result["fix_log"], result["unknown"])
    exclude = _show_flags(result["flags"])
    metrics = result["metrics"][exclude]
    _show_headline(metrics)
    _show_knowledge(metrics)
    _show_know_call(metrics)
    _show_gain(metrics)

    df = drop_flagged(cleaned) if exclude else cleaned
    st.header("Comparisons and trends")
    metric = st.selectbox("Measure", list(METRIC_LABELS),
                          format_func=lambda k: METRIC_LABELS[k][0], key="m3_metric")
    _show_compare(df, metric)
    _show_funnel(df, metric)
    _show_trend(df, metric)
    _show_alerts(df)

    st.subheader("Preview")
    n = min(PREVIEW_ROWS, len(raw))
    tab_clean, tab_raw = st.tabs(["Cleaned", "As exported"])
    with tab_clean:
        st.caption(f"First {n} of {len(cleaned)} rows after cleaning.")
        st.dataframe(cleaned.head(PREVIEW_ROWS), width="stretch")
    with tab_raw:
        st.caption(f"First {n} of {len(raw)} rows, exactly as in the file(s).")
        st.dataframe(raw.head(PREVIEW_ROWS), width="stretch")

    st.subheader("Still to come")
    for milestone, title, text in LATER_MILESTONES:
        _placeholder(milestone, title, text)


main()
