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
from pipeline.recommend import CHECK_METRIC, DECISIONS_PATH, load_decisions, run_checks, validate_decisions
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
    (5, "Comment board", "Comments sorted into Keep / Fix / Add themes."),
    (6, "Summary", "A copy-ready written summary of the findings."),
]
SECTIONS: dict[int, tuple[str, str]] = {
    1: ("Data check", "What was loaded, how columns were matched, and what was fixed."),
    2: ("Headline results", "Overall numbers for everyone in the data."),
    3: ("Comparisons and trends", "Groups, sessions, and months side by side."),
    4: ("What to act on", "Insight checks run on this data, and what they found."),
    5: ("Comment board", "What people said they valued, would fix, or want added."),
    6: ("Summary", "A copy-ready write-up of the findings."),
}
# Make the three heading levels easy to tell apart: section > subsection > label.
HEADING_CSS = """<style>
[data-testid="stMain"] h1 {font-size: 2.4rem; font-weight: 800;}
[data-testid="stMain"] h2 {font-size: 1.9rem; font-weight: 750; margin-top: 2.2rem;}
[data-testid="stMain"] h3 {font-size: 1.3rem; font-weight: 650; margin-top: 1rem;}
[data-testid="stMain"] h4 {font-size: 1.05rem; font-weight: 600; color: #555;}
.section-tag {font-size: 0.8rem; font-weight: 600; letter-spacing: 0.06em;
  text-transform: uppercase; color: #2a78d6; margin: 2.2rem 0 -1.6rem 0;}
</style>"""


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


@st.cache_data
def _load_decisions(path: str) -> dict[str, Any]:
    """Read config/decisions.yaml once per session."""
    return load_decisions(path)


# ----- display helpers ------------------------------------------------

def _section(milestone: int) -> None:
    """Top-level section header for one milestone, with a divider and short blurb."""
    title, blurb = SECTIONS[milestone]
    st.markdown(f'<div class="section-tag">Section {milestone}</div>', unsafe_allow_html=True)
    st.header(title, divider="blue")
    st.caption(blurb)


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
    st.markdown(f"#### Matched questions ({len(matched)})")
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
    st.markdown("#### Fixes made")
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

    st.markdown("#### Unknown values")
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


def _show_preview(raw: pd.DataFrame, cleaned: pd.DataFrame | None) -> None:
    """First rows as exported and, when available, after cleaning."""
    n = min(PREVIEW_ROWS, len(raw))
    tabs = st.tabs(["Cleaned", "As exported"] if cleaned is not None else ["As exported"])
    if cleaned is not None:
        with tabs[0]:
            st.caption(f"First {n} of {len(cleaned)} rows after cleaning.")
            st.dataframe(cleaned.head(PREVIEW_ROWS), width="stretch")
    with tabs[-1]:
        st.caption(f"First {n} of {len(raw)} rows, exactly as in the file(s).")
        st.dataframe(raw.head(PREVIEW_ROWS), width="stretch")


def _show_data_check(result: dict[str, Any]) -> None:
    """Section 1 as collapsible dropdowns; a summary sits in each label.

    A dropdown opens by itself when it holds a problem that stops the analysis.
    """
    stats, mapping = result["stats"], result["mapping"]
    cleaned = result["clean"]
    n_files = len(stats["files"])
    with st.expander(f"Files · {n_files} file{'s' if n_files != 1 else ''}"):
        _show_files(stats)
    with st.expander(f"Preview · first {min(PREVIEW_ROWS, len(result['raw']))} rows"):
        _show_preview(result["raw"], cleaned)
    with st.expander(f"Rows loaded · {stats['rows_in']:,} read, {stats['rows_out']:,} kept"):
        _show_counts(stats)
        _show_health(stats)
    n_q = len(mapping["matched"]) + len(mapping["missing"])
    blocked = bool(mapping["missing_required"])
    with st.expander(f"Column matching · {len(mapping['matched'])} of {n_q} questions matched"
                     + (" · required columns missing" if blocked else ""), expanded=blocked):
        _show_mapping(mapping)
    if cleaned is None:
        return
    n_fix, n_unknown = len(result["fix_log"]), len(result["unknown"])
    with st.expander(f"Fixes and unknown values · {n_fix} fix{'es' if n_fix != 1 else ''}, "
                     f"{n_unknown} unknown value{'s' if n_unknown != 1 else ''}"):
        _show_cleaning(result["fix_log"], result["unknown"])


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


def _flag_toggle(flags: dict[str, dict[str, int]]) -> bool:
    """The leave-out toggle; kept outside the dropdowns because it changes every number."""
    any_ = flags["flag_any"]
    exclude = st.toggle(
        f"Leave out the {any_['flagged']} flagged responses (of {any_['n']:,})",
        key="exclude_flagged",
    )
    st.caption("Flagged rows are never deleted. \"Could not check\" rows are kept. "
               "See 'Rushed or doubtful answers' below for what was flagged.")
    return exclude


def _show_flags(flags: dict[str, dict[str, int]]) -> None:
    """Quality-flag counts by check."""
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


def _show_headline(m: dict[str, dict[str, Any]]) -> None:
    """Headline tiles; every number shows its n."""
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
    st.caption("Self-reported retrospective change.")
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


CHANGE_GROUPS: list[tuple[str, str, str]] = [
    ("at_top", "Already at top", "#9ec5f0"),
    ("went_up", "Went up", BAR_COLOR),
    ("same", "Stayed the same", "rgba(128,128,128,0.45)"),
    ("went_down", "Went down", ACCENT_COLOR),
]


def _show_change_groups(m: dict[str, dict[str, Any]]) -> None:
    """One stacked bar per knowledge item: what happened to each person's rating."""
    st.write("For each person, did their rating go up, stay the same, or go down from before "
             "to after? People already at the top (5 of 5) had no room to go up.")
    items = list(KNOWLEDGE_LABELS)[::-1]          # plotly draws the first row at the bottom
    results = [m[f"change_groups.{k}"] for k in items]
    y = [f"{KNOWLEDGE_LABELS[k]}<br>{_n(r)}" + ("" if r.get("value") else f" · {_fmt(r)}")
         for k, r in zip(items, results)]
    fig = go.Figure()
    for key, name, color in CHANGE_GROUPS:
        x = [r["value"][key] if r.get("value") else None for r in results]
        fig.add_trace(go.Bar(
            y=y, x=x, name=name, orientation="h", marker_color=color,
            text=[f"{v:.0%}" if v is not None and v >= 0.06 else "" for v in x],
            textposition="inside", insidetextanchor="middle",
            hovertemplate=name + ": %{x:.0%}<extra></extra>"))
    fig.update_layout(
        barmode="stack", height=300, margin=dict(l=10, r=10, t=10, b=10), bargap=0.35,
        legend=dict(orientation="h", y=-0.15, traceorder="normal"),
        xaxis=dict(tickformat=".0%", range=[0, 1], showgrid=False),
        yaxis=dict(showgrid=False))
    st.plotly_chart(fig, width="stretch", key="change_groups_chart")
    st.caption("n = people who answered both before and after. Before is recalled after the "
               "training, so this is self-reported retrospective change.")


def _show_results(flags: dict[str, dict[str, int]], m: dict[str, dict[str, Any]],
                  exclude: bool) -> None:
    """Section 2 as dropdowns. Headline numbers start open; the rest start closed."""
    any_ = flags["flag_any"]
    with st.expander(f"Rushed or doubtful answers · {any_['flagged']} of {any_['n']:,} flagged"
                     + (" · left out" if exclude else " · included")):
        _show_flags(flags)
    post = m["agree_pct.post_when"]
    with st.expander(f"Headline numbers · {_n(post)}", expanded=True):
        _show_headline(m)
    with st.expander("Before and after, by knowledge item"):
        _show_knowledge(m)
    with st.expander(f"Know when to use 988 vs. likely to call · {_n(m['know_call_2x2'])}"):
        _show_know_call(m)
    with st.expander("What changed for each person"):
        _show_change_groups(m)


# ----- M3: comparisons, sessions, trend, data health ------------------

def _val(v: Any, kind: str) -> str:
    """Format a plain value from a segments/trends table."""
    if v is None or pd.isna(v):
        return "too few responses"
    return _fmt({"value": float(v)}, kind)


def _show_compare(df: pd.DataFrame, metric: str) -> None:
    """Table of one measure by the chosen group."""
    st.subheader("Compare groups")
    st.write("One measure, split into groups, so you can see which group scores higher or lower.")
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
    st.write("Each dot is one session. A dot outside the grey lines scored unusually high or "
             "low; small sessions get wider lines because their scores swing more by chance.")
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
    st.write("Orange: the score each month as measured. Blue: the score if every month had the "
             "same mix of roles. If only orange moves, the change came from who attended.")
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
    """Data-health alerts (blank answers by month, falling comment rate), in a dropdown."""
    found = alerts(df)
    label = (f"Data health · {len(found)} alert{'s' if len(found) != 1 else ''}" if found
             else "Data health · no issues found")
    with st.expander(label):
        st.write("Checks whether some answers are missing more often than usual, which can make "
                 "a score look low just because fewer people answered.")
        _show_alert_list(found)


def _show_alert_list(found: list[dict[str, Any]]) -> None:
    """One warning per alert, then a ✓ line for each check that found nothing."""
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


# ----- M4: insight cards ------------------------------------------------

CHART_SECTIONS = {
    "trend_adjusted": "Monthly trend, adjusted for who attended",
    "know_call_2x2": "Know when to use 988 vs. likely to call",
    "intent_drivers": "Headline numbers", "gain_by_baseline": "What changed for each person",
    "data_health": "Data health", "trend_comments": "Data health",
    "session_funnel": "Sessions compared with the average", "headline": "Headline numbers",
    "keep_fix_add": "Comment board (section 5, coming soon)",
}
RESULT_TEXT = {
    "finding": "found something, see card above",
    "no_finding": "checked, no issue",
    "too_few": "too few responses to check",
    "not_collected": "not collected in this file",
    "needs_themes": "needs comment themes (coming in milestone 5)",
}
RESULT_ICON = {"finding": "●", "no_finding": "✓", "too_few": "–", "not_collected": "–",
               "needs_themes": "…"}
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _tag(text: str, color: str) -> str:
    return f":{color}-background[{text}]"


def _show_card(c: dict[str, Any]) -> None:
    """One insight card: what we found, the decision it feeds, options, measure."""
    with st.container(border=True):
        st.markdown(f"#### {c['title']}")
        tags = [_tag(c["lever"].replace("_", " "), "blue") if c["lever"]
                else _tag("no lever set", "gray")]
        if c.get("secondary_lever"):
            tags.append(_tag(c["secondary_lever"].replace("_", " "), "violet"))
        tags += [_tag(f"{c['confidence']} evidence", "green"), _tag(f"{c['priority']} priority", "orange")]
        st.markdown(" ".join(tags))
        st.write(c["text"])
        st.caption(f"n = {c['n']}")
        st.write(f"Decision: {c['decision']}")
        st.write(f"Options to consider: {c['action'] or 'none configured'}")
        st.write(f"How we'd know: {c['success_measure'] or 'none configured'}")
        notes = [f"Chart: see '{CHART_SECTIONS.get(c['chart_key'], c['chart_key'])}'."]
        if c["owner_role"] != "Not assigned" and not c.get("confirmed"):
            notes.append("Owner role not yet confirmed by EveryMind.")
        st.caption(" ".join(notes))


def _show_insights(cards: list[dict[str, Any]], runs: list[dict[str, Any]],
                   errors: list[str]) -> None:
    """Cards grouped by owner role, then the list of every check run."""
    owners = sorted({c["owner_role"] for c in cards},
                    key=lambda o: (o == "Not assigned",
                                   min(PRIORITY_ORDER[c["priority"]] for c in cards
                                       if c["owner_role"] == o), o))
    if cards:
        pick = st.selectbox("Show cards for", ["Everyone"] + owners, key="m4_owner")
        for owner in owners if pick == "Everyone" else [pick]:
            mine = [c for c in cards if c["owner_role"] == owner]
            st.subheader(f"{owner} ({len(mine)})")
            for c in mine:
                _show_card(c)
    else:
        st.info("No check found anything to act on in this data.")

    st.subheader("Checks run")
    lines = []
    for x in runs:
        line = f"{RESULT_ICON[x['result']]} {x['id']} · {x['question']}: {RESULT_TEXT[x['result']]}"
        if x["result"] == "no_finding" and x["text"]:
            line += f". {x['text']}"
        elif x["result"] == "too_few" and x["n"]:
            line += f" (n = {x['n']})"
        lines.append(f"- {line}")
    st.markdown("\n".join(lines))

    problems = list(dict.fromkeys(errors + [f"{x['id']}: {g}" for x in runs
                                            for g in x["config_gaps"]]))
    if problems:
        st.warning("decisions.yaml needs attention:\n" + "\n".join(f"- {p}" for p in problems))


# ----- page -----------------------------------------------------------

def main() -> None:
    """Build the page."""
    st.set_page_config(page_title="988 Training Insights", layout="wide")
    st.markdown(HEADING_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Survey files")
        uploads = st.file_uploader(
            "Survey exports (.xlsx or .csv)", type=["xlsx", "csv"], accept_multiple_files=True
        )
        use_sample = st.checkbox("Use sample data", key="use_sample")
        st.caption("Files are processed in this session only and are never saved.")

    st.title("988 Training Insights")
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

    cleaned = result["clean"]

    _section(1)
    _show_data_check(result)
    if cleaned is None:
        return

    _section(2)
    exclude = _flag_toggle(result["flags"])
    metrics = result["metrics"][exclude]
    _show_results(result["flags"], metrics, exclude)

    df = drop_flagged(cleaned) if exclude else cleaned
    _section(3)
    metric = st.selectbox("Measure", list(METRIC_LABELS),
                          format_func=lambda k: METRIC_LABELS[k][0], key="m3_metric")
    _show_compare(df, metric)
    _show_funnel(df, metric)
    _show_trend(df, metric)
    _show_alerts(df)

    _section(4)
    decisions = _load_decisions(str(DECISIONS_PATH))
    cards, runs = run_checks(
        {"headline": metrics, "all": result["metrics"][False], "unflagged": result["metrics"][True]},
        {"session_id": compare(df, "session_id", CHECK_METRIC), "metric": CHECK_METRIC},
        {"monthly": monthly(df, CHECK_METRIC), "metric": CHECK_METRIC, "alerts": alerts(df)},
        None, decisions)
    _show_insights(cards, runs, validate_decisions(decisions))

    for milestone, title, text in LATER_MILESTONES:
        _section(milestone)
        _placeholder(milestone, title, text)


main()
