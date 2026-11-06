"""EveryMind 988 Training Insights: Streamlit front end (display only).

All calculations live in ``pipeline/``. This file only calls pipeline
functions and shows what they return.
"""
from __future__ import annotations

import io
from typing import Any

import pandas as pd
import streamlit as st

from pipeline import ROOT, load_schema
from pipeline.cleaning import clean
from pipeline.ingest import load_files
from pipeline.mapping import map_columns

SAMPLE_PATH = ROOT / "data" / "sample_30rows.xlsx"
PREVIEW_ROWS = 20

LATER_MILESTONES: list[tuple[int, str, str]] = [
    (2, "Headline numbers", "Agree %, NPS, likely to call, before/after change, quality flags."),
    (3, "Comparisons and trends", "Compare by role, county or session; monthly trend adjusted for audience."),
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
        result["clean"], result["fix_log"], result["unknown"] = clean(mapped, schema)
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
