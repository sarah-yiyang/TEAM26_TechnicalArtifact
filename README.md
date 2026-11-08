# 988 Training Insights

A Streamlit app and Python pipeline that turns EveryMind's 988 post-training survey exports into trustworthy metrics, fair comparisons, comment themes, and recommended actions.

Built for BuildFest 2026, answering: how can EveryMind use its 988 training feedback to evaluate training effectiveness, find gaps across communities, and keep improving future trainings?

## Why it exists

EveryMind runs live virtual 988 Awareness Trainings. Feedback piles up faster than staff can read it. This tool is built around two needs:

- Continuous: upload each new export and get the same checks, run the same way, every time.
- Actionable: every finding comes with a decision, an owner, an action to consider, and a way to measure success.

It is written for nontechnical staff, so every number shows how many responses it is based on (n), and small groups show "too few responses" instead of a misleading figure.

## What's in the app

| Section | What it shows |
|---|---|
| Data check | Rows loaded, duplicates removed, how columns were matched, what was fixed, and any unknown answers |
| Headline results | Agreement before vs. after training, NPS with a 95% range, likelihood to call 988, the "know when to call but wouldn't" gap, change by starting level |
| Comparisons and trends | Results by role, county, language, month, or session; a session funnel; monthly trend adjusted for audience mix; data-health alerts |
| What to act on | Insight cards from 10 yes/no checks on the data, plus a list of checks that found no issue |
| Comment board | Comments sorted into Keep / Fix / Add by topic, with every comment redacted and readable |
| Program brief | A one-page summary in internal or funder versions, ready to copy or download as Markdown |

Rushed or careless answers (speeders, straight-liners, reversed answers) are flagged, never deleted. One toggle shows results with or without them.

## Insight checks

Each upload runs these questions. A "yes" becomes a card; a "no" is listed as checked.

| ID | Question |
|---|---|
| A | Did the score change only because the audience changed? |
| B | Do people know when to use 988 but still wouldn't call? |
| C | Is delivery more linked to intent to call than knowledge? |
| D | Were many already at the top before training? |
| E | Is a score low only because answers are blank? |
| F | Are unhappy respondents silent about why? |
| G | Are fewer people writing comments? |
| H | Is any session outside the expected range? |
| I | Do flagged rushed answers move the headline numbers? |
| J | Are people asking questions the training didn't answer? |

Findings are never hardcoded. Each card's text is filled only from the numbers in its evidence, and each carries a confidence level (strong n ≥ 100, moderate n ≥ 50, otherwise thin).

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload one or more survey exports (.xlsx or .csv), or tick "Use sample data". Files are processed in the session only and never saved.

Run the tests:

```bash
pytest -q
```

## How it works

```
upload -> ingest -> mapping -> cleaning -> quality flags -> metrics
                                                 |-> segments, trends
                                                 |-> themes (comments)
                                                       -> recommend (insight cards) -> summarize (brief)
```

| Path | Purpose |
|---|---|
| `app.py` | Display only; no calculations |
| `pipeline/` | One module per step, pure functions |
| `config/schema.yaml` | Survey wording, column keywords, answer scales, theme keywords, `min_n` |
| `config/decisions.yaml` | What each check means: decision, lever, owner, action, and wording templates |
| `data/sample_30rows.xlsx` | 30 real rows plus planted messy cases, used by the tests |
| `data/test_changed.xlsx` | A simulated survey revision (renamed, removed, and new questions) |
| `scripts/` | Rebuild the sample and changed-survey files |
| `tests/` | pytest suite, including hand-checked reference numbers from the full Excel file |

Survey wording lives only in the YAML files, so if the survey changes, staff update config rather than code. Unknown columns and answers are reported, not dropped.

## Honest limits

- Before/after scores are self-reported in one post-training survey, so they are labeled "self-reported retrospective change", not impact.
- Likely-to-call was answered by a minority of respondents, so related findings show their coverage.
- Comment themes use keyword rules, not AI, so some comments land in "Other".
- Owner roles in `decisions.yaml` are placeholders until EveryMind confirms them.

## Use of AI

AI (Claude Code) was used during development only, for coding and for proposing config such as theme keywords. Every output was checked by a person or a test. The app makes no network calls and needs no API key.
