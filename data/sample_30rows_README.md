# sample_30rows.xlsx — what each row tests

30 rows of the real export (`BuildFest_EveryMind_Survey_Data_FULL.xlsx`),
picked because each already shows a messy case. Headers and values are exactly
as exported (non-breaking spaces, `‎` after Language, trailing `;` on roles).
**Hand-edited rows are marked ✎.** Rows whose comments name a person were not used.

Rebuild: `python scripts/build_sample.py` (needs the full export locally).
The recipe is `tests/fixtures/sample_spec.yaml`; `tests/test_setup.py` checks
the counts below.

| Row | Excel row in full file | Case |
|---|---|---|
| 1 | 4 | normal row; praise comment ("It was awesome!") |
| 2 | 127 | ✎ unknown answer: post_describe = "Somewhat agree" (must be reported, not dropped) |
| 3 | 128 | **Very agree** (well_organized) |
| 4 | 130 | **Satisfied** (presenter_knew) + 2× Very agree |
| 5 | 27 | **"Neither agree or disagree"** (well_organized) |
| 6 | 63 | **missing training date**; standard "Neither agree nor disagree" |
| 7 | 36 | **multi-role** Other;Caregiver; · county `Montgomery\xa0` (**\xa0** value) |
| 8 | 22 | **multi-role** School staff;Caregiver; · county "Fairfax " (out of area) |
| 9 | 211 | **multi-role** Caregiver/parent;Student → Student · 3× Satisfied |
| 10 | 1031 | **blank presenter block** (Oct 2026 form) |
| 11 | 133 | blank presenter block + blank nps + blank likely_call |
| 12 | 233 | **"nothing" comment** |
| 13 | 262 | **question comment** ("what if someone doesn't want help?") |
| 14 | 1028 | open question without "?" ("How the calls usually go") |
| 15 | 32 | **county variant** "Montgomery County with NAMI MC" |
| 16 | 24 | county variant "Fairfax county " |
| 17 | 60 | county "Harris" → Other / unclear |
| 18 | 196 | county variant "Queen Anne's County "; Spanish |
| 19 | 861 | **speeder** (18 s) |
| 20 | 15 | **straight-liner** (all 9 = Strongly agree); also at ceiling (pre = 5) |
| 21 | 8 | **reversed** (post all Strongly disagree, nps 10) |
| 22 | 34 | Spanish language |
| 23 | 1030 | knows when to use 988 (5) but likely_call 6 |
| 24 | 1027 | declined (pre_identify 4 → post 3); question comment |
| 25 | 148 | blank nps, likely_call 10 |
| 26 | 5 | ✎ Id changed to 3 = same Id as row 1, different start time → **not** a duplicate |
| 27 | 115 | **duplicate** — original |
| 28 | 115 | ✎ **duplicate** — exact copy of row 27 (same Id + start time) |
| 29 | 423 | normal (May 2026 student) |
| 30 | 665 | normal (Sep 2026 student); confidentiality question |

**Blank likely_call:** 21 of 30 rows. The real form only asked it from April 2026.

## Expected totals
- Ingest: rows_in 30 · duplicates_removed 1 · rows_out 29
- Flags: speeder 1 (row 19) · straight-liner 1 (row 20) · reversed 1 (row 21)
- Likert cells: Very agree 3 · Satisfied 4 · "Neither agree or disagree" 1 · unknown "Somewhat agree" 1
- Multi-role rows 3 (7, 8, 9) · missing date 1 (6) · blank presenter block 5 (10, 11, 14, 23, 24) · Spanish 2 (18, 22)
- Months covered: 2025-10 → 2026-10 (10 months)
