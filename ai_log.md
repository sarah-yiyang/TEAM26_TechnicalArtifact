# AI log

What AI (Claude Code) did on each milestone, and what I checked or changed.

| Date | Module | What I asked | What AI did | What I changed / decided | How verified |
|---|---|---|---|---|---|
| 2026-11-04 | M0 setup | Plan M0, then build it; sample from real rows + hand-edited messy cases | Wrote requirements.txt, pytest.ini, `pipeline/__init__.py` (MIN_N from schema.yaml), tests/conftest.py, tests/test_setup.py, sample recipe (`tests/fixtures/sample_spec.yaml`) + `scripts/build_sample.py`, data/sample_30rows.xlsx + README; .gitignore exception for the sample and secrets.toml | Approved plan; sample built from real rows (not synthetic); 3 hand edits (unknown value, reused Id, exact duplicate). AI swapped out rows that were accidentally also straight-liners/reversed, and one with a sensitive comment | `pytest -q` → 12 passed |
