"""Build data/sample_30rows.xlsx from tests/fixtures/sample_spec.yaml.

Picks real rows from the full export and applies the hand edits in the
spec. All survey wording lives in the YAML files, not here. Needs the
full export locally (it is not committed).

Usage: python scripts/build_sample.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "tests" / "fixtures" / "sample_spec.yaml"


def build(spec: dict[str, Any], source: pd.DataFrame) -> pd.DataFrame:
    """Return the sample frame described by ``spec`` (pure; no I/O)."""
    cols = spec["columns"]
    out = []
    for i, row in enumerate(spec["rows"], start=1):
        src = row["excel_row"] - 2  # Excel row 2 = first data row
        if not 0 <= src < len(source):
            raise IndexError(f"sample row {i}: excel_row {row['excel_row']} out of range")
        record = source.iloc[src].copy()
        for field, value in (row.get("edit") or {}).items():
            record.iloc[cols[field]] = value
        out.append(record)
    return pd.DataFrame(out, columns=source.columns).reset_index(drop=True)


def main() -> None:
    """Read the spec and source file, write the sample workbook."""
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    source = pd.read_excel(ROOT / spec["source_file"], dtype=object)
    sample = build(spec, source)
    out_path = ROOT / spec["output_file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_excel(out_path, index=False, sheet_name=spec["sheet_name"])
    print(f"wrote {len(sample)} rows -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
