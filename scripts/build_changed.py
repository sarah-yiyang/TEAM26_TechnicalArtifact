"""Build data/test_changed.xlsx from tests/fixtures/changed_spec.yaml.

Takes the 30-row sample and applies a simulated survey revision (renamed,
removed and added questions). All survey wording lives in the YAML spec.

Usage: python scripts/build_changed.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "tests" / "fixtures" / "changed_spec.yaml"


def build(spec: dict[str, Any], source: pd.DataFrame) -> pd.DataFrame:
    """Return ``source`` with the spec's rename/remove/add applied (pure)."""
    out = source.copy()
    for r in spec.get("rename") or []:
        if r["from"] not in out.columns:
            raise KeyError(f"rename: column not found: {r['from']!r}")
        out = out.rename(columns={r["from"]: r["to"]})
    for col in spec.get("remove") or []:
        if col not in out.columns:
            raise KeyError(f"remove: column not found: {col!r}")
        out = out.drop(columns=[col])
    for a in spec.get("add") or []:
        vals = a["values"]
        out[a["header"]] = [vals[i % len(vals)] for i in range(len(out))]
    return out


def main() -> None:
    """Read the spec and sample, write the changed workbook."""
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    source = pd.read_excel(ROOT / spec["source_file"], dtype=object)
    changed = build(spec, source)
    path = ROOT / spec["output_file"]
    changed.to_excel(path, index=False)
    print(f"wrote {path} ({len(changed)} rows, {len(changed.columns)} columns)")


if __name__ == "__main__":
    main()
