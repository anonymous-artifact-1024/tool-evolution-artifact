"""Render the canonical primary-contrast CSV as a compact Markdown table."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "results" / "primary_contrasts.csv"


def main() -> int:
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 20:
        raise SystemExit(f"expected 20 primary contrasts, found {len(rows)}")
    print("| Model | RQ | Instance | Contrast | Estimate | Simultaneous 95% interval | Class |")
    print("|---|---|---:|---|---:|---:|---|")
    for row in rows:
        interval = (f"[{float(row['simultaneous_interval_low']):+.4f}, "
                    f"{float(row['simultaneous_interval_high']):+.4f}]")
        print(f"| {row['model']} | {row['rq']} | {row['instance']} | "
              f"{row['left_condition']} - {row['right_condition']} | "
              f"{float(row['estimate']):+.4f} | {interval} | "
              f"{row['primary_delta_0_02_class']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
