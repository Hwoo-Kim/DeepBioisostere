#!/usr/bin/env python3
"""Prepare the filtered candidate summary used before drawing Figure 5.

The final Figure 5 candidate images were generated from a notebook cell that
read the DeepICL target-68 docking summary, applied the filters below, sorted
the remaining rows by SA improvement, and then drew one image per retained row.

This script provides the same filtering step as a reproducible, notebook-free
entry point. It uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CSV = PACKAGE_ROOT / "figure_5" / "summary_DeepICL_68_0.10_-1.0.csv"
DEFAULT_OUTPUT_CSV = PACKAGE_ROOT / "figure_5" / "figure5_candidate_summary.csv"


def row_key(row: dict[str, str]) -> tuple[int, int]:
    return (int(row["INPUT-MOL-IDX"]), int(row["GEN-MOL-IDX"]))


def score_diff(row: dict[str, str]) -> float:
    return float(row["GEN-MOL-DOCKED-SCORE"]) - float(row["INPUT-MOL-DOCKED-SCORE"])


def is_figure5_candidate(row: dict[str, str]) -> bool:
    diff = score_diff(row)
    return (
        float(row["QED"]) > 0.4
        and float(row["QED_DIFF"]) > 0.0
        and float(row["QED_DIFF"]) <= 0.1
        and float(row["SA_DIFF"]) < -0.5
        and float(row["SA_DIFF"]) >= -1.0
        and abs(diff) < 0.2
        and float(row["QED_DIFF"]) > 0.08
    )


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        return list(reader.fieldnames), list(reader)


def prepare_candidate_summary(
    source_csv: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames, rows = read_rows(source_csv)
    output_fieldnames = [*fieldnames, "SCORE_DIFF"]
    candidate_rows = []
    for row in rows:
        output_row = dict(row)
        output_row["SCORE_DIFF"] = str(score_diff(row))
        if is_figure5_candidate(output_row):
            candidate_rows.append(output_row)
    candidate_rows.sort(key=lambda row: (float(row["SA_DIFF"]), *row_key(row)))
    return output_fieldnames, candidate_rows


def write_rows(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-csv",
        default=str(DEFAULT_SOURCE_CSV),
        help="DeepICL target-68 summary CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Filtered candidate summary CSV to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = Path(args.output_csv)
    fieldnames, rows = prepare_candidate_summary(Path(args.source_csv))
    write_rows(output_csv, fieldnames, rows)
    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
