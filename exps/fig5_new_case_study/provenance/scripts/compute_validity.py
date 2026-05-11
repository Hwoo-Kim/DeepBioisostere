#!/usr/bin/env python3
"""Compute SBDD validity summaries from docked summary CSV files.

The manuscript success-rate tables use the following per-row criteria:

- QED success: generated QED improves over the input molecule.
- SA success: generated SA is lower than the input molecule.
- Docking success: generated docking score stays within 1.36 of input docking.
- Joint success: all three criteria are satisfied.

This script reads either an extracted ``summary_*.csv`` file or a docking
tarball containing such a file and writes the compact ``validity_*.csv`` format
used by Main Table 2 and Supplementary Table 7.
"""

from __future__ import annotations

import argparse
import csv
import io
import tarfile
from pathlib import Path


STRATEGIES: tuple[tuple[str, str], ...] = (
    ("Random", "random"),
    ("Frequent", "frequency"),
    ("Rank-filtered MMPA", "rank_filtered_mmpa_qed_sa_10"),
    ("DeepBioisostere", "0.10_-1.0"),
)

SUMMARY_COLUMNS = (
    "QED_DIFF",
    "SA_DIFF",
    "INPUT-MOL-DOCKED-SCORE",
    "GEN-MOL-DOCKED-SCORE",
)


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    """Read a summary CSV directly or from a docking result tarball."""
    if path.suffix == ".tar":
        with tarfile.open(path) as tar:
            members = [
                member
                for member in tar.getmembers()
                if Path(member.name).name.startswith("summary_")
                and member.name.endswith(".csv")
            ]
            if len(members) != 1:
                raise ValueError(f"Expected one summary CSV in {path}, got {members}")
            handle = tar.extractfile(members[0])
            if handle is None:
                raise ValueError(f"Could not extract {members[0].name} from {path}")
            with io.TextIOWrapper(handle, encoding="utf-8", newline="") as text:
                return list(csv.DictReader(text))

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def require_summary_columns(rows: list[dict[str, str]], source: Path) -> None:
    if not rows:
        raise ValueError(f"No rows found in {source}")
    missing = [column for column in SUMMARY_COLUMNS if column not in rows[0]]
    if missing:
        raise ValueError(f"Missing required columns in {source}: {missing}")


def is_qed_valid(row: dict[str, str]) -> bool:
    return float(row["QED_DIFF"]) > 0.0


def is_sa_valid(row: dict[str, str]) -> bool:
    return float(row["SA_DIFF"]) < 0.0


def is_score_valid(row: dict[str, str]) -> bool:
    score_diff = float(row["GEN-MOL-DOCKED-SCORE"]) - float(
        row["INPUT-MOL-DOCKED-SCORE"]
    )
    return abs(score_diff) < 1.36


def count_csv_data_rows(path: Path) -> int:
    with path.open(newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def compute_validity_row(
    label: str,
    rows: list[dict[str, str]],
    generated_ratio_denominator: int | None = None,
) -> dict[str, str]:
    n_rows = len(rows)
    if n_rows == 0:
        raise ValueError("Cannot compute validity from an empty row set")

    qed_valid = [is_qed_valid(row) for row in rows]
    sa_valid = [is_sa_valid(row) for row in rows]
    score_valid = [is_score_valid(row) for row in rows]
    joint_valid = [
        qed_ok and sa_ok and score_ok
        for qed_ok, sa_ok, score_ok in zip(
            qed_valid,
            sa_valid,
            score_valid,
            strict=True,
        )
    ]

    if generated_ratio_denominator:
        generated_ratio = f"{n_rows / generated_ratio_denominator * 100:.15g}"
    else:
        generated_ratio = ""

    return {
        "": label,
        "QED_valid": str(sum(qed_valid) / n_rows),
        "SA_valid": str(sum(sa_valid) / n_rows),
        "SCORE_valid": str(sum(score_valid) / n_rows),
        "is_valid (joint)": str(sum(joint_valid) / n_rows),
        "Generated Ratio": generated_ratio,
    }


def find_summary_source(
    run_root: Path, model: str, target_idx: int, condition: str
) -> Path:
    docking_dir = run_root / "docking_gpu_results"
    extracted = docking_dir / f"summary_{model}_{target_idx}_{condition}.csv"
    if extracted.exists():
        return extracted

    tarballs = sorted(
        docking_dir.glob(f"GPU_{model}_{target_idx}_{condition}_*_docking_results.tar")
    )
    if not tarballs:
        raise FileNotFoundError(
            f"No summary CSV or docking tarball for {model} {target_idx} {condition}"
        )
    if len(tarballs) > 1:
        return tarballs[-1]
    return tarballs[0]


def write_validity_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "",
        "QED_valid",
        "SA_valid",
        "SCORE_valid",
        "is_valid (joint)",
        "Generated Ratio",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_single(args: argparse.Namespace) -> None:
    source = Path(args.summary_source)
    rows = read_summary_rows(source)
    require_summary_columns(rows, source)
    denominator = args.generated_ratio_denominator
    if denominator is None and args.candidate_csv:
        denominator = count_csv_data_rows(Path(args.candidate_csv))
    output_row = compute_validity_row(args.strategy_label, rows, denominator)
    write_validity_csv(Path(args.output_csv), [output_row])


def compute_from_run_root(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root)
    output_rows = []
    for label, condition in STRATEGIES:
        source = find_summary_source(run_root, args.model, args.target_idx, condition)
        rows = read_summary_rows(source)
        require_summary_columns(rows, source)
        output_rows.append(
            compute_validity_row(label, rows, args.generated_ratio_denominator)
        )

    output_csv = Path(args.output_dir) / f"validity_{args.model}_{args.target_idx}.csv"
    write_validity_csv(output_csv, output_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single")
    single.add_argument("--summary-source", required=True)
    single.add_argument("--strategy-label", required=True)
    single.add_argument("--output-csv", required=True)
    single.add_argument("--candidate-csv")
    single.add_argument("--generated-ratio-denominator", type=int)
    single.set_defaults(func=compute_single)

    run_root = subparsers.add_parser("run-root")
    run_root.add_argument("--run-root", required=True)
    run_root.add_argument("--model", required=True)
    run_root.add_argument("--target-idx", required=True, type=int)
    run_root.add_argument("--output-dir", required=True)
    run_root.add_argument("--generated-ratio-denominator", type=int)
    run_root.set_defaults(func=compute_from_run_root)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
