#!/usr/bin/env python3
"""Prepare docking-only rerun inputs from packaged SBDD source data.

The packaged ``summary_*.csv`` files are the authoritative raw summaries used
for manuscript table reproduction. They also contain the candidate SMILES and
indices needed to rerun docking, but passing those files directly back into the
docking script would keep stale score columns in the input. This helper writes
clean candidate CSVs with only the columns consumed by ``run_gpu_docking.py``.
"""

from __future__ import annotations

import argparse
import csv
import tarfile
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "docking_rerun_inputs"
DEFAULT_REFERENCE_DIR = PACKAGE_ROOT / "raw_references" / "reference"

REQUIRED_COLUMNS = (
    "INPUT-MOL-IDX",
    "INPUT-MOL-SMI",
    "GEN-MOL-IDX",
    "GEN-MOL-SMI",
)

SUMMARY_DIRS = (
    PACKAGE_ROOT / "table_2" / "raw_summaries",
    PACKAGE_ROOT / "supplementary_table_7" / "raw_summaries",
)


def parse_summary_name(path: Path) -> tuple[str, str, str]:
    stem = path.stem
    prefix = "summary_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected summary filename: {path.name}")

    parts = stem[len(prefix) :].split("_")
    if len(parts) < 3:
        raise ValueError(f"Could not parse model, target, condition: {path.name}")

    model = parts[0]
    target_idx = parts[1]
    condition = "_".join(parts[2:])
    return model, target_idx, condition


def read_clean_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {missing}")
        return [{name: row[name] for name in REQUIRED_COLUMNS} for row in reader]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(REQUIRED_COLUMNS),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def iter_summary_paths() -> list[Path]:
    paths: list[Path] = []
    for summary_dir in SUMMARY_DIRS:
        paths.extend(sorted(summary_dir.glob("summary_*.csv")))
    return paths


def prepare_input_csvs(output_root: Path) -> list[dict[str, str]]:
    manifest_rows: list[dict[str, str]] = []
    for source_path in iter_summary_paths():
        model, target_idx, condition = parse_summary_name(source_path)
        rows = read_clean_rows(source_path)
        output_path = output_root / "gen" / model / target_idx / f"{condition}.csv"
        write_rows(output_path, rows)
        manifest_rows.append(
            {
                "source_summary": str(source_path.relative_to(PACKAGE_ROOT)),
                "docking_input": str(output_path.relative_to(output_root)),
                "model": model,
                "target_idx": target_idx,
                "condition": condition,
                "rows": str(len(rows)),
            }
        )
    return manifest_rows


def write_manifest(output_root: Path, rows: list[dict[str, str]]) -> None:
    manifest_path = output_root / "docking_input_manifest.csv"
    fieldnames = [
        "source_summary",
        "docking_input",
        "model",
        "target_idx",
        "condition",
        "rows",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_reference_tar(reference_dir: Path, output_root: Path) -> Path:
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {reference_dir}")

    tar_path = output_root / "reference.tar"
    with tarfile.open(tar_path, "w") as tar:
        for path in sorted(reference_dir.rglob("*")):
            arcname = Path("reference") / path.relative_to(reference_dir)
            tar.add(path, arcname=str(arcname))
    return tar_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where /gen inputs and manifest will be written.",
    )
    parser.add_argument(
        "--reference-dir",
        default=str(DEFAULT_REFERENCE_DIR),
        help="Packaged reference directory to archive as reference.tar.",
    )
    parser.add_argument(
        "--skip-reference-tar",
        action="store_true",
        help="Only write docking input CSVs, not reference.tar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = prepare_input_csvs(output_root)
    write_manifest(output_root, manifest_rows)
    print(f"Wrote {len(manifest_rows)} docking input CSVs under {output_root}")

    if not args.skip_reference_tar:
        tar_path = write_reference_tar(Path(args.reference_dir), output_root)
        print(f"Wrote reference tarball: {tar_path}")


if __name__ == "__main__":
    main()
