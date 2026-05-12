#!/usr/bin/env python3
"""Reproduce packaged SBDD table, figure, and provenance checks.

This script intentionally uses only the Python standard library so it can run in
minimal environments. It does not rerun GPU docking; use the Slurm workflow in
README.md for full computational regeneration.
"""

from __future__ import annotations

import csv
import glob
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable


def find_package_root() -> Path:
    cwd = Path.cwd()
    script_root = Path(__file__).resolve().parent
    candidates = [
        cwd,
        cwd / "exps" / "fig5_new_case_study",
        script_root,
    ]
    for candidate in candidates:
        if (candidate / "MANIFEST.tsv").exists():
            return candidate
    return script_root


ROOT = find_package_root()

MODELS = ("DeepICL", "pocket2mol", "targetdiff", "decompdiff")

STRATEGIES = (
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

VALIDITY_COLUMNS = (
    ("QED_valid", "QED"),
    ("SA_valid", "SA"),
    ("SCORE_valid", "Docking"),
    ("is_valid (joint)", "Joint"),
)

DOCKING_SCORE_TOLERANCE = 1.36
FLOAT_TOLERANCE = 1e-8

MODEL_DISPLAY_NAMES = {
    "DeepICL": "DeepICL",
    "pocket2mol": "Pocket2Mol",
    "targetdiff": "TargetDiff",
    "decompdiff": "DecompDiff",
}


def read_csv_dict(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_tsv_dict(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_validity_csv(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            strategy = row.pop("", None)
            row["strategy"] = strategy or ""
            rows.append(row)
    return rows


def round_half_up(value: object, places: str = "0.01") -> str:
    return str(
        Decimal(str(float(value))).quantize(
            Decimal(places),
            rounding=ROUND_HALF_UP,
        )
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_summary_columns(rows: list[dict[str, str]], source: Path) -> None:
    require(rows, f"No rows found in summary CSV: {source}")
    missing = [column for column in SUMMARY_COLUMNS if column not in rows[0]]
    require(not missing, f"Missing summary columns in {source}: {missing}")


def is_qed_valid(row: dict[str, str]) -> bool:
    return float(row["QED_DIFF"]) > 0.0


def is_sa_valid(row: dict[str, str]) -> bool:
    return float(row["SA_DIFF"]) < 0.0


def is_score_valid(row: dict[str, str]) -> bool:
    score_diff = float(row["GEN-MOL-DOCKED-SCORE"]) - float(
        row["INPUT-MOL-DOCKED-SCORE"]
    )
    return abs(score_diff) < DOCKING_SCORE_TOLERANCE


def compute_validity_from_summary(path: Path) -> dict[str, float]:
    rows = read_csv_dict(path)
    require_summary_columns(rows, path)
    n_rows = len(rows)

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

    return {
        "QED_valid": sum(qed_valid) / n_rows,
        "SA_valid": sum(sa_valid) / n_rows,
        "SCORE_valid": sum(score_valid) / n_rows,
        "is_valid (joint)": sum(joint_valid) / n_rows,
    }


def require_close(
    actual: float,
    expected: str,
    context: str,
    tolerance: float = FLOAT_TOLERANCE,
) -> None:
    expected_float = float(expected)
    require(
        abs(actual - expected_float) <= tolerance,
        f"{context}: computed {actual:.17g} != source CSV {expected}",
    )


def check_manifest() -> None:
    missing = []
    for row in read_tsv_dict(ROOT / "MANIFEST.tsv"):
        rel_path = row["path"]
        if "*" in rel_path:
            if not glob.glob(str(ROOT / rel_path)):
                missing.append(rel_path)
        elif not (ROOT / rel_path).exists():
            missing.append(rel_path)
    require(not missing, f"Missing manifest paths: {missing}")
    print("manifest: ok")


def summarize_validity(
    folder: str, target_indices: Iterable[int]
) -> list[dict[str, str]]:
    rows = []
    for target_idx in target_indices:
        for model in MODELS:
            validity_path = ROOT / folder / f"validity_{model}_{target_idx}.csv"
            require(validity_path.exists(), f"Missing validity CSV: {validity_path}")
            aggregate_rows = {
                row["strategy"]: row for row in read_validity_csv(validity_path)
            }
            expected_strategies = {label for label, _condition in STRATEGIES}
            require(
                set(aggregate_rows) == expected_strategies,
                f"Unexpected strategies in {validity_path}: {set(aggregate_rows)}",
            )
            for strategy_label, condition in STRATEGIES:
                summary_path = (
                    ROOT
                    / folder
                    / "raw_summaries"
                    / f"summary_{model}_{target_idx}_{condition}.csv"
                )
                require(summary_path.exists(), f"Missing raw summary: {summary_path}")
                computed = compute_validity_from_summary(summary_path)
                aggregate_row = aggregate_rows[strategy_label]
                for aggregate_column, _display_column in VALIDITY_COLUMNS:
                    require_close(
                        computed[aggregate_column],
                        aggregate_row[aggregate_column],
                        (
                            f"{model} target {target_idx} {strategy_label} "
                            f"{aggregate_column}"
                        ),
                    )
                rows.append(
                    {
                        "target_idx": str(target_idx),
                        "source_model": model,
                        "strategy": strategy_label,
                        "QED": round_half_up(computed["QED_valid"]),
                        "SA": round_half_up(computed["SA_valid"]),
                        "Docking": round_half_up(computed["SCORE_valid"]),
                        "Joint": round_half_up(computed["is_valid (joint)"]),
                        "aggregate_source_file": str(validity_path.relative_to(ROOT)),
                        "raw_summary_file": str(summary_path.relative_to(ROOT)),
                    }
                )
    return rows


def check_table_2() -> None:
    rows = summarize_validity("table_2", [68])
    require(len(rows) == 16, f"Expected 16 Main Table 2 rows, got {len(rows)}")
    print("main_table_2: ok")


def check_supplementary_table_6() -> None:
    selected_rows = read_csv_dict(
        ROOT / "supplementary_table_6" / "table6_selected_targets.csv"
    )
    all_rows = read_csv_dict(
        ROOT / "supplementary_table_6" / "avg_result_all_targets.csv"
    )
    mapping = {
        row["target_idx"]: row["pdb_id"]
        for row in read_csv_dict(ROOT / "target_mapping.csv")
    }
    all_rows_by_key = {
        (row["test_idx"], MODEL_DISPLAY_NAMES[row["MODEL"]]): row for row in all_rows
    }

    require(
        len(selected_rows) == 12,
        f"Expected 12 selected-target rows, got {len(selected_rows)}",
    )
    for row in selected_rows:
        key = (row["target_idx"], row["model"])
        source_row = all_rows_by_key[key]
        transformed_sa = 10.0 - 9.0 * float(source_row["SA"])

        require(
            row["source_file"] == "supplementary_table_6/avg_result_all_targets.csv",
            f"Unexpected Table 6 source file for {key}: {row['source_file']}",
        )
        require(mapping[row["target_idx"]] == row["pdb_id"], f"PDB mismatch: {row}")
        require_close(float(row["qed_mean"]), source_row["QED"], f"Table 6 {key} QED")
        require_close(float(row["sa_raw"]), source_row["SA"], f"Table 6 {key} raw SA")
        require_close(
            float(row["sa_score_transformed"]),
            str(transformed_sa),
            f"Table 6 {key} transformed SA",
        )
    print("supplementary_table_6: ok")


def check_supplementary_table_7() -> None:
    rows = summarize_validity("supplementary_table_7", [36, 84])
    require(len(rows) == 32, f"Expected 32 Supplementary Table 7 rows, got {len(rows)}")
    mapping = {
        row["target_idx"]: row["pdb_id"]
        for row in read_csv_dict(ROOT / "target_mapping.csv")
    }
    require(mapping == {"36": "4P77", "68": "4RV4", "84": "2EWY"}, str(mapping))
    print("supplementary_table_7: ok")


def index_rows(path: Path) -> set[tuple[str, str]]:
    rows = read_csv_dict(path)
    return {(str(row["INPUT-MOL-IDX"]), str(row["GEN-MOL-IDX"])) for row in rows}


def ordered_row_keys(rows: list[dict[str, str]]) -> list[tuple[str, str]]:
    return [(str(row["INPUT-MOL-IDX"]), str(row["GEN-MOL-IDX"])) for row in rows]


def figure5_score_diff(row: dict[str, str]) -> float:
    return float(row["GEN-MOL-DOCKED-SCORE"]) - float(row["INPUT-MOL-DOCKED-SCORE"])


def is_figure5_candidate(row: dict[str, str]) -> bool:
    score_diff = figure5_score_diff(row)
    return (
        float(row["QED"]) > 0.4
        and float(row["QED_DIFF"]) > 0.0
        and float(row["QED_DIFF"]) <= 0.1
        and float(row["SA_DIFF"]) < -0.5
        and float(row["SA_DIFF"]) >= -1.0
        and abs(score_diff) < 0.2
        and float(row["QED_DIFF"]) > 0.08
    )


def compute_figure5_candidate_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    for row in read_csv_dict(path):
        if is_figure5_candidate(row):
            output_row = dict(row)
            output_row["SCORE_DIFF"] = str(figure5_score_diff(row))
            rows.append(output_row)
    rows.sort(
        key=lambda row: (
            float(row["SA_DIFF"]),
            int(row["INPUT-MOL-IDX"]),
            int(row["GEN-MOL-IDX"]),
        )
    )
    return rows


def check_figure5_candidate_summary(summary_path: Path) -> set[tuple[str, str]]:
    candidate_path = ROOT / "figure_5" / "figure5_candidate_summary.csv"
    legacy_path = ROOT / "figure_5" / "filtered_deepicl_68.csv"
    require(
        candidate_path.exists(), f"Missing Figure 5 candidate CSV: {candidate_path}"
    )
    candidate_rows = read_csv_dict(candidate_path)
    legacy_rows = read_csv_dict(legacy_path)
    computed_rows = compute_figure5_candidate_rows(summary_path)

    require(
        len(candidate_rows) == 19,
        f"Expected 19 Figure 5 candidate rows, got {len(candidate_rows)}",
    )
    require(
        ordered_row_keys(candidate_rows) == ordered_row_keys(legacy_rows),
        "Figure 5 candidate summary row order differs from the legacy CSV",
    )
    require(
        ordered_row_keys(candidate_rows) == ordered_row_keys(computed_rows),
        "Figure 5 candidate row order does not match the raw-summary filter",
    )
    computed_by_key = {
        (row["INPUT-MOL-IDX"], row["GEN-MOL-IDX"]): row for row in computed_rows
    }
    for row in candidate_rows:
        key = (row["INPUT-MOL-IDX"], row["GEN-MOL-IDX"])
        require(is_figure5_candidate(row), f"Figure 5 candidate fails filters: {key}")
        computed_row = computed_by_key[key]
        for column in (
            "QED",
            "SA",
            "QED_DIFF",
            "SA_DIFF",
            "INPUT-MOL-DOCKED-SCORE",
            "GEN-MOL-DOCKED-SCORE",
        ):
            require_close(
                float(row[column]),
                computed_row[column],
                f"Figure 5 candidate {key} {column}",
            )
        require(
            round_half_up(row["SCORE_DIFF"], "0.000000000000001")
            == round_half_up(figure5_score_diff(row), "0.000000000000001"),
            f"Figure 5 SCORE_DIFF mismatch: {key}",
        )

    return set(ordered_row_keys(candidate_rows))


def check_figure_5() -> None:
    examples = read_csv_dict(ROOT / "figure_5" / "figure5_selected_examples.csv")
    require(len(examples) == 2, f"Expected two Figure 5 examples, got {len(examples)}")
    summary_path = ROOT / "figure_5" / "summary_DeepICL_68_0.10_-1.0.csv"
    summary_rows = read_csv_dict(summary_path)
    summary_by_key = {
        (str(row["INPUT-MOL-IDX"]), str(row["GEN-MOL-IDX"])): row
        for row in summary_rows
    }
    candidate_keys = check_figure5_candidate_summary(summary_path)
    for row in examples:
        key = (str(row["input_mol_idx"]), str(row["gen_mol_idx"]))
        source_csv = row["source_csv"]
        require(
            source_csv == "figure_5/summary_DeepICL_68_0.10_-1.0.csv",
            f"Unexpected Figure 5 source CSV for {key}: {source_csv}",
        )
        require(key in summary_by_key, f"Figure 5 row missing from summary CSV: {key}")
        source_row = summary_by_key[key]

        source_original_qed = float(source_row["QED"]) - float(source_row["QED_DIFF"])
        source_original_sa = float(source_row["SA"]) - float(source_row["SA_DIFF"])
        source_score_diff = figure5_score_diff(source_row)

        require(row["target_idx"] == "68", f"Unexpected Figure 5 target: {row}")
        require(row["pdb_id"] == "4RV4", f"Unexpected Figure 5 PDB ID: {row}")
        require(row["source_model"] == "DeepICL", f"Unexpected Figure 5 model: {row}")
        require(row["input_mol_smi"] == source_row["INPUT-MOL-SMI"], str(key))
        require(row["gen_mol_smi"] == source_row["GEN-MOL-SMI"], str(key))
        require(row["leaving_frag_smi"] == source_row["LEAVING-FRAG-SMI"], str(key))
        require(row["inserting_frag_smi"] == source_row["INSERTING-FRAG-SMI"], str(key))
        require_close(
            float(row["predicted_prob"]),
            source_row["PREDICTED-PROB"],
            f"Figure 5 {key} predicted probability",
        )
        require_close(
            float(row["original_qed"]),
            str(source_original_qed),
            f"Figure 5 {key} original QED",
        )
        require_close(
            float(row["generated_qed"]),
            source_row["QED"],
            f"Figure 5 {key} generated QED",
        )
        require_close(
            float(row["original_sa"]),
            str(source_original_sa),
            f"Figure 5 {key} original SA",
        )
        require_close(
            float(row["generated_sa"]),
            source_row["SA"],
            f"Figure 5 {key} generated SA",
        )
        require_close(
            float(row["original_docking_score"]),
            source_row["INPUT-MOL-DOCKED-SCORE"],
            f"Figure 5 {key} original docking",
        )
        require_close(
            float(row["generated_docking_score"]),
            source_row["GEN-MOL-DOCKED-SCORE"],
            f"Figure 5 {key} generated docking",
        )
        require_close(
            float(row["qed_diff"]),
            source_row["QED_DIFF"],
            f"Figure 5 {key} QED diff",
        )
        require_close(
            float(row["sa_diff"]),
            source_row["SA_DIFF"],
            f"Figure 5 {key} SA diff",
        )
        require(
            round_half_up(row["score_diff"], "0.000000000000001")
            == round_half_up(source_score_diff, "0.000000000000001"),
            f"Figure 5 {key} score diff mismatch",
        )
        require(
            key in candidate_keys,
            f"Figure 5 row missing from candidate summary CSV: {key}",
        )
        image_path = ROOT / "figure_5" / row["image_file"]
        require(
            image_path.exists() and image_path.stat().st_size > 0,
            f"Missing or empty Figure 5 image: {image_path}",
        )
    print("main_figure_5: ok")


def check_raw_eval() -> None:
    for model in ["DeepICL", "decompdiff", "pocket2mol", "targetdiff"]:
        path = ROOT / "raw_eval" / f"{model}_eval_result.csv"
        rows = read_csv_dict(path)
        targets = {int(float(row["test_idx"])) for row in rows if row.get("test_idx")}
        require(len(rows) == 10000, f"{path} has {len(rows)} rows")
        require(targets == set(range(1, 101)), f"{path} target coverage is wrong")
    print("raw_eval: ok")


def check_slurm_provenance() -> None:
    jobs = read_csv_dict(ROOT / "provenance" / "final_docking_job_index.csv")
    job_ids = [int(row["job_id"]) for row in jobs]
    require(len(job_ids) == 24, f"Expected 24 jobs, got {len(job_ids)}")
    require(job_ids == list(range(262943, 262967)), f"Unexpected job IDs: {job_ids}")
    inventory = (ROOT / "provenance" / "docking_tarball_inventory.tsv").read_text()
    for row in jobs:
        require(row["exit_code"] == "0", f"Nonzero exit code for job {row['job_id']}")
        require(row["tarball_created"] == "yes", f"No tarball for job {row['job_id']}")
        stdout = (ROOT / "provenance" / row["stdout_log"]).read_text(errors="replace")
        require(
            "Python script finished with exit code 0" in stdout,
            f"Missing success exit message for job {row['job_id']}",
        )
        require(
            "Successfully created tarball" in stdout,
            f"Missing tarball success message for job {row['job_id']}",
        )
        require(
            row["job_id"] in inventory,
            f"Job {row['job_id']} missing from tar inventory",
        )
    print("slurm_provenance: ok")


def main() -> None:
    print(f"Using source data: {ROOT}")
    check_manifest()
    check_raw_eval()
    check_table_2()
    check_supplementary_table_6()
    check_supplementary_table_7()
    check_figure_5()
    check_slurm_provenance()
    print("All SBDD source-data reproduction checks passed.")


if __name__ == "__main__":
    main()
