#!/usr/bin/env python
# coding: utf-8

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# --- Configuration ---
DEFAULT_BASE_PATH = Path(os.environ.get("SBDD_EVAL_ROOT", "raw_eval"))
# TARGET_INDICES_RANGE defines the range of target_idx to process, as in the original script.
TARGET_INDICES_RANGE = range(1, 101)

# Keys for property extraction and calculation. "SMILES" is included as in the original.
PROPERTY_KEYS_TO_EXTRACT = [
    "ref_QED",
    "QED",
    "ref_SA",
    "SA",
    "ref_min_aff",
    "ref_dock_aff",
    "gen_dock_aff",
    "SMILES",
]
# Primary properties to analyze for mean differences (e.g., QED, SA)
PRIMARY_PROPERTIES_TO_ANALYZE = ["qed", "sa"]


def load_evaluation_data(
    base_path: Path, model_name: str
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Loads the primary evaluation data (result.csv) and optionally the secondary
    data (result_.csv) for a given model.

    Args:
        base_path: The root directory containing model evaluation data.
        model_name: The name of the model.

    Returns:
        A tuple containing:
            - df_primary: DataFrame loaded from result.csv, or None if not found.
            - df_secondary: DataFrame loaded from result_.csv, or None if not found.
    """
    model_eval_path = base_path / f"{model_name}_eval"

    data_csv_primary_path = model_eval_path / "result.csv"
    data_csv_secondary_path = model_eval_path / "result_.csv"

    df_primary = None
    if data_csv_primary_path.exists():
        df_primary = pd.read_csv(data_csv_primary_path)

    df_secondary = None
    if data_csv_secondary_path.exists():
        df_secondary = pd.read_csv(data_csv_secondary_path)

    return df_primary, df_secondary


def compute_target_statistics(df: pd.DataFrame) -> defaultdict:
    """
    Gathers statistics for specified properties from the DataFrame,
    grouped by 'test_idx'. Handles NaN values in QED and invalid SMILES strings.
    This function is based on In[6] of the original script.

    Args:
        df: The input DataFrame (typically from result.csv).

    Returns:
        A defaultdict containing lists of property values for each test_idx.
        Structure: stats[test_idx][property_key] = [values_list]
    """
    stats = defaultdict(lambda: defaultdict(list))

    if df is None or "test_idx" not in df.columns:
        return stats

    for test_idx in sorted(df["test_idx"].unique()):
        df_subset = df[df["test_idx"] == test_idx].copy()

        # Determine invalid SMILES strings (NaN or containing '.')
        if "SMILES" in df_subset.columns:
            # Ensure SMILES is string before checking for '.' to avoid errors with non-string types
            is_invalid_smiles = df_subset["SMILES"].apply(
                lambda s: pd.isna(s) or (isinstance(s, str) and "." in s)
            )
            nan_smi_mask = is_invalid_smiles.values
        else:
            nan_smi_mask = np.zeros(len(df_subset), dtype=bool)

        # Determine NaN QED values
        if "QED" in df_subset.columns:
            nan_qed_mask = np.isnan(df_subset["QED"].values)
        else:
            nan_qed_mask = np.zeros(len(df_subset), dtype=bool)

        # Original assertion: assert (not np.all(np.logical_xor(nan_qed_mask, nan_smi_mask)))
        # This assertion passes if nan_qed_mask and nan_smi_mask are:
        #   1. Identical for all entries.
        #   2. Mixed (some same, some different).
        # It only fails if nan_qed_mask and nan_smi_mask are different for ALL entries.
        if len(nan_qed_mask) > 0 and len(nan_smi_mask) == len(
            nan_qed_mask
        ):  # Avoid error on empty or mismatched arrays
            assert not np.all(np.logical_xor(nan_qed_mask, nan_smi_mask)), (
                f"Assertion failed for test_idx {test_idx}: np.all(np.logical_xor(nan_qed_mask, nan_smi_mask)) is True"
            )

        # Filter out entries with invalid SMILES, as per original script's logic
        valid_entries_mask = ~nan_smi_mask

        for key in PROPERTY_KEYS_TO_EXTRACT:
            if key in df_subset.columns:
                stats[test_idx][key].extend(
                    df_subset.loc[valid_entries_mask, key].tolist()
                )
            # else:
            # Silently skip if key is not in subset, matching original behavior for df_

    return stats


def calculate_property_means_and_diff(
    stats: defaultdict,
    target_idx: int,
    prop_name: str,
    model_name: str,  # Kept for consistency, was used for plot paths
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calculates the mean of a generated property, its reference, and their difference.
    This function is based on `draw_prop_dists` from In[27]. Plotting is omitted here
    but path creation logic was originally present.

    Args:
        stats: The statistics dictionary.
        target_idx: The specific target index to analyze.
        prop_name: The base name of the property (e.g., "qed", "sa").
        model_name: Name of the model.

    Returns:
        A tuple (mean_gen, mean_ref, mean_diff). Returns (None, None, None) if data is missing/empty.
    """
    prop_upper = prop_name.upper()
    ref_prop_key = f"ref_{prop_upper}"

    if not (
        target_idx in stats
        and prop_upper in stats[target_idx]
        and ref_prop_key in stats[target_idx]
    ):
        return None, None, None

    gen_values = np.array(stats[target_idx][prop_upper])
    ref_values = np.array(stats[target_idx][ref_prop_key])

    if gen_values.size == 0 or ref_values.size == 0:
        return None, None, None

    mean_gen = np.mean(gen_values)
    mean_ref = np.mean(ref_values)
    mean_diff = mean_gen - mean_ref

    return mean_gen, mean_ref, mean_diff


def analyze_properties_per_target(stats: defaultdict, model_name: str) -> pd.DataFrame:
    """
    Analyzes specified properties (e.g., QED, SA) for each target index by calculating
    means of generated values, reference values, and their differences.
    This function replaces the loop in In[27].

    Args:
        stats: The statistics dictionary from `compute_target_statistics`.
        model_name: The name of the model.

    Returns:
        A DataFrame summarizing the analyzed property metrics for each target.
    """
    analysis_results = []

    # Iterate through a defined range of target indices, as in the original script.
    for target_idx in TARGET_INDICES_RANGE:
        target_data = {"target_idx": target_idx, "model_name": model_name}

        # Check if stats exist for this target_idx before proceeding
        if target_idx not in stats:
            for prop_name in PRIMARY_PROPERTIES_TO_ANALYZE:
                target_data[f"{prop_name}_gen"] = np.nan
                target_data[f"{prop_name}_ref"] = np.nan
                target_data[f"{prop_name}_diff"] = np.nan
        else:
            for prop_name in PRIMARY_PROPERTIES_TO_ANALYZE:
                gen_mean, ref_mean, diff_mean = calculate_property_means_and_diff(
                    stats, target_idx, prop_name, model_name
                )
                target_data[f"{prop_name}_gen"] = gen_mean
                target_data[f"{prop_name}_ref"] = ref_mean
                target_data[f"{prop_name}_diff"] = diff_mean

        analysis_results.append(target_data)

    return pd.DataFrame(analysis_results)


def filter_analysis_dataframe(
    analyzed_df: pd.DataFrame,
    qed_diff_threshold: float = -0.1,
    sa_diff_threshold: float = -0.1,
) -> pd.DataFrame:
    """
    Filters the analyzed property DataFrame based on QED and SA difference thresholds.
    This function is based on In[32] of the original script.

    Args:
        analyzed_df: DataFrame from `analyze_properties_per_target`.
        qed_diff_threshold: Threshold for 'qed_diff'.
        sa_diff_threshold: Threshold for 'sa_diff'.

    Returns:
        Filtered DataFrame.
    """
    if analyzed_df.empty:
        return pd.DataFrame()  # Return empty DataFrame

    # Ensure required columns exist
    required_cols = ["qed_diff", "sa_diff"]
    if not all(col in analyzed_df.columns for col in required_cols):
        return analyzed_df

    # Drop rows where diffs are NaN before comparison, as NaN comparisons are tricky
    # and original script's direct comparison would exclude NaNs implicitly.
    # (e.g., `np.nan < -0.1` is False)
    clean_df = analyzed_df.dropna(subset=["qed_diff", "sa_diff"])

    filtered_df = clean_df[
        (clean_df["qed_diff"] < qed_diff_threshold)
        & (clean_df["sa_diff"] < sa_diff_threshold)
    ].copy()  # Use .copy() to avoid SettingWithCopyWarning on slices

    # sort values from min value of min(qed, sa) to max
    filtered_df["min_qed_sa"] = filtered_df[["qed_diff", "sa_diff"]].min(axis=1)
    filtered_df = filtered_df.sort_values(by="min_qed_sa").reset_index(drop=True)
    filtered_df.drop(columns=["min_qed_sa"], inplace=True)
    return filtered_df


def main_processing_pipeline(
    model_name: str, base_path_str: Optional[str] = None
) -> pd.DataFrame:
    """
    Main processing pipeline that loads data, computes statistics, analyzes properties,
    and filters the results for a specified model.

    Args:
        model_name: The name of the model to process.
        base_path_str: Optional. String path to the root directory of datasets.
                       If None, uses DEFAULT_BASE_PATH.

    Returns:
        A DataFrame containing targets filtered by QED and SA difference thresholds.
        This is equivalent to `filtered_df` in In[32] of the original notebook.
    """
    current_base_path = Path(base_path_str) if base_path_str else DEFAULT_BASE_PATH

    df_primary, df_secondary = load_evaluation_data(current_base_path, model_name)
    target_stats = compute_target_statistics(df_primary)
    analyzed_properties_df = analyze_properties_per_target(target_stats, model_name)
    final_filtered_df = filter_analysis_dataframe(analyzed_properties_df)
    return final_filtered_df


if __name__ == "__main__":
    # Configure pandas display options to match notebook's typical output for all data
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 2000)  # Adjust display width for wider tables

    # Specify the model name, as in the original script
    model_to_process = sys.argv[1]

    # To use a custom base path, provide it as a string:
    # custom_data_path = "/path/to/your/data"
    # filtered_dataframe = main_processing_pipeline(model_name=model_to_process, base_path_str=custom_data_path)

    # Run the main pipeline with the default base path
    filtered_dataframe = main_processing_pipeline(model_name=model_to_process)
    print(filtered_dataframe)

    # The 'filtered_dataframe' variable now holds the final result, equivalent to
    # 'filtered_df' from In[32] of your notebook.
    # You can perform further actions with it below if needed.
    # Example: Save to CSV
    # filtered_dataframe.to_csv(f"{model_to_process}_filtered_results.csv", index=False)
    # print(f"\nSaved filtered results to {model_to_process}_filtered_results.csv")
