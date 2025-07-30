import argparse
from pathlib import Path

import pandas as pd

from scripts.utils import FrequencySampler, set_seed


def unnormalize_sa(norm_sa):
    return 10 - (norm_sa * 9)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate baseline data using DeepBioisostere model."
    )
    parser.add_argument(
        "model_name",
        help="Path to the SBDD generation CSV file containing input data.",
        type=str,
    )
    parser.add_argument(
        "target_idx",
        help="Index of the target protein in the SBDD generation CSV file.",
        type=int,
    )
    parser.add_argument(
        "--sbdd_gen_csv_dir",
        help="Directory containing the SBDD generation CSV file.",
        type=Path,
        default=Path("/home/share/DATA/wonho_SBDD_models_eval"),
    )
    parser.add_argument(
        "--result_dir",
        help="Directory to save the generated results.",
        type=Path,
        default=Path(
            "/home/mseok/work/DL/DeepBioisostere/Resubmission_DeepBioisostere/exps/fig5_new_case_study/20250531"
        ),
    )
    parser.add_argument(
        "--ranking_mode",
        type=str,
        default="frequency",
        choices=["frequency", "random", "rank_filtered_mmpa_qed_sa"],
        help="Sampling ranking strategy.",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=10,
        help="Min frequency for filtering replacement",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    assert args.sbdd_gen_csv_dir.is_dir(), (
        "SBDD generation CSV directory does not exist."
    )
    args.sbdd_gen_csv = args.sbdd_gen_csv_dir / f"{args.model_name}_eval/result.csv"

    # USER SETTINGS
    num_cores = 4
    batch_size = 512
    num_sample_each_mol = 100

    # Set paths
    filename = f"{args.ranking_mode}.csv"
    if args.ranking_mode == "rank_filtered_mmpa_qed_sa":
        filename = f"{args.ranking_mode}_{args.min_frequency}.csv"
    output_path = args.result_dir / args.model_name / str(args.target_idx) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # make inputs
    df = pd.read_csv(args.sbdd_gen_csv, low_memory=False)
    target_df = df[df["test_idx"] == args.target_idx]
    target_df: pd.DataFrame = target_df.copy()
    target_df.dropna(subset=["SMILES"], inplace=True)
    target_df.drop_duplicates(subset=["SMILES"], inplace=True)
    target_df["SA"] = target_df["SA"].apply(unnormalize_sa)
    input_list = target_df["SMILES"].tolist()
    assert len(input_list) == len(set(input_list)), "Input SMILES must be unique."

    sampler = FrequencySampler(
        smis=input_list,
        replacement_lib_path="/home/share/DATA/swkim/DeepBioisostere/replacement_library.csv",
        generate_all_attachments=True,
        ranking_mode=args.ranking_mode,
        min_frequency=args.min_frequency,
    )
    gen_df = sampler.sample(num_samples=num_sample_each_mol)

    merged_df = gen_df.merge(
        target_df[["SMILES", "QED", "SA"]],
        left_on="INPUT-MOL-SMI",
        right_on="SMILES",
        suffixes=("", "_ref"),
        how="left",
    )

    merged_df["QED_DIFF"] = merged_df["QED"] - merged_df["QED_ref"]
    merged_df["SA_DIFF"] = merged_df["SA"] - merged_df["SA_ref"]
    merged_df = merged_df.drop(["SMILES", "QED_ref", "SA_ref"], axis=1)
    if args.ranking_mode == "rank_filtered_mmpa_qed_sa":
        assert (merged_df["USED-REPLACEMENT-FREQ"] >= args.min_frequency).all(), (
            "Used replacement frequency must be greater than min_frequency."
        )

    merged_df.to_csv(output_path, index=False)
