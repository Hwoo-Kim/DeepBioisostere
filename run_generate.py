import argparse
import os
import time
from pathlib import Path

import pandas as pd
import torch

from scripts.generate import Generator
from scripts.conditioning import Conditioner
from scripts.model import DeepBioisostere
from scripts.utils import set_seed


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
            "/home/mseok/work/DL/DeepBioisostere/Resubmission_DeepBioisostere/exps/fig5_new_case_study/20250611_final"
        ),
    )
    parser.add_argument(
        "--properties",
        help="Properties to control during generation (comma-separated).",
        type=str,
        nargs="+",
        default=["qed", "sa"],
    )
    parser.add_argument(
        "--target_properties",
        help="Target properties for the generation (comma-separated).",
        type=str,
        nargs="+",
        default=["0.10", "-1.0"],
    )
    parser.add_argument(
        "--seed",
        help="SEED",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    assert args.sbdd_gen_csv_dir.is_dir(), (
        "SBDD generation CSV directory does not exist."
    )
    args.sbdd_gen_csv = args.sbdd_gen_csv_dir / f"{args.model_name}_eval/result.csv"
    set_seed(args.seed)

    # USER SETTINGS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_cores = 4
    batch_size = 512
    num_sample_each_mol = 100
    new_frag_type = "all"

    # Set paths
    properties = sorted(args.properties)
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = f"{proj_dir}/model_save/DeepBioisostere_{'_'.join(properties)}.pt"
    frag_lib_path = f"{proj_dir}/fragment_library/"
    output_path = (
        args.result_dir
        / "gen"
        / args.model_name
        / str(args.target_idx)
        / ("_".join(args.target_properties) + ".csv")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("=== Configuration Values ===")
    print(f"seed: {args.seed}")
    print(f"device: {device}")
    print(f"num_cores: {num_cores}")
    print(f"batch_size: {batch_size}")
    print(f"num_sample_each_mol: {num_sample_each_mol}")
    print(f"new_frag_type: {new_frag_type}")
    print(f"properties: {properties}")
    print(f"proj_dir: {proj_dir}")
    print(f"model_path: {model_path}")
    print(f"frag_lib_path: {frag_lib_path}")
    print(f"output_path: {output_path}")
    print("=" * 30)

    # Initialize model and generator
    model = DeepBioisostere.from_trained_model(model_path, properties=properties)
    conditioner = Conditioner(
        phase="generation",
        properties=properties,
    )
    generator = Generator(
        model=model,
        processed_frag_dir=frag_lib_path,
        conditioner=conditioner,
        device=device,
        num_cores=num_cores,
        batch_size=batch_size,
        new_frag_type=new_frag_type,
        num_sample_each_mol=num_sample_each_mol,
        properties=properties,
    )

    # make inputs
    df = pd.read_csv(args.sbdd_gen_csv, low_memory=False)
    target_df = df[df["test_idx"] == args.target_idx]
    target_df = target_df.copy()

    # apply unnormalize function to SA property
    target_df["SA"] = target_df["SA"].apply(unnormalize_sa)
    target_df.dropna(subset=["SMILES"], inplace=True)

    input_list = target_df["SMILES"].tolist()
    prop_input = dict(zip(args.properties, list(map(float, args.target_properties))))
    input_list = [(inp, prop_input) for inp in input_list]
    print(input_list)

    start_time = time.time()
    result_df = generator.generate(input_list)
    elapsed_time = time.time() - start_time
    print(f"Generation completed in {elapsed_time:.2f} seconds.")
    print(f"Generated {len(result_df)} molecules.")

    result_df.to_csv(output_path.parent / f"raw_{output_path.name}", index=False)

    merged_df = result_df.merge(
        target_df[["SMILES", "QED", "SA"]],
        left_on="INPUT-MOL-SMI",
        right_on="SMILES",
        suffixes=("", "_ref"),
        how="left",
    )

    merged_df["QED_DIFF"] = merged_df["QED"] - merged_df["QED_ref"]
    merged_df["SA_DIFF"] = merged_df["SA"] - merged_df["SA_ref"]
    merged_df = merged_df.drop(["SMILES", "QED_ref", "SA_ref"], axis=1)

    merged_df.to_csv(output_path, index=False)
