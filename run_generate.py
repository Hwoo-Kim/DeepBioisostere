import argparse
import os
import time
from pathlib import Path

import pandas as pd
import torch

from deepbioisostere.conditioning import Conditioner
from deepbioisostere.generate import Generator
from deepbioisostere.model import DeepBioisostere
from deepbioisostere.utils import set_seed

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


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
        help=(
            "Directory holding <model_name>_eval/result.csv for the SBDD "
            "baselines. This is study-specific input with no sensible default."
        ),
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--result_dir",
        help="Directory to save the generated results (default: ./fig5_results).",
        type=Path,
        default=Path("fig5_results"),
    )
    parser.add_argument(
        "--trained_model_dir",
        help=(
            "Directory holding the checkpoints. Omit to resolve them the normal "
            "way: local checkout, then $DEEPBIOISOSTERE_ASSET_DIR, then the Hub."
        ),
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--frag_lib_dir",
        help=(
            "Directory holding the fragment library. Omit to resolve it the "
            "normal way, as above."
        ),
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--ablation",
        help="Ablation study flag (default: False).",
        action="store_true",
        default=False,
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
    if args.trained_model_dir is None:
        model_path = None  # resolved by from_pretrained below
    elif args.ablation:
        model_path = (
            args.trained_model_dir
            / f"DeepBioisostere_{'_'.join(properties)}_ablation.pt"
        )
    else:
        model_path = (
            args.trained_model_dir / f"DeepBioisostere_{'_'.join(properties)}.pt"
        )

    # None here means "let Generator resolve it", not "no library".
    frag_lib_path = args.frag_lib_dir
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
    print(f"model_path: {model_path or '(resolved: checkout / env / Hub)'}")
    print(f"frag_lib_path: {frag_lib_path or '(resolved: checkout / env / Hub)'}")
    print(f"output_path: {output_path}")
    print("=" * 30)

    # Initialize model and generator
    use_subgraph_AMPN = (
        not args.ablation
    )  # False is for ablation model. default is True.
    model_args = {
        "mol_node_hid_dim": 256,
        "mol_edge_hid_dim": 256,
        "mol_num_emb_layer": 5,
        "frag_message_passing_num_layer": 3,
        "use_subgraph_AMPN": use_subgraph_AMPN,
    }
    if model_path is None:
        model = DeepBioisostere.from_pretrained(
            properties=properties, ablation=args.ablation, **model_args
        )
    else:
        model = DeepBioisostere.from_trained_model(
            model_path, properties=properties, **model_args
        )
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
    target_df: pd.DataFrame = target_df.copy()
    target_df.dropna(subset=["SMILES"], inplace=True)
    target_df.drop_duplicates(subset=["SMILES"], inplace=True)
    target_df["SA"] = target_df["SA"].apply(unnormalize_sa)
    input_list = target_df["SMILES"].tolist()
    assert len(input_list) == len(set(input_list)), "Input SMILES must be unique."

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
