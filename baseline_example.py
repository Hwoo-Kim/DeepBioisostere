import os
import time

from rdkit import Chem

from scripts.baseline_generator import BaselineGenerator
from scripts.conditioning import Conditioner
from scripts.model import DeepBioisostere
from scripts.property import calc_logP, calc_Mw, calc_QED, calc_SAscore


def print_properties(smi: str):
    print(f"SMILES: {smi}", end=", ")
    mol = Chem.MolFromSmiles(smi)
    print(f"logP: {calc_logP(mol):.3f}", end=", ")
    print(f"QED: {calc_QED(mol):.3f}", end=", ")
    print(f"Mw: {calc_Mw(mol):.3f}", end=", ")
    print(f"SAscore: {calc_SAscore(mol):.3f}")


if __name__ == "__main__":
    smi1 = "ClC(Cc1c(C(Nc2c(Br)cccc2)=O)cccc1)=O"
    smi2 = "Cc1ccc2cnc(N(C)CCc3ccccn3)nc2c1"
    print("Original molecules:")
    print_properties(smi1)
    print_properties(smi2)
    print()

    # USER SETTINGS
    device = "cpu"
    num_cores = 4
    batch_size = 512
    num_sample_each_mol = 100
    new_frag_type = "all"
    properties_to_control = ["mw", "logp"]

    # Set paths
    properties = sorted(properties_to_control)
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = f"{proj_dir}/model_save/DeepBioisostere_{'_'.join(properties)}.pt"
    frag_lib_path = f"{proj_dir}/fragment_library/"

    # Initialize conditioner for property control
    conditioner = Conditioner(
        phase="generation",
        properties=properties,
    )

    # Initialize model for strategy 2
    model = DeepBioisostere.from_trained_model(model_path, properties=properties)

    # Initialize baseline generator
    baseline_generator = BaselineGenerator(
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

    # Prepare input with property constraints
    input_list = [smi1, smi2]

    print("=" * 80)
    print("STRATEGY 1: Random leaving fragment + Frequency-based insertion fragment")
    print("=" * 80)

    start_time = time.time()
    result_df_1 = baseline_generator.generate_strategy_1(input_list)
    elapsed_time_1 = time.time() - start_time

    print(f"Generated {len(result_df_1)} molecules")
    print(f"Elapsed time: {elapsed_time_1:.2f} seconds")

    if len(result_df_1) > 0:
        print("\nTop 5 generated molecules:")
        for idx, row in result_df_1.head(5).iterrows():
            print(f"Input: {row['INPUT-MOL-SMI']}")
            print(f"Generated: {row['GEN-MOL-SMI']}")
            print(f"Leaving fragment: {row['LEAVING-FRAG-SMI']}")
            print(f"Inserting fragment: {row['INSERTING-FRAG-SMI']}")
            print(
                f"LogP: {row['LOGP']:.3f}, MW: {row['MW']:.3f}, QED: {row['QED']:.3f}"
            )
            print("-" * 40)

    result_df_1.to_csv("baseline_strategy_1_results.csv", index=False)
    print("Results saved to baseline_strategy_1_results.csv\n")

    print("=" * 80)
    print(
        "STRATEGY 2: DeepBioisostere leaving fragment + Frequency-based insertion fragment"
    )
    print("=" * 80)

    start_time = time.time()
    result_df_2 = baseline_generator.generate_strategy_2(input_list)
    elapsed_time_2 = time.time() - start_time

    print(f"Generated {len(result_df_2)} molecules")
    print(f"Elapsed time: {elapsed_time_2:.2f} seconds")

    if len(result_df_2) > 0:
        print("\nTop 5 generated molecules:")
        for idx, row in result_df_2.head(5).iterrows():
            print(f"Input: {row['INPUT-MOL-SMI']}")
            print(f"Generated: {row['GEN-MOL-SMI']}")
            print(f"Leaving fragment: {row['LEAVING-FRAG-SMI']}")
            print(f"Inserting fragment: {row['INSERTING-FRAG-SMI']}")
            print(
                f"LogP: {row['LOGP']:.3f}, MW: {row['MW']:.3f}, QED: {row['QED']:.3f}"
            )
            print("-" * 40)

    result_df_2.to_csv("baseline_strategy_2_results.csv", index=False)
    print("Results saved to baseline_strategy_2_results.csv\n")

    print("=" * 80)
    print("STRATEGY 3: Completely random selection")
    print("=" * 80)

    start_time = time.time()
    result_df_3 = baseline_generator.generate_strategy_3(input_list)
    elapsed_time_3 = time.time() - start_time

    print(f"Generated {len(result_df_3)} molecules")
    print(f"Elapsed time: {elapsed_time_3:.2f} seconds")

    if len(result_df_3) > 0:
        print("\nTop 5 generated molecules:")
        for idx, row in result_df_3.head(5).iterrows():
            print(f"Input: {row['INPUT-MOL-SMI']}")
            print(f"Generated: {row['GEN-MOL-SMI']}")
            print(f"Leaving fragment: {row['LEAVING-FRAG-SMI']}")
            print(f"Inserting fragment: {row['INSERTING-FRAG-SMI']}")
            print(
                f"LogP: {row['LOGP']:.3f}, MW: {row['MW']:.3f}, QED: {row['QED']:.3f}"
            )
            print("-" * 40)

    result_df_3.to_csv("baseline_strategy_3_results.csv", index=False)
    print("Results saved to baseline_strategy_3_results.csv\n")

    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(
        f"Strategy 1 (Random + Frequency): {len(result_df_1)} molecules in {elapsed_time_1:.2f}s"
    )
    print(
        f"Strategy 2 (Model + Frequency): {len(result_df_2)} molecules in {elapsed_time_2:.2f}s"
    )
    print(
        f"Strategy 3 (Random + Random): {len(result_df_3)} molecules in {elapsed_time_3:.2f}s"
    )

    if len(result_df_1) > 0:
        print("\nStrategy 1 - Average properties:")
        print(
            f"  LogP: {result_df_1['LOGP'].mean():.3f} ± {result_df_1['LOGP'].std():.3f}"
        )
        print(f"  MW: {result_df_1['MW'].mean():.3f} ± {result_df_1['MW'].std():.3f}")
        print(
            f"  QED: {result_df_1['QED'].mean():.3f} ± {result_df_1['QED'].std():.3f}"
        )

    if len(result_df_2) > 0:
        print("\nStrategy 2 - Average properties:")
        print(
            f"  LogP: {result_df_2['LOGP'].mean():.3f} ± {result_df_2['LOGP'].std():.3f}"
        )
        print(f"  MW: {result_df_2['MW'].mean():.3f} ± {result_df_2['MW'].std():.3f}")
        print(
            f"  QED: {result_df_2['QED'].mean():.3f} ± {result_df_2['QED'].std():.3f}"
        )

    if len(result_df_3) > 0:
        print("\nStrategy 3 - Average properties:")
        print(
            f"  LogP: {result_df_3['LOGP'].mean():.3f} ± {result_df_3['LOGP'].std():.3f}"
        )
        print(f"  MW: {result_df_3['MW'].mean():.3f} ± {result_df_3['MW'].std():.3f}")
        print(
            f"  QED: {result_df_3['QED'].mean():.3f} ± {result_df_3['QED'].std():.3f}"
        )

    print("\nGeneration completed successfully!")
