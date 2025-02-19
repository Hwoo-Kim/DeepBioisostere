import time
from pathlib import Path

from rdkit import Chem

from conditioning import Conditioner
from generate import Generator
from model import DeepBioisostere
from property import calc_logP, calc_Mw, calc_QED, calc_SAscore


def print_properties(smi: str):
    print(f"SMILES: {smi}", end=", ")
    mol = Chem.MolFromSmiles(smi)
    print(f"logP: {calc_logP(mol):.3f}", end=", ")
    print(f"QED: {calc_QED(mol):.3f}", end=", ")
    print(f"Mw: {calc_Mw(mol):.3f}", end=", ")
    print(f"SAscore: {calc_SAscore(mol):.3f}")


def main():
    # USER SETTINGS
    device = "cpu"
    num_cores = 4
    batch_size = 512
    num_sample_each_mol = "all"  # or integer
    new_frag_type = "all" # one of ["test", "train", "valid", "all"]
    properties_to_control = ["mw", "logp", "qed", "sa"]  # You don't need to worry about the order!

    # Set model and fragment library paths
    properties = sorted(properties_to_control)
    model_path = f"../../model_save/DeepBioisostere_{'_'.join(properties)}.pt"
    frag_lib_path = "../../fragment_library"

    # Initialize model and generator
    print(f"Loading model from: {model_path}")
    print(f"Loading fragment library from: {frag_lib_path}")
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

    smi = "Cn1cnc(Cn2c(=O)nc(Nc3cc4cn(C)nc4cc3Cl)n(Cc3cc(F)c(F)cc3F)c2=O)n1"
    print_properties(smi)

    # Generate with DeepBioisostere
    print("Generate with DeepBioisostere.")
    start_time = time.time()
    frag_smi =  "[14*]c1ncn(C)n1"
    frag_atom_idx = 0
    input_list = [
        (smi, frag_smi, frag_atom_idx, {"mw": 0, "logp": 1, "qed": 0, "sa": 0}),
    ]

    result_df = generator.generate_with_leaving_frag(input_list)
    result_df.to_csv("GenMol_all_MW0_logP1_QED0_SA0.csv", index=False)
    return


if __name__ == "__main__":
    main()
