"""Optimize two molecules with DeepBioisostere.

Run with::

    python example.py

The checkpoint and fragment library are resolved automatically: a local
``model_save/`` and ``fragment_library/`` are used if present, otherwise they
are downloaded from the Hugging Face Hub and cached. See
``deepbioisostere.assets``.

The equivalent one-liner is::

    deepbioisostere generate -s "<smiles>" -t mw=0 -t logp=-1
"""

import time

from rdkit import Chem

from deepbioisostere import (
    Conditioner,
    DeepBioisostere,
    Generator,
    calc_logP,
    calc_Mw,
    calc_QED,
    calc_SAscore,
)


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
    print_properties(smi1)
    print_properties(smi2)

    # USER SETTINGS
    device = "cpu"
    num_cores = 4
    batch_size = 512
    num_sample_each_mol = 100
    new_frag_type = "all"  # one of ["test", "train", "val", "all"]
    properties = ["mw", "logp"]  # order does not matter

    # Initialize model and generator. from_pretrained() picks the checkpoint
    # matching this property set and fetches it if it is not already local.
    model = DeepBioisostere.from_pretrained(properties=properties)
    conditioner = Conditioner(phase="generation", properties=sorted(properties))
    generator = Generator(
        model=model,
        conditioner=conditioner,
        device=device,
        num_cores=num_cores,
        batch_size=batch_size,
        new_frag_type=new_frag_type,
        num_sample_each_mol=num_sample_each_mol,
        properties=sorted(properties),
    )

    # Option 1. Let the model choose which fragment to replace.
    print("Option 1. Generate with DeepBioisostere.")
    start_time = time.time()
    input_list = [
        (smi1, {"mw": 0, "logp": -1}),
        (smi2, {"mw": 0, "logp": -1}),
    ]
    result_df = generator.generate(input_list)
    result_df.to_csv("generation_result.csv", index=False)
    print("Elapsed time: ", time.time() - start_time)

    # Option 2. Specify the leaving fragment yourself.
    print("Option 2. Generate with a specific leaving fragment.")
    start_time = time.time()
    input_list = [
        (smi1, "[*]c1ccccc1[*]", 4, {"mw": 0, "logp": -1}),
        (smi2, "[*]c1ccccn1", 12, {"mw": 0, "logp": -1}),
    ]
    result_df = generator.generate_with_leaving_frag(input_list)
    result_df.to_csv("generation_result_with_leaving_frag.csv", index=False)
    print("Elapsed time: ", time.time() - start_time)
