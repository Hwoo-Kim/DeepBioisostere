import os
import time
from rdkit import Chem
from pathlib import Path
import pandas as pd

from scripts.conditioning import Conditioner
from scripts.generate import Generator
from scripts.model import DeepBioisostere      
from scripts.property import calc_logP, calc_Mw, calc_QED, calc_SAscore

def print_properties(smi: str):
    print(f"SMILES: {smi}", end=", ")
    mol = Chem.MolFromSmiles(smi)
    print(f"logP: {calc_logP(mol):.3f}", end=", ")
    print(f"QED: {calc_QED(mol):.3f}", end=", ")
    print(f"Mw: {calc_Mw(mol):.3f}", end=", ")
    print(f"SAscore: {calc_SAscore(mol):.3f}")

def main():
    device = "cpu"
    num_cores = 16
    batch_size = 512
    num_sample_each_mol = 100
    new_frag_type = "all"      # one of ["test", "train", "valid", "all"]
    properties_to_control = ["qed", "sa"]

    # Set model and fragment library paths
    properties = sorted(properties_to_control)
    proj_dir = os.path.abspath("")
    model_path = f"{proj_dir}/model_save/DeepBioisostere_{'_'.join(properties)}.pt"
    frag_lib_path = f"{proj_dir}/data/"

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

    gen_path = Path("/home/share/DATA/wonho_SBDD_models_eval")
    model_list = ["DeepICL", "targetdiff", "pocket2mol", "decompdiff"]
    target_index_list = [67, 81, 93]

    sa_list = [-(0.5*i + 1) for i in range(5)]         # need to use minus value
    qed_list = [0.05*i + 0.1 for i in range(5)]

    # fname : gen_{mol_idx}.sdf
    # path : gen_path / model_name + _eval / gen / {target_idx} / gen_{mol_idx}.sdf
    # savepath : proj_dir / gen / model_name / {target_idx} / {QED}_{SA}_{mol_idx}.sdf

    proj_dir = Path(proj_dir)
    zip_list = []

    for model_name in model_list:
        for target_idx in target_index_list:
            zip_list.append((model_name, target_idx))

    zip_list = zip_list[:1]
    # zip_list += [("DeepICL", 98), ("targetdiff", 84), ("pocket2mol", 27), ("decompdiff", 56)]

    for model_name, target_idx in zip_list:
        print(f"Model: {model_name}")
        gen_dir = gen_path / f"{model_name}_eval" / "result.csv"
        df = pd.read_csv(gen_dir)
        target_index_list = [target_idx]
        df = df[df["test_idx"].isin(target_index_list)]
        # use test_idx and SMILES only
        df = df[["test_idx", "gen_idx", "SMILES", "QED", "SA"]]

        for target_idx in target_index_list:
            print(f"Target index: {target_idx}")
            save_dir = proj_dir / "gen" / model_name / str(target_idx)
            os.makedirs(save_dir, exist_ok=True)
            df_target = df[df["test_idx"] == target_idx]
            input_list = []
            input_smi_to_idx = dict(zip(df_target["SMILES"], df_target["gen_idx"]))
            input_smi_to_qed = dict(zip(df_target["SMILES"], df_target["QED"]))
            # real SA : 10 - df SA * 9
            input_smi_to_sa = dict(zip(df_target["SMILES"], 10 - df_target["SA"] * 9))
            for (_, row) in df_target.iterrows():
                gen_idx = row["gen_idx"]
                smi = row["SMILES"]
                if not isinstance(smi, str):
                    print(f"Invalid SMILES: {gen_idx}, {smi}")
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    print(f"Invalid SMILES: {smi}")
                    continue
                input_list.append(smi)
            for qed in qed_list:
                for sa in sa_list:
                    print(f"QED: {qed}, SA: {sa}")
                    result_df = generator.generate([(smi, {"qed": qed, "sa": sa}) for smi in input_list])
                    result_df["INPUT-MOL-IDX"] = result_df["INPUT-MOL-SMI"].map(input_smi_to_idx)

                    # add QED, SA diff column
                    result_df["INPUT-MOL-QED"] = result_df["INPUT-MOL-SMI"].map(input_smi_to_qed)
                    result_df["INPUT-MOL-SA"] = result_df["INPUT-MOL-SMI"].map(input_smi_to_sa)
                    result_df["QED_DIFF"] = result_df["QED"] - result_df["INPUT-MOL-QED"]
                    result_df["SA_DIFF"] = result_df["SA"] - result_df["INPUT-MOL-SA"]

                    # drop unnecessary columns
                    result_df = result_df.drop(columns=["INPUT-MOL-QED", "INPUT-MOL-SA"])

                    result_df.to_csv(f"{save_dir}/{qed:.2f}_{sa:.1f}.csv", index=False)

if __name__ == "__main__":
    main()
