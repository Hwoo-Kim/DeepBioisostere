import argparse
import multiprocessing as mp
import os
import shutil
import subprocess
from itertools import islice
from typing import Dict, List, Optional, Tuple

import AutoDockTools
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

INPUT_MOL_IDX_COL = "INPUT-MOL-IDX"
INPUT_MOL_SMI_COL = "INPUT-MOL-SMI"
INPUT_MOL_DOCKED_SCORE_COL = "INPUT-MOL-DOCKED-SCORE"
INPUT_ERROR_COL = "INPUT-ERROR"

GEN_MOL_IDX_COL = "GEN-MOL-IDX"
GEN_MOL_SMI_COL = "GEN-MOL-SMI"
GEN_MOL_DOCKED_SCORE_COL = "GEN-MOL-DOCKED-SCORE"
GEN_ERROR_COL = "GEN-ERROR"

MK_PREPARE_RECEPTOR = "mk_prepare_receptor.py"
MK_PREPARE_LIGAND = "mk_prepare_ligand.py"
OPENCL_BINARY_PATH = os.environ.get(
    "VINA_GPU_OPENCL_BINARY_PATH",
    "AutoDock-Vina-GPU-2-1",
)


def prepare_receptor(receptor_pdb_path: str, output_pdbqt_path: str) -> bool:
    pdb2pqr_cmd = [
        "pdb2pqr30",
        "--ff=AMBER",
        receptor_pdb_path,
        receptor_pdb_path.replace(".pdb", ".pqr"),
    ]
    try:
        subprocess.run(
            pdb2pqr_cmd, capture_output=True, text=True, check=True, timeout=600
        )
    except subprocess.CalledProcessError:
        # print(f"Receptor prep error ({receptor_pdb_path}):\n{e.stderr}")
        return False

    _prepare_receptor = os.path.join(
        AutoDockTools.__path__[0], "Utilities24/prepare_receptor4.py"
    )
    pdbqt_cmd = [
        "python",
        _prepare_receptor,
        "-r",
        receptor_pdb_path.replace(".pdb", ".pqr"),
        "-o",
        output_pdbqt_path,
    ]
    try:
        subprocess.run(
            pdbqt_cmd, capture_output=True, text=True, check=True, timeout=600
        )
    except subprocess.CalledProcessError:
        # print(f"Receptor prep error ({receptor_pdb_path}):\n{e.stderr}")
        return False
    return True


def prepare_receptor_meeko(
    receptor_pdb_path: str, output_pdbqt_path: str, mk_prepare_receptor_script: str
) -> bool:
    cmd = [
        "python",
        mk_prepare_receptor_script,
        "--default_altloc",
        "A",
        "--read_pdb",
        receptor_pdb_path,
        "-o",
        output_pdbqt_path.replace(".pdbqt", ""),
        "-p",
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        return True
    except subprocess.CalledProcessError:
        # print(f"Receptor prep error ({receptor_pdb_path}):\n{e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        # print(f"Receptor prep timeout ({receptor_pdb_path})")
        return False


def prepare_ligand_meeko(
    smiles_tuple: Tuple[str, int],
    base_ligand_dir: str,
    mk_prepare_ligand_script: str,  # Added this argument
) -> Optional[str]:
    smiles, ligand_unique_id = smiles_tuple
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        # print(f"SMILES parse failed: {smiles}")
        return None

    mol_h = Chem.AddHs(mol)
    embed_params = AllChem.ETKDGv3()
    embed_params.randomSeed = 42
    embed_params.numThreads = 0
    if AllChem.EmbedMolecule(mol_h, embed_params) == -1:
        embed_params_v2 = AllChem.ETKDG()
        embed_params_v2.randomSeed = 42
        embed_params_v2.numThreads = 0
        if AllChem.EmbedMolecule(mol_h, embed_params_v2) == -1:
            # print(f"3D conformer generation failed: {smiles}")
            AllChem.Compute2DCoords(mol_h)  # Fallback to 2D
            # return None # If 3D is strictly required

    try:
        if mol_h.GetNumConformers() > 0:
            AllChem.UFFOptimizeMolecule(mol_h)
    except Exception:
        # print(f"Error occured in ligand optimization ({smiles}): {e}")
        pass

    temp_sdf_path = os.path.join(
        base_ligand_dir, f"lig_{ligand_unique_id}_prep_temp.sdf"
    )
    output_pdbqt_path = os.path.join(base_ligand_dir, f"lig_{ligand_unique_id}.pdbqt")

    try:
        writer = Chem.SDWriter(temp_sdf_path)
        writer.write(mol_h)
        writer.close()
    except Exception:
        # print(f"SDF write failed for {smiles}: {e}")
        return None

    cmd = [
        "python",
        mk_prepare_ligand_script,
        "--rigid_macrocycles",
        "-i",
        temp_sdf_path,
        "-o",
        output_pdbqt_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
        return output_pdbqt_path
    except subprocess.CalledProcessError:
        # print(f"Ligand prep error ({smiles}):\n{e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        # print(f"Ligand prep timeout ({smiles})")
        return None
    finally:
        if os.path.exists(temp_sdf_path):
            os.remove(temp_sdf_path)


def load_molecule_for_center(filepath: str) -> Optional[Chem.Mol]:
    if not os.path.exists(filepath):
        return None
    file_ext = os.path.splitext(filepath)[1].lower()
    mol = None
    try:
        if file_ext == ".sdf":
            suppl = Chem.SDMolSupplier(filepath, removeHs=False)
            mol = next(suppl, None)
        elif file_ext == ".mol2":
            mol = Chem.MolFromMol2File(filepath, removeHs=False)
        elif file_ext == ".mol":
            mol = Chem.MolFromMolFile(filepath, removeHs=False)
        elif file_ext in [".pdb", ".pdbqt"]:
            mol = Chem.MolFromPDBFile(filepath, removeHs=False)
    except Exception:
        return None
    return mol


def get_ligand_center(mol: Chem.Mol, conformer_id: int = 0) -> np.ndarray:
    conf = mol.GetConformer(
        conformer_id
        if conformer_id < mol.GetNumConformers() and conformer_id >= 0
        else 0
    )
    centroid_point3d = AllChem.ComputeCentroid(conf)
    return np.array([centroid_point3d.x, centroid_point3d.y, centroid_point3d.z])


def run_vina_gpu(
    vina_gpu_executable: str, config_file_path: str, run_log_path: str
) -> bool:
    cmd = [vina_gpu_executable, "--config", config_file_path]
    try:
        with open(run_log_path, "w") as log_f:
            subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                check=True,
                text=True,
                timeout=7200,
            )
        return True
    except subprocess.CalledProcessError:
        # print(f"Vina GPU run error. Log: {run_log_path}")
        return False
    except subprocess.TimeoutExpired:
        # print(f"Vina GPU run timeout. Log: {run_log_path}")
        return False


def parse_vina_gpu_output(output_pdbqt_path: str) -> Optional[float]:
    try:
        with open(output_pdbqt_path, "r") as f:
            for line in f:
                if line.startswith("REMARK VINA RESULT:"):
                    parts = line.split()
                    return float(parts[3])
        return None
    except Exception:
        return None


def process_docking_tasks(
    tasks_for_this_batch: List[Dict],
    common_receptor_pdbqt: str,
    box_info: Dict[str, float],
    common_vina_exec: str,
    common_opencl_binary_path: str,
    num_processes_for_ligprep: int,
    batch_processing_scratch_dir: str,
    mk_prepare_ligand_script_path: str,
) -> List[Dict]:
    all_individual_results = []
    batch_ligands_input_dir = os.path.join(
        batch_processing_scratch_dir, "batch_ligands_to_dock"
    )
    batch_vina_outputs_dir = os.path.join(
        batch_processing_scratch_dir, "batch_vina_outputs"
    )
    os.makedirs(batch_ligands_input_dir, exist_ok=True)
    os.makedirs(batch_vina_outputs_dir, exist_ok=True)

    ligand_prep_args_list = []
    for i, task_meta in enumerate(tasks_for_this_batch):
        ligand_prep_args_list.append(
            (
                (task_meta["smiles"], i),
                batch_ligands_input_dir,
                mk_prepare_ligand_script_path,
            )
        )

    prepared_ligand_batch_paths = [None] * len(tasks_for_this_batch)
    if ligand_prep_args_list:
        actual_ligprep_procs = min(
            num_processes_for_ligprep, len(ligand_prep_args_list)
        )
        if actual_ligprep_procs > 0:
            with mp.Pool(processes=actual_ligprep_procs) as pool:
                pdbqt_paths_results = pool.starmap(
                    prepare_ligand_meeko, ligand_prep_args_list
                )
            for i, path_result in enumerate(pdbqt_paths_results):
                prepared_ligand_batch_paths[i] = path_result

    vina_config_path = os.path.join(
        batch_processing_scratch_dir, "vina_batch_config.txt"
    )
    vina_run_log_path = os.path.join(batch_processing_scratch_dir, "vina_batch_run.log")
    vina_config_content = [
        f"opencl_binary_path = {os.path.abspath(common_opencl_binary_path)}",
        f"\nreceptor = {os.path.abspath(common_receptor_pdbqt)}",
        f"ligand_directory = {os.path.abspath(batch_ligands_input_dir)}",
        f"output_directory = {os.path.abspath(batch_vina_outputs_dir)}",
        f"\ncenter_x = {box_info['center_x']:.4f}",
        f"center_y = {box_info['center_y']:.4f}",
        f"center_z = {box_info['center_z']:.4f}",
        "\nsize_x = 20.0",
        "size_y = 20.0",
        "size_z = 20.0",
        "\nnum_modes = 1",
        f"log = {os.path.abspath(vina_run_log_path)}",
        "thread = 8000",
        "seed = 0",
    ]
    with open(vina_config_path, "w") as f_cfg:
        f_cfg.write("\n".join(vina_config_content))

    docking_batch_successful = run_vina_gpu(
        common_vina_exec, vina_config_path, vina_run_log_path
    )

    for i, task_meta in enumerate(tasks_for_this_batch):
        result_template = task_meta | {"score": None, "error": None}
        result_template["id"] = result_template.pop("mol_id")
        batch_prepared_ligand_path = prepared_ligand_batch_paths[i]
        if not batch_prepared_ligand_path:
            result_template["error"] = "Ligand preparation failed"
            all_individual_results.append(result_template)
            continue

        if not docking_batch_successful:
            result_template["error"] = (
                result_template["error"] or ""
            ) + ";Vina GPU batch execution failed"
            all_individual_results.append(result_template)
            continue

        prepared_ligand_basename = os.path.basename(batch_prepared_ligand_path)
        expected_docked_filename = (
            os.path.splitext(prepared_ligand_basename)[0] + "_out.pdbqt"
        )
        docked_output_path_in_batch_dir = os.path.join(
            batch_vina_outputs_dir, expected_docked_filename
        )

        if os.path.exists(docked_output_path_in_batch_dir):
            score = parse_vina_gpu_output(docked_output_path_in_batch_dir)
            if score is not None:
                result_template["score"] = str(score)
            else:
                result_template["error"] = (
                    result_template["error"] or ""
                ) + ";Score parsing failed"
        else:
            result_template["error"] = (
                (result_template["error"] or "")
                + f";Docked output file not found from options: {expected_docked_filename}"
            )
        all_individual_results.append(result_template)
    return all_individual_results


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv_path", required=True)
    parser.add_argument("--output_subdir_name", default="docking_results_gpu")
    parser.add_argument("--job_scratch_dir", required=True)
    parser.add_argument("--references_dir", required=True)

    parser.add_argument("--mk_prepare_receptor_script", default=MK_PREPARE_RECEPTOR)
    parser.add_argument("--mk_prepare_ligand_script", default=MK_PREPARE_LIGAND)
    parser.add_argument("--vina_gpu_executable", required=True)
    parser.add_argument("--opencl_binary_path", default=OPENCL_BINARY_PATH)

    parser.add_argument("--num_conformers", type=int, default=1)
    parser.add_argument("--docking_chunk_size", type=int, default=80)
    parser.add_argument(
        "--num_processes", type=int, default=max(1, mp.cpu_count() // 2)
    )
    parser.add_argument("--archive_intermediate_files", action="store_true")

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    global MK_PREPARE_RECEPTOR_SCRIPT_PATH, MK_PREPARE_LIGAND_SCRIPT_PATH
    MK_PREPARE_RECEPTOR_SCRIPT_PATH = args.mk_prepare_receptor_script
    MK_PREPARE_LIGAND_SCRIPT_PATH = args.mk_prepare_ligand_script

    if not os.path.isfile(args.vina_gpu_executable) or not os.access(
        args.vina_gpu_executable, os.X_OK
    ):
        print(
            f"CRITICAL ERROR: Vina GPU executable not found or not executable: {args.vina_gpu_executable}"
        )
        exit(1)

    script_output_base_in_scratch = os.path.join(
        args.job_scratch_dir, args.output_subdir_name
    )
    os.makedirs(script_output_base_in_scratch, exist_ok=True)

    MODEL_NAME, TARGET_IDX, filename = args.input_csv_path.split("/")[-3:]
    df = pd.read_csv(args.input_csv_path)
    try:
        df[INPUT_MOL_IDX_COL] = df[INPUT_MOL_IDX_COL].astype(int).astype(str)
    except Exception:
        # rewrite index based on the index of unique index of input-mol-smi
        unique_input_smis = df[INPUT_MOL_SMI_COL].unique()
        smi2index = {smi: i for i, smi in enumerate(unique_input_smis)}
        df[INPUT_MOL_IDX_COL] = (
            df[INPUT_MOL_SMI_COL].map(smi2index).astype(int).astype(str)
        )

    if GEN_MOL_IDX_COL not in df.columns:
        df[GEN_MOL_IDX_COL] = df.groupby(INPUT_MOL_IDX_COL).cumcount().astype(str)

    required_cols = [
        INPUT_MOL_IDX_COL,
        INPUT_MOL_SMI_COL,
        GEN_MOL_IDX_COL,
        GEN_MOL_SMI_COL,
    ]
    for col_name in required_cols:
        if col_name not in df.columns:
            print(
                f"CRITICAL ERROR: Required column '{col_name}' not found in input CSV."
            )
            exit(1)
        df[col_name] = df[col_name].astype(str)
    df.to_csv("input_data.csv", index=False)

    # prepare receptor
    receptor_pdb_original_path = os.path.join(
        args.references_dir, str(TARGET_IDX), "pro.pdb"
    )
    receptor_prep_dir = os.path.join(
        args.job_scratch_dir, "prepared_receptors_cache", str(TARGET_IDX)
    )
    os.makedirs(receptor_prep_dir, exist_ok=True)
    prepared_receptor_pdbqt = os.path.join(receptor_prep_dir, "receptor.pdbqt")
    prepare_receptor(receptor_pdb_original_path, prepared_receptor_pdbqt)

    # prepare box info
    ref_lig_sdf_original_path = os.path.join(
        args.references_dir, str(TARGET_IDX), "ref.sdf"
    )
    ref_mol = load_molecule_for_center(ref_lig_sdf_original_path)
    center_coords = get_ligand_center(ref_mol)
    box_info = {
        "center_x": center_coords[0],
        "center_y": center_coords[1],
        "center_z": center_coords[2],
    }

    tasks = []
    input_filtered_df = df.drop_duplicates(
        subset=[INPUT_MOL_IDX_COL, INPUT_MOL_SMI_COL], keep="first"
    )
    for index, row in input_filtered_df.iterrows():
        input_mol_id_str = row[INPUT_MOL_IDX_COL]
        input_smiles_str = row[INPUT_MOL_SMI_COL]
        task = {
            "type": "input",
            "mol_id": input_mol_id_str,
            "smiles": input_smiles_str,
        }
        tasks.append(task)

    for index, row in df.iterrows():
        input_mol_id_str = row[INPUT_MOL_IDX_COL]
        gen_mol_id_within_input_str = row[GEN_MOL_IDX_COL]
        gen_smiles_str = row[GEN_MOL_SMI_COL]
        task = {
            "type": "gen",
            "mol_id": input_mol_id_str,
            "original_gen_idx": gen_mol_id_within_input_str,  # To map back later
            "smiles": gen_smiles_str,
        }
        tasks.append(task)

    print("TOTAL TASKS TO DOCK:", len(tasks))
    print("TARGET: ", TARGET_IDX)

    all_results_list = []
    batch_run_counter = 0
    for batched_task_meta in batched(tasks, args.docking_chunk_size):
        current_batch_scratch_dir = os.path.join(
            args.job_scratch_dir, f"batch_run_{batch_run_counter}"
        )
        batch_results = process_docking_tasks(
            tasks_for_this_batch=batched_task_meta,
            common_receptor_pdbqt=prepared_receptor_pdbqt,
            box_info=box_info,
            common_vina_exec=args.vina_gpu_executable,
            common_opencl_binary_path=args.opencl_binary_path,
            num_processes_for_ligprep=args.num_processes,
            batch_processing_scratch_dir=current_batch_scratch_dir,
            mk_prepare_ligand_script_path=args.mk_prepare_ligand_script,
        )
        all_results_list.extend(batch_results)
        if args.archive_intermediate_files:
            batch_archive_dir = os.path.join(
                script_output_base_in_scratch, f"batch_run_{batch_run_counter}"
            )
            shutil.copytree(
                current_batch_scratch_dir,
                batch_archive_dir,
                dirs_exist_ok=True,
            )
        batch_run_counter += 1

    temp_results_df = pd.DataFrame(all_results_list)
    # input data
    input_results_df = temp_results_df[temp_results_df["type"] == "input"].copy()
    input_results_df.drop(columns=["type", "error", "original_gen_idx"], inplace=True)
    input_results_df.dropna(subset=["score"], inplace=True)
    input_results_df.rename(
        {
            "id": INPUT_MOL_IDX_COL,
            "smiles": INPUT_MOL_SMI_COL,
            "score": INPUT_MOL_DOCKED_SCORE_COL,
        },
        inplace=True,
        axis=1,
    )
    # gen data
    gen_results_df = temp_results_df[temp_results_df["type"] == "gen"].copy()
    gen_results_df.drop(columns=["type", "error"], inplace=True)
    gen_results_df.dropna(subset=["score"], inplace=True)
    gen_results_df.rename(
        {
            "id": INPUT_MOL_IDX_COL,
            "original_gen_idx": GEN_MOL_IDX_COL,
            "smiles": GEN_MOL_SMI_COL,
            "score": GEN_MOL_DOCKED_SCORE_COL,
        },
        inplace=True,
        axis=1,
    )
    # merge input - gen data (product)
    merged_data = pd.merge(
        input_results_df,
        gen_results_df,
        on=[INPUT_MOL_IDX_COL],
        how="inner",
    )
    merged_data.to_csv("out.csv", index=False)

    # original data
    output_df = df.copy()  # Start with the original CSV structure
    output_df = pd.merge(
        output_df,
        merged_data,
        on=[INPUT_MOL_IDX_COL, GEN_MOL_IDX_COL, INPUT_MOL_SMI_COL, GEN_MOL_SMI_COL],
        how="inner",
    )

    # save
    csv_basename = os.path.basename(args.input_csv_path)
    output_summary_filename = (
        f"summary_{MODEL_NAME}_{TARGET_IDX}_{os.path.splitext(csv_basename)[0]}.csv"
    )
    output_summary_csv_path = os.path.join(
        script_output_base_in_scratch, output_summary_filename
    )
    output_df.to_csv(output_summary_csv_path, index=False)
    print(f"Docking results summary saved to: {output_summary_csv_path}")

    print("GPU Docking script completed.")


if __name__ == "__main__":
    args = parse_args()

    main(args)
