#!/usr/bin/env python

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED

# Attempt to import SA Score functionality from RDKit Contrib
try:
    from rdkit.Contrib.SA_Score import sascorer

    SA_SCORE_AVAILABLE = True
except ImportError:
    print(
        "Warning: RDKit SA_Score module (rdkit.Contrib.SA_Score) not found. "
        "SA scores will be calculated as NaN. \n"
        "Please ensure your RDKit installation includes contrib modules, "
        "or install rdkit-pypi if you haven't."
    )
    SA_SCORE_AVAILABLE = False


def read_affinity_from_file(filepath: Path) -> Optional[float]:
    """
    Reads a single float affinity value from a text file.
    Assumes the affinity is the first space-separated token on the first line.
    Returns None if the file doesn't exist or parsing fails.
    """
    if not filepath.is_file():  # Check if it's a file and exists
        # print(f"Debug: Affinity file not found: {filepath}")
        return None
    try:
        with open(filepath, "r") as f:
            line = f.readline().strip()
            if line:  # Ensure line is not empty
                return float(
                    line.split()[0]
                )  # Takes the first part, e.g., "-7.5" from "-7.5 kcal/mol"
            else:
                # print(f"Debug: Affinity file is empty: {filepath}")
                return None
    except (ValueError, IndexError) as e:
        print(
            f"Warning: Could not parse affinity from {filepath}. Error: {e}. Line content: '{line}'"
        )
        return None
    except Exception as e:
        print(f"Warning: Error reading or processing affinity file {filepath}: {e}")
        return None


def compute_sa_score(rdmol):
    rdmol = Chem.MolFromSmiles(Chem.MolToSmiles(rdmol))
    sa = sascorer.calculateScore(rdmol)
    sa_norm = round((10 - sa) / 9, 2)
    return sa_norm


def has_radical_atom(mol):
    """Checks if a molecule contains a radical atom.

    Args:
      mol: An RDKit Mol object.

    Returns:
      True if the molecule contains a radical atom, False otherwise.
    """
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            return True
    return False


def get_molecule_properties(
    sdf_path: Path, is_generated: bool = False
) -> tuple[Optional[Chem.Mol], Optional[str], float, float, Union[int, float]]:
    """
    Reads an SDF file and returns the RDKit Mol object, SMILES string,
    QED score, SA score, and number of atoms.
    Returns NaNs for scores/counts if molecule reading fails.
    """
    if not sdf_path.is_file():
        # print(f"Debug: SDF file not found: {sdf_path}")
        return None, None, np.nan, np.nan, np.nan

    mol = None
    # RDKit's SDMolSupplier can be a bit sensitive.
    try:
        suppl = Chem.SDMolSupplier(str(sdf_path))
        if suppl and len(suppl) > 0:
            mol = suppl[0]  # Take the first molecule from the SDF
        if is_generated:
            for a in mol.GetAtoms():
                a.SetNumExplicitHs(a.GetNumRadicalElectrons())
                a.SetNumRadicalElectrons(0)
    except Exception as e:
        print(f"Warning: Could not initialize SDMolSupplier for {sdf_path}. Error: {e}")
        return None, None, np.nan, np.nan, np.nan

    if has_radical_atom(mol):
        print("Warning: Molecule contains radical atoms.")

    if mol is None:
        # print(f"Warning: No molecule could be read from SDF file: {sdf_path}")
        return None, None, np.nan, np.nan, np.nan

    try:
        smiles = Chem.MolToSmiles(mol)
        qed_val = QED.qed(mol)
        # sa_val = sascorer.calculateScore(mol) if SA_SCORE_AVAILABLE else np.nan
        sa_val = compute_sa_score(mol) if SA_SCORE_AVAILABLE else np.nan
        num_atoms = mol.GetNumAtoms()
        return mol, smiles, qed_val, sa_val, num_atoms
    except Exception as e:
        print(
            f"Warning: Could not calculate properties for molecule from {sdf_path}. Error: {e}"
        )
        return mol, None, np.nan, np.nan, np.nan


def process_dataset(
    base_data_dir: str = "gen", num_test_indices: int = 100, num_gen_indices: int = 100
) -> pd.DataFrame:
    """
    Processes the dataset structured under base_data_dir to extract molecular properties,
    scores, and file paths, returning them as a Pandas DataFrame.

    Args:
        base_data_dir: The root directory of the dataset (e.g., "gen").
        num_test_indices: The number of test_idx subdirectories (e.g., 100 for 1 to 100).
        num_gen_indices: The number of generated molecule indices per test_idx (e.g., 100 for gen_1 to gen_100).

    Returns:
        A Pandas DataFrame containing the processed data.
    """
    root_path = Path(base_data_dir)
    all_records = []

    print(f"Starting dataset processing from: {root_path.resolve()}")

    for test_idx in range(1, num_test_indices + 1):
        target_dir = root_path / str(test_idx)
        print(f"\nProcessing Test Index (Target): {test_idx}")

        if not target_dir.is_dir():
            print(
                f"  Warning: Directory not found: {target_dir}, skipping this target."
            )
            continue

        # --- Reference Molecule Data ---
        ref_sdf_file = target_dir / "ref.sdf"
        _, _, ref_qed, ref_sa, ref_n_atoms = get_molecule_properties(ref_sdf_file)

        ref_score_aff_val = read_affinity_from_file(
            target_dir / "ref_vina_score_aff.txt"
        )
        ref_min_aff_val = read_affinity_from_file(target_dir / "ref_vina_min_aff.txt")
        ref_dock_aff_val = read_affinity_from_file(target_dir / "ref_vina_dock_aff.txt")

        # Common file paths for this test_idx
        pro_file_path = target_dir / "pro.pdb"
        rec_file_path = target_dir / "rec.pdb"
        ref_score_sdf_path = target_dir / "ref_vina_score.sdf"
        ref_min_sdf_path = target_dir / "ref_vina_min.sdf"
        ref_dock_sdf_path = target_dir / "ref_vina_dock.sdf"

        for gen_idx in range(1, num_gen_indices + 1):
            # print(f"  Processing Generated Index: {gen_idx}")
            current_record = {
                "error": ""
            }  # Placeholder for any errors specific to this combo

            # Populate with test_idx and gen_idx
            current_record["test_idx"] = test_idx
            current_record["gen_idx"] = gen_idx

            # File paths
            current_record["pro_fn"] = (
                str(pro_file_path) if pro_file_path.exists() else None
            )
            current_record["rec_fn"] = (
                str(rec_file_path) if rec_file_path.exists() else None
            )
            current_record["ref_fn"] = (
                str(ref_sdf_file) if ref_sdf_file.exists() else None
            )
            current_record["ref_score_fn"] = (
                str(ref_score_sdf_path) if ref_score_sdf_path.exists() else None
            )
            current_record["ref_min_fn"] = (
                str(ref_min_sdf_path) if ref_min_sdf_path.exists() else None
            )
            current_record["ref_dock_fn"] = (
                str(ref_dock_sdf_path) if ref_dock_sdf_path.exists() else None
            )

            # Reference properties and scores
            current_record["ref_QED"] = ref_qed
            current_record["ref_SA"] = ref_sa
            current_record["ref_n_atom"] = ref_n_atoms
            current_record["ref_score_aff"] = ref_score_aff_val
            current_record["ref_min_aff"] = ref_min_aff_val
            current_record["ref_dock_aff"] = ref_dock_aff_val

            # --- Generated Molecule Data ---
            gen_sdf_file = target_dir / f"gen_{gen_idx}.sdf"
            _, gen_smiles, gen_qed, gen_sa, gen_n_atoms = get_molecule_properties(
                gen_sdf_file, is_generated=True
            )

            current_record["gen_fn"] = (
                str(gen_sdf_file) if gen_sdf_file.exists() else None
            )
            current_record["SMILES"] = gen_smiles  # For generated molecule
            current_record["n_atom"] = gen_n_atoms  # For generated molecule
            current_record["QED"] = gen_qed  # For generated molecule
            current_record["SA"] = gen_sa  # For generated molecule

            current_record["gen_score_aff"] = read_affinity_from_file(
                target_dir / f"gen_{gen_idx}_vina_score_aff.txt"
            )
            current_record["gen_min_aff"] = read_affinity_from_file(
                target_dir / f"gen_{gen_idx}_vina_min_aff.txt"
            )
            current_record["gen_dock_aff"] = read_affinity_from_file(
                target_dir / f"gen_{gen_idx}_vina_dock_aff.txt"
            )

            gen_score_sdf_path = target_dir / f"gen_{gen_idx}_vina_score.sdf"
            gen_min_sdf_path = target_dir / f"gen_{gen_idx}_vina_min.sdf"
            gen_dock_sdf_path = target_dir / f"gen_{gen_idx}_vina_dock.sdf"
            current_record["gen_score_fn"] = (
                str(gen_score_sdf_path) if gen_score_sdf_path.exists() else None
            )
            current_record["gen_min_fn"] = (
                str(gen_min_sdf_path) if gen_min_sdf_path.exists() else None
            )
            current_record["gen_dock_fn"] = (
                str(gen_dock_sdf_path) if gen_dock_sdf_path.exists() else None
            )

            # Calculate diffs (as in your example CSV)
            current_record["ref_score_min_aff_diff"] = (
                (ref_score_aff_val - ref_min_aff_val)
                if pd.notnull(ref_score_aff_val) and pd.notnull(ref_min_aff_val)
                else np.nan
            )
            current_record["ref_score_dock_aff_diff"] = (
                (ref_score_aff_val - ref_dock_aff_val)
                if pd.notnull(ref_score_aff_val) and pd.notnull(ref_dock_aff_val)
                else np.nan
            )

            current_record["gen_score_min_aff_diff"] = (
                (current_record["gen_score_aff"] - current_record["gen_min_aff"])
                if pd.notnull(current_record["gen_score_aff"])
                and pd.notnull(current_record["gen_min_aff"])
                else np.nan
            )
            current_record["gen_score_dock_aff_diff"] = (
                (current_record["gen_score_aff"] - current_record["gen_dock_aff"])
                if pd.notnull(current_record["gen_score_aff"])
                and pd.notnull(current_record["gen_dock_aff"])
                else np.nan
            )

            all_records.append(current_record)

    df = pd.DataFrame(all_records)

    # Define a preferred column order based on your example CSV structure.
    # This helps ensure compatibility with your next analysis script.
    preferred_column_order = [
        "error",
        "test_idx",
        "gen_idx",
        "pro_fn",
        "rec_fn",
        "ref_fn",
        "ref_score_fn",
        "ref_min_fn",
        "ref_dock_fn",
        "gen_fn",
        "gen_score_fn",
        "gen_min_fn",
        "gen_dock_fn",
        "ref_QED",
        "ref_SA",
        "ref_n_atom",
        "ref_score_aff",
        "ref_min_aff",
        "ref_dock_aff",
        "ref_score_min_aff_diff",
        "ref_score_dock_aff_diff",
        "SMILES",
        "n_atom",
        "QED",
        "SA",
        "gen_score_aff",
        "gen_min_aff",
        "gen_dock_aff",
        "gen_score_min_aff_diff",
        "gen_score_dock_aff_diff",
    ]

    # Reorder DataFrame columns, adding missing ones from preferred_order as NaN columns
    # and keeping any extra columns generated that are not in preferred_order.
    final_columns = []
    existing_columns = df.columns.tolist()

    for col in preferred_column_order:
        if col in existing_columns:
            final_columns.append(col)
        else:
            df[col] = np.nan  # Add as new column with NaNs if not present
            final_columns.append(col)

    for col in existing_columns:  # Add any columns not in preferred_order to the end
        if col not in final_columns:
            final_columns.append(col)

    df = df[final_columns]
    return df


if __name__ == "__main__":
    data_directory = "gen"
    output_filename = "result.csv"

    print("Running data processing script...")
    print(f"SA Score module available: {SA_SCORE_AVAILABLE}")

    # Process the dataset
    results_df = process_dataset(
        base_data_dir=data_directory,
        num_test_indices=100,  # Assuming 100 target directories (1 to 100)
        num_gen_indices=100,  # Assuming 100 generated molecules per target (gen_1 to gen_100)
    )

    # Save the DataFrame to a CSV file
    try:
        results_df.to_csv(output_filename, index=False)
        print(
            f"\nProcessing complete. Output saved to: {Path(output_filename).resolve()}"
        )
    except Exception as e:
        print(f"\nError saving DataFrame to CSV: {e}")

    # Display the first few rows of the generated DataFrame
    if not results_df.empty:
        print("\nFirst 5 rows of the generated data:")
        print(results_df.head())
    else:
        print("\nNo data was processed or DataFrame is empty.")
