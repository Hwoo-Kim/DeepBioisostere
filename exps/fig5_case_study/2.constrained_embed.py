import os
import re
from subprocess import run, DEVNULL
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import (
    AllChem,
    Atom,
    Draw,
    Mol,
    PyMol,
    rdDistGeom,
    rdFMCS,
    rdForceFieldHelpers,
    rdMolAlign,
)
from rdkit.Chem.rdMolAlign import AlignMol

SMILES = str
ATOM_INDEX = int


def local_opt(protein_file: Path, ligand_file: Path):
    new_file = ligand_file.parent / f"opt_{ligand_file.stem}.sdf"
    command = f"smina -r {str(protein_file)} -l {str(ligand_file)} --minimize -o {str(new_file)} -q"
    command = command.split()
    run(command, stdout=DEVNULL, stderr=DEVNULL)
    return


def write_mols(mol: Mol, confids: List[int], file: Path):
    file.parent.mkdir(parents=True, exist_ok=True)
    # rdMolAlign.AlignMolConformers(mol)
    writer = Chem.SDWriter(str(file))
    for confid in confids:
        writer.write(mol, confId=confid)
    writer.close()
    return


def constrained_embed(
    mol: Mol,
    core: Mol,
    core_conf_id: int = -1,
    **kwargs,
) -> Tuple[Mol, List[int]]:
    # Get the atom indices of the mol that match the core
    constrained_indices = mol.GetSubstructMatch(core)
    if not constrained_indices:
        raise ValueError("molecule doesn't match the core")

    coord_map = {}
    core_conf = core.GetConformer(core_conf_id)
    for i, idx in enumerate(constrained_indices):
        # idx: mol atom index, i: core atom index
        core_pos_i = core_conf.GetAtomPosition(i)
        coord_map[idx] = core_pos_i

    cids = AllChem.EmbedMultipleConfs(mol, coordMap=coord_map, **kwargs)
    if len(cids) == 0:
        raise ValueError("Could not embed molecule.")

    # Check the core pos is not changed with numpy array
    for conf_id in cids:
        conf = mol.GetConformer(conf_id)
        conf_pos = conf.GetPositions()[constrained_indices, :]
        core_pos = core_conf.GetPositions()
        assert np.allclose(conf_pos, core_pos), "[EMBED] Core position is changed"

    return mol, cids


def constrained_optimize(
    mol: Mol, core: Mol, cids: List[int], core_conf_id: int = -1
) -> List[int]:
    # Get the atom indices of the mol that match the core
    constrained_indices = mol.GetSubstructMatch(core)
    if not constrained_indices:
        raise ValueError("molecule doesn't match the core")

    # Define the MMFF properties
    mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)

    conf_ids = []
    for conf_id in cids:
        FF = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            mol, pyMMFFMolProperties=mp, confId=conf_id
        )
        # Fix the position of the core atoms
        for i in constrained_indices:
            FF.MMFFAddPositionConstraint(i, 0, 1.0e4)
        failed = FF.Minimize(maxIts=10000)
        if not failed:
            conf_ids.append(conf_id)

    # Check the core pos is not changed with numpy array
    core_conf = core.GetConformer(core_conf_id)
    for conf_id in conf_ids:
        conf = mol.GetConformer(conf_id)
        conf_pos = conf.GetPositions()[constrained_indices, :]
        core_pos = core_conf.GetPositions()
        assert (np.abs(conf_pos - core_pos) < 0.1).all(), "[OPT] Core position is changed"

    return conf_ids


def main(args):
    if not args.csv_path.exists():
        raise FileExistsError(f"{args.csv_path} not exists")
    args.base_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)

    # Check removal fragment
    assert (
        df.iloc[:, 3].values[0] == "[14*]c1ncn(C)n1"
    ), "The removal fragment is not correct"

    # get 3'th column (GEN-MOL-SMI)
    gen_smis = df.iloc[:, 2].values
    ori_smi = df.iloc[:, 1].values[0]
    ori_sdf = Chem.SDMolSupplier(str(args.original_ligand_file))[0]

    # Original ligand
    ori_ligand = Chem.MolFromSmiles(ori_smi)
    ori_ligand = Chem.AddHs(ori_ligand)

    # Handcrafted remaining fragment
    remaining_fragment = Chem.MolFromSmiles(
        "Cn(c1)nc2c1cc(Nc(nc(=O)n3C)n(Cc4c(F)cc(F)c(F)c4)c3=O)c(Cl)c2"
    )
    leaving_fragment = AllChem.DeleteSubstructs(ori_sdf, remaining_fragment)
    remaining_fragment = AllChem.DeleteSubstructs(ori_sdf, leaving_fragment)

    for seed in args.seeds:
        (args.base_path / str(seed)).mkdir(parents=True, exist_ok=True)

        # Original ligand
        ori_ligand, ori_cids = constrained_embed(
            ori_ligand,
            remaining_fragment,
            useRandomCoords=True,
            useSmallRingTorsions=True,
            ETversion=2,
            numConfs=args.n_confs,
            numThreads=args.num_threads,
            pruneRmsThresh=0.1,
            randomSeed=seed,
        )

        _ori_ligand = Chem.Mol(ori_ligand)
        ori_opt_conf_ids = constrained_optimize(
            _ori_ligand, remaining_fragment, ori_cids
        )

        ori_mmff_path = args.base_path / str(seed) / "mmff_ori.sdf"
        _ori_ligand = Chem.RemoveHs(_ori_ligand)
        write_mols(_ori_ligand, ori_opt_conf_ids[:100], ori_mmff_path)
        local_opt(args.original_protein_file, ori_mmff_path)
        ori_etkdg_path = args.base_path / str(seed) / "etkdg_ori.sdf"
        ori_ligand = Chem.RemoveHs(ori_ligand)
        write_mols(ori_ligand, ori_opt_conf_ids[:100], ori_etkdg_path)
        local_opt(args.original_protein_file, ori_etkdg_path)

        # Generated molecules
        failed_count = 0
        for idx, smi in enumerate(gen_smis):
            if idx - failed_count == args.max_n_mols:
                break

            ligand = Chem.MolFromSmiles(smi)
            ligand = Chem.AddHs(ligand)

            if not rdForceFieldHelpers.MMFFHasAllMoleculeParams(ligand):
                print(smi, idx, "failed")
                failed_count += 1
                continue

            try:
                ligand, cids = constrained_embed(
                    ligand,
                    remaining_fragment,
                    useRandomCoords=True,
                    useSmallRingTorsions=True,
                    ETversion=2,
                    numConfs=args.n_confs,
                    numThreads=args.num_threads,
                    pruneRmsThresh=0.1,
                    randomSeed=seed,
                )

                _ligand = Chem.Mol(ligand)
                opt_conf_ids = constrained_optimize(_ligand, remaining_fragment, cids)

                mmff_path = args.base_path / str(seed) / f"mmff_{idx - failed_count}.sdf"
                _ligand = Chem.RemoveHs(_ligand)
                print(smi, idx, len(list(cids)))
                write_mols(_ligand, opt_conf_ids[:100], mmff_path)
                local_opt(args.original_protein_file, mmff_path)
                # Save corresponding etkdg conformers
                etkdg_path = args.base_path / str(seed) / f"etkdg_{idx - failed_count}.sdf"
                ligand = Chem.RemoveHs(ligand)
                write_mols(ligand, opt_conf_ids[:100], etkdg_path)
                local_opt(args.original_protein_file, etkdg_path)

            except Exception as e:
                print(smi, idx, "failed")
                print(e)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", type=Path)
    parser.add_argument(
        "-c",
        "--csv_path",
        type=Path,
    )
    parser.add_argument(
        "-l",
        "--original_ligand_file",
        type=str,
    )
    parser.add_argument(
        "--max_n_mols",
        type=int,
        default=100,
        help="Default: 100",
    )
    parser.add_argument(
        "--n_confs",
        type=int,
        default=1000,
        help="Default: 1000",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2021, 2022, 2023, 2024, 2025],
        help="Default: [2021, 2022, 2023, 2024, 2025]",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Default: 4",
    )
    parser.add_argument(
        "-p",
        "--original_protein_file",
        type=Path,
        default="./8h3g_protein_no_alt.pdb",
    )
    args = parser.parse_args()

    main(args)
