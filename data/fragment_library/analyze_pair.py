from pathlib import Path
from typing import List

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, Mol
from rdkit.Chem.rdchem import BondType

ATOM_INDEX = int
CID = SMILES = str


def create_bond(
    rwmol: Chem.RWMol, idx1: ATOM_INDEX, idx2: ATOM_INDEX, bondtype: BondType
) -> None:
    """
    idx1 and idx2 are the indice of 'dummy atoms'.
    """
    rwmol.AddBond(idx1, idx2, bondtype)
    for idx in [idx1, idx2]:
        atom = rwmol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "N" and atom.GetIsAromatic() is True:
            atom.SetNumExplicitHs(0)


def get_submol_by_indice(mol: Mol, indices: List[ATOM_INDEX]) -> Mol:
    sub_mol = Chem.RWMol()

    atom_map = {}
    for idx in indices:
        atom = mol.GetAtomWithIdx(idx)
        atom_map[idx] = sub_mol.AddAtom(atom)

    for start_idx in indices:
        start_atom = mol.GetAtomWithIdx(start_idx)
        for end_atom in start_atom.GetNeighbors():
            end_idx = end_atom.GetIdx()
            if end_idx not in indices:
                continue
            bond = mol.GetBondBetweenAtoms(start_idx, end_idx)
            bond_type = bond.GetBondType()
            if start_idx < end_idx:  # each bond is enumerated twice
                create_bond(sub_mol, atom_map[start_idx], atom_map[end_idx], bond_type)
                # sub_mol.AddBond(atom_map[start_idx], atom_map[end_idx], bond_type)
    return sub_mol.GetMol()


def parse_df(df: pd.DataFrame, ref_cid: CID, prb_cid: CID) -> pd.DataFrame:
    df = df.loc[df["REF-CID"] == ref_cid]
    df = df.loc[df["PRB-CID"] == prb_cid]
    return df


def get_smi_from_cid(df: pd.DataFrame, cid: CID) -> SMILES:
    df = df.loc[df["REF-CID"] == cid]
    return df["SMI"].iloc[0]


def get_whole_smis_and_key_frags(
    df: pd.DataFrame, ref_smi: SMILES, prb_smi: SMILES
) -> None:
    ref_mol = Chem.MolFromSmiles(ref_smi)

    results = []
    for row in df.iterrows():
        idx, series = row

        old_frag_smi = series.get("OLD-FRAG")
        new_frag_smi = series.get("NEW-FRAG")

        results.append([ref_smi, prb_smi, old_frag_smi, new_frag_smi])

    max_len = get_max_len(results)
    print_results(results, max_len)
    return


def get_max_len(results: List[List[SMILES]]) -> List[int]:
    df = pd.DataFrame(data=results)
    len_df = df.apply(lambda series: list(map(len, series)), axis=0)
    max_len = len_df.apply(max, axis=0)
    return max_len.values.tolist()


def print_results(
    results: List[List[SMILES]], max_len: List[int], print_as_row: bool = False
) -> None:
    print(f"REF-CID: {args.ref_cid}")
    print(f"PRB-CID: {args.prb_cid}")
    if print_as_row:
        for result in results:
            result = [elem.ljust(length) for length, elem in zip(max_len, result)]
            print("\t".join(result))
    else:
        for idx, result in enumerate(results):
            ref_smi, prb_smi, old_frag_smi, new_frag_smi = result
            print(f"Sample # {idx}")
            print(f"REF-SMI      -> {ref_smi}")
            print(f"PRB-SMI      -> {prb_smi}")
            print(f"OLD-FRAG-SMI -> {old_frag_smi}")
            print(f"NEW-FRAG-SMI -> {new_frag_smi}")
    return


def mol_with_atom_index(mol: Mol) -> Mol:
    from copy import deepcopy

    _mol = deepcopy(mol)

    for atom in _mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return _mol


def main():
    df = pd.read_csv(args.pair_file, delimiter="\t", low_memory=False)
    unique_df = df.drop_duplicates(subset=["SMI"])
    ref_smi = get_smi_from_cid(unique_df, args.ref_cid)
    prb_smi = get_smi_from_cid(unique_df, args.prb_cid)
    df = parse_df(df, args.ref_cid, args.prb_cid)
    get_whole_smis_and_key_frags(df, ref_smi, prb_smi)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_file", type=Path, default="./8.txt")
    parser.add_argument("--ref_cid", type=str, default="CHEMBL3326633")
    parser.add_argument("--prb_cid", type=str, default="CHEMBL3326635")
    args, _ = parser.parse_known_args()

    main()
