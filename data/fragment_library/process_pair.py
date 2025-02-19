from collections import defaultdict
from contextlib import ExitStack
from functools import partial
from multiprocessing import cpu_count, Lock, Pool
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.FragmentMatcher import FragmentMatcher

from make_frag_db import BRICS_BOND

COLS = [
    "REF-CID",
    "PRB-CID",
    "REF-SMI",
    "PRB-SMI",
    "ASSAY-ID",
    "REF-TARGET-ID",
    "PRB-TARGET-ID",
    "KEY-FRAG-ATOM-INDICE",
    "ATOM-FRAG-INDICE",
    "OLD-FRAG",
    "NEW-FRAG",
    "BRICS-EDGE-INDICE",
    "CUTTING-EDGE-INDICE",
]


def cut_brics_bonds(mol: Mol, brics_bonds: List[BRICS_BOND]) -> Mol:
    """
    Returns:
      broken_mol: Mol. fragmented mol object.
    """

    bond_indice, brics_types = [], []
    for atom_indices, brics_type in brics_bonds:
        bond = mol.GetBondBetweenAtoms(*atom_indices)
        bond_indice.append(bond.GetIdx())

        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()

        if (begin_atom_idx, end_atom_idx) == atom_indices:
            brics_type = (brics_type[1], brics_type[0])
        brics_types.append(tuple(map(int, brics_type)))

    broken_mol = Chem.FragmentOnBonds(
        mol, bondIndices=bond_indice, dummyLabels=brics_types
    )
    return broken_mol


def get_attachment_information(elems: List, file: Path) -> None:
    (
        ref_smi,
        prb_smi,
        assay_id,
        ref_target_id,
        prb_target_id,
        key_frag_atom_indice,
        _,
        old_frag_smi,
        new_frag_smi,
        _,
        cutting_edge_indice,
    ) = elems[2:]
    leaving_atom_indice = [int(i) for i in key_frag_atom_indice.split(",")]

    cutting_edges_in_prb = cutting_edge_indice.split(",")
    cutting_edges_in_prb = [
        (int(edge.split("-")[0]), int(edge.split("-")[1]))
        for edge in cutting_edges_in_prb
    ]

    # for ref_mol
    ref_mol = Chem.MolFromSmiles(ref_smi)

    ref_cut_edges = []
    ref_brics_bonds = list(BRICS.FindBRICSBonds(ref_mol))
    for brics_bond in ref_brics_bonds:
        atom_indices, _ = brics_bond
        in_leaving_atoms = [
            atom_idx in leaving_atom_indice for atom_idx in atom_indices
        ]
        if in_leaving_atoms.count(True) == 1:
            ref_cut_edges.append(brics_bond)

    ref_frags = []
    ref_broken_mol = cut_brics_bonds(ref_mol, ref_cut_edges)
    ref_broken_frag_mols = Chem.GetMolFrags(
        ref_broken_mol, asMols=True, frags=ref_frags
    )
    ref_broken_frag_smis = [Chem.MolToSmiles(mol) for mol in ref_broken_frag_mols]

    # for prb_mol
    prb_mol = Chem.MolFromSmiles(prb_smi)

    new_cut_edges = []
    new_brics_bonds = list(BRICS.FindBRICSBonds(prb_mol))
    for brics_bond in new_brics_bonds:
        atom_indices, _ = brics_bond
        if (
            atom_indices in cutting_edges_in_prb
            or (atom_indices[1], atom_indices[0]) in cutting_edges_in_prb
        ):
            new_cut_edges.append(brics_bond)

    new_frags = []
    new_broken_mol = cut_brics_bonds(prb_mol, new_cut_edges)
    new_broken_frag_mols = Chem.GetMolFrags(
        new_broken_mol, asMols=True, frags=new_frags
    )
    new_broken_frag_smis = [Chem.MolToSmiles(mol) for mol in new_broken_frag_mols]

    # find the answer
    new_frag_idx = new_broken_frag_smis.index(new_frag_smi)
    new_frag_atoms = []
    for atom_idx, frag_idx in enumerate(new_frags):
        if frag_idx == new_frag_idx:
            new_frag_atoms.append(atom_idx)

    # get index match between ref molecule and prb molecule
    # new_atom_to_ref_atom = dict()
    new_atom_to_ref_atom = defaultdict(list)
    for cut_edge in cutting_edges_in_prb:
        atom_idx1, atom_idx2 = cut_edge
        if atom_idx1 in new_frag_atoms:
            atom_in_new_smi = atom_idx1
            adj_frag_idx = new_frags[atom_idx2]

        elif atom_idx2 in new_frag_atoms:
            atom_in_new_smi = atom_idx2
            adj_frag_idx = new_frags[atom_idx1]

        adj_smi = new_broken_frag_smis[adj_frag_idx]
        adj_frag_atoms_indice = []  # symmetry in remaining part resolved here
        for ref_broken_frag_idx, ref_broken_frag_smi in enumerate(ref_broken_frag_smis):
            if ref_broken_frag_smi == adj_smi:
                adj_frag_atoms_indice += [
                    atom_idx
                    for atom_idx, frag_idx in enumerate(ref_frags)
                    if frag_idx == ref_broken_frag_idx
                ]

        # search the bond (in ref mol) corresponding to cut_edge (in new mol)
        for brics_edge in ref_cut_edges:
            atom_indices, _ = brics_edge
            for ref_atom_idx in atom_indices:
                if ref_atom_idx in adj_frag_atoms_indice:
                    new_atom_to_ref_atom[atom_in_new_smi].append(ref_atom_idx)

    # mapping indices with checking symmetry in new fragment
    attachment_dict = dict()  # new_frag to old_frag
    p = FragmentMatcher()
    p.Init(new_frag_smi)

    res_frag_mol = Chem.MolFromSmiles(new_frag_smi)
    matches = p.GetMatches(new_broken_mol, uniquify=0)  # allow duplicates

    cut_atom_matches = []  # new fragment atom to ref molecule atom!!!!
    for match in matches:
        new_to_atom_map = {}
        for new_atom_idx, res_atom_idx in enumerate(match):
            if res_atom_idx in new_atom_to_ref_atom:
                new_to_atom_map[new_atom_idx] = new_atom_to_ref_atom[res_atom_idx]
        cut_atom_matches.append(new_to_atom_map)

    # new fragment atom to ref molecule atom!!!!
    atom_matches_dict = defaultdict(list)
    for match in cut_atom_matches:
        for k, v in match.items():
            atom_matches_dict[k] += v

    # get the result in the form able to be parsed
    allowed_attach_new_frag_to_ref_mol = []
    for k, v in atom_matches_dict.items():
        v = list(set(map(str, v)))
        atoms_in_ref_mol = "o".join(v)
        allowed_attach_new_frag_to_ref_mol.append(f"{str(k)}-{atoms_in_ref_mol}")
    allowed_attach_new_frag_to_ref_mol = ",".join(allowed_attach_new_frag_to_ref_mol)

    # REF-CID PRB-CID REF-SMI ASSAY-ID REF-TARGET-ID PRB-TARGET-ID PRB-SMI KEY-FRAG-ATOM-INDICE ATOM-FRAG-INDICE OLD-FRAG NEW-FRAG ALLOWED-ATTACHMENT
    del elems[-1]
    elems[-1] = allowed_attach_new_frag_to_ref_mol

    with ExitStack() as stack, file.open("a") as f:
        lock = globals().get("lock", False)
        if lock:
            stack.enter_context(lock)
        f.write("\t".join(elems) + "\n")
    return


def init_pool(lock_: Lock):
    global lock
    lock = lock_


def main():
    lock = Lock()

    df = pd.read_csv(args.pair_file, delimiter="\t", low_memory=False, names=COLS)
    data = df.to_dict("split")["data"]

    with Pool(cpu_count(), initializer=init_pool, initargs=(lock,)) as pool:
        worker = partial(get_attachment_information, file=args.result_file)
        pool.map(worker, data)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Process parser")
    parser.add_argument("--pair_file", type=Path)
    parser.add_argument("--result_file", type=Path)
    args, _ = parser.parse_known_args()

    main()
