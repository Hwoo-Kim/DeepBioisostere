import collections
import time
from contextlib import ExitStack
from copy import deepcopy
from functools import partial
from itertools import combinations
from multiprocessing import Lock, Manager, Pool, cpu_count, get_context, synchronize
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, Atom, BondType, Mol

SMILES = str
INDEX = int
COUNT = int 
ATOM_INDEX = Union[List[INDEX], str]
BOND_INDICE = Tuple[INDEX, INDEX]
BRICS_TYPE = Tuple[str, str]
BRICS_BOND = Tuple[BOND_INDICE, BRICS_TYPE]  # ((3, 2), ('3', '4'))
FRAG_INFO = Dict[SMILES, SMILES]  # fragment: remaining fragments

COLS = [
    "CID",
    "SMI",
    "ASSAY-ID",
    "TARGET-ID",
    "KEY-FRAG",
    "REST-FRAG",
    "NUM-FRAGS",
    "KEY-FRAG-ATOM-INDICE",
    "ATOM-FRAG-INDICE",
    "BRICS-EDGE-INDICE",
    "CUTTING-EDGE-INDICE",
]


def remove_bond(rwmol: Chem.RWMol, idx1: ATOM_INDEX, idx2: ATOM_INDEX):
    rwmol.RemoveBond(idx1, idx2)
    for idx in [idx1, idx2]:
        atom = rwmol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "N" and atom.GetIsAromatic() is True:
            atom.SetNumExplicitHs(1)


def get_dummy_indice_from_mol(mol: Mol) -> List[ATOM_INDEX]:
    dummy_indice = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            dummy_indice.append(atom.GetIdx())
    return dummy_indice


def remove_dummy_atom_from_mol(mol: Mol) -> Mol:
    mol = Chem.RWMol(mol)
    dummy_indice = get_dummy_indice_from_mol(mol)

    dummy_bonds = []
    for dummy_idx in dummy_indice:
        dummy_bonds.append(mol.GetAtomWithIdx(dummy_idx).GetBonds()[0])

    for dummy_bond in dummy_bonds:
        remove_bond(mol, dummy_bond.GetBeginAtomIdx(), dummy_bond.GetEndAtomIdx())
    for dummy_idx in sorted(dummy_indice, reverse=True):
        mol.RemoveAtom(dummy_idx)

    return mol


def remove_match_from_mol(mol: Mol, match: Tuple[ATOM_INDEX]) -> Mol:
    mol = Chem.RWMol(mol)

    remove_bonds = []
    for atom_idx in match:
        atom: Chem.Atom = mol.GetAtomWithIdx(atom_idx)
        for bond in atom.GetBonds():
            remove_bonds.append(bond)

    for bond in remove_bonds:
        remove_bond(mol, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    for idx in sorted(match, reverse=True):
        mol.RemoveAtom(idx)

    return mol


class FragmentParser:
    def __init__(self, mol: Mol, max_natoms: int):
        self.mol = mol
        self.max_natoms = max_natoms

    def get_brics_bonds_info(self) -> Tuple[List[BRICS_BOND], List[BOND_INDICE]]:
        """\
        Get BRICS bonds information of original molecule.
        BRICS bonds information, which comprises atom indices of bond and
        corresponding BRICS type for atoms, is sorted by atom index.
        
        Returns
        -------
        brics_bonds: List[BRICS_BOND]
        brics_bonds_indice: List[BOND_INDICE]
        """
        brics_bonds = list(BRICS.FindBRICSBonds(self.mol))
        brics_bonds_indice = [bond[0] for bond in brics_bonds]
        return brics_bonds, brics_bonds_indice

    def get_atom_frag_indice(self, verbose: bool = False) -> List[INDEX]:
        """\
        Get atom-wise fragment index of original molecule.
        List of atom-wise fragment index is sorted by atom index.

        Parameters
        ----------
        verbose : bool, default False

        Returns
        -------
        atom_frag_indice: List[INDEX]
        """
        frags = []
        broken_mol = BRICS.BreakBRICSBonds(self.mol)
        frag_mols = list(Chem.GetMolFrags(broken_mol, asMols=True, frags=frags))
        atom_frag_indice = frags[: self.mol.GetNumAtoms()]

        if verbose:
            frag_smis = [Chem.MolToSmiles(mol) for mol in frag_mols]
            print(f"frag_smis: {frag_smis}")
            print(f"atom_frag_indice: {atom_frag_indice}")
        return atom_frag_indice

    def extract_frag_wise_edges(
        self,
        brics_bonds_indice: List[BOND_INDICE],
        verbose: bool = False,
    ) -> List[BOND_INDICE]:
        """\
        Extract fragment-wise edges from BRICS bonds.
        The original `brics_bonds_indice` comprises atom indice of molecule for given edge.
        The output `frag_wise_edges` comprises fragment indice of molecule for given edge.
        
        Parameters
        ----------
        brics_bonds_indice: List[BOND_INDICE]

        Returns
        -------
        frag_wise_edges: List[BOND_INDICE]
        """
        frag_wise_edges = []
        for bond in brics_bonds_indice:
            frag_wise_edges.append(
                (
                    self.atom_frag_indice[bond[0]],
                    self.atom_frag_indice[bond[1]],
                )
            )
        if verbose:
            print(f"node_wise_edges: {brics_bonds_indice}")
            print(f"frag_wise_edges: {frag_wise_edges}")
        return frag_wise_edges

    def filter_valid(self, key_frag_mol: Mol, rest_frag_mol: Mol) -> bool:
        """\
        Filter out invalid fragments.
        Currently only check the number of atoms.

        Parameters
        ----------
        key_frag_mol: Mol
        rest_frag_mol: Mol

        Returns
        -------
        valid: bool
        """
        key_natoms = key_frag_mol.GetNumHeavyAtoms()
        rest_natoms = rest_frag_mol.GetNumHeavyAtoms()
        max_natoms_cond = key_natoms < self.max_natoms
        cmp_natoms_cond = key_natoms <= rest_natoms
        return max_natoms_cond & cmp_natoms_cond

    def get_removing_indice(
        self, key_frag_mol: Mol, rest_frag_mol: Mol
    ) -> Optional[Set[ATOM_INDEX]]:
        """\
        Get removing indice of rest fragment.
        The removing indice is the atom indice of rest fragment which is not
        included in key fragment.

        Parameters
        ----------
        key_frag_mol: Mol
        rest_frag_mol: Mol

        Returns
        -------
        removing_indice: Optional[Set[ATOM_INDEX]]
        """

        key_frag_mol = remove_dummy_atom_from_mol(key_frag_mol)
        rest_frag_mol = remove_dummy_atom_from_mol(rest_frag_mol)
        if key_frag_mol is None or rest_frag_mol is None:
            return
        for match in self.mol.GetSubstructMatches(key_frag_mol):
            key_frag_removed_mol = remove_match_from_mol(self.mol, match)
            if key_frag_removed_mol is None:
                continue
            key_frag_removed_smi = Chem.MolToSmiles(key_frag_removed_mol)
            rest_frag_smi = Chem.MolToSmiles(rest_frag_mol)
            if key_frag_removed_smi == rest_frag_smi:
                return match

    def get_fragmentation_result(
        self, ref_smi: SMILES, prb_smi: SMILES, atom_wise_cut_edges: List[BOND_INDICE]
    ) -> Optional[Dict[SMILES, Tuple[SMILES, ATOM_INDEX]]]:
        """\
        Get fragmentation result of given reference and probe molecules.
        The result is a dictionary which has key as reference fragment and
        value as list of probe fragment and removing indice of probe fragment.
        
        Parameters
        ----------
        ref_smi: SMILES
        prb_smi: SMILES
        atom_wise_cut_edges: List[BOND_INDICE]
            
        Returns
        -------
        fragmentation_result: Optional[Dict[SMILES, List[SMILES, ATOM_INDEX]]]
        """

        ref_mol = Chem.MolFromSmiles(ref_smi)
        prb_mol = Chem.MolFromSmiles(prb_smi)
        if self.filter_valid(ref_mol, prb_mol):
            key_frag_atom_indice = self.get_removing_indice(ref_mol, prb_mol)
            try:
                key_frag_atom_indice = ",".join(list(map(str, key_frag_atom_indice)))
                atom_frag_indice = ",".join(list(map(str, self.atom_frag_indice)))
                brics_bonds_indice = [
                    list(map(str, indice)) for indice in self.brics_bonds_indice
                ]
                brics_bonds_indice = ",".join(list(map("-".join, brics_bonds_indice)))
                atom_wise_cut_edges = [
                    list(map(str, indice)) for indice in atom_wise_cut_edges
                ]
                atom_wise_cut_edges = ",".join(list(map("-".join, atom_wise_cut_edges)))
            except TypeError:  # None case
                pass
            else:
                return {
                    ref_smi: (
                        prb_smi,
                        key_frag_atom_indice,
                        atom_frag_indice,
                        brics_bonds_indice,
                        atom_wise_cut_edges,
                    )
                }
        return

    def fragmentation(
        self, frag_wise_edges: List[Tuple[BOND_INDICE]]
    ) -> List[FRAG_INFO]:
        """\
        Fragmentation of molecule.
        Iterate for possible fragmentation plans.

        Parameters
        ----------
        frag_wise_edges: List[Tuple[BOND_INDICE]]

        Returns
        -------
        frag_info: List[FRAG_INFO]
        """

        fragmentation_results = []
        frag_edges_indice = list(range(len(frag_wise_edges)))
        for num_cut_edges in range(1, self.num_frags):
            fragmentation_plans = combinations(frag_wise_edges, num_cut_edges)
            for cut_edges in fragmentation_plans:
                frag_dummy_atom_counts = self.count_dummy_atoms_per_frag(cut_edges)

                remained_edges = [
                    set(edge) for edge in frag_wise_edges if edge not in cut_edges
                ]
                integrated_frags = self.connect_adjacent_frags(remained_edges)

                key_frag = self.scatter_sum_frag_dummy_atom_counts(
                    integrated_frags, frag_dummy_atom_counts, num_cut_edges
                )
                if key_frag is None:
                    continue

                frag_smis = self.get_frag_smis_by_cut_edges(cut_edges, frag_wise_edges)
                atom_wise_cut_edges = [
                    self.brics_bonds_indice[frag_wise_edges.index(edge)]
                    for edge in cut_edges
                ]
                if num_cut_edges == 1:
                    result = self.get_fragmentation_result(
                        frag_smis[0], frag_smis[1], atom_wise_cut_edges
                    )
                    if result:
                        fragmentation_results.append(result)
                    result = self.get_fragmentation_result(
                        frag_smis[1], frag_smis[0], atom_wise_cut_edges
                    )
                    if result:
                        fragmentation_results.append(result)
                else:
                    for frag in frag_smis:
                        if frag.count("*") == num_cut_edges:
                            frag_smis_ = deepcopy(frag_smis)
                            frag_smis_.remove(frag)
                            frag_smis_ = ".".join(sorted(frag_smis_))
                            result = self.get_fragmentation_result(
                                frag, frag_smis_, atom_wise_cut_edges
                            )
                            if result:
                                fragmentation_results.append(result)

        return fragmentation_results

    def count_dummy_atoms_per_frag(self, cut_edges: Tuple[BOND_INDICE]) -> List[COUNT]:
        """\
        Count dummy atoms of fragment by counting cutting edges.
            
        Parameters
        ----------
        cut_edges: Tuple[BOND_INDICE]
        
        Returns
        -------
        frag_dummy_atom_counts: List[COUNT]
        """

        cut_edges_arr = np.array(cut_edges).reshape(-1)
        counts = collections.Counter(cut_edges_arr)
        frag_dummy_atom_counts = []
        for frag_idx in range(self.num_frags):
            frag_dummy_atom_counts.append(counts[frag_idx])
        return frag_dummy_atom_counts

    def connect_adjacent_frags(
        self, remained_edges: List[Set[BOND_INDICE]]
    ) -> List[Set[INDEX]]:
        """\
        Connect adjacent fragments by remained edges.
        Generate the list of connected fragment indice set.

        Parameters
        ----------
        remained_edges: List[Set[BOND_INDICE]]

        Returns
        -------
        integrated_frags: List[Set[INDEX]]
        """
        integrated_frags = []
        for edge in remained_edges:
            for integrated_frag in integrated_frags:
                # Update edge if edge is connected to integrated_frag
                if integrated_frag.intersection(edge):
                    integrated_frag.update(edge)
                    break
            else:  # Initially add edge
                integrated_frags.append(edge)

        for frag_idx in range(self.num_frags):
            for integrated_frag in integrated_frags:
                if frag_idx in integrated_frag:
                    break
            else:
                integrated_frags.append({frag_idx})
        return integrated_frags

    def scatter_sum_frag_dummy_atom_counts(
        self,
        intergrated_frags: List[Set[INDEX]],
        frag_dummy_atom_counts: List[COUNT],
        num_cut_edges: int,
    ) -> Optional[Set[INDEX]]:
        """\
        Scatter sum dummy atoms of fragments.
        If the number of dummy atoms is not equal to the number of cut edges,
        return None.

        Parameters
        ----------
        intergrated_frags: List[Set[INDEX]]
        frag_dummy_atom_counts: List[COUNT]
        num_cut_edges: int
        
        Returns
        -------
        key_frag: Optional[Set[INDEX]]
        """
        # NOTE: OK for using just initial fragment since it is connected frag
        # from remained edges?

        for intergrated_frag in intergrated_frags:
            frag_wise_dummy_atom_count = 0
            for frag_idx in intergrated_frag:
                frag_wise_dummy_atom_count += frag_dummy_atom_counts[frag_idx]
            if frag_wise_dummy_atom_count == num_cut_edges:
                return intergrated_frag

    def get_frag_smis_by_cut_edges(
        self,
        cut_edges: Tuple[BOND_INDICE],
        frag_wise_edges: List[Tuple[BOND_INDICE]],
    ) -> List[SMILES]:
        """\
        Get fragment SMILES by cutting edges.
        Using the atom indice of BRICS bond, fragment the molecule,
        and return the SMILES of each fragment.

        Parameters
        ----------
        cut_edges: Tuple[BOND_INDICE]
        frag_wise_edges: List[Tuple[BOND_INDICE]]

        Returns
        -------
        frag_smis: List[SMILES]
        """

        cut_edge_indice = [frag_wise_edges.index(edge) for edge in cut_edges]
        cut_brics_bonds = [self.brics_bonds[idx] for idx in cut_edge_indice]

        bond_indice = []
        brics_types = []
        for cut_brics_bond in cut_brics_bonds:
            brics_atom_idx, brics_type = cut_brics_bond
            brics_atom_idx = tuple(map(int, brics_atom_idx))
            bond = self.mol.GetBondBetweenAtoms(*brics_atom_idx)
            bond_idx = bond.GetIdx()
            begin_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()

            bond_indice.append(bond_idx)
            if (begin_atom_idx, end_atom_idx) == brics_atom_idx:
                brics_type = (brics_type[1], brics_type[0])
            brics_types.append(tuple(map(int, brics_type)))

        frag_mols = Chem.FragmentOnBonds(
            self.mol,
            bondIndices=bond_indice,
            dummyLabels=brics_types,
        )
        frag_smis = Chem.MolToSmiles(frag_mols).split(".")
        return frag_smis

    def mol_to_frag_info(self, verbose: bool = False) -> List[FRAG_INFO]:
        """\
        Convert a molecule to a list of fragment information.
        1. `self.get_brics_bonds_info`
        2. `self.get_atom_frag_indice`
        3. `self.extract_frag_wise_edges`
        4. `self.fragmentation`
        
        Parameters
        ----------
        verbose : bool, default False

        Returns
        -------
        frag_info : List[FRAG_INFO]
        """
        brics_bonds, brics_bonds_indice = self.get_brics_bonds_info()
        self.brics_bonds = brics_bonds
        self.brics_bonds_indice = brics_bonds_indice

        self.atom_frag_indice = self.get_atom_frag_indice(verbose)
        self.num_frags = len(set(self.atom_frag_indice))

        frag_wise_edges = self.extract_frag_wise_edges(brics_bonds_indice, verbose)

        fragmentation_results = self.fragmentation(frag_wise_edges)
        return fragmentation_results


def write_data(line: str, dic: Dict[str, int]):
    cid, pchembl, assayid, targetid, smi = line.split()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return
    frag_parser = FragmentParser(mol, max_natoms=args.max_natoms)
    fragmentation_results = frag_parser.mol_to_frag_info()
    if not fragmentation_results:
        return

    with ExitStack() as stack:
        lock = globals().get("lock", False)
        if lock:
            stack.enter_context(lock)
        for fragmentation in fragmentation_results:
            for key_frag, values in fragmentation.items():
                (
                    rest_frag,
                    key_frag_atom_indice,
                    atom_frag_indice,
                    brics_bonds_indice,
                    cut_edges_indice,
                ) = values
                num_frags = len(rest_frag.split(".")) + 1
                line = [
                    cid,
                    smi,
                    assayid,
                    targetid,
                    key_frag,
                    rest_frag,
                    str(num_frags),
                    key_frag_atom_indice,
                    atom_frag_indice,
                    brics_bonds_indice,
                    cut_edges_indice,
                ]
                line = ";".join(line)
                dic[line] = 1
    return


def filter_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.duplicated(subset=["REST-FRAG"], keep=False)]
    df = df.drop_duplicates()
    return df


def init_pool(lock_: synchronize.Lock):
    global lock
    lock = lock_


def main():
    args.result_file.parent.mkdir(exist_ok=True, parents=True)

    manager = Manager()
    dic = manager.dict()
    lock = Lock()
    nprocs = cpu_count() if not args.nprocs else args.nprocs
    with args.chembl_file.open("r") as f:
        f.readline()  # skip first line

        worker = partial(write_data, dic=dic)
        with Pool(nprocs, initializer=init_pool, initargs=(lock,)) as pool:
            pool.map(worker, f)

    data = [key.split(";") for key in dic.keys()]
    df = pd.DataFrame(data, columns=COLS)
    df = filter_duplicates(df)

    df["NUM-FRAGS"].astype(int)
    df.to_csv(args.result_file, index=False)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Fragment library argparser")
    parser.add_argument("--chembl_file", type=Path, default="./mini_chembl.txt")
    parser.add_argument("--result_file", type=Path, default="./result.csv")
    parser.add_argument("--max_natoms", type=int, default=12)
    parser.add_argument("-n", "--nprocs", type=int, default=1)
    parser.add_argument("--cols", type=str, nargs="+", default=COLS)
    args, _ = parser.parse_known_args()

    main()
