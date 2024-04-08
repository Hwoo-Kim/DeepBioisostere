from typing import List, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS
from rdkit.Chem.rdchem import Mol
from torch_geometric.data import Batch

RDLogger.DisableLog("rdApp.*")

import numpy as np
import torch

from .data import PairData

BOND_INDICE = Tuple[int, int]
BRICS_TYPE = Tuple[str, str]
BRICS_BOND = Tuple[BOND_INDICE, BRICS_TYPE]  # ((3, 2), ('3', '4'))

# zeros in period, group, degree are for '*' symbol in BRICS.
x_map = {
    "period": list(range(0, 6)),
    "group": list(range(0, 9)) + ["ELSE"],
    "degree": list(range(0, 7)) + ["ELSE"],
    "formal_charge": list(range(-2, 3)) + ["ELSE"],
    "num_hs": list(range(0, 5)) + ["ELSE"],
    "hybridization": [
        Chem.rdchem.HybridizationType.UNSPECIFIED,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.OTHER,
    ],
    "chirality": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    "is_aromatic": [True],
    "is_in_ring": [True],
    "brics_type": list(range(0, 17)),
}

e_map = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "stereo": [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
        Chem.rdchem.BondStereo.STEREOANY,
    ],
    "is_conjugated": [True],
    "is_brics_bond": [True],
}

NUM_ATOM_FEATURES_BRICS = sum(len(choices) for choices in x_map.values())
# 6 + 10 + 8 + 6 + 6 + 7 + 4 + 1 + 1 + 17 = 66
NUM_ATOM_FEATURES = NUM_ATOM_FEATURES_BRICS - 17
# 6 + 9 + 0  + 7 + 10 + 6 + 6 + 7 + 1 + 2 = 49
NUM_EDGE_FEATURES = sum(len(choices) for choices in e_map.values())
# 4 + 6 + 1 + 1= 12

_periodic_table = Chem.GetPeriodicTable()
_periodic_list = [0, 2, 10, 18, 36, 54]


def _get_periodic_inform(atomic_num: int):
    if atomic_num == 0:
        return 0, 0
    for i, crr in enumerate(_periodic_list):
        if crr >= atomic_num:
            break
        period = i + 1
        group = _periodic_table.GetNOuterElecs(atomic_num)
        if _periodic_table.GetDefaultValence(atomic_num) == -1:
            group = -1  # for transition metals
    return period, group


def _one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


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

    # No BRICS bond
    if len(bond_indice) == 0:
        return None

    broken_mol = Chem.FragmentOnBonds(
        mol, bondIndices=bond_indice, dummyLabels=brics_types
    )
    Chem.rdmolops.SanitizeMol(broken_mol)
    return broken_mol


def from_mol(mol: Mol, type: str, return_broken_mol=False, original_smiles=None):
    """
    original_smiles: str. original SMILES string. Used for atom indexing.
    """
    if original_smiles is None:
        smi_to_return = Chem.MolToSmiles(mol)
    else:
        smi_to_return = original_smiles

    assert any([type == "Mol", type == "Frag"]), ValueError(
        "'type' argument must be given 'Mol' or 'Frag'."
    )

    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    if type == "Mol":
        mol_to_parse = cut_brics_bonds(mol, brics_bonds)
        if mol_to_parse is None:
            print(f"No BRICS bond in {Chem.MolToSmiles(mol)}")
            return None, None
    else:
        mol_to_parse = mol

    xs = []
    for atom in mol_to_parse.GetAtoms():
        x = []
        period, group = _get_periodic_inform(atom.GetAtomicNum())

        x.append(_one_of_k_encoding(period, x_map["period"]))
        x.append(_one_of_k_encoding(group, x_map["group"]))
        x.append(_one_of_k_encoding(atom.GetTotalDegree(), x_map["degree"]))
        x.append(_one_of_k_encoding(atom.GetFormalCharge(), x_map["formal_charge"]))
        x.append(_one_of_k_encoding(atom.GetTotalNumHs(), x_map["num_hs"]))
        x.append(_one_of_k_encoding(atom.GetHybridization(), x_map["hybridization"]))
        x.append(_one_of_k_encoding(atom.GetChiralTag(), x_map["chirality"]))
        x.append([atom.GetIsAromatic() == x_map["is_aromatic"][0]])
        x.append([atom.IsInRing() == x_map["is_in_ring"][0]])
        x.append(_one_of_k_encoding(atom.GetIsotope(), x_map["brics_type"]))
        xs.append(np.concatenate(x, axis=0))

    x = np.stack(xs, axis=0)
    x = torch.from_numpy(x).to(torch.float).view(-1, NUM_ATOM_FEATURES_BRICS)

    brics_bond_indices = [
        brics_bond[0] for brics_bond in brics_bonds
    ]  # only atom_indices
    brics_bond_types = [brics_bond[1] for brics_bond in brics_bonds]  # only brics_types
    adj_dummy_inform, edge_indices, edge_attrs = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        if type == "Frag":
            if mol.GetAtomWithIdx(i).GetSymbol() == "*":
                adj_dummy_inform.append((j, mol.GetAtomWithIdx(i).GetIsotope()))
            if mol.GetAtomWithIdx(j).GetSymbol() == "*":
                adj_dummy_inform.append((i, mol.GetAtomWithIdx(j).GetIsotope()))

        e = []
        e.append(_one_of_k_encoding(bond.GetBondType(), e_map["bond_type"]))
        e.append(_one_of_k_encoding(bond.GetStereo(), e_map["stereo"]))
        e.append([bond.GetIsConjugated() == e_map["is_conjugated"][0]])
        e.append(
            [
                (i, j) in brics_bond_indices
                or (j, i) in brics_bond_indices == e_map["is_brics_bond"][0]
            ]
        )
        e = np.concatenate(e, axis=0)

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_attrs = np.stack(edge_attrs, axis=0)
    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.int).view(2, -1)
    edge_attr = torch.from_numpy(edge_attrs).view(-1, NUM_EDGE_FEATURES).float()

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    if type == "Frag":
        return PairData(
            x_n=x,
            edge_index_n=edge_index,
            edge_attr_n=edge_attr,
            smiles=smi_to_return,
            num_atoms=mol.GetNumAtoms(),  # including dummy atoms
            adj_dummy_inform=adj_dummy_inform,
        )

    edge_indices_w_dummy, edge_attrs_w_dummy = [], []
    for bond in mol_to_parse.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(_one_of_k_encoding(bond.GetBondType(), e_map["bond_type"]))
        e.append(_one_of_k_encoding(bond.GetStereo(), e_map["stereo"]))
        e.append([bond.GetIsConjugated() == e_map["is_conjugated"][0]])
        e.append(
            [
                (i, j) in brics_bond_indices
                or (j, i) in brics_bond_indices == e_map["is_brics_bond"][0]
            ]
        )
        e = np.concatenate(e, axis=0)

        edge_indices_w_dummy += [[i, j], [j, i]]
        edge_attrs_w_dummy += [e, e]

    edge_attrs_w_dummy = np.stack(edge_attrs_w_dummy, axis=0)
    edge_index_w_dummy = torch.tensor(edge_indices_w_dummy)
    edge_index_w_dummy = edge_index_w_dummy.t().to(torch.int).view(2, -1)
    edge_attr_w_dummy = (
        torch.from_numpy(edge_attrs_w_dummy).view(-1, NUM_EDGE_FEATURES).float()
    )

    if edge_index_w_dummy.numel() > 0:  # Sort indices.
        perm = (edge_index_w_dummy[0] * x.size(0) + edge_index_w_dummy[1]).argsort()
        edge_index_w_dummy, edge_attr_w_dummy = (
            edge_index_w_dummy[:, perm],
            edge_attr_w_dummy[perm],
        )

    parsed_data = PairData(
        x_n=x,
        edge_index_n=edge_index,
        edge_index_n_w_dummy=edge_index_w_dummy,
        edge_attr_n=edge_attr,
        edge_attr_n_w_dummy=edge_attr_w_dummy,
        smiles=smi_to_return,
        num_atoms=mol.GetNumAtoms(),
        brics_bond_indices=brics_bond_indices,
        brics_bond_types=brics_bond_types,
    )

    if return_broken_mol:
        return parsed_data, mol_to_parse
    else:
        return parsed_data


def from_smiles(smiles: str, type: str, kekulize: bool = False):
    r"""Converts a SMILES string to a data.PairData instance.

    Args:
        smiles (string): The SMILES string.
        type (string): Which type of graph to parse.
            We use differnt features for "Current target molecule" and
            "Fragments".
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """

    assert any([type == "Mol", type == "Frag"]), ValueError(
        "'type' argument must be given 'Mol' or 'Frag'."
    )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if kekulize:
        mol = Chem.Kekulize(mol)

    return from_mol(mol, type, original_smiles=smiles)


if __name__ == "__main__":
    # smis = ["[7*]CC", "C(=O)CNC", "c1ccccc1", "C=CC=CC=C", "C1CCCCC1"]
    # smis = ["O=C(Nc1nnc(SCc2ccc(Br)cc2)s1)c1ccc(Cl)cc1"]
    # for smi in smis:
    #    # frag_data = from_smiles(smi, type="Frag")
    #    # print(frag_data.x_n)
    #    # print(frag_data.edge_index_n)
    #    # print(frag_data.edge_attr_n)
    #    # print()
    #    mol_data = from_smiles(smi, type="Mol")
    #    # print(mol_data.x_n)
    #    # print(mol_data.edge_index_n)
    #    # print(mol_data.edge_index_n_w_dummy)
    #    # print(mol_data.edge_attr_n)
    #    # print(mol_data.edge_attr_n_w_dummy)
    #    print(mol_data.brics_bonds)
    frag = "[13*]C1OC2([13*])OC([13*])([C@H]([15*])[C@H]2O)C1([15*])O"
    data = from_smiles(frag, type="Frag")
    print(data)
    print(data.adj_dummy_inform)
    batch = Batch.from_data_list([data, data])
    print(np.array(batch.adj_dummy_inform, dtype=list)[0])
