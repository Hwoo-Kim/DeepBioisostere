import math
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.FragmentMatcher import FragmentMatcher as FM
from torch_geometric.data import Batch, Dataset

from .brics.brics import BRICSModule
from .conditioning import Conditioner
from .data import PairData
from .feature import from_mol
from .model import DeepBioisostere

mp.set_sharing_strategy("file_system")


COLS = [
    "INDEX",
    "REF-CID",
    "PRB-CID",
    "REF-SMI",
    "PRB-SMI",
    "KEY-FRAG-ATOM-INDICE",
    "ATOM-FRAG-INDICE",
    "OLD-FRAG",
    "NEW-FRAG-FREQ",
    "NEW-FRAG",
    "NEW-FRAG-IDX",
    "ALLOWED-ATTACHMENT",
    "DATATYPE",
]
MAX_NUM_CHANGE_ATOMS = 12
SMILES = str
PROPERTY = str


class TrainDataset(Dataset):
    # properties = PROPERTIES

    def __init__(self, data_df: pd.DataFrame, conditioner: Conditioner):
        super().__init__()
        self.data_df = data_df
        self.conditioner = conditioner

    @classmethod
    def get_datasets(
        cls,
        data_path: Path,
        conditioner: Conditioner,
        modes: Union[list, tuple],
    ):
        # Read the input parameters
        data_df = pd.read_csv(
            data_path,
            sep="\t",
            dtype={"KEY-FRAG-ATOM-INDICE": str, "ATOM-FRAG-INDICE": str},
        )

        # Get train / val data keys
        dataset_list = []
        for mode in modes:
            keys = (
                (
                    (data_df["DATATYPE"] == mode)
                    & (data_df["ALLOWED-ATTACHMENT"].isnull() == False)
                )
                .to_numpy()
                .nonzero()[0]
            )
            if len(keys) > int(math.pow(2, 24)):
                random.shuffle(keys)
                keys = keys[: int(math.pow(2, 24))]
                keys.sort()
            dataset_list.append(cls(data_df.iloc[keys], conditioner))

        return dataset_list

    def get_data_weights(self, alpha2: float):
        # Read frag_lib
        data_freq = self.data_df["OLD-FRAG-FREQ"].to_numpy()

        # Apply exponential function to data_frequencies to get data_weights
        data_freq = torch.tensor(data_freq).float()
        data_weights = data_freq.pow(-alpha2)

        return data_weights

    def len(self):
        return len(self.data_df)

    def get(self, idx):
        item = dict()
        input_data = self.data_df.iloc[idx]

        # Read input information
        ref_smi = input_data["REF-SMI"]
        new_smi = input_data["PRB-SMI"]
        new_frag_smi = input_data["NEW-FRAG"]
        key_frag_atom_indice = input_data["KEY-FRAG-ATOM-INDICE"]
        atom_frag_indice = input_data["ATOM-FRAG-INDICE"]
        allowed_attachment = input_data["ALLOWED-ATTACHMENT"]

        # Parse the input data
        ref_mol = Chem.MolFromSmiles(ref_smi)
        data = from_mol(ref_mol, type="Mol", original_smiles=ref_smi)  # PairData object
        change_atom_indice = list(map(int, key_frag_atom_indice.split(",")))
        atom_frag_indice = list(map(int, atom_frag_indice.split(",")))
        atom_frag_indice_w_dummy = self.addDummyAtoms(
            data, data.num_atoms, atom_frag_indice
        )
        leaving_frag = list(
            set([atom_frag_indice[atom_idx] for atom_idx in change_atom_indice])
        )
        data.new_frag_num_atoms = Chem.MolFromSmiles(new_frag_smi).GetNumAtoms()

        # Get fragment-atom index mapping
        frag_atom_indice = dict()
        for i, f_id in enumerate(atom_frag_indice):
            try:
                frag_atom_indice[int(f_id)].append(i)
            except KeyError:
                frag_atom_indice[int(f_id)] = [i]
        frag_atom_indice = {
            key: tuple(value) for key, value in frag_atom_indice.items()
        }

        # Get fragment-level edges
        edge_index_f, edge_attr_f = [], []
        for edge in data.brics_bond_indices:
            idx1, idx2 = edge
            edge_index_f.append((atom_frag_indice[idx1], atom_frag_indice[idx2]))
            edge_index_f.append((atom_frag_indice[idx2], atom_frag_indice[idx1]))
            edge_attr_f.append([idx1, idx2])
            edge_attr_f.append([idx2, idx1])

        # Tensors for fragment-level edges
        data.x_n_mask = torch.zeros(len(atom_frag_indice_w_dummy)).bool()
        data.x_n_mask[: data.num_atoms] = True
        data.x_f = torch.tensor(atom_frag_indice_w_dummy)
        data.edge_index_f = torch.tensor(edge_index_f).t().contiguous()
        data.edge_attr_f = torch.tensor(edge_attr_f)

        # Modification answers
        data.y_fragID = input_data["NEW-FRAG-IDX"]
        data.y_pos_subgraph = torch.tensor(leaving_frag)
        data.y_pos_subgraph_idx = torch.zeros_like(data.y_pos_subgraph).long()
        (
            data.y_neg_subgraph,
            data.y_neg_subgraph_idx,
        ) = BRICSModule.get_allowed_neg_subgraph(
            frag_atom_indice,
            edge_index_f,
            leaving_frag,
            data.num_atoms,
            MAX_NUM_CHANGE_ATOMS,
        )
        if data.y_neg_subgraph_idx.size(0):
            data.y_neg_subgraph_batch = torch.zeros(
                data.y_neg_subgraph_idx.max() + 1
            ).long()
        else:
            data.y_neg_subgraph_batch = torch.tensor([]).long()

        # Get brics types of adjacent subgraphs
        data.adj_brics_type = BRICSModule.get_adj_frag_brics_type(
            ref_mol, change_atom_indice
        )

        # Parse attachment position informatoin
        new_to_ref_atom_pairs = [
            pair.split("-") for pair in allowed_attachment.split(",")
        ]
        new_to_ref_atom_pairs = {
            frag_atom: ref_atoms.split("o")
            for frag_atom, ref_atoms in new_to_ref_atom_pairs
        }

        new_frag_indices = set()
        ref_atom_indices = set()
        for k, v in new_to_ref_atom_pairs.items():
            new_frag_indices.add(k)
            ref_atom_indices.update(set(v))

        new_frag_indices = list(new_frag_indices)
        ref_atom_indices = list(ref_atom_indices)

        compose_original_idx = []
        compose_fragment_idx = []
        for ref_atom_idx in ref_atom_indices:
            compose_original_idx += [ref_atom_idx] * len(new_frag_indices)
            compose_fragment_idx += new_frag_indices
        compose_allowed_bool = [
            original_atom in new_to_ref_atom_pairs[frag_atom]
            for frag_atom, original_atom in zip(
                compose_fragment_idx, compose_original_idx
            )
        ]

        # Tensors for attachment position prediction
        data.compose_original_idx = torch.tensor(
            list(map(int, compose_original_idx))
        ).long()
        data.compose_fragment_idx = torch.tensor(
            list(map(int, compose_fragment_idx))
        ).long()
        data.compose_allowed_bool = torch.tensor(compose_allowed_bool).bool()

        # Property conditioning
        data.num_frags = max(atom_frag_indice) + 1
        if self.conditioner:
            encoded_condition_dict = self.conditioner(
                ref_smi=ref_smi, new_smi=new_smi
            )  # [1] or [F_node-17]: 17 is the number of BRICS types
            for prop, encoded_prop in encoded_condition_dict.items():
                encoded_condition_dict[prop] = encoded_prop.repeat(
                    data.num_frags, 1
                )  # [F, 1] or [F, F_node-17], depending on use_soft_one_hot option of conditioner

        # Return the result
        item["data"] = data
        # item["brics_type"] = data.adj_brics_type
        # item["brics_type"] = tuple(
        #     list(map(lambda x: int(x), data.adj_brics_type.split(",")))
        # )
        if self.conditioner:
            item.update(encoded_condition_dict)
        return item

    @staticmethod
    def addDummyAtoms(data: PairData, num_atoms: int, atom_frag_indice: List[int]):
        edges_w_dummy = data.edge_index_n_w_dummy.t().tolist()
        atom_frag_indice = {
            atom_idx: frag_idx for atom_idx, frag_idx in enumerate(atom_frag_indice)
        }
        for edge in edges_w_dummy:
            i, j = sorted(edge)  # i < j
            if j < num_atoms:
                continue
            atom_frag_indice[j] = atom_frag_indice[
                i
            ]  # j: dummy atom, i: adjacent atom to j

        atom_indice, frag_indice = list(atom_frag_indice.keys()), list(
            atom_frag_indice.values()
        )
        atom_indice, frag_indice = np.array(atom_indice), np.array(frag_indice)
        atom_frag_indice_w_dummy = frag_indice[atom_indice.argsort()].tolist()
        return atom_frag_indice_w_dummy


class TrainCollator:
    def __init__(
        self,
        fragment_library,
        num_neg_sample,
        mode: str,
        alpha1: float,
        use_conditioning: bool,
        properties: List[PROPERTY] = None,
        follow_batch=None,
        exclude_keys=None,
    ):
        # Read the fragment library
        self.frags_freq = fragment_library.frags_freq  # torch.LongTensor
        self.frag_features = fragment_library.frag_features  # List[PairData]

        self.frag_brics_maskings = fragment_library.frag_brics_maskings  # dict
        self.data_type = fragment_library.data_type  # np.array
        self.num_neg_sample = num_neg_sample
        self.mode = mode

        self.alpha1 = alpha1
        self.conditioning = use_conditioning
        self.properties = properties

        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        # Set train/val/test mode
        frag_datatype_mask = torch.from_numpy(self.data_type == mode).bool()
        for k, v in self.frag_brics_maskings.items():
            self.frag_brics_maskings[k] = torch.mul(v, frag_datatype_mask)

    def __call__(self, batch):
        # Initialize
        collated_batch = dict()
        parsed_data, pos_frags_data, frags_mask = [], [], []
        if self.conditioning:
            prop_dict = {prop: [] for prop in self.properties}

        # Read the data from dataset 'get' method
        pos_frags_IDs = []
        for item in batch:  # Tuple[PairData, Tuple]
            data = item["data"]
            brics_type = data.adj_brics_type
            frag_brics_type = self.frag_brics_maskings[brics_type]
            if int(frag_brics_type.sum(0)) == 1:
                continue
            parsed_data.append(data)
            pos_frags_data.append(
                self.frag_features[data.y_fragID]
            )
            pos_frags_IDs.append(data.y_fragID)
            frags_mask.append(frag_brics_type)

            if self.conditioning:
                for prop in self.properties:
                    prop_dict[prop].append(
                        item[prop]
                    )  # [F, 1] or [F, F_node-17], depending on use_soft_one_hot option of conditioner

        # The answer fragment IDs
        pos_frags_IDs = torch.tensor(pos_frags_IDs).long().unsqueeze(-1)  # [B,1]

        # Mask fragment frequency
        mask = torch.zeros_like(pos_frags_IDs).bool()
        frags_mask = torch.stack(frags_mask, dim=0).bool()  # [B,frags_lib_size]
        frags_mask.scatter_(dim=1, index=pos_frags_IDs, src=mask)
        masked_freq = torch.mul(self.frags_freq, frags_mask).pow(self.alpha1)

        # Negative sampling
        neg_indice = torch.multinomial(
            input=masked_freq, num_samples=self.num_neg_sample, replacement=True
        )  # [B,n]
        neg_indice = neg_indice.reshape(-1).long().tolist()
        neg_frags_data = [self.frag_features[idx] for idx in neg_indice]

        # Batch all the data
        collated_batch["data"] = Batch.from_data_list(
            parsed_data, self.follow_batch, self.exclude_keys
        )
        collated_batch["pos"] = Batch.from_data_list(
            pos_frags_data, self.follow_batch, self.exclude_keys
        )
        collated_batch["neg"] = Batch.from_data_list(
            neg_frags_data, self.follow_batch, self.exclude_keys
        )

        # Concat Condition Embedding
        if self.conditioning:
            for prop, condition_list in prop_dict.items():
                collated_batch[prop] = torch.concat(condition_list, dim=0)

        return collated_batch


class InferenceDataset(Dataset):
    def __init__(
        self,
        smiles_list: List[SMILES],
        conditioner: Union[Conditioner, None],
        prop_dict_list: Union[List[Dict[str, float]], None],
    ):
        super().__init__()
        assert all(list(map(lambda x: isinstance(x, str), smiles_list))), ValueError(
            "molecule_list must be list of SMILES."
        )
        self.data_list = smiles_list
        self.conditioner = conditioner
        self.prop_dict_list = prop_dict_list
        self.exist_leaving_frag_inform = False
        return

    def set_leaving_frag_inform(
        self, leaving_frag_list: List[SMILES], atom_idx_list: List[int]
    ):
        self.exist_leaving_frag_inform = True
        self.leaving_frag_list = (
            leaving_frag_list  # each frag smi involves one or more [*]
        )
        self.atom_idx_list = atom_idx_list
        return

    def reset_leaving_frag_inform(self):
        self.exist_leaving_frag_inform = False
        self.leaving_frag_list = None
        self.atom_idx_list = None

    def len(self):
        return len(self.data_list)

    def get(self, idx, verbose=False) -> Union[PairData, None]:
        smi = self.data_list[idx]
        data = self.parse_smiles(smi, verbose)

        if data is None:
            return None

        # Specifying leaving fragment
        if self.exist_leaving_frag_inform:
            try:
                data = self.set_leaving_frag(data, idx)
            except Exception as e:
                print(e)
                return None

        if data is None:
            return None

        # Property conditioning
        if self.conditioner:
            prop_dict = self.prop_dict_list[idx]
            encoded_condition_dict = self.conditioner(
                prop_dict=prop_dict
            )  # [1] or [F_node-17]: 17 is the number of BRICS types
            for prop, encoded_prop in encoded_condition_dict.items():
                encoded_condition_dict[prop] = encoded_prop.repeat(
                    data.num_frags, 1
                )  # [F, 1]

        item = {"data": data}
        if self.conditioner:
            item.update(encoded_condition_dict)
        return item

    def set_leaving_frag(self, data: PairData, data_idx: int) -> PairData:
        """
        Specify the leaving fragment information for the data.
        Below attributes are modified:
            allowed_subgraph
            allowed_subgraph_idx
            allowed_subgraph_batch
            num_allowed_subgraph
            brics_types
        """
        # Read input data
        original_smi = data.smiles
        leaving_frag_smi = self.leaving_frag_list[data_idx]
        atom_idx = self.atom_idx_list[data_idx]
        mol = Chem.MolFromSmiles(original_smi)

        # Remove BRICS types
        leaving_frag_mol = Chem.MolFromSmiles(leaving_frag_smi)
        for atom in leaving_frag_mol.GetAtoms():
            if atom.GetSymbol() == "*":  # dummy atom
                atom.SetIsotope(0)
        removed_leaving_frag_smi = Chem.MolToSmiles(leaving_frag_mol)

        leaving_frag_mol = Chem.MolFromSmiles(removed_leaving_frag_smi)
        dummy_atom_idxs = []
        for atom in leaving_frag_mol.GetAtoms():
            if atom.GetSymbol() == "*":  # dummy atom
                dummy_atom_idxs.append(atom.GetIdx())

        # Get subgraph matching
        matcher_with_dummy = FM()
        matcher_with_dummy.Init(removed_leaving_frag_smi)
        matches_with_dummy = matcher_with_dummy.GetMatches(mol)  # tuple of tuples

        for match_with_dummy in matches_with_dummy:
            if atom_idx in match_with_dummy and match_with_dummy.index(atom_idx) not in dummy_atom_idxs:
                break
        else:
            print("Specified atom is not a member of the given fragment.")
            print(f"original_smi: {original_smi}")
            print(f"leaving_frag_smi: {leaving_frag_smi}")
            print(f"atom_idx: {atom_idx}\n")
            return None

        # Remove dummy atoms, which are adjacent atoms of target subgraph
        match = list(
            filter(lambda x: x[0] not in dummy_atom_idxs, enumerate(match_with_dummy))
        )
        match = sorted(list(map(lambda x: x[1], match)))

        # Get atom to frag indice
        atom_frag_indice = list(map(int, data.atom_frag_indice.split(",")))
        atom_frag_indice = np.array(atom_frag_indice)

        frag_atom_indice = dict()
        for i, f_id in enumerate(atom_frag_indice):
            try:
                frag_atom_indice[int(f_id)].append(i)
            except KeyError:
                frag_atom_indice[int(f_id)] = [i]
        frag_atom_indice = {
            key: tuple(value) for key, value in frag_atom_indice.items()
        }

        # Get involved fragment indices
        involved_frag_idxs = atom_frag_indice[match]
        involved_frag_idxs = sorted(list(set(involved_frag_idxs)))
        involved_atoms = []
        for frag in involved_frag_idxs:
            involved_atoms.extend(frag_atom_indice[frag])
        if sorted(involved_atoms) != match:
            print("Error occured when retrieving a subgraph.")
            print("Please check whether the subgraph is allowed.")
            print(f"Match: {match}")
            return None

        # Get subgraph index -> subgraph_idx
        num_subgraphs = data.allowed_subgraph_idx.max() + 1
        for subgraph_idx in range(num_subgraphs):
            frags_in_subgraph = torch.where(data.allowed_subgraph_idx == subgraph_idx)[
                0
            ]
            frags_in_subgraph = sorted(
                data.allowed_subgraph[frags_in_subgraph].tolist()
            )
            if frags_in_subgraph == involved_frag_idxs:
                break
        else:
            print("Error occured when retrieving a subgraph.")
            print("Please check whether the subgraph is allowed.")
            print(f"original_smi: {original_smi}")
            print(f"leaving_frag_smi: {leaving_frag_smi}")
            print(f"involved_frag_idxs: {involved_frag_idxs}")
            print(f"data.allowed_subgraph: {data.allowed_subgraph.tolist()}")
            print(f"data.allowed_subgraph_idx: {data.allowed_subgraph_idx.tolist()}\n")
            return None

        # Set data attributes
        data.allowed_subgraph = torch.tensor(sorted(involved_frag_idxs))
        data.allowed_subgraph_idx = torch.tensor([0] * len(data.allowed_subgraph))
        data.allowed_subgraph_batch = torch.zeros(
            data.allowed_subgraph_idx.max() + 1
        ).long()
        data.num_allowed_subgraph = int(data.allowed_subgraph_idx.max()) + 1
        data.min_allowed_subgraph = int(data.allowed_subgraph.min())
        data.brics_types = [data.brics_types[subgraph_idx]]  # list of tuple of int

        return data

    @staticmethod
    def parse_smiles(smi, verbose=False) -> Union[PairData, None]:
        # Read input smiles
        mol = Chem.MolFromSmiles(smi)
        data, broken_mol = from_mol(mol, type="Mol", return_broken_mol=True, original_smiles=smi)

        # if mol.GetNumAtoms() == broken_mol.GetNumAtoms():  # no BRICS bond
        if data is None:  # no BRICS bond
            return None

        if verbose:
            print("broken_mol:", Chem.MolToSmiles(broken_mol))

        # Get atom-fragment index mapping
        atom_frag_indice_w_dummy = torch.zeros(broken_mol.GetNumAtoms()).long()
        frag_indice_list = list(Chem.GetMolFrags(broken_mol, asMols=False))
        for frag_idx, indices in enumerate(frag_indice_list):
            atom_frag_indice_w_dummy[list(indices)] = frag_idx
        atom_frag_indice_w_dummy = atom_frag_indice_w_dummy.tolist()
        atom_frag_indice = atom_frag_indice_w_dummy[: data.num_atoms]

        # Get fragment-atom index mapping
        frag_atom_indice = dict()
        for i, f_id in enumerate(atom_frag_indice):
            try:
                frag_atom_indice[int(f_id)].append(i)
            except KeyError:
                frag_atom_indice[int(f_id)] = [i]
        frag_atom_indice = {
            key: tuple(value) for key, value in frag_atom_indice.items()
        }

        # Get fragment-level edges
        edge_index_f, edge_attr_f = [], []
        brics_bonds = BRICS.FindBRICSBonds(mol)
        for edge, _ in brics_bonds:
            idx1, idx2 = edge
            edge_index_f.append((atom_frag_indice[idx1], atom_frag_indice[idx2]))
            edge_index_f.append((atom_frag_indice[idx2], atom_frag_indice[idx1]))
            edge_attr_f.append([idx1, idx2])
            edge_attr_f.append([idx2, idx1])

        # Tensors for fragment-level edges
        data.x_n_mask = torch.zeros(len(atom_frag_indice_w_dummy)).bool()
        data.x_n_mask[: data.num_atoms] = True
        data.x_f = torch.tensor(atom_frag_indice_w_dummy)
        data.edge_index_f = torch.tensor(edge_index_f).t().contiguous()
        data.edge_attr_f = torch.tensor(edge_attr_f)

        # Allowed leaving_fragments
        (
            data.allowed_subgraph,
            data.allowed_subgraph_idx,
        ) = BRICSModule.get_allowed_subgraph(
            frag_atom_indice, edge_index_f, data.num_atoms, MAX_NUM_CHANGE_ATOMS
        )
        data.min_allowed_subgraph = int(data.allowed_subgraph.min())
        if data.allowed_subgraph_idx.size(0):
            data.allowed_subgraph_batch = torch.zeros(
                data.allowed_subgraph_idx.max() + 1
            ).long()
            data.num_allowed_subgraph = int(data.allowed_subgraph_idx.max()) + 1
        else:
            data.allowed_subgraph_batch = torch.tensor([]).long()
            data.num_allowed_subgraph = 0

        # Get brics types of adjacent subgraphs
        subgraph_idx_to_brics_type = []
        for subgraph_idx in range(
            int(max(data.allowed_subgraph_idx)) + 1
        ):  # number of subgraphs
            subgraph_fragments_indices = torch.argwhere(
                data.allowed_subgraph_idx == subgraph_idx
            ).squeeze(dim=-1)
            fragments = data.allowed_subgraph[subgraph_fragments_indices].tolist()
            atoms_in_subgraph = []
            for frag in fragments:
                atoms_in_subgraph += frag_atom_indice[frag]

            other_frags, _, _ = BRICSModule.get_adjacent_fragments(
                mol, atoms_in_subgraph
            )
            if other_frags is None:
                return None  # No allowed modification position

            brics_types = []
            for frag in other_frags:
                for atom in frag.GetAtoms():
                    if atom.GetSymbol() == "*":
                        brics_types.append(int(atom.GetIsotope()))
            brics_types.sort()
            subgraph_idx_to_brics_type.append(tuple(brics_types))

        data.brics_types = subgraph_idx_to_brics_type
        data.atom_frag_indice = ",".join(list(map(str, atom_frag_indice)))
        data.num_frags = max(atom_frag_indice) + 1
        return data


class InferenceCollator:
    node_hid_dim = DeepBioisostere.default_args["frag_node_hid_dim"]

    def __init__(
        self,
        use_conditioning: bool,
        properties: List[PROPERTY] = None,
        follow_batch=None,
        exclude_keys=None,
    ):
        self.conditioning = use_conditioning
        self.properties = properties
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        collated_batch = dict()
        parsed_data = []
        if self.conditioning:
            prop_dict = {prop: [] for prop in self.properties}

        # Read the data from dataset 'get' method
        for item in batch:
            if item is None:
                continue
            data = item["data"]
            if data.num_allowed_subgraph == 0:
                continue
            parsed_data.append(data)

            num_frags = data.num_frags
            nan_tensor = torch.tensor([torch.nan] * self.node_hid_dim)
            nan_tensor = nan_tensor.repeat(num_frags, 1)
            if self.conditioning:
                for prop in self.properties:
                    prop_dict[prop].append(item[prop])

        collated_batch["data"] = Batch.from_data_list(
            parsed_data, self.follow_batch, self.exclude_keys
        )

        # Concat Condition Embedding
        if self.conditioning:
            for prop, condition_list in prop_dict.items():
                collated_batch[prop] = torch.concat(condition_list, dim=0)

        return collated_batch


class FragmentLibrary(Dataset):
    FRAGMENT_LIBRARY_CSV = "fragment_library.csv"
    FRAG_FEATURES = "frag_features.pkl"
    FRAG_BRICS_MASKINGS = "frag_brics_maskings.pkl"

    def __init__(
        self,
        frag_lib: pd.DataFrame,
        smi_to_frag_features: Dict[SMILES, PairData],
        frag_brics_maskings: Dict[Tuple[int], List[torch.BoolTensor]] = None,
        num_cores: int = None,          # used only when parsing fragment library
    ):
        super().__init__()

        # Save the data in attributes
        self.frags_smis = frag_lib["FRAG-SMI"].to_numpy()
        self.frags_freq = torch.from_numpy(frag_lib["FRAG-FREQ"].to_numpy()).long()
        self.data_type = frag_lib["DATA-TYPE"].to_numpy()
        self.brics_type = frag_lib["BRICS-TYPE"].tolist()  # for sampling

        # FragmentLibrary for new fragments
        self.frag_features = [smi_to_frag_features[smi] for smi in self.frags_smis]
        self.frag_brics_maskings = frag_brics_maskings      # only for training

    @classmethod
    def get_insertion_frag_library(cls, data_dir: Union[str, Path], new_frag_type="all", with_maskings=False, num_cores=None):
        assert new_frag_type == "all" or new_frag_type in ["train", "val", "test"]

        # Columns: INDEX FRAG-SMI FRAG-FREQ NEW-OLD DATA-TYPE BRICS-TYPE
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        fragment_library_df = pd.read_csv(
            os.path.join(data_dir, cls.FRAGMENT_LIBRARY_CSV), sep="\t", dtype={"DATA-TYPE": str, "BRICS-TYPE": str}
        )
        fragment_library_df = fragment_library_df[
            fragment_library_df["NEW-OLD"] == "new"
        ]

        # Open the pickle file of fragment features and brics types
        frag_feature_path = os.path.join(data_dir, cls.FRAG_FEATURES)
        frag_brics_masking_path = os.path.join(data_dir, cls.FRAG_BRICS_MASKINGS)

        if not os.path.exists(frag_feature_path):
            print("Tensor files for fragment library not found.")
            print("Do not worry! This is intended due to the large size of the fragment lirary to upload.")
            print("Parsing fragment library from the existing csv file. It may take a few minutes...")

            # Parse fragment library
            PROJ_DIR = Path(__file__).parents[1]
            sys.path.append(str(PROJ_DIR))
            from data.fragment_library.parse_fragments import FragLibProcessor
            FragLibProcessor.process_frag_library(frag_lib_dir=data_dir, num_cores=num_cores)
            print("Fragment library parsing has been done.")

        with open(frag_feature_path, "rb") as fr:
            smi_to_frag_features = pickle.load(fr)  # Dict[SMILES, PairData]
        if with_maskings:
            with open(frag_brics_masking_path, "rb") as fr:
                frag_brics_maskings = pickle.load(fr)
        else:
            frag_brics_maskings = None

        if new_frag_type == "all":  # all insertion fragments
            fragment_library_df = fragment_library_df
        else:
            fragment_library_df = fragment_library_df[
                fragment_library_df["DATA-TYPE"] == new_frag_type
            ]

        return cls(fragment_library_df, smi_to_frag_features, frag_brics_maskings)

    # @classmethod
    # def get_old_frag_libraries(cls, data_dir, *modes):
    #     assert modes in [
    #         "train",
    #         "val",
    #         "test",
    #         "all",
    #     ], "mode must be one between train, val, or test."

    #     # Open the csv file of smiles and frequency
    #     smiles_freq_df = pd.read_csv(
    #         os.path.join(data_dir, cls.SMILES_FREQ_FILE), sep="\t"
    #     )

    #     # Get fragment libraries of train and val
    #     if isinstance(modes, str == str):
    #         frag_libraries = cls(smiles_freq_df, "old", modes)
    #     elif isinstance(modes, str == tuple):
    #         frag_libraries = []
    #         for mode in modes:
    #             frag_libraries.append(cls(smiles_freq_df, "old", mode))

    #     return frag_libraries

    def len(self):
        return len(self.frag_features)

    def get(self, idx):
        return self.frag_features[idx]


def debugTrainDataset(data_path):
    data_df = pd.read_csv(data_path, sep="\t")
    key_list = [len(data_df) - 1]
    dataset = TrainDataset(data_df[-2:], key_list)
    dataset.get(0)
    return


def debugInferenceDataset():
    smi = "O=S(=O)(NCCN1CCCC1)c1ccc(Br)cc1"
    data = InferenceDataset.parse_smiles(smi)
    print(data)
    print(data.allowed_subgraph)
    print(data.allowed_subgraph_idx)


if __name__ == "__main__":
    # data_path = "/home/share/DATA/mseok/FRAGMOD/230208/processed_data.csv"
    # debugDeepBioDataset(data_path)
    # debugGenerationDataset()
    dataset = InferenceDataset(smiles_list=["C1CCCCC1O"], cond_module=None)
    data = dataset.get(0)
    print(data)
