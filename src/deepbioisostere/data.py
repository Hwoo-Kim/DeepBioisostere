from typing import Dict, List

import torch
from torch import Tensor
from torch_geometric.data import Data


class PairData(Data):
    """
    PairData for both AMPN and FMPN.
    Notations:
      N: number of nodes (atoms).
      F: number of fragments.
      E_n: number of node edges.
      E_f: number of frag edges.
      F_n: dim of node feature vector.
      F_e: dim of edge feature vector.
      E_b: number change bonds.

    Attributes (shape is for batched data):
      x_n                   (float) : [N,F_n], for node (atom) %
      x_n_mask              (bool)  : [N], whether the node is atom or dummy %
      edge_index_n          (int)   : [2,E_n], for node (atom) and model %
      edge_index_n_w_dummy  (int)   : [2,E_n_dummy], for node (atom) and model %
      edge_attr_n           (float) : [E_n,F_e], for node (atom) %
      edge_attr_n_w_dummy   (float) : [E_n_dummy,F_e], for node (atom) %
      x_f                   (int)   : [N], node to frag indice matcher, which will be changed into frag emb vectors *
      edge_index_f          (int)   : [2,E_f], frag-level connectivity information *
      edge_attr_f           (int)   : [E_f,2], atom indice for each frag-level edges *
      x_n_batch             (int)   : [N], batch index for x_n
      x_f_batch             (int)   : [F], batch index for x_f
      compose_original_idx  (int)   : [E_b*E_b], original atom index for composition position prediction *
      compose_fragment_idx  (int)   : [E_b*E_b], fragment atom index for composition position prediction *
      compose_allowed_bool  (bool)  : [E_b*E_b], whether the indice pair is allowed or not *

      num_atoms             (int)   : [B], Number of atoms in the original molecule %
      new_frag_num_atoms    (int)   : [B], Number of atoms in the new fragment %
      smiles                (str)   : [B], SMILES of original molecule %
      brics_bonds           (str)   : [B], atom idx pairs for each BRICS bond.
      y_fragID              (int)   : [B], Fragment ID for which will be inserted (answer frag ID) *
      y_pos_subgraph        (bool)  : [pos_F], positive subgraphs for original molecule *
      y_pos_subgraph_idx    (bool)  : [pos_F], positive subgraphs scatter indices for original molecule *
      y_neg_subgraph        (bool)  : [neg_F], allowed but negative subgraphs for original molecule *
      y_neg_subgraph_idx    (bool)  : [neg_F], allowed but negative subgraphs scatter indices for original molecule *
      y_neg_subgraph_batch  (bool)  : [<B], batch index for y_neg_subgraph *

      *Below are for generation*
      adj_dummy_inform      (list)  : [num_dummy], list of tuple
      atom_frag_indice      (list)  : [N], node to frag indice matcher *
      frag_atom_indice      (dict)  : [F], frag to node indice matcher *
      allowed_subgraph      (bool)  : [all_F], allowed subgraphs for original molecule *
      allowed_subgraph_idx  (bool)  : [all_F], allowed subgraphs scatter indices for original molecule *
      num_allowed_subgraph  (int)   : [B], number of allowed subgraphs for original molecule *
    %: obtained from .feature.from_smiles function
    *: obtained from data preprocessing
    """

    def __init__(
        self,
        x_n: Tensor = None,
        x_n_mask: Tensor = None,
        edge_index_n: Tensor = None,
        edge_index_n_w_dummy: Tensor = None,
        edge_attr_n: Tensor = None,
        edge_attr_n_w_dummy: Tensor = None,
        x_f: Tensor = None,
        edge_index_f: Tensor = None,
        edge_attr_f: Tensor = None,
        compose_original_idx: Tensor = None,
        compose_fragment_idx: Tensor = None,
        compose_allowed_bool: Tensor = None,
        num_atoms: int = None,
        new_frag_num_atoms: int = None,
        smiles: str = None,
        brics_bond_indices: list = None,
        brics_bond_types: list = None,
        y_fragID: int = None,
        y_pos_subgraph: Tensor = None,
        y_pos_subgraph_idx: Tensor = None,
        y_neg_subgraph: Tensor = None,
        y_neg_subgraph_idx: Tensor = None,
        y_neg_subgraph_batch: Tensor = None,
        adj_dummy_inform: List = None,
        atom_frag_indice: List = None,
        frag_atom_indice: Dict = None,
        allowed_subgraph: Tensor = None,
        allowed_subgraph_idx: Tensor = None,
        num_allowed_subgraph: int = None,
    ):
        super().__init__()
        self.x_n = x_n
        self.x_n_mask = x_n_mask
        self.edge_index_n = edge_index_n
        self.edge_index_n_w_dummy = edge_index_n_w_dummy
        self.edge_attr_n = edge_attr_n
        self.edge_attr_n_w_dummy = edge_attr_n_w_dummy
        self.x_f = x_f
        self.edge_index_f = edge_index_f
        self.edge_attr_f = edge_attr_f
        self.compose_original_idx = compose_original_idx
        self.compose_fragment_idx = compose_fragment_idx
        self.compose_allowed_bool = compose_allowed_bool
        self.num_atoms = num_atoms
        self.new_frag_num_atoms = new_frag_num_atoms
        self.smiles = smiles
        self.brics_bond_indices = brics_bond_indices
        self.brics_bond_types = brics_bond_types
        self.y_fragID = y_fragID
        self.y_pos_subgraph = y_pos_subgraph
        self.y_pos_subgraph_idx = y_pos_subgraph_idx
        self.y_neg_subgraph = y_neg_subgraph
        self.y_neg_subgraph_idx = y_neg_subgraph_idx
        self.y_neg_subgraph_batch = y_neg_subgraph_batch
        self.adj_dummy_inform = adj_dummy_inform
        self.atom_frag_indice = atom_frag_indice
        self.frag_atom_indice = frag_atom_indice
        self.allowed_subgraph = allowed_subgraph
        self.allowed_subgraph_idx = allowed_subgraph_idx
        self.num_allowed_subgraph = num_allowed_subgraph

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_n":
            return self.x_n.size(0)  # increase by num atoms (w/ dummy)
        if key == "edge_index_n_w_dummy":
            return self.x_n.size(0)  # increase by num atoms (w/ dummy)
        if key == "edge_index_f":
            return self.x_f.max() + 1  # increase by num fragments
        if key == "edge_attr_f":
            return self.x_n.size(0)  # increase by num atoms (w/ dummy)
        if key == "x_f":
            return self.x_f.max() + 1  # increase by num atoms

        if key == "compose_original_idx":
            return self.x_n.size(0)  # increase by num atoms (w/ dummy)
        if key == "compose_fragment_idx":
            return self.new_frag_num_atoms  # increase by num atoms in new_frag

        if key == "y_pos_subgraph":
            return self.x_f.max() + 1  # increase by num atoms
        if key == "y_pos_subgraph_idx":
            return self.y_pos_subgraph_idx.max() + 1
        if key == "y_neg_subgraph":
            return self.x_f.max() + 1
        if key == "y_neg_subgraph_idx":
            if self.y_neg_subgraph_idx.size(0) != 0:
                return self.y_neg_subgraph_idx.max() + 1
            else:
                return torch.tensor(0)
        if key == "y_neg_subgraph_batch":
            if self.y_neg_subgraph_batch.size(0) != 0:
                return self.y_neg_subgraph_batch.max() + 1
            else:
                return torch.tensor(0)

        if key == "allowed_subgraph":
            return self.x_f.max() + 1
        if key == "allowed_subgraph_idx":
            if self.allowed_subgraph_idx.size(0) != 0:
                return self.allowed_subgraph_idx.max() + 1
            else:
                return torch.tensor(0)
        if key == "allowed_subgraph_batch":
            if self.allowed_subgraph_batch.size(0) != 0:
                return self.allowed_subgraph_batch.max() + 1
            else:
                return torch.tensor(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


if __name__ == "__main__":
    from torch_geometric.loader import DataLoader

    edge_index_n = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
        ]
    )
    edge_attr_n = torch.tensor(
        [
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [2, 1, 3, 4],
            [3, 1, 2, 1],
            [1, 0, 2, 1],
            [0, 0, 1, 0],
        ]
    )
    x_n = torch.randn(7, 16)  # 7 nodes.

    edge_index_f = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 3],
        ]
    )
    edge_attr_f = torch.tensor([2, 4, 5])
    x_f = torch.tensor([0, 0, 0, 1, 1, 2, 3])  # 4 frags.

    pos_subgraph = torch.tensor([1, 2])
    pos_subgraph_idx = torch.tensor([0, 0])

    neg_subgraph = torch.tensor([0, 1, 2, 0, 1])
    neg_subgraph_idx = torch.tensor([0, 1, 2, 3, 3])
    neg_subgraph_batch = torch.zeros(neg_subgraph_idx.max() + 1)

    data = PairData(
        x_n,
        edge_index_n,
        edge_attr_n,
        x_f,
        edge_index_f,
        edge_attr_f,
        y_pos_subgraph=pos_subgraph,
        y_pos_subgraph_idx=pos_subgraph_idx,
        y_neg_subgraph=neg_subgraph,
        y_neg_subgraph_idx=neg_subgraph_idx,
        y_neg_subgraph_batch=neg_subgraph_batch,
    )
    data2 = PairData(
        x_n,
        edge_index_n,
        edge_attr_n,
        x_f,
        edge_index_f,
        edge_attr_f,
        y_pos_subgraph=pos_subgraph,
        y_pos_subgraph_idx=pos_subgraph_idx,
        y_neg_subgraph=torch.tensor([]),
        y_neg_subgraph_idx=torch.tensor([]),
        y_neg_subgraph_batch=torch.tensor([]),
    )
    datas = [data, data, data]
    datas = [data, data, data2, data2, data, data]

    data_loader = DataLoader(datas, batch_size=6, follow_batch=["x_n"])
    data_batch = next(iter(data_loader))
    print(data_batch.y_pos_subgraph)
    print(data_batch.y_pos_subgraph_idx)
    print(data_batch.y_neg_subgraph)
    print(data_batch.y_neg_subgraph_idx)
    print(data_batch.y_neg_subgraph_batch)

    print(data_batch)
    print(data_batch.edge_index_n)
    print(data_batch.edge_index_f)
    print(data_batch.x_f)
    print(data_batch.x_n_batch)
