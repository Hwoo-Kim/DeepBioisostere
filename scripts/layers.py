import torch
import torch.nn as nn

# from rdkit import Chem
# from rdkit.Chem import rdmolops
from torch import FloatTensor, LongTensor, Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_sum

from .data import PairData


class MPNNLayer(MessagePassing):
    """
    MPNN layer without embedding.
    """

    def __init__(self, node_hidden_dim, edge_hidden_dim):
        super().__init__(aggr="add")
        self.bias = nn.Parameter(torch.Tensor(node_hidden_dim))
        self.message_layer = nn.Linear(
            in_features=node_hidden_dim * 2 + edge_hidden_dim,
            out_features=node_hidden_dim,
        )
        self.gru_cell = nn.GRUCell(
            input_size=node_hidden_dim, hidden_size=node_hidden_dim
        )
        self.act = nn.SiLU()

    def forward(self, x: FloatTensor, edge_index: LongTensor, edge_emb: FloatTensor):
        """
        x: [N, F], already embedded.
        edge_index: [2, E]
        edge_feature: [E, F], already embedded.
        """

        out = self.propagate(edge_index, x=x, edge=edge_emb)
        # out += self.bias
        return out

    def message(self, x_i, x_j, edge):
        # all are in shape of [E,F]
        # x_i: target, x_j: source
        message = self.message_layer(torch.concat([x_j, edge, x_i], dim=-1))
        return message

    def update(self, aggregated, x) -> Tensor:
        new_x = self.gru_cell(aggregated, x)
        # new_x = self.gru_cell(x, aggregated)

        # new_x = self.act(new_x)
        return new_x


class MPNNEmbedding(nn.Module):
    """
    Embedding layer for FragModNet model.
    Will be used to embed given molecular graphs and fragments library.
    Args:
      node_feature_dim
      edge_feature_dim
      node_hidden_dim
      edge_hidden_dim
      num_layer
      dropout

    """

    def __init__(
        self,
        node_feature_dim,
        edge_feature_dim,
        node_hidden_dim,
        edge_hidden_dim,
        num_layer,
        dropout,
    ):
        super().__init__()
        self.num_layers = num_layer

        self.node_embedding = nn.Linear(node_feature_dim, node_hidden_dim, bias=False)
        self.edge_embedding = nn.Linear(edge_feature_dim, edge_hidden_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)

        self.mpnn_layers = nn.ModuleList(
            [
                MPNNLayer(
                    node_hidden_dim=node_hidden_dim, edge_hidden_dim=edge_hidden_dim
                )
                for _ in range(self.num_layers)
            ]
        )

        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr) -> Tensor:
        x, edge_index, edge_attr = x.float(), edge_index.long(), edge_attr.float()

        x_emb = self.node_embedding(x)
        edge_emb = self.edge_embedding(edge_attr)

        x_hid = x_emb
        for layer in self.mpnn_layers:
            x_hid = layer(x_hid, edge_index, edge_emb)
            x_hid = self.dropout(x_hid)

        return x_hid


class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, dropout):
        super().__init__()
        self.num_hidden = len(hidden_dims)
        self.fcs = nn.ModuleList()
        for i in range(self.num_hidden + 1):
            if i == 0:
                self.fcs.append(nn.Linear(in_dim, hidden_dims[i]))
            elif i == self.num_hidden:
                self.fcs.append(nn.Linear(hidden_dims[-1], out_dim))
            else:
                self.fcs.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for layer in self.fcs[:-1]:
            x = self.act(layer(x))
            x = self.dropout(x)
        x = self.fcs[-1](x)  # prediction layer
        return x


class AMPN(nn.Module):
    """
    Atom MessagePassing Network.
    """

    def __init__(
        self,
        mol_node_features: int,
        atom_embedding: MPNNEmbedding,
        frag_embedding: MPNNEmbedding,
    ):
        super().__init__()
        self.mol_node_features = mol_node_features
        self.atom_embedding = atom_embedding
        self.frag_embedding = frag_embedding

    def forward(self, data: PairData):
        """
        Args:
          x_n                  (bool) : [N,F_n], for node (atom)
          x_f                  (int)  : [N], node to frag indice matcher
          edge_index_n         (int)  : [2,E_n], for node (atom) and model
          edge_index_n_w_dummy (int)  : [2,E_n_dummy], for node (atom) and model
          edge_attr_n          (bool) : [E_n,F_e], for node (atom)
          edge_attr_n_w_dummy  (bool) : [E_n_dummy,F_e], for node (atom)
          edge_attr_f          (int)  : [E_f], edge indices for each frag-level edge
          edge_attr_f          (int)  : [E_f], edge indices for each frag-level edge
        Returns:
          x_f                  (bool) : [F,F_n], for fragment
          edge_attr_f          (int)  : [E_f,F_n], for fragment
        """

        x_n_emb = self.atom_embedding(
            data.x_n[:, : self.mol_node_features], data.edge_index_n, data.edge_attr_n
        )
        x_f_emb = self.frag_embedding(
            data.x_n, data.edge_index_n_w_dummy, data.edge_attr_n_w_dummy
        )

        return self.node_to_frag_level(data, x_n_emb, x_f_emb)

    def node_to_frag_level(self, data: PairData, x_n_emb, x_f_emb):
        """
        Getting fragment-level embeddings from node featrues.

        Args:
          x_n_emb (bool):
            [N,F_n]
          x_f_emb (int):
            [N]
          data.x_n_mask (bool):
            (bool) [N]
          data.edge_attr_f:
            (int)  : [E_f,2], atom indices for each frag-level edges
        Returns:
          x_n          (bool) : [N,2*F_n], for node
          x_f          (bool) : [F,2*F_n], for fragment
          edge_attr_f  (int)  : [E_f,F_n], for fragment
        """

        h_f = scatter_sum(src=x_n_emb, index=data.x_f, dim=0)
        s_f = scatter_sum(src=x_f_emb, index=data.x_f, dim=0)

        data.x_n = torch.cat([x_n_emb, x_f_emb], dim=1).float()
        data.x_f = torch.cat([h_f, s_f], dim=1)

        data.edge_attr_f = data.x_n[data.edge_attr_f.t().contiguous()].sum(
            dim=0
        )  # [2,E_f] -> [2,E_f,F_n] -> [E_f,F_n]
        # x_n_emb = torch.mul(x_n_emb, data.x_n_mask.unsqueeze(-1)) # TODO: 마스킹 하는게 맞나?

        return data


class FMPN(nn.Module):
    """
    Fragment MessagePassing Network.
    """

    def __init__(self, frag_message_passing: MPNNEmbedding):
        super().__init__()
        self.frag_message_passing = frag_message_passing

    def forward(self, data: PairData):
        """
        Args:
          x_f          (bool) : [F,2*F_n], for fragment
          edge_index_f (int)  : [2,E_f],   for fragment
          edge_attr_f  (int)  : [E_f,F_n], for fragment
        Returns:
        """
        data.x_f = self.frag_message_passing(
            data.x_f, data.edge_index_f, data.edge_attr_f
        )
        return data


if __name__ == "__main__":
    edge_index = torch.tensor([[0, 1], [1, 0], [1, 2]], dtype=torch.long)
    x = torch.tensor([[1, 2, -3], [2, 1, -1], [1, 2, 0]], dtype=torch.float)
    edge_attr = torch.tensor([[2, 0, 1], [1, 2, 1], [-1, 2, 1]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)

    mpnn_emb = MPNNLayer(3, 3)

    print(mpnn_emb(data.x, data.edge_index, data.edge_attr))
