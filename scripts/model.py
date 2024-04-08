import argparse
import pathlib
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from rdkit.Chem.rdchem import Mol
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_scatter import scatter_mean, scatter_sum

from .feature import PairData, from_mol, from_smiles
from .layers import AMPN, FMPN, FeedForward, MPNNEmbedding

# from .property import PROPERTIES

SMILES = str
PROPERTY = str


class DeepBioisostere(nn.Module):
    """
    Fragment-level structural modification model to explore bioisosteres.
    You can use this model by one of the following codes:
    1. model = FragModNet(<args>)
    2. model = FragModNet.from_trained_model(<path_to_model>, <args>)
    """

    default_args = {
        "path_to_model": "Not restored",
        "mol_node_features": 49,
        "mol_edge_features": 12,
        "mol_node_hid_dim": 128,
        "mol_edge_hid_dim": 128,
        "mol_num_emb_layer": 4,
        "frag_node_features": 66,
        "frag_edge_features": 12,
        "frag_node_hid_dim": 128,
        "frag_edge_hid_dim": 128,
        "frag_num_emb_layer": 4,
        "position_score_hid_dim": 128,
        "num_mod_position_score_layer": 3,
        "frag_score_hid_dim": 128,
        "num_frag_score_layer": 3,
        "attach_score_hid_dim": 128,
        "num_attach_score_layer": 3,
        "frag_message_passing_num_layer": 2,
        "dropout": 0.0,
        "num_neg_sample": 20,
        "conditioning": True,
        "properties": [],
        "properties_dim": 1,
        "device": torch.device("cpu"),
    }

    def __init__(self, args: argparse.Namespace = None):
        # model initialize
        super().__init__()
        self._init_default_setting()
        if args:
            model_args = dict()
            for key, value in vars(args).items():
                if key in self.__dict__:
                    model_args[key] = value
            self.__dict__.update(model_args)
        if self.properties is None:
            self.properties = []

        # 1. MessagePassing Networks
        atom_embedding = MPNNEmbedding(
            node_feature_dim=self.mol_node_features,
            edge_feature_dim=self.mol_edge_features,
            node_hidden_dim=self.mol_node_hid_dim,
            edge_hidden_dim=self.mol_edge_hid_dim,
            num_layer=self.mol_num_emb_layer,
            dropout=self.dropout,
        )
        frag_embedding = MPNNEmbedding(
            node_feature_dim=self.frag_node_features,
            edge_feature_dim=self.frag_edge_features,
            node_hidden_dim=self.frag_node_hid_dim,
            edge_hidden_dim=self.frag_edge_hid_dim,
            num_layer=self.frag_num_emb_layer,
            dropout=self.dropout,
        )
        frag_message_passing = MPNNEmbedding(
            node_feature_dim=self.mol_node_hid_dim
            + self.frag_node_hid_dim
            + self.properties_dim * len(self.properties),
            edge_feature_dim=self.mol_node_hid_dim * 2,
            node_hidden_dim=self.mol_node_hid_dim,
            edge_hidden_dim=self.mol_node_hid_dim,
            num_layer=self.frag_message_passing_num_layer,
            dropout=self.dropout,
        )

        self.ampn = AMPN(self.mol_node_features, atom_embedding, frag_embedding)
        self.fmpn = FMPN(frag_message_passing)
        self.frag_embedding = frag_embedding

        # 2. Determine Leaving Fragment
        self.position_scoring_model = FeedForward(
            in_dim=self.mol_node_hid_dim,
            out_dim=1,
            hidden_dims=[
                self.position_score_hid_dim
                for _ in range(self.num_mod_position_score_layer)
            ],
            dropout=self.dropout,
        )

        # 3. Determine Inserting Fragment
        self.frag_scoring_model = FeedForward(
            in_dim=self.mol_node_hid_dim + self.frag_node_hid_dim,
            out_dim=1,
            hidden_dims=[
                self.frag_score_hid_dim for _ in range(self.num_frag_score_layer)
            ],
            dropout=self.dropout,
        )

        # 4. Attachment Prediction
        self.attach_scoring_model = FeedForward(
            in_dim=self.mol_node_hid_dim * 2 + self.frag_node_hid_dim,
            out_dim=1,
            hidden_dims=[
                self.attach_score_hid_dim for _ in range(self.num_attach_score_layer)
            ],
            dropout=self.dropout,
        )

        # 5. Condition Embedding
        # self.condition_embeddings = nn.ModuleDict(
        #     {
        #         prop: nn.Linear(
        #             in_features=self.frag_node_hid_dim,
        #             out_features=self.frag_node_hid_dim,
        #         )
        #         for prop in self.properties
        #     }
        # )

        self.Sigmoid = nn.Sigmoid()
        # self.BCELoss = nn.BCELoss(reduction="mean")

        self.init_params()

    def _init_default_setting(self):
        self._PARAMS = list(self.default_args.keys())
        self.__dict__.update(self.default_args)
        return

    def init_params(self):
        for param in self.parameters():
            if param.dim() == 1:
                continue
            else:
                nn.init.xavier_normal_(param)

    def frags_embedding(self, frags_lib_data: Batch) -> Tuple[Tensor, Tensor]:
        """
        Returns:
          frags_node_emb: FloatTensor. (N, F)
          frags_graph_emb: FloatTensor. (B, F)
        """
        frags_node_emb = self.frag_embedding(
            frags_lib_data.x_n, frags_lib_data.edge_index_n, frags_lib_data.edge_attr_n
        )
        frags_graph_emb = scatter_sum(
            src=frags_node_emb, index=frags_lib_data.x_n_batch, dim=0
        )

        return frags_node_emb, frags_graph_emb

    def forward(
        # self, data: Batch, pos_frags: Batch, neg_frags: Union[Batch, torch.LongTensor]
        self,
        batch_data: Batch,
    ) -> Tuple[float, float, float, float, float, float]:
        """Main function for training.

        Args:
          data: .data.PairData object (shape is for batched data).
            x_n              (bool) : [N,F_n], for node (atom)
            edge_index1_n    (int)  : [2,E_n], for node (atom) and model1
            edge_index2_n    (int)  : [2,E_n], for node (atom) and model2
            edge_attr_n      (bool) : [E_n,F_e], for node (atom)
            x_f              (int)  : [N], node to frag indice matcher
            edge_index_f     (int)  : [2,E_f], frag-level connectivity information
            edge_attr_f      (int)  : [2,E_f], atom indice for each frag-level edges
            x_n_batch        (int)  : [N], batch index for x_n
            x_f_batch        (int)  : [F], batch index for x_f

            smiles           (str)  : [B], SMILES of original molecule
            y_frag_bool      (bool) : [F], leaving frags index
            y_fragID         (int)  : [B], Fragment ID for which will be inserted (answer frag ID)
            allowed_subgraph (str)  : [B], allowed but negative subgraphs for original molecule
          pos_frags: torch_geometric.data.Batch obejct.
          neg_frags: torch_geometric.data.Batch obejct(for train), or torch.LongTensor(for val).

        Returns:
          Four losses and two probabilities.
        """

        # Read data
        data = batch_data["data"]
        pos_frags = batch_data["pos"]
        neg_frags = batch_data["neg"]

        # 1. Embedding given molecule by AMPN and FMPN
        ampn_emb = self.ampn(data)
        # NOTE: implementation choice: condition embedding vector is added to AMPN embedding vector.
        if self.conditioning:
            cond_embeddings = []
            # for prop, embedding_layer in self.condition_embeddings.items():
            #     cond_embeddings.append(embedding_layer(batch_data[prop]))
            for prop in self.properties:
                cond_embeddings.append(batch_data[prop])
            condition_embedding = torch.cat(cond_embeddings, dim=1)  # [F, num_props]
            ampn_emb.x_f = torch.cat(
                [ampn_emb.x_f, condition_embedding], dim=1
            )  # [F, F_node+F_frag+num_props]
        mol_emb = self.fmpn(ampn_emb)

        # 2. Score leaving Posision
        pos_mod_scores, removal_subgraph_vector = self.mod_pos_scoring(
            mol_emb, data.y_pos_subgraph, data.y_pos_subgraph_idx
        )
        neg_mod_scores, _ = self.mod_pos_scoring(
            mol_emb, data.y_neg_subgraph, data.y_neg_subgraph_idx
        )

        pPosLoss = (pos_mod_scores + 1e-10).log().mean().neg()
        nPosLoss = scatter_mean(
            src=(1 - neg_mod_scores + 1e-10).log(), index=data.y_neg_subgraph_batch
        )  # datapoint-wise mean
        nPosLoss = nPosLoss.mean().neg()

        # 3. Score inserting fragments
        # for positive samples
        pos_frag_node_emb, pos_frag_graph_emb = self.frags_embedding(pos_frags)
        pos_frag_graph_emb = pos_frag_graph_emb.unsqueeze(0)  # [1,B,F]
        pos_frag_scores = self.frags_scoring(
            removal_subgraph_vector, pos_frag_graph_emb
        )  # [1,B]
        pFragsLoss = (pos_frag_scores + 1e-12).log().mean().neg()

        # for negative samples
        _, neg_frag_graph_emb = self.frags_embedding(neg_frags)  # [nB,F]
        neg_frag_graph_emb = neg_frag_graph_emb.view(  # [n,B,F]
            self.num_neg_sample, len(data.smiles), neg_frag_graph_emb.size(-1)
        )
        neg_frag_scores = self.frags_scoring(
            removal_subgraph_vector, neg_frag_graph_emb
        )  # [n,B]
        nFragsLoss = (1 - neg_frag_scores + 1e-12).log().mean().neg()

        # 4. Attachment position prediction
        attachment_scores = self.attachment_scoring(
            data, ampn_emb.x_n, pos_frag_node_emb
        )
        attachment_allowed = mol_emb.compose_allowed_bool
        attachment_not_allowed = attachment_allowed == False
        # print(f"attachment_allowed:\n{attachment_scores*attachment_allowed}")
        # print(f"attachment_not_allowed:\n{attachment_scores*attachment_not_allowed}")
        attPredLoss = (
            (
                attachment_scores * attachment_allowed
                + ((1 - attachment_scores) * attachment_not_allowed)
                + 1e-12
            )
            .log()
            .mean()
            .neg()
        )

        return (
            pPosLoss,
            nPosLoss,
            pFragsLoss,
            nFragsLoss,
            attPredLoss,
            pos_mod_scores.mean(),
            neg_mod_scores.mean(),
            pos_frag_scores.mean(),
            neg_frag_scores.mean(),
        )

    def mod_pos_scoring(
        self, data: Batch, subgraph, subgraph_idx
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          y_pos_subgraph       (int) : [pos_F], positive subgraphs for original molecule *
          y_pos_subgraph_idx   (int) : [pos_F], positive subgraphs scatter indices for original molecule *
          y_neg_subgraph       (int) : [neg_F], allowed but negative subgraphs for original molecule *
          y_neg_subgraph_idx   (int) : [neg_F], allowed but negative subgraphs scatter indices for original molecule *
          y_neg_subgraph_batch (int) : [neg_F], batch_indices
          Ex)
            y_neg_subgraph       = tensor([ 0,  1,  2,  0,  1,  4,  5,  6,  4,  5, 16, 17, 18, 16, 17])
            y_neg_subgraph_idx   = tensor([ 0,  1,  2,  3,  3,  4,  5,  6,  7,  7,  8,  9, 10, 11, 11])
            y_neg_subgraph_batch = tensor([ 0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2])

        Returns:
          pos_frag_level_mod_score : [B]
          neg_frag_level_mod_score : [?], number of neg frag for each molecule is not necessarily same
          pos_subgraph_vector : [B,F_n]

        """

        subgraph_vector = scatter_sum(
            src=data.x_f[subgraph], index=subgraph_idx, dim=0
        )  # [B, F_n]
        frag_level_mod_score = self.position_scoring_model(subgraph_vector).squeeze(
            -1
        )  # [B]
        frag_level_mod_prob = self.Sigmoid(frag_level_mod_score)

        return frag_level_mod_prob, subgraph_vector

    def frags_scoring(self, removal_subgraph_vector: Tensor, frag_h: Tensor) -> Tensor:
        """
        Args:
          removal_subgraph_vector:
            frag-level hidden vector embedded by AMPN and FMPN.
            shape: [B,F_n]. B: num_batches
          frag_h:
            fragment hidden vector embedded by self.frag_embedding.
            shape: [n,B,F_f]. n: num_samples, B: num_batches
              For positive samples, n = 1
              For negative samples, n = n_sample
        Returns:
          graph_wise_frags_score: [n,B]
        """

        n_sample = frag_h.size(0)
        removal_subgraph_vector = removal_subgraph_vector.unsqueeze(0).repeat(
            n_sample, 1, 1
        )  # [n,B,F_n]
        concat_query = torch.cat(
            [removal_subgraph_vector, frag_h], dim=2
        )  # [n,B,F_n+F_f]
        graph_wise_frags_score = self.frag_scoring_model(concat_query).squeeze(-1)
        graph_wise_frags_score = self.Sigmoid(graph_wise_frags_score)  # [n,B]

        return graph_wise_frags_score

    def attachment_scoring(
        self, data: PairData, mol_node_emb: Tensor, frag_node_emb: Tensor
    ) -> Tensor:
        """
        Args:
          data:
            compose_original_idx, compose_fragment_idx, compose_allowed_bool
            shape: [B]. B: num_batches
          mol_node_emb:
            embedded molecule by AMPN.
            shape: [N_n, 2*F_n]. N_n: num atoms in original molecule
          frag_node_emb:
            embedded fragment by frags_embedding.
            shape: [N_f, F_f]. N_f: num atoms in new fragment

        Returns:
          attachment_scores: [n,B]
        """

        atom_emb_in_original = mol_node_emb[data.compose_original_idx]
        atom_emb_in_new_frag = frag_node_emb[data.compose_fragment_idx]

        concat_query = torch.cat([atom_emb_in_original, atom_emb_in_new_frag], dim=1)
        attachment_scores = self.attach_scoring_model(concat_query).squeeze(-1)
        attachment_scores = self.Sigmoid(attachment_scores)  # [num_combinations]

        return attachment_scores

    @staticmethod
    def parse_smi(smi: str, type="Mol") -> Data:
        return from_smiles(smi, type=type)

    @staticmethod
    def parse_mol(mol: Mol, type="Mol") -> Data:
        return from_mol(mol, type=type)

    @classmethod
    def from_trained_model(
        cls,
        path_to_model: Union[str, pathlib.Path],
        properties: List[PROPERTY] = [],
        **kwargs,
    ):
        if len(properties) > 0:
            properties = [prop.lower() for prop in properties]

        model_args = argparse.ArgumentParser()
        if kwargs:
            for k, v in kwargs.items():
                vars(model_args)[k] = v

        vars(model_args)["properties"] = properties
        model = cls(model_args)
        model.restore(path_to_model)
        return model

    @staticmethod
    def save_model(model_state_dict, save_dir, name="Best_model"):
        torch.save(model_state_dict, f"{save_dir}/{name}.pt")

    def restore(self, path_to_model: Union[str, pathlib.Path]):
        path_to_model = str(path_to_model)
        if self.device == torch.device("cpu"):
            model_params = torch.load(path_to_model, map_location=torch.device("cpu"))
        else:
            model_params = torch.load(path_to_model)
        self.load_state_dict(model_params)
        self.path_to_model = path_to_model

    def cuda(self, device: str = "cuda:0"):
        super().cuda(device)
        self.device = self.frag_scoring_model.fcs[0].weight.device
        return self

    def to(self, torch_device):
        super().to(torch_device)
        self.device = self.frag_scoring_model.fcs[0].weight.device
        return self

    def __repr__(self):
        model_repr = f"{self.__class__.__name__}(\n"
        for arg, value in self.__dict__.items():
            if arg in self._PARAMS:
                model_repr += f"  {arg}: {value}\n"
        model_repr += ")\n"
        model_repr += str(self._modules)
        return model_repr
