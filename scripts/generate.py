from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from rdkit import Chem
from torch import Tensor
from torch.multiprocessing import Lock
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from brics.brics import BRICSModule, BRICSTypeMapper
from conditioning import Conditioner
from dataset import FragmentLibrary, InferenceCollator, InferenceDataset
from model import DeepBioisostere
from property import PROPERTIES, calc_logP, calc_Mw, calc_QED, calc_SAscore

mp.set_sharing_strategy("file_system")


SMILES = str
PROPERTY = str
GEN_COLUMNS = [
    "INPUT-MOL-IDX",
    "INPUT-MOL-SMI",
    "GEN-MOL-SMI",
    # "LEAVING-ATOM-INDICES",
    "LEAVING-FRAG-SMI",
    "INSERTING-FRAG-SMI",
    "PREDICTED-PROB",
    "LOGP",
    "MW",
    "QED",
    "SA",
]

LEAV_DF_COLUMNS = [
    "INPUT-MOL-IDX",
    "INPUT-MOL-SMI",
    "LEAV-ATOM-INDICES",
    "LEAV-FRAG-SMI",
    "PREDICTED-PROB",
]


class Generator:
    r"""
    Molecule generator with fragment sampling.
    """

    # LEAVINGFRAG_CRITERION = 0.0
    # INSERTINGFRAG_CRITERION = 0.0

    GENERATION_RESULT_PATH = "results.csv"
    LEAVING_FRAG_RESULT_PATH = "leaving_frag_results.csv"

    BRICS_TYPE_MAPPER = BRICSTypeMapper(inp_type=int)

    def __init__(
        self,
        model: DeepBioisostere,
        processed_frag_dir: Union[str, Path],
        num_sample_each_mol: Union[int, str],
        device: torch.device,
        num_cores: int,
        batch_size: int,
        new_frag_type: str,
        conditioner: Union[Conditioner, None] = None,
        properties: List[PROPERTY] = None,
        logger=None,
    ):
        assert (
            isinstance(num_sample_each_mol, int) or num_sample_each_mol == "all"
        ), "num_sample_each_mol should be 'all' or an integer."

        # super().__init__()
        self.model = model
        self.num_sample_each_mol = num_sample_each_mol
        self.conditioner = conditioner
        self.properties = sorted(properties) if properties else None
        self.device = device
        self.num_cores = num_cores
        self.batch_size = batch_size
        self.model = self.model.to(device)
        if logger:
            self.logger = logger
        else:
            self.logger = print

        # Fragment library dataset
        self.logger("Loading the fragment library...")
        frags_lib_dataset = FragmentLibrary.get_insertion_frag_library(
            processed_frag_dir, new_frag_type, with_maskings=False, num_cores=num_cores
        )
        self.frag_lib_size = len(frags_lib_dataset)
        self.frags_smis = frags_lib_dataset.frags_smis
        self.brics_type = frags_lib_dataset.brics_type

        self.brics_type_to_insertion_frags = dict()
        brics_types, positions = np.unique(self.brics_type, return_inverse=True)
        for i, brics_type in enumerate(brics_types):
            self.brics_type_to_insertion_frags[brics_type] = np.where(positions == i)[0]

        # Embed fragment library into frag_node_emb, frag_graph_emb
        self.frag_lib_dl = DataLoader(
            dataset=frags_lib_dataset,
            batch_size=batch_size,
            num_workers=num_cores,
            follow_batch=["x_n"],
        )
        self.save_frags_embeddings()
        del frags_lib_dataset

        self.logger("Generator initialization finished.")

    @torch.no_grad()
    def generate(
        self, input_list: List[Tuple[SMILES, Dict[str, float]]]
    ) -> pd.DataFrame:
        """
        Generate modified molecules for a given SMILES list.
        TODO

        Parameters
        ----------
        About batch_data.
          x_n                (bool) : [N,F_n], for node (atom)
          edge_index_n       (int)  : [2,E_n], for node (atom) and model
          edge_attr_n        (bool) : [E_n,F_e], for node (atom)
          x_f                (int)  : [N], node to frag indice matcher
          edge_index_f       (int)  : [2,E_f], frag-level connectivity information
          edge_attr_f        (int)  : [2,E_f], atom indice for each frag-level edges

          smiles             (str)  : [B], SMILES of original molecule
        TODO:
          About batch_data.
          x (bool): [N,F_n]
          edge_index (int): [2,E]
          edge_attr (bool): [E,F_e]
          smiles (str): [B]
          batch (int): [N]

          y_fragID           (int)  : [B], Fragment ID for which will be inserted (answer frag ID)
          y_pos_subgraph     (bool) : [pos_F], positive subgraphs for original molecule
          y_pos_subgraph_idx (bool) : [pos_F], positive subgraphs scatter indices for original molecule
          y_neg_subgraph     (bool) : [neg_F], allowed but negative subgraphs for original molecule
          y_neg_subgraph_idx (bool) : [neg_F], allowed but negative subgraphs scatter indices for original molecule

        """
        # Read input_list
        if self.conditioner:
            smiles_list, prop_dict_list = zip(*input_list)
            conditioning = True
        else:
            smiles_list = input_list
            prop_dict_list = None
            conditioning = False

        # initialize generate dataset
        dataset = InferenceDataset(
            smiles_list=smiles_list,
            conditioner=self.conditioner,
            prop_dict_list=prop_dict_list,
        )

        # initialize dataloader
        data_dl = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_cores
        )
        data_dl.collate_fn = InferenceCollator(
            use_conditioning=conditioning,
            properties=self.properties,
            follow_batch=["x_n", "allowed_subgraph"],
        )

        # Main Generation Part
        batch_result = []
        for batch_idx, batch in enumerate(data_dl):
            data = batch["data"].to(self.device)
            if self.conditioner:
                for prop in self.properties:
                    batch[prop] = batch[prop].to(self.device)

            # 1. Embedding data
            ampn_emb = self.model.ampn(data)
            # NOTE: implementation choice: condition embedding vector is added to AMPN embedding vector.
            if self.conditioner:
                cond_embeddings = []
                for prop in self.properties:
                    cond_embeddings.append(batch[prop])
                condition_embedding = torch.cat(
                    cond_embeddings, dim=1
                )  # [F, num_props]
                ampn_emb.x_f = torch.cat(
                    [ampn_emb.x_f, condition_embedding], dim=1
                )  # [F, F_node+F_frag+num_props]
            mol_emb = self.model.fmpn(ampn_emb)

            # 2. Score modiciation position
            # leaving_subgraph_probs: list of Tensor[subgraph]
            # subgraph_embed_vector: Tensor[∑subgraph, n_F]
            prediction_result = self.score_modification_position(mol_emb, data)
            leaving_subgraph_probs, subgraph_embed_vector = prediction_result

            # 3. Score fragment for the selected position
            # inserting_frag_probs: list of Tensor[subgraph, frag_lib]
            inserting_frag_probs = self.score_fragment_for_position(
                subgraph_embed_vector, data
            )

            # 4. Select from joint probability
            sampling_result_list = self.select_from_joint_prob(
                leaving_subgraph_probs, inserting_frag_probs
            )

            # 5. select attachment orientation for the selected position & fragment
            # NOTE: Most time-consuming part.
            model_inference_results = self.select_attachment_orientation(
                sampling_result_list, ampn_emb, data
            )

            # 6. merge the fragment to the molecule
            result_df = self.merge_fragment(
                model_inference_results, sampling_result_list, batch_idx
            )
            batch_result.append(result_df)
            continue

        # Merge and return
        result_df = pd.concat(batch_result, axis=0, ignore_index=True)
        return result_df

    @torch.no_grad()
    def generate_with_leaving_frag(
        self, input_list: List[Tuple[SMILES, SMILES, int, Dict[str, float]]]
    ) -> pd.DataFrame:
        """
        Generate modified moleucles for a given SMILES list by modifying specific fragments.

        Args:
            input_list: List of modification information, (original_molecule, leaving_frag, atom_idx).
              original_molecule: SMILES of original molecule
              leaving_frag: SMILES of fragment to be removed
              atom_idx: Index of a random atom in the leaving fragment

        Returns:
            [TODO:return]
        """
        # Read input_list
        if self.conditioner:
            smiles_list, leaving_frag_list, atom_idx_list, prop_dict_list = zip(
                *input_list
            )
            conditioning = True
        else:
            smiles_list, leaving_frag_list, atom_idx_list = zip(*input_list)
            prop_dict_list = None
            conditioning = False

        # initialize generate dataset
        dataset = InferenceDataset(
            smiles_list=smiles_list,
            conditioner=self.conditioner,
            prop_dict_list=prop_dict_list,
        )
        dataset.set_leaving_frag_inform(leaving_frag_list, atom_idx_list)

        # initialize dataloader
        data_dl = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_cores
        )
        data_dl.collate_fn = InferenceCollator(
            use_conditioning=conditioning,
            properties=self.properties,
            follow_batch=["x_n", "allowed_subgraph"],
        )

        # Main Generation Part
        batch_result = []
        for batch_idx, batch in enumerate(data_dl):
            data = batch["data"].to(self.device)
            if self.conditioner:
                for prop in self.properties:
                    batch[prop] = batch[prop].to(self.device)

            # 1. Embedding data
            ampn_emb = self.model.ampn(data)
            # NOTE: implementation choice: condition embedding vector is added to AMPN embedding vector.
            if self.conditioner:
                cond_embeddings = []
                for prop in self.properties:
                    cond_embeddings.append(batch[prop])
                condition_embedding = torch.cat(
                    cond_embeddings, dim=1
                )  # [F, num_props]
                ampn_emb.x_f = torch.cat(
                    [ampn_emb.x_f, condition_embedding], dim=1
                )  # [F, F_node+F_frag+num_props]
            mol_emb = self.model.fmpn(ampn_emb)

            # 2. Score modiciation position
            # leaving_subgraph_probs: list of Tensor[subgraph]
            # subgraph_embed_vector: Tensor[∑subgraph, n_F]
            prediction_result = self.score_modification_position(mol_emb, data)
            leaving_subgraph_probs, subgraph_embed_vector = prediction_result

            # 3. Score fragment for the selected position
            # inserting_frag_probs: list of Tensor[subgraph, frag_lib]
            inserting_frag_probs = self.score_fragment_for_position(
                subgraph_embed_vector, data
            )

            # 4. Select from joint probability
            sampling_result_list = self.select_from_joint_prob(
                leaving_subgraph_probs, inserting_frag_probs
            )

            # 5. select attachment orientation for the selected position & fragment
            # NOTE: Most time-consuming part.
            model_inference_results = self.select_attachment_orientation(
                sampling_result_list, ampn_emb, data
            )

            # 6. merge the fragment to the molecule
            result_df = self.merge_fragment(
                model_inference_results, sampling_result_list, batch_idx
            )
            batch_result.append(result_df)
            continue

        # reset leaving fragment information
        dataset.reset_leaving_frag_inform()

        # Merge and return
        result_df = pd.concat(batch_result, axis=0, ignore_index=True)
        return result_df

    # TODO: removal fragment selection 용 method 만들기.
    @torch.no_grad()
    def score_removal_fragment(
        self, input_list: List[Tuple[SMILES, Dict[str, float]]]
    ) -> pd.DataFrame:
        """
        Generate modified molecules for a given SMILES list.
        TODO

        Parameters
        ----------
        About batch_data.
          x_n                (bool) : [N,F_n], for node (atom)
          edge_index_n       (int)  : [2,E_n], for node (atom) and model
          edge_attr_n        (bool) : [E_n,F_e], for node (atom)
          x_f                (int)  : [N], node to frag indice matcher
          edge_index_f       (int)  : [2,E_f], frag-level connectivity information
          edge_attr_f        (int)  : [2,E_f], atom indice for each frag-level edges

          smiles             (str)  : [B], SMILES of original molecule
        TODO:
          About batch_data.
          x (bool): [N,F_n]
          edge_index (int): [2,E]
          edge_attr (bool): [E,F_e]
          smiles (str): [B]
          batch (int): [N]

          y_fragID           (int)  : [B], Fragment ID for which will be inserted (answer frag ID)
          y_pos_subgraph     (bool) : [pos_F], positive subgraphs for original molecule
          y_pos_subgraph_idx (bool) : [pos_F], positive subgraphs scatter indices for original molecule
          y_neg_subgraph     (bool) : [neg_F], allowed but negative subgraphs for original molecule
          y_neg_subgraph_idx (bool) : [neg_F], allowed but negative subgraphs scatter indices for original molecule

        """
        # Read input_list
        if self.conditioner:
            smiles_list, prop_dict_list = zip(*input_list)
            conditioning = True
        else:
            smiles_list = input_list
            prop_dict_list = None
            conditioning = False

        # initialize generate dataset
        dataset = InferenceDataset(
            smiles_list=smiles_list,
            conditioner=self.conditioner,
            prop_dict_list=prop_dict_list,
        )

        # initialize dataloader
        data_dl = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_cores
        )
        data_dl.collate_fn = InferenceCollator(
            use_conditioning=conditioning,
            properties=self.properties,
            follow_batch=["x_n", "allowed_subgraph"],
        )

        # Main Generation Part
        batch_result = []
        for batch_idx, batch in enumerate(data_dl):
            data = batch["data"].to(self.device)
            if self.conditioner:
                for prop in self.properties:
                    batch[prop] = batch[prop].to(self.device)

            # 1. Embedding data
            ampn_emb = self.model.ampn(data)
            # NOTE: implementation choice: condition embedding vector is added to AMPN embedding vector.
            if self.conditioner:
                cond_embeddings = []
                for prop in self.properties:
                    cond_embeddings.append(batch[prop])
                condition_embedding = torch.cat(
                    cond_embeddings, dim=1
                )  # [F, num_props]
                ampn_emb.x_f = torch.cat(
                    [ampn_emb.x_f, condition_embedding], dim=1
                )  # [F, F_node+F_frag+num_props]
            mol_emb = self.model.fmpn(ampn_emb)

            # 2. Score modiciation position
            # leaving_subgraph_probs: list of Tensor[subgraph]
            # subgraph_embed_vector: Tensor[∑subgraph, n_F]
            prediction_result = self.score_modification_position(mol_emb, data)
            leaving_subgraph_probs, subgraph_embed_vector = prediction_result

            # 3. Score fragment for the selected position
            # inserting_frag_probs: list of Tensor[subgraph, frag_lib]
            inserting_frag_probs = self.score_fragment_for_position(
                subgraph_embed_vector, data
            )

            # 4. Select from joint probability
            sampling_result_list = self.select_from_joint_prob(
                leaving_subgraph_probs, inserting_frag_probs
            )

            # 5. select attachment orientation for the selected position & fragment
            # NOTE: Most time-consuming part.
            model_inference_results = self.select_attachment_orientation(
                sampling_result_list, ampn_emb, data
            )

            # 6. merge the fragment to the molecule
            result_df = self.merge_fragment(model_inference_results, batch_idx)
            batch_result.append(result_df)
            continue

        # Merge and return
        result_df = pd.concat(batch_result, axis=0, ignore_index=True)
        return result_df

    @staticmethod
    def brics_compose(merge_plan: tuple, batch_idx: int, batch_size: int) -> List[str]:
        """
        Compose all the given information to yield a complete molecule.
        All the composing procedure follows the BRICS rule.
        This method utilzes the methods of BRICSModule; get_adjacent_fragments and enumerate_composing_mols.

        Parameters
        ----------
        composing_inform: (smiles, subgraph_idx, frag_indices, insert_frag_smi, position_frag_joint_prob)

        Returns
        -------
        Tuple[int, np.array]
        """
        # read merge_plan
        (
            data_idx,
            original_smi,
            subgraph_idx,
            change_indices,
            new_frag_smi,
            attachment,
            prob,
        ) = merge_plan
        idx = data_idx + batch_size * batch_idx

        # Generate molecule using BRICSModule
        generated_smi, leaving_frag_smi = BRICSModule.compose_mols_with_attachment(
            original_smi,
            change_indices,
            new_frag_smi,
            attachment,
            get_leaving_frag_smi=True,
        )
        generated_mol = Chem.MolFromSmiles(generated_smi)
        logp = calc_logP(generated_mol)
        mw = calc_Mw(generated_mol)
        qed = calc_QED(generated_mol)
        sa = calc_SAscore(generated_mol)

        # return the results
        res_list = [
            idx,
            original_smi,
            generated_smi,
            leaving_frag_smi,
            new_frag_smi,
            prob,
            logp,
            mw,
            qed,
            sa,
            subgraph_idx,
        ]

        return res_list

    @torch.no_grad()
    def save_frags_embeddings(self) -> None:
        """
        frag_node_emb: [N_f, F_f] on CPU.
        frag_graph_emb: [N_f, F_f] on CPU.
        frag_adj_dummy_inform:
        """
        frag_node_emb_list = []
        frag_graph_emb_list = []
        adj_dummy_inform = []  # List[List[Tuple[int,int]]]
        # frags_num_atoms = []
        frag_node_batch_list = []
        prev_max = 0

        for batch in self.frag_lib_dl:
            batch = batch.to(self.device)
            frag_node_emb, frag_graph_emb = self.model.frags_embedding(batch)
            frag_node_emb, frag_graph_emb = frag_node_emb.to("cpu"), frag_graph_emb.to(
                "cpu"
            )

            frag_node_emb_list.append(frag_node_emb)
            frag_graph_emb_list.append(frag_graph_emb)
            adj_dummy_inform += batch.adj_dummy_inform
            # frags_num_atoms += batch.num_atoms
            frag_node_batch_list.append(batch.x_n_batch.to("cpu") + prev_max)
            prev_max = frag_node_batch_list[-1].max() + 1

        frag_node_emb = torch.concat(frag_node_emb_list, dim=0)  # [N_f, F_f]
        frag_graph_emb = torch.concat(frag_graph_emb_list, dim=0)  # [num_frags, F_f]
        frag_adj_dummy_inform = np.array(adj_dummy_inform, dtype=list)
        frag_node_batch = torch.concat(frag_node_batch_list, dim=0)

        # Make dictionary which maps a fragment idx into node indices in frag_node_emb
        frag_idx_to_node_idxs = dict()
        for node_idx, frag_idx in enumerate(frag_node_batch.tolist()):
            if frag_idx in frag_idx_to_node_idxs:
                frag_idx_to_node_idxs[frag_idx].append(node_idx)
            else:
                frag_idx_to_node_idxs[frag_idx] = [node_idx]

        self.frag_node_emb = frag_node_emb
        self.frag_graph_emb = frag_graph_emb.unsqueeze(
            dim=1
        )  # [frag_lib, 1, F_n] # TODO: device?
        self.frag_adj_dummy_inform = frag_adj_dummy_inform
        self.frag_idx_to_node_idxs = frag_idx_to_node_idxs

        return

    @torch.no_grad()
    def score_modification_position(
        self, mol_emb: Tensor, batch: Batch
    ) -> Tuple[Tensor, Tensor]:
        # Scoring with DeepBio model
        subgraph_mod_score, subgraph_embed_vector = self.mod_pos_scoring(mol_emb)

        # Get leaving fragment probabilities
        leaving_subgraph_probs = []
        for data_idx in range(batch.num_graphs):
            leaving_frag_scores = subgraph_mod_score[  # cuda
                batch.allowed_subgraph_batch == data_idx
            ]  # [subgraph], self.device
            leaving_frag_probs = F.normalize(  # cuda
                leaving_frag_scores, p=1, dim=0
            )  # [subgraph], self.device
            leaving_subgraph_probs.append(leaving_frag_probs)

        return leaving_subgraph_probs, subgraph_embed_vector

    @torch.no_grad()
    def score_fragment_for_position(
        self, batch_subgraph_embed_vectors: Tensor, batch: Batch
    ) -> Tuple[Tensor, Tensor]:
        batch_inserting_frag_probs = []
        for data_idx in range(batch.num_graphs):
            # Get fragments masks based on BRICS rule
            frag_masks = []  # [num_subgraph, frag_lib_size]
            for brics_types in batch.brics_types[           # data.brics_types: [subgraph_idx], brics types of adjacent frags
                data_idx
            ]:  # brics_types: tuple of int, corresponding to each subgrpah
                frag_mask = torch.zeros(self.frag_lib_size, dtype=bool)
                allowed_insertion_types = self.BRICS_TYPE_MAPPER.getMapping(sorted(brics_types))
                for insertion_brics_type in allowed_insertion_types:
                    insertion_brics_type = ",".join(list(map(str, insertion_brics_type)))
                    if insertion_brics_type in self.brics_type_to_insertion_frags:
                        frag_mask[self.brics_type_to_insertion_frags[insertion_brics_type]] = True
                    else:
                        continue
                frag_masks.append(frag_mask)

            # Score inserting fragments with the DeepBio model
            subgraph_embed_vectors = batch_subgraph_embed_vectors[  # cuda
                batch.allowed_subgraph_batch == data_idx
            ]  # [subgraph, F_n], self.device
            inserting_frag_scores = []
            for subgraph_idx, emb_vector in enumerate(subgraph_embed_vectors):
                frag_mask = frag_masks[subgraph_idx]
                allowed_frag_graph_emb = self.frag_graph_emb[frag_mask]
                allowed_frag_graph_emb = allowed_frag_graph_emb.to(self.device)

                # NOTE: "num_allowed_frag" becomes different depending on the subgraph.
                # emb_vector: [F_n], allowed_frag_graph_emb: [num_allowed_frag, 1, F_f]
                allowed_frag_score = self.model.frags_scoring(
                    emb_vector.unsqueeze(0), allowed_frag_graph_emb
                ).squeeze(
                    dim=-1
                )  # [num_allowed_frag]

                frag_score = torch.zeros(self.frag_lib_size).to(self.device)
                frag_score.masked_scatter_(
                    frag_mask.to(self.device), allowed_frag_score
                ).to(
                    "cpu"
                )  # [frag_lib]
                inserting_frag_scores.append(frag_score)

            # Stack the results
            inserting_frag_scores = torch.stack(
                inserting_frag_scores, dim=0
            )  # [subgraph, frag_lib]
            inserting_frag_probs = F.normalize(inserting_frag_scores, p=1, dim=1)
            batch_inserting_frag_probs.append(inserting_frag_probs)

        return batch_inserting_frag_probs  # list of [subgraph, frag_lib] with batch_size)

    @torch.no_grad()
    def select_from_joint_prob(
        self, leaving_subgraph_probs, inserting_frag_probs
    ) -> List[dict]:
        """
        Main parts of DeepBioisostere are two prediction models:
            1) leaving fragment prediction model, 2) inserting fragment prediction model.
        DeepBioisostere provides the exact likelihoods of leaving fragments and inserting fragments.
        Because the numbers of choices are not too many, we can choose by top-k sampling.

        However, top-k sampling does not guarantee the sampled molecules follow the first model's prediction result.
        Thus, we explicitly adopts the first model's prediction result,
            and apply top-k sampling only on the seconed model's prediction.

        Args:
            leaving_subgraph_probs: list of Tensor[num_subgraph]. Already normalized on dim=0.
            inserting_subgraph_probs: list of Tensor[num_subgraph, frag_lib_size]. Already normalized on dim=1.
        """
        sampling_result_list = []
        for (
            leaving_prob,
            inserting_prob,
        ) in zip(  # [num_subgraph], [num_subgraph, frag_lib_size]
            leaving_subgraph_probs,
            inserting_frag_probs,
        ):
            num_allowed_subgraph = len(leaving_prob)
            # Sampling from joint probability
            each_sampling_result = dict()
            each_sampling_result["num_sample"] = {
                _: None for _ in range(leaving_prob.size(0))
            }
            each_sampling_result["inserting"] = {
                _: [] for _ in range(leaving_prob.size(0))
            }
            each_sampling_result["prob"] = {_: [] for _ in range(leaving_prob.size(0))}

            if self.num_sample_each_mol == "all":
                joint_prob = torch.mul(
                    leaving_prob.unsqueeze(dim=1), inserting_prob
                )  # [num_subgraph, frag_lib]

                for subgraph_idx, joint_probs_for_subgraph in enumerate(joint_prob):
                    inserting_frag_idxs = joint_probs_for_subgraph.nonzero().view(-1)
                    inserting_frag_probs = joint_probs_for_subgraph[inserting_frag_idxs]

                    each_sampling_result["num_sample"][
                        subgraph_idx
                    ] = inserting_frag_idxs.size(0)
                    each_sampling_result["inserting"][
                        subgraph_idx
                    ] = inserting_frag_idxs.to("cpu").tolist()
                    each_sampling_result["prob"][
                        subgraph_idx
                    ] = inserting_frag_probs.to("cpu").tolist()

            else:
                # Calculate the number of molecules to be sampled for each subgraph
                leaving_frag_sampling_result = torch.multinomial(
                    input=leaving_prob,
                    num_samples=self.num_sample_each_mol,
                    replacement=True,
                )
                unique_subgraph, subgraph_counts = torch.unique(
                    leaving_frag_sampling_result, return_counts=True
                )
                num_sample_each_subgraph = torch.zeros(
                    num_allowed_subgraph, dtype=torch.int
                )
                num_sample_each_subgraph = num_sample_each_subgraph.to(self.device)
                num_sample_each_subgraph.scatter_(
                    dim=0, index=unique_subgraph, src=subgraph_counts.int()
                )

                for subgraph_idx, inserting_probs_for_subgraph in enumerate(
                    inserting_prob
                ):
                    num_to_sample = int(num_sample_each_subgraph[subgraph_idx])
                    num_max_choices = torch.where(
                        inserting_probs_for_subgraph != 0, True, False
                    ).sum()  # < num_subgraph * frag_lib
                    each_sampling_result["num_sample"][subgraph_idx] = min(
                        int(num_max_choices), num_to_sample
                    )

                    # NOTE: Samples 100 times more than the number of molecules to be sampled,
                    # because the likelihood might be largely changed by the third model.
                    num_to_sample = min(int(num_max_choices), num_to_sample * 100)
                    sampling_result = torch.topk(
                        inserting_probs_for_subgraph, k=num_to_sample
                    )
                    each_sampling_result["inserting"][
                        subgraph_idx
                    ] = sampling_result.indices.tolist()
                    each_sampling_result["prob"][
                        subgraph_idx
                    ] = sampling_result.values.tolist()

            sampling_result_list.append(each_sampling_result)

        return sampling_result_list

    @torch.no_grad()
    def select_attachment_orientation(
        self,
        sampling_result_list: Tensor,
        ampn_emb,
        batch: Batch,
    ) -> Tuple[Tensor, Tensor]:
        """
        allowed_subgraph       = tensor([ 0,  1,  2,  0,  1,  4,  5,  6,  4,  5, 16, 17, 18, 16, 17])
        allowed_subgraph_idx   = tensor([ 0,  1,  2,  3,  3,  4,  5,  6,  7,  7,  8,  9, 10, 11, 11])
        allowed_subgraph_batch = tensor([ 0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2])
        """
        model_inference_results = []
        for data_idx in range(batch.num_graphs):
            # Read information from Batch object
            atom_frag_indice = batch.atom_frag_indice[data_idx]  # Dict[int:List[int]]
            brics_bond_indices = batch.brics_bond_indices[
                data_idx
            ]  # List[Tuple[int, int]]
            brics_bond_types = batch.brics_bond_types[data_idx]  # List[str]
            query_smi = batch.smiles[data_idx]

            # A dictionary mapping fragment index to involved atom indices
            frag_atom_indice = dict()
            for i, f_id in enumerate(list(map(int, atom_frag_indice.split(",")))):
                if f_id in frag_atom_indice:
                    frag_atom_indice[f_id].append(i)
                else:
                    frag_atom_indice[f_id] = [i]

            # Retrive allowed subgraphs for the current molecule
            retrieved_subgraph_idxs = (
                batch.allowed_subgraph_batch == data_idx
            ).nonzero()
            retrieved_subgraph_idxs = retrieved_subgraph_idxs.squeeze(-1)

            mask = torch.isin(batch.allowed_subgraph_idx, retrieved_subgraph_idxs)
            allowed_subgraph = batch.allowed_subgraph[mask]
            allowed_subgraph_idx = batch.allowed_subgraph_idx[mask]

            # Translation
            allowed_subgraph -= (
                allowed_subgraph.min() - batch.min_allowed_subgraph[data_idx]
            )  # NOTE: can start with nonzero value
            allowed_subgraph_idx -= (
                allowed_subgraph_idx.min()
            )  # NOTE: must start with zero
            num_allowed_subgraph = allowed_subgraph_idx.max() + 1

            # Get allowed fragment indices for the current molecule
            # subgraph_to_frag_prob_dict = sampling_result_list[data_idx]
            each_sampling_result = sampling_result_list[data_idx]
            subgraph_to_num_samples = each_sampling_result["num_sample"]
            subgraph_to_insertings = each_sampling_result["inserting"]
            subgraph_to_probs = each_sampling_result["prob"]

            assert num_allowed_subgraph == len(subgraph_to_num_samples)
            for subgraph_idx in range(num_allowed_subgraph):
                each_subgraph_result = []
                # Get fragment idxs
                # frag_idxs, probs = [], []
                # for pair in frag_prob_pair_list:
                #     frag_idxs.append(pair[0])
                #     probs.append(pair[1])
                num_samples = subgraph_to_num_samples[subgraph_idx]
                frag_idxs = subgraph_to_insertings[subgraph_idx]
                probs = subgraph_to_probs[subgraph_idx]

                # Get atom indices in the subgraph
                frags_in_subgraph = allowed_subgraph[
                    allowed_subgraph_idx == subgraph_idx
                ]
                frags_in_subgraph = frags_in_subgraph.tolist()
                atoms_in_subgraph = []
                for frag_idx in frags_in_subgraph:
                    break_flag = False
                    try:
                        atoms_in_subgraph += frag_atom_indice[frag_idx]
                    except KeyError as e:
                        self.logger(f"Error message: {e}")
                        self.logger(f"smiles: {batch.smiles[data_idx]}")
                        self.logger(f"allowed_subgraph: {allowed_subgraph}")
                        self.logger(f"allowed_subgraph_idx: {allowed_subgraph_idx}")
                        self.logger(f"frag_atom_indice: {frag_atom_indice}")
                        self.logger(f"frags_in_subgraph: {frags_in_subgraph }")
                        break_flag = True
                        break
                if break_flag:
                    break

                # Get atom indices which will form new bonds with inserting fragments
                attach_atom_indice, attach_brics_types = [], []
                for bond_idx, bond_indice in enumerate(brics_bond_indices):
                    bond_types = brics_bond_types[bond_idx]
                    atom_is_in_subgraph = [
                        atom_idx in atoms_in_subgraph for atom_idx in bond_indice
                    ]

                    if not all(atom_is_in_subgraph) and any(
                        atom_is_in_subgraph
                    ):  # this bond will be broken!
                        atom_not_in_subgraph = atom_is_in_subgraph.index(False)
                        attach_atom_indice.append(bond_indice[atom_not_in_subgraph])
                        attach_brics_types.append(bond_types[atom_not_in_subgraph])

                # Check whether BRICS compose is allowed or not
                allowed_adj_dummy_inform = self.frag_adj_dummy_inform[
                    frag_idxs
                ]  # each one is allowed, but their combinations should be examined.
                allowed_combinations = BRICSModule.enumerate_allowed_combinations(
                    attach_atom_indice,
                    attach_brics_types,
                    allowed_adj_dummy_inform,
                    frag_idxs,
                    logger=self.logger,
                )
                if (
                    type(allowed_combinations) is int
                ):  # Error occured in BRICSModule.enumerate_allowed_combinations
                    frag_idx = allowed_combinations
                    error_frag_smi = self.frags_smis[frag_idx]
                    self.logger(f"Error query smiles: {query_smi}")
                    self.logger(f"Error inserting fragment: {error_frag_smi}")
                    continue

                # Calculate attachment likelihood by the model
                final_likelihoods = torch.tensor([]).to(self.device)
                atom_emb_in_original = ampn_emb.x_n[(batch.x_n_batch == data_idx)]
                for sample_idx, (frag_idx, combinations) in enumerate(
                    allowed_combinations.items()
                ):
                    new_frag_smi = self.frags_smis[frag_idx]
                    new_frag_node_idxs = self.frag_idx_to_node_idxs[
                        frag_idx
                    ]  # example: [10234, 10235, ... 10240]
                    concat_query_list = []
                    for (
                        ref_atom_inform,
                        new_frag_atom_inform,
                    ) in combinations:  # new_frag_atom_inform: [2, 4, 7, 8]
                        ref_atom_emb = atom_emb_in_original[ref_atom_inform]
                        new_atom_emb = self.frag_node_emb[
                            [new_frag_node_idxs[_] for _ in new_frag_atom_inform]
                        ]
                        concat_query_list.append(
                            torch.cat(
                                [ref_atom_emb, new_atom_emb.to(self.device)], dim=1
                            )
                        )

                    # Calculate probability of each combination
                    concat_query = torch.stack(concat_query_list, dim=0)
                    attachment_scores = self.model.attach_scoring_model(
                        concat_query
                    ).squeeze(-1)
                    attachment_probs = torch.sigmoid(attachment_scores)

                    # NOTE: Each combination is scored by averaging the scores of involved attachment points.
                    combination_probs = torch.mean(
                        attachment_probs, dim=1
                    )  # [num_allowed_attachment]
                    final_likelihoods_each_inserting = (
                        combination_probs * probs[sample_idx]
                    )  # probs[sample_idx]: likelihood of LG*IG

                    # 5. Save the selection results which will be used in molecule generation
                    sample_results = [
                        (
                            data_idx,
                            query_smi,
                            subgraph_idx,
                            atoms_in_subgraph,
                            new_frag_smi,
                            combinations[attach_idx],
                            final_likelihoods_each_inserting[attach_idx].item(),
                        )
                        for attach_idx in range(len(combinations))
                    ]
                    # model_inference_results.extend(sample_results)
                    each_subgraph_result.extend(sample_results)
                    final_likelihoods = torch.cat(
                        [final_likelihoods, final_likelihoods_each_inserting], dim=0
                    )

                # 6. Top-k sampling from final_likelihoods
                if self.num_sample_each_mol == "all":
                    selected_merge_plans = torch.argsort(
                        final_likelihoods, descending=True
                    ).to("cpu")
                else:
                    # HACK: debugging (24.01.13)
                    if len(final_likelihoods) < num_samples * 2:
                        selected_merge_plans = torch.sort(-final_likelihoods).indices.to("cpu").tolist()
                    else:
                        selected_merge_plans = (
                            torch.topk(final_likelihoods, k=num_samples * 2)
                            .indices.to("cpu")
                            .tolist()
                        )

                model_inference_results.extend(
                    [
                        each_subgraph_result[plan_idx]
                        for plan_idx in selected_merge_plans
                    ]
                )

        return model_inference_results

    def merge_fragment(
        self, merge_plans: List[tuple], sampling_result_list: List[dict], batch_idx: int
    ) -> pd.DataFrame:
        # Select only self.num_sample_each_mol
        selected_merge_plans = []
        data_idx_list = torch.tensor(list(zip(*merge_plans))[0])
        for data_idx in data_idx_list.unique():
            corresponding_merge_plans = list(
                filter(lambda x: x[0] == data_idx, merge_plans)
            )

            # joint_probs = [merge_plan[-1] for merge_plan in corresponding_merge_plans]
            # joint_probs = torch.tensor(joint_probs, requires_grad=False)
            # num_max_choices = torch.where(
            #     joint_probs != 0, True, False
            # ).sum()  # < num_subgraph * frag_lib

            # if self.num_sample_each_mol == "all":
            #     num_to_sample = num_max_choices
            # else:
            #     num_to_sample = min(num_max_choices, self.num_sample_each_mol)
            # selected_merge_plan_idxs = torch.topk(
            #     joint_probs, k=num_to_sample
            # ).indices.tolist()
            # selected_merge_plans.extend(
            #     [corresponding_merge_plans[idx] for idx in selected_merge_plan_idxs]
            # )
            selected_merge_plans.extend(corresponding_merge_plans)

        # Generate molecules with multiprocessing
        lock = Lock()
        brics_compose = partial(
            self.brics_compose, batch_idx=batch_idx, batch_size=self.batch_size
        )
        with mp.Pool(
            processes=self.num_cores, initializer=init_pool, initargs=(lock,)
        ) as p:
            generation_results = p.map(brics_compose, selected_merge_plans)

        # Merge the generated results into one pandas dataframe
        result_df = pd.DataFrame(
            generation_results, columns=GEN_COLUMNS + ["SUBGRAPH-IDX"]
        )

        # Group the dataframe by the 'INPUT-MOL-IDX' column and sort them by probability in descending order
        grouped_df = result_df.groupby("INPUT-MOL-IDX")
        sorted_groups = []
        for data_idx, group in grouped_df:
            each_sampling_result = sampling_result_list[data_idx]
            subgraph_to_num_samples = each_sampling_result["num_sample"]

            sorted_group = group.sort_values(by="PREDICTED-PROB", ascending=False)
            sorted_group = sorted_group.drop_duplicates(subset="GEN-MOL-SMI")

            groupd_by_subgraph_idx = sorted_group.groupby("SUBGRAPH-IDX")

            subgraph_groups = []
            for subgraph_idx, subgraph_group in groupd_by_subgraph_idx:
                num_samples = subgraph_to_num_samples[subgraph_idx]
                top_subgraph_group = subgraph_group.head(num_samples)
                subgraph_groups.append(top_subgraph_group)

            merged_subgraph_group = pd.concat(subgraph_groups, axis=0, ignore_index=True)
            sorted_group = merged_subgraph_group.sort_values(
                by="PREDICTED-PROB", ascending=False
            )
            sorted_groups.append(sorted_group)

        sorted_result_df = pd.concat(sorted_groups, axis=0, ignore_index=True)
        sorted_result_df = sorted_result_df.drop("SUBGRAPH-IDX", axis=1)
        return sorted_result_df

    @torch.no_grad()
    def mod_pos_scoring(self, mol_emb):
        return self.model.mod_pos_scoring(
            mol_emb, mol_emb.allowed_subgraph, mol_emb.allowed_subgraph_idx
        )


def init_pool(lock_: Lock):
    global lock
    lock = lock_
