import random
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.loader import DataLoader

from brics.brics import BRICSModule
from conditioning import Conditioner
from dataset import InferenceCollator, InferenceDataset
from generate import Generator
from model import DeepBioisostere
from property import calc_logP, calc_Mw, calc_QED, calc_SAscore

SMILES = str
PROPERTY = str
GEN_COLUMNS = [
    "INPUT-MOL-IDX",
    "INPUT-MOL-SMI",
    "GEN-MOL-SMI",
    "LEAVING-FRAG-SMI",
    "INSERTING-FRAG-SMI",
    "PREDICTED-PROB",
    "LOGP",
    "MW",
    "QED",
    "SA",
]


class BaselineGenerator(Generator):
    """
    Baseline molecule generator with three strategies:
    1. Random leaving fragment + frequency-based insertion fragment
    2. DeepBioisostere leaving fragment + frequency-based insertion fragment
    3. Completely random selection
    """

    def __init__(
        self,
        model: DeepBioisostere = None,
        processed_frag_dir: Union[str, Path] = None,
        num_sample_each_mol: Union[int, str] = None,
        device: torch.device = None,
        num_cores: int = None,
        batch_size: int = None,
        new_frag_type: str = None,
        conditioner: Union[Conditioner, None] = None,
        properties: List[PROPERTY] = None,
        logger=None,
    ):
        super().__init__(
            model=model,
            processed_frag_dir=processed_frag_dir,
            num_sample_each_mol=num_sample_each_mol,
            device=device,
            num_cores=num_cores,
            batch_size=batch_size,
            new_frag_type=new_frag_type,
            conditioner=conditioner,
            properties=properties,
            logger=logger,
        )

    def generate_strategy_1(
        self, input_list: List[Tuple[SMILES, Dict[str, float]]]
    ) -> pd.DataFrame:
        """
        Strategy 1: Random leaving fragment selection + frequency-based insertion fragment selection
        """
        return self._generate_with_strategy(
            input_list, strategy="random_leaving_freq_insertion"
        )

    def generate_strategy_2(
        self, input_list: List[Tuple[SMILES, Dict[str, float]]]
    ) -> pd.DataFrame:
        """
        Strategy 2: DeepBioisostere leaving fragment selection + frequency-based insertion fragment selection
        """
        if self.model is None:
            raise ValueError("Model is required for strategy 2")
        return self._generate_with_strategy(
            input_list, strategy="model_leaving_freq_insertion"
        )

    def generate_strategy_3(
        self, input_list: List[Tuple[SMILES, Dict[str, float]]]
    ) -> pd.DataFrame:
        """
        Strategy 3: Completely random selection for both leaving and insertion fragments
        """
        return self._generate_with_strategy(input_list, strategy="random_both")

    def _generate_with_strategy(
        self, input_list: List[Tuple[SMILES, Dict[str, float]]], strategy: str
    ) -> pd.DataFrame:
        """
        Main generation method that handles all three strategies
        """
        if self.conditioner:
            smiles_list, prop_dict_list = zip(*input_list)
            conditioning = True
        else:
            smiles_list = input_list
            prop_dict_list = None
            conditioning = False

        dataset = InferenceDataset(
            smiles_list=smiles_list,
            conditioner=self.conditioner,
            prop_dict_list=prop_dict_list,
        )

        data_dl = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_cores
        )
        data_dl.collate_fn = InferenceCollator(
            use_conditioning=conditioning,
            properties=self.properties,
            follow_batch=["x_n", "allowed_subgraph"],
        )

        batch_result = []
        for batch_idx, batch in enumerate(data_dl):
            data = batch["data"].to(self.device)
            if self.conditioner:
                for prop in self.properties:
                    batch[prop] = batch[prop].to(self.device)

            if strategy == "random_leaving_freq_insertion":
                sampling_result_list = self._random_leaving_freq_insertion(data)
            elif strategy == "model_leaving_freq_insertion":
                sampling_result_list = self._model_leaving_freq_insertion(batch, data)
            elif strategy == "freq_leaving_freq_insertion":
                sampling_result_list = self._freq_leaving_freq_insertion(batch, data)
            elif strategy == "random_both":
                sampling_result_list = self._random_both_selection(data)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            model_inference_results = self._select_attachment_orientation_baseline(
                sampling_result_list, data
            )

            result_df = self._merge_fragment_baseline(
                model_inference_results, sampling_result_list, batch_idx
            )
            batch_result.append(result_df)

        result_df = pd.concat(batch_result, axis=0, ignore_index=True)
        return result_df

    def _freq_leaving_freq_insertion(self, batch, data):
        """
        Frequency-based leaving fragment selection + frequency-based insertion fragment selection.
        
        This method implements frequency-based selection for both leaving and insertion fragments,
        considering the frequency of "old" fragments (leaving fragments) in the training data.
        """
        raise NotImplementedError("Not implemented yet")

    def _model_leaving_freq_insertion(self, batch, data):
        """
        Use DeepBioisostere for leaving fragment selection, frequency-based for insertion.

        This method:
        1. Uses the trained model to score leaving fragment positions
        2. Applies BRICS type compatibility rules for insertion fragment selection
        3. Uses frequency-based sampling from compatible insertion fragments
        """
        ampn_emb = self.model.ampn(data)
        if self.conditioner:
            cond_embeddings = []
            for prop in self.properties:
                cond_embeddings.append(batch[prop])
            condition_embedding = torch.cat(cond_embeddings, dim=1)
            ampn_emb.x_f = torch.cat([ampn_emb.x_f, condition_embedding], dim=1)
        mol_emb = self.model.fmpn(ampn_emb)

        leaving_subgraph_probs, subgraph_embed_vector = (
            self.score_modification_position(mol_emb, data)
        )

        sampling_result_list = []
        for data_idx in range(data.num_graphs):
            leaving_prob = leaving_subgraph_probs[data_idx]
            num_allowed_subgraph = len(leaving_prob)

            each_sampling_result = dict()
            each_sampling_result["num_sample"] = {
                _: None for _ in range(num_allowed_subgraph)
            }
            each_sampling_result["inserting"] = {
                _: [] for _ in range(num_allowed_subgraph)
            }
            each_sampling_result["prob"] = {_: [] for _ in range(num_allowed_subgraph)}

            if self.num_sample_each_mol == "all":
                for subgraph_idx in range(num_allowed_subgraph):
                    brics_types = data.brics_types[data_idx][subgraph_idx]
                    self.logger(
                        f"Processing subgraph {subgraph_idx} with BRICS types: {brics_types}"
                    )

                    frag_indices, frag_probs = self._get_frequency_based_fragments(
                        brics_types
                    )

                    if not frag_indices:
                        self.logger(
                            f"No compatible fragments found for BRICS types: {brics_types}"
                        )
                        each_sampling_result["num_sample"][subgraph_idx] = 0
                        continue

                    each_sampling_result["num_sample"][subgraph_idx] = len(frag_indices)
                    each_sampling_result["inserting"][subgraph_idx] = frag_indices
                    each_sampling_result["prob"][subgraph_idx] = [
                        leaving_prob[subgraph_idx].item() * p for p in frag_probs
                    ]
            else:
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

                for subgraph_idx in range(num_allowed_subgraph):
                    num_to_sample = int(num_sample_each_subgraph[subgraph_idx])
                    if num_to_sample > 0:
                        brics_types = data.brics_types[data_idx][subgraph_idx]
                        frag_indices, frag_probs = self._get_frequency_based_fragments(
                            brics_types, num_to_sample
                        )

                        if not frag_indices:
                            self.logger(
                                f"No compatible fragments found for BRICS types: {brics_types}"
                            )
                            each_sampling_result["num_sample"][subgraph_idx] = 0
                            continue

                        each_sampling_result["num_sample"][subgraph_idx] = len(
                            frag_indices
                        )
                        each_sampling_result["inserting"][subgraph_idx] = frag_indices
                        each_sampling_result["prob"][subgraph_idx] = [
                            leaving_prob[subgraph_idx].item() * p for p in frag_probs
                        ]
                    else:
                        each_sampling_result["num_sample"][subgraph_idx] = 0

            sampling_result_list.append(each_sampling_result)

        return sampling_result_list

    def _random_leaving_freq_insertion(self, data):
        """
        Random leaving fragment selection + frequency-based insertion fragment selection
        """
        sampling_result_list = []
        for data_idx in range(data.num_graphs):
            num_allowed_subgraph = (
                data.num_allowed_subgraph[data_idx]
                if hasattr(data, "num_allowed_subgraph")
                else len(data.brics_types[data_idx])
            )

            each_sampling_result = dict()
            each_sampling_result["num_sample"] = {
                _: None for _ in range(num_allowed_subgraph)
            }
            each_sampling_result["inserting"] = {
                _: [] for _ in range(num_allowed_subgraph)
            }
            each_sampling_result["prob"] = {_: [] for _ in range(num_allowed_subgraph)}

            if self.num_sample_each_mol == "all":
                for subgraph_idx in range(num_allowed_subgraph):
                    frag_indices, frag_probs = self._get_frequency_based_fragments(
                        data.brics_types[data_idx][subgraph_idx]
                    )
                    each_sampling_result["num_sample"][subgraph_idx] = len(frag_indices)
                    each_sampling_result["inserting"][subgraph_idx] = frag_indices
                    each_sampling_result["prob"][subgraph_idx] = frag_probs
            else:
                subgraph_probs = (
                    torch.ones(num_allowed_subgraph).to(self.device)
                    / num_allowed_subgraph
                )
                leaving_frag_sampling_result = torch.multinomial(
                    input=subgraph_probs,
                    num_samples=self.num_sample_each_mol,
                    replacement=True,
                )
                unique_subgraph, subgraph_counts = torch.unique(
                    leaving_frag_sampling_result, return_counts=True
                )

                for i, subgraph_idx in enumerate(unique_subgraph.tolist()):
                    num_to_sample = subgraph_counts[i].item()
                    frag_indices, frag_probs = self._get_frequency_based_fragments(
                        data.brics_types[data_idx][subgraph_idx], num_to_sample
                    )
                    each_sampling_result["num_sample"][subgraph_idx] = len(frag_indices)
                    each_sampling_result["inserting"][subgraph_idx] = frag_indices
                    each_sampling_result["prob"][subgraph_idx] = frag_probs

                for subgraph_idx in range(num_allowed_subgraph):
                    if subgraph_idx not in unique_subgraph.tolist():
                        each_sampling_result["num_sample"][subgraph_idx] = 0

            sampling_result_list.append(each_sampling_result)

        return sampling_result_list

    def _random_both_selection(self, data):
        """
        Completely random selection for both leaving and insertion fragments
        """
        sampling_result_list = []
        for data_idx in range(data.num_graphs):
            num_allowed_subgraph = (
                data.num_allowed_subgraph[data_idx]
                if hasattr(data, "num_allowed_subgraph")
                else len(data.brics_types[data_idx])
            )

            each_sampling_result = dict()
            each_sampling_result["num_sample"] = {
                _: None for _ in range(num_allowed_subgraph)
            }
            each_sampling_result["inserting"] = {
                _: [] for _ in range(num_allowed_subgraph)
            }
            each_sampling_result["prob"] = {_: [] for _ in range(num_allowed_subgraph)}

            if self.num_sample_each_mol == "all":
                for subgraph_idx in range(num_allowed_subgraph):
                    frag_indices = self._get_random_fragments(
                        data.brics_types[data_idx][subgraph_idx]
                    )
                    each_sampling_result["num_sample"][subgraph_idx] = len(frag_indices)
                    each_sampling_result["inserting"][subgraph_idx] = frag_indices
                    each_sampling_result["prob"][subgraph_idx] = (
                        [1.0 / len(frag_indices)] * len(frag_indices)
                        if frag_indices
                        else []
                    )
            else:
                subgraph_probs = (
                    torch.ones(num_allowed_subgraph).to(self.device)
                    / num_allowed_subgraph
                )
                leaving_frag_sampling_result = torch.multinomial(
                    input=subgraph_probs,
                    num_samples=self.num_sample_each_mol,
                    replacement=True,
                )
                unique_subgraph, subgraph_counts = torch.unique(
                    leaving_frag_sampling_result, return_counts=True
                )

                for i, subgraph_idx in enumerate(unique_subgraph.tolist()):
                    num_to_sample = subgraph_counts[i].item()
                    frag_indices = self._get_random_fragments(
                        data.brics_types[data_idx][subgraph_idx], num_to_sample
                    )
                    each_sampling_result["num_sample"][subgraph_idx] = len(frag_indices)
                    each_sampling_result["inserting"][subgraph_idx] = frag_indices
                    each_sampling_result["prob"][subgraph_idx] = [1.0] * len(
                        frag_indices
                    )

                for subgraph_idx in range(num_allowed_subgraph):
                    if subgraph_idx not in unique_subgraph.tolist():
                        each_sampling_result["num_sample"][subgraph_idx] = 0

            sampling_result_list.append(each_sampling_result)

        return sampling_result_list

    def _get_frequency_based_fragments(self, brics_types, num_samples=None):
        """
        Get fragment indices based on frequency distribution with BRICS type compatibility.

        This method implements the core BRICS type matching logic:
        1. Takes leaving fragment BRICS types as input
        2. Maps to compatible insertion BRICS types using BRICS rules
        3. Filters fragment library to only include compatible fragments
        4. Applies frequency-based probability weighting
        5. Samples from the weighted distribution
        """
        allowed_insertion_types = self.BRICS_TYPE_MAPPER.getMapping(sorted(brics_types))
        valid_frag_indices = []

        for insertion_brics_type in allowed_insertion_types:
            insertion_brics_type_str = ",".join(list(map(str, insertion_brics_type)))
            if insertion_brics_type_str in self.brics_type_to_insertion_frags:
                matching_frags = self.brics_type_to_insertion_frags[
                    insertion_brics_type_str
                ].tolist()
                valid_frag_indices.extend(matching_frags)
                # self.logger(
                #     f"Found {len(matching_frags)} fragments for BRICS type: {insertion_brics_type_str}"
                # )

        if not valid_frag_indices:
            self.logger(f"No valid fragments found for BRICS types: {brics_types}")
            return [], []

        valid_frag_indices = list(set(valid_frag_indices))
        # self.logger(f"Total unique compatible fragments: {len(valid_frag_indices)}")

        frag_frequencies = self.frags_freq[valid_frag_indices].float()
        frag_probs = F.normalize(frag_frequencies, p=1, dim=0)

        if num_samples is None or num_samples >= len(valid_frag_indices):
            return valid_frag_indices, frag_probs.tolist()
        else:
            sampled_indices = torch.multinomial(
                frag_probs, num_samples, replacement=True
            )
            selected_frag_indices = [
                valid_frag_indices[i] for i in sampled_indices.tolist()
            ]
            selected_probs = [frag_probs[i].item() for i in sampled_indices.tolist()]
            return selected_frag_indices, selected_probs

    def _get_random_fragments(self, brics_types, num_samples=None):
        """
        Get random fragment indices that satisfy BRICS rules
        """
        allowed_insertion_types = self.BRICS_TYPE_MAPPER.getMapping(sorted(brics_types))
        valid_frag_indices = []

        for insertion_brics_type in allowed_insertion_types:
            insertion_brics_type_str = ",".join(list(map(str, insertion_brics_type)))
            if insertion_brics_type_str in self.brics_type_to_insertion_frags:
                valid_frag_indices.extend(
                    self.brics_type_to_insertion_frags[
                        insertion_brics_type_str
                    ].tolist()
                )

        if not valid_frag_indices:
            return []

        valid_frag_indices = list(set(valid_frag_indices))

        if num_samples is None or num_samples >= len(valid_frag_indices):
            return valid_frag_indices
        else:
            return random.sample(valid_frag_indices, num_samples)

    def _select_attachment_orientation_baseline(self, sampling_result_list, batch):
        """
        Select attachment orientations randomly (simplified version of original method)
        """
        model_inference_results = []
        for data_idx in range(batch.num_graphs):
            atom_frag_indice = batch.atom_frag_indice[data_idx]
            brics_bond_indices = batch.brics_bond_indices[data_idx]
            brics_bond_types = batch.brics_bond_types[data_idx]
            query_smi = batch.smiles[data_idx]

            frag_atom_indice = dict()
            for i, f_id in enumerate(list(map(int, atom_frag_indice.split(",")))):
                if f_id in frag_atom_indice:
                    frag_atom_indice[f_id].append(i)
                else:
                    frag_atom_indice[f_id] = [i]

            retrieved_subgraph_idxs = (
                batch.allowed_subgraph_batch == data_idx
            ).nonzero()
            retrieved_subgraph_idxs = retrieved_subgraph_idxs.squeeze(-1)

            mask = torch.isin(batch.allowed_subgraph_idx, retrieved_subgraph_idxs)
            allowed_subgraph = batch.allowed_subgraph[mask]
            allowed_subgraph_idx = batch.allowed_subgraph_idx[mask]

            allowed_subgraph -= (
                allowed_subgraph.min() - batch.min_allowed_subgraph[data_idx]
            )
            allowed_subgraph_idx -= allowed_subgraph_idx.min()
            num_allowed_subgraph = allowed_subgraph_idx.max() + 1

            each_sampling_result = sampling_result_list[data_idx]
            subgraph_to_num_samples = each_sampling_result["num_sample"]
            subgraph_to_insertings = each_sampling_result["inserting"]
            subgraph_to_probs = each_sampling_result["prob"]

            for subgraph_idx in range(num_allowed_subgraph):
                num_samples = subgraph_to_num_samples[subgraph_idx]
                if num_samples == 0:
                    continue

                frag_idxs = subgraph_to_insertings[subgraph_idx]
                probs = subgraph_to_probs[subgraph_idx]

                frags_in_subgraph = allowed_subgraph[
                    allowed_subgraph_idx == subgraph_idx
                ]
                frags_in_subgraph = frags_in_subgraph.tolist()
                atoms_in_subgraph = []
                for frag_idx in frags_in_subgraph:
                    try:
                        atoms_in_subgraph += frag_atom_indice[frag_idx]
                    except KeyError:
                        continue

                attach_atom_indice, attach_brics_types = [], []
                for bond_idx, bond_indice in enumerate(brics_bond_indices):
                    bond_types = brics_bond_types[bond_idx]
                    atom_is_in_subgraph = [
                        atom_idx in atoms_in_subgraph for atom_idx in bond_indice
                    ]

                    if not all(atom_is_in_subgraph) and any(atom_is_in_subgraph):
                        atom_not_in_subgraph = atom_is_in_subgraph.index(False)
                        attach_atom_indice.append(bond_indice[atom_not_in_subgraph])
                        attach_brics_types.append(bond_types[atom_not_in_subgraph])

                allowed_attachments = BRICSModule.enumerate_allowed_combinations(
                    attach_atom_indice,
                    attach_brics_types,
                    self.frag_adj_dummy_inform[frag_idxs],
                    frag_idxs,
                    logger=self.logger,
                )

                for sample_idx, frag_idx in enumerate(frag_idxs):
                    new_frag_smi = self.frags_smis[frag_idx]

                    attachments = allowed_attachments.get(frag_idx, [])
                    if not attachments:
                        self.logger(
                            "No allowed combinations found for fragment index: "
                            f"{frag_idx} in subgraph {subgraph_idx}"
                        )

                    # Randomly select one of the allowed combinations
                    selected_attachment = random.choice(attachments)

                    if selected_attachment:
                        model_inference_results.append(
                            (
                                data_idx,
                                query_smi,
                                subgraph_idx,
                                atoms_in_subgraph,
                                new_frag_smi,
                                selected_attachment,
                                probs[sample_idx] if sample_idx < len(probs) else 1.0,
                            )
                        )

        return model_inference_results

    def _merge_fragment_baseline(self, merge_plans, sampling_result_list, batch_idx):
        """
        Merge fragments to create final molecules
        """
        selected_merge_plans = []
        if merge_plans:
            data_idx_list = torch.tensor(list(zip(*merge_plans))[0])
            for data_idx in data_idx_list.unique():
                corresponding_merge_plans = list(
                    filter(lambda x: x[0] == data_idx, merge_plans)
                )
                selected_merge_plans.extend(corresponding_merge_plans)

        generation_results = []
        for merge_plan in selected_merge_plans:
            try:
                result = self._compose_molecule_baseline(merge_plan, batch_idx)
                if result:
                    generation_results.append(result)
            except Exception as e:
                self.logger(f"Error in molecule composition: {e}")
                continue

        if not generation_results:
            return pd.DataFrame(columns=GEN_COLUMNS)

        result_df = pd.DataFrame(generation_results, columns=GEN_COLUMNS)

        grouped_df = result_df.groupby("INPUT-MOL-IDX")
        sorted_groups = []
        for data_idx, group in grouped_df:
            if data_idx < len(sampling_result_list):
                each_sampling_result = sampling_result_list[data_idx]
                subgraph_to_num_samples = each_sampling_result["num_sample"]

                sorted_group = group.sort_values(by="PREDICTED-PROB", ascending=False)
                sorted_group = sorted_group.drop_duplicates(subset="GEN-MOL-SMI")
                sorted_groups.append(sorted_group)

        if sorted_groups:
            sorted_result_df = pd.concat(sorted_groups, axis=0, ignore_index=True)
        else:
            sorted_result_df = pd.DataFrame(columns=GEN_COLUMNS)

        return sorted_result_df

    def _compose_molecule_baseline(self, merge_plan, batch_idx):
        """
        Compose individual molecule using BRICS rules
        """
        (
            data_idx,
            original_smi,
            subgraph_idx,
            change_indices,
            new_frag_smi,
            attachment,
            prob,
        ) = merge_plan
        idx = data_idx + self.batch_size * batch_idx

        try:
            generated_smi, leaving_frag_smi = BRICSModule.compose_mols_with_attachment(
                original_smi,
                change_indices,
                new_frag_smi,
                attachment,
                get_leaving_frag_smi=True,
            )

            generated_mol = Chem.MolFromSmiles(generated_smi)
            if generated_mol is None:
                return None

            logp = calc_logP(generated_mol)
            mw = calc_Mw(generated_mol)
            qed = calc_QED(generated_mol)
            sa = calc_SAscore(generated_mol)

            return [
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
            ]
        except Exception as e:
            self.logger(f"Error in BRICS composition: {e}")
            return None
