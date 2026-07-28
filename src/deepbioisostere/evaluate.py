from pathlib import Path
from typing import Union

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .conditioning import Conditioner
from .dataset import FragmentLibrary, InferenceCollator, InferenceDataset
from .generate import Generator
from .model import DeepBioisostere


class Evaluator(Generator):
    def __init__(
        self,
        model: DeepBioisostere,
        processed_frag_dir: Union[str, Path],
        cond_module: Union[Conditioner, None],
        device: torch.device,
        num_cores: int,
        batch_size: int,
        new_frag_type: str = "test",
        logger=None,
    ):
        self.model = model
        self.cond_module = cond_module
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
        frags_lib_dataset = FragmentLibrary.get_new_frag_libraries(
            processed_frag_dir, new_frag_type
        )
        self.frag_lib_size = len(frags_lib_dataset)
        self.frag_smis = frags_lib_dataset.frags_smis
        self.frag_brics_dict = frags_lib_dataset.frags_brics_dict

        # Embed fragment library into frag_node_emb, frag_graph_emb
        self.frag_lib_dl = DataLoader(
            dataset=frags_lib_dataset,
            batch_size=batch_size,
            num_workers=num_cores,
            follow_batch=["x_n"],
        )
        self.save_frags_embeddings()
        del frags_lib_dataset

        self.logger("Evaluator initialization finished.")

    def generate(self):
        raise AttributeError("'Evaluator' object has no attribute 'generate'.")

    def generate_with_leaving_frag(self):
        raise AttributeError(
            "'Evaluator' object has no attribute 'generate_with_leaving_frag'."
        )

    def select_from_joint_prob(self):
        raise AttributeError(
            "'Evaluator' object has no attribute 'select_from_joint_prob'."
        )

    def select_attachment_orientation(self):
        raise AttributeError(
            "'Evaluator' object has no attribute 'select_attachment_orientation'."
        )

    def merge_fragment(self):
        raise AttributeError("'Evaluator' object has no attribute 'merge_fragment'.")

    @torch.no_grad()
    def evaluate_ground_truth(self, gt_data: pd.DataFrame) -> pd.DataFrame:
        smiles_list = gt_data["REF-SMI"].to_list()

        # initialize generate dataset
        dataset = InferenceDataset(
            smiles_list=smiles_list, cond_module=self.cond_module
        )

        # initialize dataloader
        data_dl = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_cores
        )
        data_dl.collate_fn = InferenceCollator(follow_batch=["x_n", "allowed_subgraph"])

        # Main Generation Part
        batch_result = []
        for batch_idx, batch in enumerate(data_dl):
            # 1. Embedding data
            batch = batch.to(self.device)
            ampn_emb = self.model.ampn(batch)
            mol_emb = self.model.fmpn(ampn_emb)

            # 2. Score modiciation position
            # leaving_subgraph_probs: list of Tensor[subgraph]
            # subgraph_embed_vector: Tensor[∑subgraph, n_F]
            prediction_result = self.score_modification_position(mol_emb, batch)
            leaving_subgraph_probs, subgraph_embed_vector = prediction_result

            # 3. Score fragment for the selected position
            # inserting_frag_probs: list of Tensor[subgraph, frag_lib]
            inserting_frag_probs = self.score_fragment_for_position(
                subgraph_embed_vector, batch
            )

            # 4. Select from joint probability
            sampling_result_list = self.select_from_joint_prob(
                leaving_subgraph_probs, inserting_frag_probs
            )

            # 5. select attachment orientation for the selected position & fragment
            # NOTE: Most time-consuming part.
            model_inference_results = self.select_attachment_orientation(
                sampling_result_list, ampn_emb, batch
            )

            # 6. merge the fragment to the molecule
            result_df = self.merge_fragment(model_inference_results, batch_idx)
            batch_result.append(result_df)
            continue

        # Merge and return
        result_df = pd.concat(batch_result, axis=0, ignore_index=True)
        return result_df


# class Evaluator_DEP:
#     def __init__(self, target_dir, property, target_value, num_cores):
#         self.target_dir = Path(target_dir)
#         self.target_value = target_value
#         self.prop = property
#         self.num_cores = num_cores
#         self._get_prop_diff = partial(self.get_prop_diff, prop=self.prop)
#
#     @staticmethod
#     def get_prop_diff(pair: Tuple[str, str], prop):
#         if prop == "logp":
#             prop_fn = logP
#         elif prop == "mw":
#             prop_fn = Mw
#         elif prop == "qed":
#             prop_fn = Qed
#         original_smi, sampled_smi = pair
#         mol1, mol2 = Mol(original_smi), Mol(sampled_smi)
#         prop1, prop2 = prop_fn(mol1), prop_fn(mol2)
#         return prop2 - prop1
#
#     @staticmethod
#     def is_valid(smi: str):
#         try:
#             mol = Mol(smi)
#             if mol is None:
#                 return False
#             else:
#                 return smi
#         except:
#             return False
#
#     def evaluate(self, sampled_data):
#         # get statistics
#         pairs = []
#         sampled_smis = []
#         for original_smi, sample_result in sampled_data.items():
#             for res in sample_result:
#                 sampled_smi, fragID, prob = res
#                 pairs.append((original_smi, sampled_smi))
#                 sampled_smis.append(sampled_smi)
#
#         with Pool(processes=self.num_cores) as p:
#             result = p.map(self._get_prop_diff, pairs)
#         self.prop_result = np.array(result, dtype=float)
#
#         # plot distribution
#         label = f"{self.prop}_{self.target_value}"
#         plt.figure()
#         plt.hist(self.prop_result, cumulative=False, bins=100, label=label)
#         plt.legend(loc="upper left")
#         plt.savefig(self.target_dir.joinpath(f"{label}.png"), format="png")
#
#         # save the calculated_values
#         np.save(self.target_dir.joinpath("prop_values.npy"), self.prop_result)
#
#         # calulate metrics
#         return self.get_metrics(sampled_smis)
#
#     def get_metrics(self, sampled_smis):
#         num_samples = len(sampled_smis)
#         with Pool(processes=self.num_cores) as p:
#             result = p.map(self.is_valid, sampled_smis)
#
#         validity = (num_samples - result.count(False)) / num_samples
#         uniqueness = len(set(result) - set([False])) / (
#             num_samples - result.count(False)
#         )
#         novelty = None  # TODO
#         return validity, uniqueness, novelty
