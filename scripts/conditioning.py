from typing import Dict, List
from functools import partial

import torch
from rdkit import Chem

from .model import DeepBioisostere
from .property import PROPERTIES, calc_logP, calc_Mw, calc_QED, calc_SAscore

SMILES = str
PROPERTY = str


class Conditioner:
    # properties = PROPERTIES

    # max, min values for logp, mw, qed
    logp_consts = [10.0, -5.0]
    mw_consts = [1000.0, 0.0]
    qed_consts = [1.0, 0.0]
    sa_consts = [6.0, 0.0]

    # max, min values for delta logp, delta mw, delta qed
    delta_logp_consts = [3.0, -3.0]
    delta_mw_consts = [150.0, -150.0]
    delta_qed_consts = [0.4, -0.4]
    delta_sa_consts = [2.0, -2.0]

    node_hid_dim = DeepBioisostere.default_args["frag_node_hid_dim"]

    def __init__(
        self,
        phase: str,
        properties: List[PROPERTY],
        use_delta: bool = True,
    ):
        """
        Args:
            phase: One of 'train' or 'generation'.
            use_delta: Whether to use delta values or not.
            soft_one_hot: Whether to use soft one-hot encoding or not.
              if true, use soft one-hot encoding. Else, yield a normalized scalar value.
        """
        # Embedding_layers: ["logp", "mw", "qed", "sa"]
        self.use_delta = use_delta
        self.norm_fn_dict = {
            # prop: self.get_norm_fn(prop, use_delta) for prop in PROPERTIES
            prop: partial(self.norm_fn, prop=prop, use_delta=use_delta) for prop in PROPERTIES
        }
        self.properties = properties

        # Functions to calculate properties
        self.calc_prop_fn_dict = {
            "logp": calc_logP,
            "mw": calc_Mw,
            "qed": calc_QED,
            "sa": calc_SAscore,
        }

        # Depending on the phase,
        if phase == "train":
            self.add_condition = self.add_cond_to_training
        elif phase == "generation":
            self.add_condition = self.add_cond_to_generation

    @classmethod
    def norm_fn(cls, x: float, prop: str, use_delta: bool) -> float:
        if prop == "logp" and use_delta:
            max, min = cls.delta_logp_consts
        elif prop == "mw" and use_delta:
            max, min = cls.delta_mw_consts
        elif prop == "qed" and use_delta:
            max, min = cls.delta_qed_consts
        elif prop == "sa" and use_delta:
            max, min = cls.delta_sa_consts
        elif prop == "logp" and not use_delta:
            max, min = cls.logp_consts
        elif prop == "mw" and not use_delta:
            max, min = cls.mw_consts
        elif prop == "qed" and not use_delta:
            max, min = cls.qed_consts
        elif prop == "sa" and not use_delta:
            max, min = cls.sa_consts
        else:
            raise ValueError("Property must be one of 'logp', 'mw', 'qed', or 'sa'.")

        normalized_x = (x - min) / (max - min)
        return normalized_x

    @classmethod
    def get_norm_fn(cls, prop: str, use_delta: bool) -> callable:
        if prop == "logp" and use_delta:
            max, min = cls.delta_logp_consts
        elif prop == "mw" and use_delta:
            max, min = cls.delta_mw_consts
        elif prop == "qed" and use_delta:
            max, min = cls.delta_qed_consts
        elif prop == "sa" and use_delta:
            max, min = cls.delta_sa_consts
        elif prop == "logp" and not use_delta:
            max, min = cls.logp_consts
        elif prop == "mw" and not use_delta:
            max, min = cls.mw_consts
        elif prop == "qed" and not use_delta:
            max, min = cls.qed_consts
        elif prop == "sa" and not use_delta:
            max, min = cls.sa_consts
        else:
            raise ValueError("Property must be one of 'logp', 'mw', 'qed', or 'sa'.")

        def norm_fn(x: float) -> float:
            normalized_x = (x - min) / (max - min)
            return normalized_x

        return norm_fn

    @staticmethod
    def soft_one_hot_encoding(
        normalized_prop: float, dim: int, sigma_ratio: float = 0.05
    ) -> torch.FloatTensor:
        x_bin = round(normalized_prop * dim)
        sigma = round(sigma_ratio * dim)

        soft_one_hot = torch.arange(dim) - x_bin
        soft_one_hot = torch.exp(-(soft_one_hot * soft_one_hot) / (2 * sigma))
        return soft_one_hot

    def add_cond_to_training(
        self, ref_smi: SMILES, new_smi: SMILES
    ) -> Dict[PROPERTY, torch.FloatTensor]:
        """
        Args:
          data (attributes):
            x_n (bool): [N,F_n]
            smiles: str
            chembl_id1: original molecule ChEMBL ID.
            chembl_id2: target molecule ChEMBL ID.
        """
        # Get RDKit mol objects
        if self.use_delta:
            original_mol = Chem.MolFromSmiles(ref_smi)
            target_mol = Chem.MolFromSmiles(new_smi)
        else:
            target_mol = Chem.MolFromSmiles(new_smi)

        encoded_prop_dict = dict()
        for prop in self.properties:
            norm_fn = self.norm_fn_dict[prop]
            calc_prop_fn = self.calc_prop_fn_dict[prop]

            # Calculate property and add to node feature
            if self.use_delta:
                value1 = calc_prop_fn(original_mol)
                value2 = calc_prop_fn(target_mol)
                norm_value = norm_fn(value2 - value1)
            else:
                value = calc_prop_fn(target_mol)
                norm_value = norm_fn(value)

            encoded_prop = torch.tensor(norm_value).unsqueeze(0)  # [1]
            encoded_prop_dict[prop] = encoded_prop

        return encoded_prop_dict

    def add_cond_to_generation(
        self, prop_dict: Dict[str, float]
    ) -> Dict[PROPERTY, torch.FloatTensor]:
        """
        Args:
          data (attributes):
            x_n (bool): [N,F_n]
        """
        encoded_prop_dict = dict()
        for prop, value in prop_dict.items():
            norm_fn = self.norm_fn_dict[prop]

            # Add embedded condition
            norm_value = norm_fn(value)
            encoded_prop = torch.tensor(norm_value).unsqueeze(0)  # [1]
            encoded_prop_dict[prop] = encoded_prop

        return encoded_prop_dict

    def __call__(self, **kwargs):
        return self.add_condition(**kwargs)


if __name__ == "__main__":
    soft_one_hot = Conditioner.soft_one_hot_encoding(x=25, dim=50)
    print(soft_one_hot)
