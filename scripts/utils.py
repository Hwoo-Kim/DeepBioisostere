import random
import re
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import BRICS
import rdkit.Chem.AllChem as AllChem
import pandas as pd
import logging
import os
import subprocess
import itertools


class Logger(logging.Logger):
    def __init__(self, name, save_path=None):
        super().__init__(name=name)
        if save_path:
            if os.path.exists(save_path):
                os.remove(save_path)
            try:
                file_handler = logging.FileHandler(filename=save_path)
                # file_handler.setFormatter(formatter)
                file_handler.setLevel(logging.DEBUG)
                self.addHandler(file_handler)
            except FileNotFoundError:
                print(f"Invalid log path {save_path}")
                exit()
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        self.addHandler(stream_handler)

    def __call__(self, message=""):
        self.info(message)

    @staticmethod
    def _get_skip_args():
        return [
            "logger",
            "save_name",
        ]

    def log_args(self, args, tab=""):
        d = vars(args)
        _skip_args = self._get_skip_args()
        for v in d:
            if v not in _skip_args:
                self.info(f"{tab}{v}: {d[v]}")


def set_seed(seed):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def set_cuda_visible_devices() -> str:
    """Set available GPU IDs as a str (e.g., '0,1,2')"""
    max_num_gpus = 8
    idle_gpus = []

    for i in range(max_num_gpus):
        cmd = ["nvidia-smi", "-i", str(i)]
        proc = subprocess.run(cmd, capture_output=True, text=True)  # after python 3.7

        if "No devices were found" in proc.stdout:
            break

        if "No running" in proc.stdout:
            idle_gpus.append(i)

    # Convert to a str to feed to os.environ.
    idle_gpus = ",".join(str(i) for i in idle_gpus)

    os.environ["CUDA_VISIBLE_DEVICES"] = idle_gpus
    return idle_gpus


def train_path_setting(args):
    args.data_dir = os.path.normpath(args.data_dir)
    # args.key_path = os.path.join(args.processed_data_dir, "keys.pkl")

    save_dir = os.path.normpath(os.path.join(args.project_dir, "model_save"))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, args.save_name)
    while save_dir[-1] == "/":
        save_dir = save_dir[:-1]

    if os.path.exists(save_dir):
        i = 2
        while os.path.exists(f"{save_dir}_{i}"):
            i += 1
        save_dir = f"{save_dir}_{i}"
    os.mkdir(save_dir)
    args.save_dir = save_dir

    return args


def generate_path_setting(args):
    args.data_dir = os.path.normpath(args.data_dir)
    # args.original_smiles_path = os.path.normpath(args.original_smiles_path)

    save_dir = os.path.normpath(os.path.join(args.project_dir, "sampling_save"))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, args.save_name)
    while save_dir[-1] == "/":
        save_dir = save_dir[:-1]

    if os.path.exists(save_dir):
        i = 2
        while os.path.exists(f"{save_dir}_{i}"):
            i += 1
        save_dir = f"{save_dir}_{i}"
    os.mkdir(save_dir)
    args.save_dir = save_dir

    return args

class FrequencySampler:
    def __init__(self, smis: list[str], replacement_lib_path: str, generate_all_attachments: bool = True):
        self.smis = smis
        self.replacement_lib = pd.read_csv(replacement_lib_path, sep="\t")
        self.generate_all_attachments = generate_all_attachments
        return

    def filter_frag(self, num_atoms, broken_frag, max_num_change_atoms=12):
        if broken_frag.GetNumAtoms() > max_num_change_atoms:
            return False
        elif broken_frag.GetNumAtoms() > num_atoms/2:
            return False
        else: 
            return True

    def sample(self, num_samples: int) -> list[str]:
        """
        Args:
            num_samples: (int) number of SMILES per input molecule.
        Returns:
            generation_df: (pd.DataFrame) DataFrame containing sampled SMILES.
        """

        generation_dict = {
            "INPUT-SMI": [],
            "GEN-SMI": [],
            "OLD-FRAG": [],
            "NEW-FRAG": [],
            "USED-REPLACEMENT-SMILES": [],
            "USED-REPLACEMENT-FREQ": [],
        }

        for smi in self.smis:
            # 1. break by BRICS rule
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logging.warning(f"Invalid SMILES: {smi}")
                continue
            brics_fragments = BRICS.BRICSDecompose(mol, returnMols=True)
            _brics_fragments = BRICS.BRICSDecompose(mol, returnMols=False)
            # brics_smis = [Chem.MolToSmiles(fragment) for fragment in brics_fragments]

            # 2. filter brics-broken SMILES
            num_atoms = mol.GetNumAtoms()
            allowed_frags = [
                frag for frag in brics_fragments
                if self.filter_frag(num_atoms, frag)
            ]

            # 3. explore replacement library
            sampled_replacements = None
            for frag in allowed_frags:
                frag_smi = Chem.MolToSmiles(frag)

                # find replacement SMILES
                # replacement_candidates = self.replacement_lib[
                #     self.replacement_lib["OLD-FRAG"].str.contains(frag_smi)
                # ]
                replacement_candidates = self.replacement_lib[self.replacement_lib["OLD-FRAG"]==frag_smi]

                if replacement_candidates.empty:
                    continue

                if sampled_replacements is None:
                    sampled_replacements = replacement_candidates
                else:
                    sampled_replacements = pd.concat(
                        [sampled_replacements, replacement_candidates]
                    )

            if sampled_replacements is None:
                print(f"No replacement found for {smi}.")
                continue

            # 4. sample from replacement library
            # NOTE: sampling based on frequency
            sampled_replacements = sampled_replacements.sample(frac=1).reset_index(drop=True)
            sampled_replacements = sampled_replacements.sort_values(
                by="FREQUENCY", ascending=False
            ).reset_index(drop=True)

            num_gen_mol = 0
            pattern = r'\[(1[0-6]|[1-9])\*\]'
            for _, row in sampled_replacements.iterrows():
                old_frag, new_frag = row["OLD-FRAG"], row["NEW-FRAG"]
                # remove isotope tag, possibly mutiple isotope tages
                old_frag = re.sub(pattern, '[*]', old_frag)
                new_frag = re.sub(pattern, '[*]', new_frag)

                perms = list(itertools.permutations(list(range(old_frag.count("[*]")))))

                replacements = []
                for perm in perms:
                    _old_frag = old_frag
                    _new_frag = new_frag

                    # for i in range(old_frag.count("[*]")):
                    # old_idx_matches = [match.start() for match in re.finditer(re.escape("[*]"), _old_frag)]
                    # new_idx_matches = [match.start() for match in re.finditer(re.escape("[*]"), _new_frag)]

                    # random.shuffle(old_idx_matches)
                    # random.shuffle(new_idx_matches)

                    _old_frag_broken = _old_frag.split("[*]")
                    _new_frag_broken = _new_frag.split("[*]")

                    # if _old_frag_broken[0] == "":
                    #     _old_frag_broken = _old_frag_broken[1:]
                    # if _new_frag_broken[0] == "":
                    #     _new_frag_broken = _new_frag_broken[1:]

                    _old_frag_numbered = _old_frag_broken[0]
                    for i in range(len(_old_frag_broken)-1):
                        _old_frag_numbered += f"[*:{i+1}]" + _old_frag_broken[i+1]

                    _new_frag_numbered = _new_frag_broken[0]
                    for i in range(len(_new_frag_broken)-1):
                        _new_frag_numbered += f"[*:{perm[i]+1}]" + _new_frag_broken[i+1]

                    # _idx = old_idx_matches[0]
                    # _old_frag = _old_frag[:_idx] + f"[*:{i+1}]" + _old_frag[_idx+3:]

                    # _idx = new_idx_matches[-perm[i]]
                    # _new_frag = _new_frag[:_idx] + f"[*:{i+1}]" + _new_frag[_idx+3:]

                    replacement = f"{_old_frag_numbered}>>{_new_frag_numbered}"
                    replacements.append(replacement)

                if not self.generate_all_attachments:
                    random.shuffle(replacements)
                    replacements = [replacements[0]]

                for replacement in replacements:
                    # 5. generate SMILES
                    rxn = AllChem.ReactionFromSmarts(replacement)
                    gen_mols = rxn.RunReactants((mol,))     # tup of tup
                    gen_mols = list(gen_mols)

                    if len(gen_mols) == 0:
                        continue

                    random.shuffle(gen_mols)
                    gen_mol = gen_mols[0][0]

                    if gen_mol is None:
                        continue

                    generation_dict["INPUT-SMI"].append(smi)
                    generation_dict["GEN-SMI"].append(Chem.MolToSmiles(gen_mol))
                    generation_dict["OLD-FRAG"].append(old_frag)
                    generation_dict["NEW-FRAG"].append(new_frag)
                    generation_dict["USED-REPLACEMENT-SMILES"].append(replacement)
                    generation_dict["USED-REPLACEMENT-FREQ"].append(row["FREQUENCY"])
                    num_gen_mol += 1

                if num_gen_mol == num_samples:
                    break
            else:
                print(f"Not enough samples generated for {smi}. " f"Generated {num_gen_mol} samples.")

        # 6. create DataFrame
        generation_df = pd.DataFrame(generation_dict)
        return generation_df

if __name__ == "__main__":
    sampler = FrequencySampler(
        smis=["c1ccccc1CC(O)CC(=O)OCCO"],
        replacement_lib_path="/home/share/DATA/swkim/DeepBioisostere/replacement_library.csv",
        generate_all_attachments=True,
    )
    gen_df = sampler.sample(num_samples=1000)
    gen_df.to_csv("sampled_molecules.csv", index=False, sep="\t")
