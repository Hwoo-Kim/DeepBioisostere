import itertools
import logging
import os
import random
import re
import subprocess
from collections import defaultdict

import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import rdkit.Chem.rdmolops as rdmolops
from rdkit.Chem import BRICS

from scripts.property import calc_logP, calc_Mw, calc_QED, calc_SAscore


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


# def exact_query_from_smiles(smiles: str) -> str:
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         raise ValueError("Invalid SMILES")
#
#     # Parameter object
#     params = rdmolops.AdjustQueryParameters()
#     params.adjustHs           = True
#     params.adjustDegree       = True
#
#     # Create strict QueryMol
#     qmol = rdmolops.AdjustQueryProperties(mol, params)
#
#     return Chem.MolToSmarts(qmol)


def process_isotopes_in_smarts(
    smarts: str, atom_h_counts: list, isotope_indices: list
) -> str:
    """
    Convert isotope notation in SMARTS to regular element symbols.
    Isotopes are represented without hydrogen counts, using only element symbols.

    Args:
        smarts (str): SMARTS string containing isotopes
        atom_h_counts (list): List of hydrogen counts for each atom
        isotope_indices (list): List of indices of isotope atoms

    Returns:
        str: SMARTS with isotopes converted to regular elements

    Examples:
        >>> process_isotopes_in_smarts("[18#9]")
        "[F]"
        >>> process_isotopes_in_smarts("[#6]")
        "[C]"
        >>> process_isotopes_in_smarts("[18F]")
        "[F]"
    """
    # Isotope pattern: [number#atomic_number] -> [element_symbol]
    isotope_pattern = re.compile(r"\[(\d+)#(\d+)\]")

    def replace_isotope(match):
        isotope_num = match.group(1)
        atomic_num = int(match.group(2))

        # Get element symbol
        pt = Chem.GetPeriodicTable()
        element_symbol = pt.GetElementSymbol(atomic_num)

        return f"[{element_symbol}]"

    # Already converted isotope pattern: [number_element_symbol] -> [element_symbol]
    converted_isotope_pattern = re.compile(r"\[(\d+)([A-Z][a-z]?)\]")

    def replace_converted_isotope(match):
        isotope_num = match.group(1)
        element_symbol = match.group(2)

        return f"[{element_symbol}]"

    # Atomic number pattern: [#atomic_number] -> [element_symbol]
    atomic_num_pattern = re.compile(r"\[#(\d+)\]")

    def replace_atomic_num(match):
        atomic_num = int(match.group(1))

        # Get element symbol
        pt = Chem.GetPeriodicTable()
        element_symbol = pt.GetElementSymbol(atomic_num)

        return f"[{element_symbol}]"

    # 1. Process isotopes (number#atomic_number format)
    result = isotope_pattern.sub(replace_isotope, smarts)

    # 2. Process already converted isotopes (number_element_symbol format)
    result = converted_isotope_pattern.sub(replace_converted_isotope, result)

    # 3. Process atomic numbers
    result = atomic_num_pattern.sub(replace_atomic_num, result)

    return result


def exact_query_from_smiles(smiles: str) -> str:
    """
    Convert SMILES to SMARTS with explicit hydrogen representation.

    This function takes a SMILES string and converts it to SMARTS format where
    implicit hydrogens are explicitly represented. It preserves chirality,
    ring information, and bond order while adding explicit hydrogen counts
    to each atom.

    Args:
        smiles (str): Input SMILES string

    Returns:
        str: SMARTS with explicit hydrogen representation

    Examples:
        >>> hydrogenated_smarts_with_bonds_and_H0("CCC")
        "[CH3]-[CH2]-[CH3]"
        >>> hydrogenated_smarts_with_bonds_and_H0("CCO")
        "[CH3]-[CH2]-[OH1]"
        >>> hydrogenated_smarts_with_bonds_and_H0("c1ccccc1")
        "[cH1]1:[cH1]:[cH]:[cH1]:[cH1]:[cH1]:1"
        >>> hydrogenated_smarts_with_bonds_and_H0("[*:1]C=C")
        "[*:1]-[CH1]=[CH2]"
    """
    # 1) Parse Mol and assign stereochemistry
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES/SMARTS: {smiles}")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    # 2) Record implicit hydrogen counts, aromaticity, isotope indices, and wildcard indices for each atom
    atom_h_counts = []
    atom_aromatic = []
    isotope_indices = []
    wildcard_indices = []
    for i, atom in enumerate(mol.GetAtoms()):
        h_count = atom.GetNumImplicitHs()
        is_aromatic = atom.GetIsAromatic()
        isotope = atom.GetIsotope()

        atom_h_counts.append(h_count)
        atom_aromatic.append(is_aromatic)

        if isotope > 0:
            isotope_indices.append(i)
        if atom.GetAtomicNum() == 0:
            wildcard_indices.append(i)

    # Map wildcard indices to atom map numbers
    wildcard_map = {idx: n + 1 for n, idx in enumerate(wildcard_indices)}

    # 3) Generate basic SMARTS using MolToSmarts (including bond information)
    smarts = Chem.MolToSmarts(mol)
    pt = Chem.GetPeriodicTable()
    pattern = re.compile(
        r"\[#(?P<at>\d+)(?P<st>@@|@)?(?P<charge>[+-])?(?P<explicit_h>H)?(?::(?P<mp>\d+))?\]"
    )
    atom_counter = 0

    def replace_atom_query(match):
        nonlocal atom_counter
        at = int(match.group("at"))
        st = match.group("st") or ""
        charge = match.group("charge") or ""
        explicit_h = match.group("explicit_h") or ""
        mp = match.group("mp") or ""
        # Handle wildcards (atomic number 0)
        if at == 0:
            # Map SMILES index to atom map number
            map_num = wildcard_map.get(atom_counter, atom_counter + 1)
            result = f"[*:{map_num}]"
            atom_counter += 1
            return result
        # Calculate actual atom index while skipping isotope atoms
        actual_atom_idx = atom_counter
        for isotope_idx in isotope_indices:
            if isotope_idx <= actual_atom_idx:
                actual_atom_idx += 1
        if actual_atom_idx < len(atom_h_counts):
            h_count = atom_h_counts[actual_atom_idx]
        else:
            h_count = 0
        if explicit_h:
            h_count += 1
        sym = pt.GetElementSymbol(at)
        if actual_atom_idx < len(atom_aromatic) and atom_aromatic[actual_atom_idx]:
            sym = sym.lower()
        result = f"[{sym}{st}{charge}H{h_count}{':' + mp if mp else ''}]"
        atom_counter += 1
        return result

    result = pattern.sub(replace_atom_query, smarts)
    result = process_isotopes_in_smarts(result, atom_h_counts, isotope_indices)
    return result


class FrequencySampler:
    def __init__(
        self,
        smis: list[str],
        replacement_lib_path: str,
        generate_all_attachments: bool = True,
        ranking_mode: str = "frequency",
        min_frequency: int = 10,
    ):
        self.smis = smis
        self.replacement_lib = pd.read_csv(replacement_lib_path, sep="\t")
        self.mmpa_lib = pd.read_csv(replacement_lib_path, sep="\t", low_memory=False)
        self.generate_all_attachments = generate_all_attachments
        self.ranking_mode = ranking_mode
        self.min_frequency = min_frequency
        return

    def filter_frag(self, num_atoms, broken_frag, max_num_change_atoms=12):
        if broken_frag.GetNumHeavyAtoms() > max_num_change_atoms:
            return False
        elif broken_frag.GetNumHeavyAtoms() > num_atoms / 2:
            return False
        else:
            return True

    def _normalize_sa(self, sa_value):
        # SA comes roughly between 1 (easy) and 10 (hard)
        # Normalize to [0,1], lower is better
        return (10.0 - sa_value) / 9.0

    def _normalize(self, arr):
        arr = pd.Series(arr)
        minv, maxv = arr.min(), arr.max()
        if maxv == minv:
            return [0.0] * len(arr)  # or 1.0, choose consistent with use
        return (arr - minv) / (maxv - minv)

    def sample(self, num_samples: int, verbose: bool = False) -> pd.DataFrame:
        """
        Args:
            num_samples: (int) number of SMILES per input molecule.
            ranking_mode: 'frequency', 'random', or 'rank_filtered_mmpa_qed_sa' (default='frequency')
        Returns:
            generation_df: (pd.DataFrame) DataFrame containing sampled SMILES.
        """

        generation_dict = defaultdict(list)

        for smi in self.smis:
            # 1. break by BRICS rule
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logging.warning(f"Invalid SMILES: {smi}")
                continue
            orig_mw = calc_Mw(mol)
            brics_fragments = BRICS.BRICSDecompose(mol, returnMols=True)
            _brics_fragments = BRICS.BRICSDecompose(mol, returnMols=False)
            # brics_smis = [Chem.MolToSmiles(fragment) for fragment in brics_fragments]

            # 2. filter brics-broken SMILES
            num_atoms = mol.GetNumAtoms()
            allowed_frags = [
                frag for frag in brics_fragments if self.filter_frag(num_atoms, frag)
            ]

            # 3. explore replacement library
            sampled_replacements = None
            for frag in allowed_frags:
                frag_smi = Chem.MolToSmiles(frag)

                # find replacement SMILES
                # replacement_candidates = self.replacement_lib[
                #     self.replacement_lib["OLD-FRAG"].str.contains(frag_smi)
                # ]
                replacement_candidates = self.replacement_lib[
                    self.replacement_lib["OLD-FRAG"] == frag_smi
                ]

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
            sampled_replacements = sampled_replacements.sample(frac=1).reset_index(
                drop=True
            )
            if self.ranking_mode == "rank_filtered_mmpa_qed_sa":
                rep = sampled_replacements.merge(
                    self.mmpa_lib[
                        ["OLD-FRAG", "NEW-FRAG", "qed_mean", "sa_mean", "FREQUENCY"]
                    ],
                    on=["OLD-FRAG", "NEW-FRAG"],
                    how="left",
                    suffixes=("", "_mmpa"),
                )
                # filter FREQUENCY < 10
                rep = rep[rep["FREQUENCY_mmpa"] >= self.min_frequency]
                rep["QED_SA_SCORE"] = rep["qed_mean"].fillna(0.0) + rep[
                    "sa_mean"
                ].fillna(0.0)
                sampled_replacements = rep.sort_values(
                    "QED_SA_SCORE", ascending=False
                ).reset_index(drop=True)
            elif self.ranking_mode == "random":
                # random sampling
                sampled_replacements = sampled_replacements.sample(frac=1).reset_index(
                    drop=True
                )
            else:
                # default: frequency
                sampled_replacements = sampled_replacements.sort_values(
                    by="FREQUENCY", ascending=False
                ).reset_index(drop=True)

            num_gen_mol = 0
            pattern = r"\[(1[0-6]|[1-9])\*\]"
            for _, row in sampled_replacements.iterrows():
                old_frag, new_frag = row["OLD-FRAG"], row["NEW-FRAG"]
                # remove isotope tag, possibly mutiple isotope tages
                old_frag = re.sub(pattern, "[*]", old_frag)
                new_frag = re.sub(pattern, "[*]", new_frag)

                # replace [*] with [H] and ([*]) with ([H])
                old_frag_without_wildcard = re.sub(r"\(\?\[\*\]\)", "([H])", old_frag)
                old_frag_without_wildcard = re.sub(
                    r"\[\*\]", "[H]", old_frag_without_wildcard
                )
                new_frag_without_wildcard = re.sub(r"\(\?\[\*\]\)", "([H])", new_frag)
                new_frag_without_wildcard = re.sub(
                    r"\[\*\]", "[H]", new_frag_without_wildcard
                )

                old_frag_without_wildcard = Chem.MolFromSmiles(
                    old_frag_without_wildcard
                )
                new_frag_without_wildcard = Chem.MolFromSmiles(
                    new_frag_without_wildcard
                )

                old_frag_mw = calc_Mw(old_frag_without_wildcard)
                new_frag_mw = calc_Mw(new_frag_without_wildcard)

                frag_mw_change = new_frag_mw - old_frag_mw

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
                    for i in range(len(_old_frag_broken) - 1):
                        _old_frag_numbered += f"[*:{i + 1}]" + _old_frag_broken[i + 1]

                    _new_frag_numbered = _new_frag_broken[0]
                    for i in range(len(_new_frag_broken) - 1):
                        _new_frag_numbered += (
                            f"[*:{perm[i] + 1}]" + _new_frag_broken[i + 1]
                        )

                    try:
                        _old_frag_numbered = exact_query_from_smiles(_old_frag_numbered)
                        _new_frag_numbered = exact_query_from_smiles(_new_frag_numbered)
                    except:
                        print(
                            f"Making exact reaction SMILES failed: R:{_old_frag_numbered} / P: {_new_frag_numbered}"
                        )
                        continue

                    # _idx = old_idx_matches[0]
                    # _old_frag = _old_frag[:_idx] + f"[*:{i+1}]" + _old_frag[_idx+3:]

                    # _idx = new_idx_matches[-perm[i]]
                    # _new_frag = _new_frag[:_idx] + f"[*:{i+1}]" + _new_frag[_idx+3:]

                    replacement = f"{_old_frag_numbered}>>{_new_frag_numbered}"
                    replacements.append(replacement)

                if not self.generate_all_attachments:
                    random.shuffle(replacements)
                    replacements = replacements[:1]

                gen_mol_list = []
                for replacement in replacements:
                    # 5. generate SMILES
                    rxn = AllChem.ReactionFromSmarts(replacement)
                    gen_mols = rxn.RunReactants((mol,))  # tup of tup
                    gen_mols = list(gen_mols)

                    if len(gen_mols) == 0:
                        continue

                    random.shuffle(gen_mols)
                    gen_mol = gen_mols[0][0]
                    try:
                        Chem.SanitizeMol(gen_mol)
                    except Exception:
                        continue

                    if gen_mol is None:
                        continue

                    gen_mol_mw = calc_Mw(gen_mol)
                    mol_mw_change = gen_mol_mw - orig_mw

                    if abs(mol_mw_change - frag_mw_change) > 2:
                        if verbose:
                            print(
                                f"Warning: Certain part of the original molecule was lost."
                            )
                            print(f"Original molecule: {Chem.MolToSmiles(mol)}")
                            print(f"Generated molecule: {Chem.MolToSmiles(gen_mol)}")
                            print(f"Removal fragment: {old_frag}")
                            print(f"Insertion fragment: {new_frag}")
                            print(f"Original molecule MW: {orig_mw:.2f}")
                            print(f"Generated molecule MW: {gen_mol_mw:.2f}")
                            print(f"Removal fragment MW: {old_frag_mw:.2f}")
                            print(f"Insertion fragment MW: {new_frag_mw:.2f}")
                            print(f"Molecule MW change: {mol_mw_change:.2f}")
                            print(f"Fragment MW change: {frag_mw_change:.2f}")
                            print(
                                f"Difference: {abs(mol_mw_change - frag_mw_change):.2f}"
                            )
                            print()
                        continue

                    gen_mol_list.append(gen_mol)

                    logp = calc_logP(gen_mol)
                    mw = calc_Mw(gen_mol)
                    qed = calc_QED(gen_mol)
                    sa = calc_SAscore(gen_mol)

                    generation_dict["INPUT-MOL-SMI"].append(smi)
                    generation_dict["GEN-MOL-SMI"].append(Chem.MolToSmiles(gen_mol))
                    generation_dict["OLD-FRAG"].append(old_frag)
                    generation_dict["NEW-FRAG"].append(new_frag)
                    generation_dict["USED-REPLACEMENT-SMILES"].append(replacement)
                    generation_dict["USED-REPLACEMENT-FREQ"].append(row["FREQUENCY"])
                    generation_dict["LOGP"].append(logp)
                    generation_dict["MW"].append(mw)
                    generation_dict["QED"].append(qed)
                    generation_dict["SA"].append(sa)
                    num_gen_mol += 1

                    if num_gen_mol == num_samples:
                        break
                if num_gen_mol == num_samples:
                    if verbose:
                        print(
                            f"Successfully generated {num_gen_mol} samples for {smi}."
                        )
                    break
            else:
                if verbose:
                    print(
                        f"Not enough samples generated for {smi}. "
                        f"Generated {num_gen_mol} samples."
                    )
                    continue

        # 6. create DataFrame
        generation_df = pd.DataFrame(generation_dict)
        return generation_df


if __name__ == "__main__":
    import random

    seed = 42
    set_seed(seed)

    num_samples = 1000
    with open("/home/hwkim/DeepBioisostere/data/chembl.smi", "r") as f:
        chembl_smis = [line.strip() for line in f.readlines()]
    random.shuffle(chembl_smis)
    test_smis = chembl_smis[:num_samples]
    print(f"Testing on {len(test_smis)} molecules with seed {seed}.")

    sampler = FrequencySampler(
        smis=test_smis,
        replacement_lib_path="/home/share/DATA/swkim/DeepBioisostere/replacement_library.csv",
        generate_all_attachments=True,
    )
    gen_df = sampler.sample(num_samples=100, verbose=True)
    gen_df.to_csv("sampled_molecules.csv", index=False, sep="\t")
