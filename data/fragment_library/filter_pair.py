from collections import defaultdict
from contextlib import ExitStack
from functools import partial
from multiprocessing import cpu_count, Lock, Pool, synchronize
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem

from make_frag_db import remove_dummy_atom_from_mol
from parse_db import COLS

SMILES = str
CID = str
ATOM_INDEX = int


def get_shortest_smi(frag_smis: List[SMILES]) -> SMILES:
    len_chars = list(map(len, frag_smis))
    shortest_smi = frag_smis[len_chars.index(min(len_chars))]
    return shortest_smi


def get_shortest_indice(key_frag_atom_indice: List[List[ATOM_INDEX]]) -> int:
    lens = list(map(len, key_frag_atom_indice))
    return lens.index(min(lens))


def get_unique_min_substruct(
    key_frag_atom_indice: List[List[ATOM_INDEX]],
    frag_smis: List[SMILES],
    substruct_smis: List[SMILES],
) -> None:
    shortest_idx = get_shortest_indice(key_frag_atom_indice)
    shortest_smi = frag_smis[shortest_idx]
    shortest_mol = Chem.MolFromSmiles(shortest_smi)
    shortest_mol = remove_dummy_atom_from_mol(shortest_mol)
    shortest_indice = key_frag_atom_indice[shortest_idx]

    key_frag_atom_indice.remove(shortest_indice)
    frag_smis.remove(shortest_smi)

    for indice, frag_smi in zip(list(key_frag_atom_indice), list(frag_smis)):
        if len(indice) == len(shortest_indice):
            continue
        else:
            frag_mol = Chem.MolFromSmiles(frag_smi)
            frag_mol = remove_dummy_atom_from_mol(frag_mol)
            if frag_mol.HasSubstructMatch(shortest_mol):
                key_frag_atom_indice.remove(indice)
                frag_smis.remove(frag_smi)

    substruct_smis.append(shortest_smi)
    if frag_smis:
        get_unique_min_substruct(key_frag_atom_indice, frag_smis, substruct_smis)
    return


def get_min_substructs_in_common_cids(item: Tuple[CID, Dict[str, str]]) -> pd.DataFrame:
    cids, row_dic = item
    ref_cid, prb_cid = cids.split(",")

    partial_df = pd.DataFrame.from_dict(row_dic)
    old_frag_smis = partial_df["OLD-FRAG"].values.tolist()
    key_frag_atom_indice = partial_df["KEY-FRAG-ATOM-INDICE"].values.tolist()
    key_frag_atom_indice = [
        list(map(int, indice.split(","))) for indice in key_frag_atom_indice
    ]

    substruct_smis = []
    get_unique_min_substruct(key_frag_atom_indice, old_frag_smis, substruct_smis)
    flag = np.zeros(partial_df.shape[0]).astype(bool)
    for substruct_smi in substruct_smis:
        flag |= partial_df["OLD-FRAG"] == substruct_smi
    partial_df = partial_df[flag]
    return partial_df


def write_min_substructs_in_common_cids(
    item: Tuple[CID, Dict[str, str]], file: Path
) -> None:
    df = get_min_substructs_in_common_cids(item)
    data = df.to_dict("split")["data"]
    with ExitStack() as stack, file.open("a") as f:
        lock = globals().get("lock", False)
        if lock:
            stack.enter_context(lock)
        for elems in data:
            f.write("\t".join(elems) + "\n")
    return


def filter_if_same_old_new_frags(df: pd.DataFrame) -> pd.DataFrame:
    values = df.values
    old_frags = values[:, df.keys().values.tolist().index("OLD-FRAG")]
    new_frags = values[:, df.keys().values.tolist().index("NEW-FRAG")]
    df = df[old_frags != new_frags]
    return df


def divide_dic(dic: Dict[str, Any]) -> List[Dict[str, Any]]:
    length = len(dic) // 100
    dics = []
    for idx, (key, value) in enumerate(dic.items()):
        if idx % length == 0:
            try:
                dics.append(new_dic)
            except UnboundLocalError:
                pass
            new_dic = {}
        new_dic[key] = value
    dics.append(new_dic)
    return dics


def init_pool(lock_: synchronize.Lock):
    global lock
    lock = lock_


def main():
    lock = Lock()

    df = pd.read_csv(args.pair_file, delimiter="\t", low_memory=False)
    df = filter_if_same_old_new_frags(df)
    df = df[df["REF-ASSAY-ID"] == df["PRB-ASSAY-ID"]]               # Filter w/ same assay id
    assay_id = df["REF-ASSAY-ID"]
    df = df.drop(columns=["REF-ASSAY-ID", "PRB-ASSAY-ID"])
    df.insert(4, "ASSAY-ID", assay_id)                              # Merge two columns
    
    records = df.to_dict("records")
    dic = defaultdict(list)
    for record in records:
        key = f"{record['REF-CID']},{record['PRB-CID']}"
        dic[key].append(record)

    with Pool(cpu_count(), initializer=init_pool, initargs=(lock,)) as pool:
        worker = partial(write_min_substructs_in_common_cids, file=args.result_file)
        pool.map(worker, dic.items())
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Filter parser")
    parser.add_argument("--pair_file", type=Path)
    parser.add_argument("--result_file", type=Path)
    args, _ = parser.parse_known_args()

    main()
