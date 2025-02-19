from functools import partial
from itertools import combinations
from multiprocessing import cpu_count, Pool, Manager
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, Draw, Mol
from tqdm.contrib.concurrent import process_map

SMILES = str
ATOM_INDEX = int

COLS = [
    "REF-CID",
    "PRB-CID",
    "REF-SMI",
    "PRB-SMI",
    "REF-ASSAY-ID",
    "PRB-ASSAY-ID",
    "REF-TARGET-ID",
    "PRB-TARGET-ID",
    "KEY-FRAG-ATOM-INDICE",
    "ATOM-FRAG-INDICE",
    "OLD-FRAG",
    "NEW-FRAG",
    "BRICS-EDGE-INDICE",
    "CUTTING-EDGE-INDICE",
]


def get_pairs_per_rest_frag(rest_frag: str, df: pd.DataFrame):
    rest_frag_subset = df.loc[df["REST-FRAG"] == rest_frag]
    pairs = []
    for ref, prb in combinations(rest_frag_subset.iterrows(), 2):
        ref_idx, ref_series = ref
        prb_idx, prb_series = prb

        ref_cid = ref_series.get("CID")
        prb_cid = prb_series.get("CID")

        ref_smi = ref_series.get("SMI")
        prb_smi = prb_series.get("SMI")

        if (ref_cid == prb_cid) or (ref_smi == prb_smi):
            continue

        data = (
            ref_cid,
            prb_cid,
            ref_smi,
            prb_smi,
            ref_series.get("ASSAY-ID"),
            prb_series.get("ASSAY-ID"),
            ref_series.get("TARGET-ID"),
            prb_series.get("TARGET-ID"),
            ref_series.get("KEY-FRAG-ATOM-INDICE"),
            ref_series.get("ATOM-FRAG-INDICE"),
            ref_series.get("KEY-FRAG"),
            prb_series.get("KEY-FRAG"),
            ref_series.get("BRICS-EDGE-INDICE"),
            prb_series.get("CUTTING-EDGE-INDICE"),
        )
        pairs.append(data)
        data = (
            prb_cid,
            ref_cid,
            prb_smi,
            ref_smi,
            prb_series.get("ASSAY-ID"),
            ref_series.get("ASSAY-ID"),
            prb_series.get("TARGET-ID"),
            ref_series.get("TARGET-ID"),
            prb_series.get("KEY-FRAG-ATOM-INDICE"),
            prb_series.get("ATOM-FRAG-INDICE"),
            prb_series.get("KEY-FRAG"),
            ref_series.get("KEY-FRAG"),
            prb_series.get("BRICS-EDGE-INDICE"),
            ref_series.get("CUTTING-EDGE-INDICE"),
        )
        pairs.append(data)
    return pairs


def get_pairs_per_num_frags(df: pd.DataFrame, num_frags: int):
    """\
    CID,SMI,KEY-FRAG,REST-FRAG,NUM-FRAGS
    1. num_frags should be same
    2. rest_frags should be same
    3. cids should be different
    """
    num_frags_subset = df.loc[df["NUM-FRAGS"] == num_frags]
    rest_frags = np.unique(num_frags_subset["REST-FRAG"].values)

    pairs = []
    worker = partial(get_pairs_per_rest_frag, df=num_frags_subset)
    with Pool(cpu_count()) as pool:
        results = pool.map(worker, rest_frags)
    for result in results:
        pairs += result
    return pairs


def main():
    df = pd.read_csv(args.fragment_file, low_memory=False)
    pairs = get_pairs_per_num_frags(df, args.num_frags)

    with args.result_file.open("w") as w:
        w.write("\t".join(COLS) + "\n")
        for pair in pairs:
            w.write("\t".join(pair) + "\n")
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Fragment library argparser")
    parser.add_argument("--fragment_file", type=Path, default="./test.csv")
    parser.add_argument("--num_frags", type=int)
    parser.add_argument("--result_file", type=Path)
    args, _ = parser.parse_known_args()

    main()
