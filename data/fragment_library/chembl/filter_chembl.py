"""
1. MW <= 800 Da
2. 0 <= pChEMBL <= 10_000 nM
3. Without "."
"""
import argparse
from multiprocessing import Pool
from pathlib import Path
from typing import List

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt


def filter_line(
    line: str,
    mw_criteria: int = 800,
    pchembl_criteria: int = 5,
) -> bool:
    try:
        cid, pchembl, assayid, targetid, smi = line.split()
    except ValueError:
        return True  # First row
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    elif ExactMolWt(mol) > mw_criteria:
        return False
    elif float(pchembl) <= pchembl_criteria:
        return False
    elif "." in smi:
        return False
    return True


def mp(nprocs: int, lines: List[str]) -> List[bool]:
    pool = Pool(nprocs)
    results = pool.map_async(filter_line, lines)
    results.wait()
    pool.close()
    pool.join()

    data = results.get()
    return data


def main():
    with args.input_file.open("r") as f:
        lines = f.readlines()
    data = mp(16, lines)

    done = set()
    with args.output_file.open("w") as f:
        for ok, line in zip(data, lines):
            if ok and (cid := line.split()[0]) not in done:
                f.write(line)
                done.add(cid)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_file",
        default="./chembl_activities_250115.txt",
        type=Path,
        help="ChEMBL data parsed result",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        default="./filtered_chembl.txt",
        type=Path,
        help="Filter result file",
    )
    args, _ = parser.parse_known_args()

    main()
