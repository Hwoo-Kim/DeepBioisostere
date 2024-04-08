import argparse
from itertools import repeat
from multiprocessing import cpu_count, Pool
from pathlib import Path
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt, qed
from rdkit.Chem.Crippen import MolLogP


def get_feature(line: str, feature: str) -> Optional[float]:
    cid, _, smi = line.split()
    mol = Chem.MolFromSmiles(smi)
    value = None
    if feature == "qed":
        value = qed(mol)
    elif feature == "mw":
        value = ExactMolWt(mol)
    elif feature == "logp":
        value = MolLogP(mol)
    return cid, value


def mp(nprocs: int, lines: List[str], feature: str) -> List[bool]:
    pool = Pool(nprocs)
    variables = zip(lines, repeat(feature))
    results = pool.starmap_async(get_feature, variables)
    results.wait()
    pool.close()
    pool.join()

    data = results.get()
    return data


def main():
    with args.input_file.open("r") as f:
        lines = f.readlines()[1:]

    args.save_dir.mkdir(exist_ok=True, parents=True)
    output_file = args.save_dir / f"{args.feature}.txt"

    data = mp(cpu_count(), lines, args.feature)
    with output_file.open("w") as f:
        for (cid, value) in data:
            if value is not None:
                f.write(f"{cid}\t{str(round(value, 3))}\n")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_file",
        default="./filtered_chembl.txt",
        type=Path,
        help="ChEMBL data parsed result",
    )
    parser.add_argument(
        "-f",
        "--feature",
        type=str,
        help="feature",
        choices=["qed", "mw", "logp"],
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=".",
        help="Save directory",
    )
    args, _ = parser.parse_known_args()

    main()
