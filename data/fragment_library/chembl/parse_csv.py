import glob
import re
from itertools import repeat
from multiprocessing import Pool
from typing import Any, Iterable, List, Optional

import pandas as pd


def read_csv(
    file: str,
    columns: Optional[pd.Index] = None,
    regex: str = "DOWNLOAD-.+=\_part\d+.csv",
) -> pd.DataFrame:
    if re.match(re.compile(regex), file):
        header = None
    else:
        header = "infer"  # first file

    df = pd.read_csv(
        file,
        delimiter=";",
        low_memory=False,
        na_filter=True,
        header=header,
    )
    if header is None:
        df.columns = columns

    df = df[COLS]
    df.dropna(axis=0, how="any", inplace=True)
    return df


def mp(nprocs: int, args: Iterable) -> List[pd.DataFrame]:
    pool = Pool(nprocs)
    results = pool.starmap_async(read_csv, args)
    pool.close()
    pool.join()
    data = results.get()
    return data


def get_whole_columns(files: List[str], regex: str = "DOWNLOAD-.+=\.csv") -> pd.Index:
    file = [file for file in files if re.match(re.compile(regex), file)][0]
    df = pd.read_csv(file, delimiter=";", low_memory=False, nrows=1)
    return df.columns


def main():
    files = sorted(glob.glob("*.csv"))
    whole_cols = get_whole_columns(files)
    args = zip(files, repeat(whole_cols))

    dfs = mp(16, args)
    df = pd.concat(dfs, ignore_index=True)
    df.sort_values(by=[COLS[0]])

    data = [df[col] for col in COLS]
    fn = "chembl_activities_250115.txt"
    with open(fn, "w") as w:
        line = "\t".join([col.ljust(20) for col in COLS])
        w.write(line + "\n")
        for d in zip(*data):
            d = list(map(str, list(d)))
            d = [dd.ljust(20) for dd in d]
            w.write("\t".join(d) + "\n")
    return


if __name__ == "__main__":
    COLS = ["Molecule ChEMBL ID", "pChEMBL Value", "Assay ChEMBL ID", "Target ChEMBL ID", "Smiles"]
    main()
