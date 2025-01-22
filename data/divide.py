import random
from functools import partial
from multiprocessing import cpu_count, Pool
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import pandas as pd

COLS = [
    "REF-CID",
    "PRB-CID",
    "REF-SMI",
    "PRB-SMI",
    "ASSAY-ID",
    "REF-TARGET-ID",
    "PRB-TARGET-ID",
    "KEY-FRAG-ATOM-INDICE",
    "ATOM-FRAG-INDICE",
    "OLD-FRAG",
    "NEW-FRAG",
    "ALLOWED-ATTACHMENT"
]


def read_files(file: Path, seed: int, cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(
        file,
        low_memory=False,
        delimiter="\t",
        names=cols,
    )
    return df


def parallel_df_filtering(
    df: pd.DataFrame, keys: List[str], valids: Set[str], nprocs: int = 0
) -> pd.DataFrame:
    num_processes = cpu_count() if not nprocs else nprocs
    df_split = np.array_split(df, num_processes)
    with Pool(num_processes) as pool:
        worker = partial(filter_df, keys=keys, valids=valids)
        df = pd.concat(pool.map(worker, df_split))
    df.reset_index(inplace=True, drop=True)
    return df


def filter_df(df: pd.DataFrame, keys: List[str], valids: Set[str]) -> pd.DataFrame:
    from collections import Counter

    filter_condition = np.ones(df.shape[0]).astype(bool)
    for key in keys:
        filter_condition &= df.apply(func=lambda elem: elem[key] in valids, axis=1)
    df = df[filter_condition]
    return df


def get_train_val_test(
    data: np.ndarray, train_ratio: float = 0.8, val_ratio: float = 0.1
) -> Tuple[np.ndarray]:
    num_data = len(data)
    num_train = int(train_ratio * num_data)
    num_val = int(val_ratio * num_data)
    return (
        data[:num_train],
        data[num_train : num_train + num_val],
        data[num_train + num_val :],
    )


def get_unique(df: pd.DataFrame, key: str) -> np.ndarray:
    unique_df = df.drop_duplicates(subset=[key])
    return unique_df[key].values


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)

    dfs = []
    print(args.pair_files)
    for file in args.pair_files:
        df = read_files(file, args.seed, args.cols)
        dfs.append(df)
    df = pd.concat(dfs, axis="index", ignore_index=True)

    unique_frags = get_unique(df, key="NEW-FRAG")
    np.random.shuffle(unique_frags)
    train_frags, val_frags, test_frags = get_train_val_test(
        unique_frags, train_ratio=0.8, val_ratio=0.1
    )

    train_df = parallel_df_filtering(
        df,
        keys=["NEW-FRAG"],
        valids=set(train_frags),
        nprocs=args.nprocs,
    )
    train_df["DATATYPE"] = "train"
    val_df = parallel_df_filtering(
        df,
        keys=["NEW-FRAG"],
        valids=set(val_frags),
        nprocs=args.nprocs,
    )
    val_df["DATATYPE"] = "val"
    test_df = parallel_df_filtering(
        df,
        keys=["NEW-FRAG"],
        valids=set(test_frags),
        nprocs=args.nprocs,
    )
    test_df["DATATYPE"] = "test"

    df = pd.concat([train_df, val_df, test_df], axis="index", ignore_index=True)

    args.data_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(args.data_dir / "data.csv", sep="\t", index_label="INDEX")
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate keys for training")
    parser.add_argument(
        "pair_files",
        nargs="+",
        metavar="PAIR_FILE",
        help="Processed pair file paths",
        type=Path,
    )
    parser.add_argument(
        "--nprocs",
        help="number of the processes",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="output data directory",
    )
    parser.add_argument("--cols", type=str, nargs="+", default=COLS)
    parser.add_argument("--seed", type=int, help="seed", default=0)
    args, _ = parser.parse_known_args()

    main()
