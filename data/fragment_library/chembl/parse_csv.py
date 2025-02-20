import glob
import re
import sqlite3
from itertools import repeat
from multiprocessing import Pool
from typing import Any, Iterable, List, Optional

import pandas as pd


def create_tables(db_name: str) -> None:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS molecule (
            molecule_id INTEGER PRIMARY KEY,
            smiles TEXT
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity (
            molecule_id INTEGER,
            pchembl_value REAL,
            assay_chembl_id INTEGER,
            target_chembl_id INTEGER,
            FOREIGN KEY (molecule_id) REFERENCES molecule (molecule_id)
        );
    """)
    cursor.execute("""
        PRAGMA journal_mode = WAL;
    """)

    conn.commit()
    conn.close()


def clean_chembl_id(value: str) -> Optional[int]:
    match = re.match(r"CHEMBL(\d+)", value)
    return int(match.group(1)) if match else None


def read_csv(
    file: str,
    columns: Optional[pd.Index] = None,
    regex: str = "DOWNLOAD-.+=\_part\d+.csv",
) -> pd.DataFrame:
    if re.match(re.compile(regex), file):
        header = None
    else:
        header = "infer"

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

    df["Molecule ChEMBL ID"] = df["Molecule ChEMBL ID"].apply(clean_chembl_id)
    df["Assay ChEMBL ID"] = df["Assay ChEMBL ID"].apply(clean_chembl_id)
    df["Target ChEMBL ID"] = df["Target ChEMBL ID"].apply(clean_chembl_id)

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


def save_to_sqlite(df: pd.DataFrame, db_name: str) -> None:
    conn = sqlite3.connect(db_name)

    molecule_df = df[["Molecule ChEMBL ID", "Smiles"]].drop_duplicates()
    molecule_df.columns = ["molecule_id", "smiles"]
    molecule_df.to_sql("molecule", conn, if_exists="append", index=False)

    activity_df = df.drop(columns=["Smiles"])
    activity_df.columns = ["molecule_id", "pchembl_value", "assay_chembl_id", "target_chembl_id"]
    activity_df.to_sql("activity", conn, if_exists="append", index=False)

    conn.close()


def main():
    files = sorted(glob.glob("*.csv"))
    whole_cols = get_whole_columns(files)
    args = zip(files, repeat(whole_cols))

    dfs = mp(16, args)
    df = pd.concat(dfs, ignore_index=True)
    df.sort_values(by=[COLS[0]])

    db_name = "chembl_activities_250115.db"

    create_tables(db_name)
    save_to_sqlite(df, db_name)
    return


if __name__ == "__main__":
    COLS = ["Molecule ChEMBL ID", "pChEMBL Value", "Assay ChEMBL ID", "Target ChEMBL ID", "Smiles"]
    main()
