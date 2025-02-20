import sqlite3
import pandas as pd
import multiprocessing
from functools import partial
from typing import List, Tuple


def fetch_activity_data(db_name: str) -> pd.DataFrame:
    """ Fetch molecule_id, pchembl_value, and assay_chembl_id from the activity table. """
    conn = sqlite3.connect(db_name)
    query = "SELECT molecule_id, pchembl_value, assay_chembl_id FROM activity"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def process_group(assay_group: Tuple[int, pd.DataFrame], p: float) -> pd.DataFrame:
    """ Process a single assay_chembl_id group to classify high/low pchembl_values. """
    assay_id, group = assay_group

    percentile_high = 1 - (p / 100)
    percentile_low = p / 100

    threshold_high = group["pchembl_value"].quantile(percentile_high)
    threshold_low = group["pchembl_value"].quantile(percentile_low)

    high_group = group[group["pchembl_value"] >= threshold_high].copy()
    high_group["high_or_low"] = "high"

    low_group = group[group["pchembl_value"] <= threshold_low].copy()
    low_group["high_or_low"] = "low"

    return pd.concat([high_group, low_group], ignore_index=True)


def classify_high_low(df: pd.DataFrame, p: float) -> pd.DataFrame:
    """ Use multiprocessing to classify each assay_chembl_id group into high/low pchembl_value categories. """
    if not (0 < p < 50):
        raise ValueError("Percentage n must be between 0 and 50 (exclusive).")

    num_cores = args.nprocs if args.nprocs else multiprocessing.cpu_count()  # Get available CPU cores
    grouped = df.groupby("assay_chembl_id")
    grouped = {key: group for key, group in grouped if len(group) > int(100/p)}
    with multiprocessing.Pool(num_cores) as pool:
        results = pool.map(partial(process_group, p=p), grouped.items())

    return pd.concat(results, ignore_index=True)[["molecule_id", "assay_chembl_id", "high_or_low"]]


def save_to_sqlite(df: pd.DataFrame, db_name: str, n: float) -> None:
    """ Save the classified data to SQLite under a dynamically named table. """
    table_name = f"group_extreme_{int(n)}"
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            molecule_id INTEGER,
            assay_chembl_id INTEGER,
            high_or_low TEXT,
            PRIMARY KEY (molecule_id, assay_chembl_id),
            FOREIGN KEY (molecule_id) REFERENCES molecule (molecule_id)
        );
    """)

    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()


def main():
    """ Main function to classify pchembl_value data using multiprocessing and store results in SQLite. """
    df = fetch_activity_data(args.db_file)
    result_df = classify_high_low(df, args.percentile)
    save_to_sqlite(result_df, args.db_file, args.percentile)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classify molecule_id into high and low categories based on pchembl_value percentiles.")
    parser.add_argument("--db_file", type=str, default="chembl/chembl_activities_250115.db")
    parser.add_argument("-p", "--percentile", type=int, help="Percentage (p) for selecting extreme values (0 < p < 50).")
    parser.add_argument("-n", "--nprocs", type=int, default=0)
    args = parser.parse_args()
    main()
