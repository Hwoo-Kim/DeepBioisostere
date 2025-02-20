"""
1. MW <= 800 Da                            -> filtered in filter_molecule()
2. 0 <= ChEMBL <= 10_000 nM (pChEMBL >= 5) -> filtered in query
3. Without "."                             -> filtered in filter_molecule()
"""
import sqlite3
import argparse
from multiprocessing import Pool
from typing import List, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt


def filter_molecule(
    molecule_id: int,
    smiles: str,
    mw_criteria: int = 800,
) -> Tuple[int, bool]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return molecule_id, False
    elif ExactMolWt(mol) > mw_criteria:
        return molecule_id, False
    elif "." in smiles:
        return molecule_id, False
    return molecule_id, True


def mp(nprocs: int, data: List[Tuple[int, str]]) -> List[Tuple[int, bool]]:
    pool = Pool(nprocs)
    results = pool.starmap_async(filter_molecule, data)
    results.wait()
    pool.close()
    pool.join()
    return results.get()


def create_filtered_table(db_name: str) -> None:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS filtered_molecule_id (
            molecule_id INTEGER PRIMARY KEY,
            brics_bond_indice TEXT,
            FOREIGN KEY (molecule_id) REFERENCES molecule (molecule_id)
        );
    """)

    conn.commit()
    conn.close()


def save_filtered_data(db_name: str, filtered_molecule_ids: List[int]) -> None:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.executemany(
        """
        INSERT OR IGNORE INTO filtered_molecule_id (molecule_id, brics_bond_indice)
        VALUES (?, ?);
        """,
        ((m, None) for m in filtered_molecule_ids)
    )

    conn.commit()
    conn.close()


def main():
    db_name = "chembl_activities_250115.db"

    conn = sqlite3.connect(db_name)
    query = """
        SELECT molecule.molecule_id, molecule.smiles
        FROM molecule
        JOIN activity ON molecule.molecule_id = activity.molecule_id
        WHERE activity.pchembl_value >= 5.0;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    data = list(df.itertuples(index=False, name=None))

    filtered_results = mp(16, data)
    filtered_molecule_ids = [mol_id for mol_id, keep in filtered_results if keep]

    create_filtered_table(db_name)
    save_filtered_data(db_name, filtered_molecule_ids)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_known_args()
    main()