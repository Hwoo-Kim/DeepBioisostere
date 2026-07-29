import argparse
from pathlib import Path
import pandas as pd
import glob

# These are the columns expected in the processed_*.txt files
INPUT_COLS = [
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


def process_file(input_file: Path, cid_to_pchembl: pd.DataFrame, output_dir: Path):
    """
    Filters a single bioisostere pair file based on the pChEMBL value difference.
    """
    print(f"\nProcessing {input_file.name}...")

    try:
        # Explicitly set dtype for CID columns to string to prevent merge errors
        pair_df = pd.read_csv(
            input_file,
            sep="\t",
            names=INPUT_COLS,
            low_memory=False,
            dtype={'REF-CID': str, 'PRB-CID': str}
        )
    except Exception as e:
        print(f"Error loading {input_file.name}: {e}")
        return

    # Merge with pair_df to get REF-PCHEMBL
    # The 'CID' column in cid_to_pchembl is already a string
    merged_df = pd.merge(
        pair_df,
        cid_to_pchembl,
        how='left',
        left_on='REF-CID',
        right_on='CID'
    ).rename(columns={'PCHEMBL': 'REF-PCHEMBL'}).drop(columns=['CID'])

    # Merge again to get PRB-PCHEMBL
    merged_df = pd.merge(
        merged_df,
        cid_to_pchembl,
        how='left',
        left_on='PRB-CID',
        right_on='CID'
    ).rename(columns={'PCHEMBL': 'PRB-PCHEMBL'}).drop(columns=['CID'])

    # Handle cases where CIDs might not be in the ChEMBL file by dropping them
    initial_rows = len(merged_df)
    merged_df.dropna(subset=['REF-PCHEMBL', 'PRB-PCHEMBL'], inplace=True)
    if len(merged_df) < initial_rows:
        print(f"Dropped {initial_rows - len(merged_df)} rows due to missing pChEMBL values.")

    # Convert pChEMBL columns to numeric, coercing errors to NaN
    merged_df['REF-PCHEMBL'] = pd.to_numeric(merged_df['REF-PCHEMBL'], errors='coerce')
    merged_df['PRB-PCHEMBL'] = pd.to_numeric(merged_df['PRB-PCHEMBL'], errors='coerce')
    merged_df.dropna(subset=['REF-PCHEMBL', 'PRB-PCHEMBL'], inplace=True)

    # Apply the pChEMBL difference filter
    # pchembl_diff_threshold = 1.36
    pchembl_diff_threshold = 1.0
    filtered_df = merged_df[abs(merged_df['REF-PCHEMBL'] - merged_df['PRB-PCHEMBL']) <= pchembl_diff_threshold].copy()

    print(f"Original pairs in {input_file.name}: {len(pair_df)}")
    print(f"Pairs after filtering: {len(filtered_df)}")

    # Drop the temporary pChEMBL columns to match original format
    final_df = filtered_df.drop(columns=['REF-PCHEMBL', 'PRB-PCHEMBL'])

    # Define output path
    output_file = output_dir / f"{input_file.stem}_pchembl_filtered.txt"
    
    print(f"Saving filtered data to {output_file}...")
    # Save the filtered dataframe with the same format as the input
    final_df.to_csv(output_file, sep='\t', index=False, header=False)


def main(args):
    """
    Finds all processed_*.txt files in the input directory and filters them.
    """
    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ChEMBL data to get pChEMBL values...")
    try:
        # Define the actual column names from the file
        chembl_file_cols = [
            'Molecule ChEMBL ID',
            'pChEMBL Value',
            'Assay ChEMBL ID',
            'Target ChEMBL ID',
            'Smiles'
        ]

        # Read the space-separated ChEMBL file, skipping the original header
        # and providing the correct column names.
        chembl_df = pd.read_csv(
            args.chembl_file,
            sep='\s+',
            engine='python',
            names=chembl_file_cols,    # Manually assign column names
            header=0,                  # Skip the first row (the original header)
            dtype={'Molecule ChEMBL ID': str}
        )
    except FileNotFoundError:
        print(f"Error: ChEMBL file not found at {args.chembl_file}")
        return
    except Exception as e:
        print(f"Error loading {args.chembl_file}: {e}")
        return

    # Create a mapping from CID to pChEMBL value once
    # Use the correct column names from the ChEMBL file
    cid_to_pchembl = chembl_df[['Molecule ChEMBL ID', 'pChEMBL Value']].copy()
    cid_to_pchembl.rename(columns={
        'Molecule ChEMBL ID': 'CID',
        'pChEMBL Value': 'PCHEMBL'
    }, inplace=True)
    cid_to_pchembl.drop_duplicates(subset=['CID'], keep='first', inplace=True)

    # Find all 'processed_*.txt' files in the input directory
    search_pattern = str(args.input_dir / "processed_*.txt")
    input_files = [Path(f) for f in glob.glob(search_pattern)]

    if not input_files:
        print(f"No 'processed_*.txt' files found in {args.input_dir}")
        return

    print(f"Found {len(input_files)} files to process.")

    for input_file in input_files:
        process_file(input_file, cid_to_pchembl, args.output_dir)

    print("\nAll files processed. Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter bioisostere pairs in a directory based on pChEMBL value difference.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Path to the directory containing 'processed_*.txt' files."
    )
    parser.add_argument(
        "--chembl_file",
        type=Path,
        required=True,
        help="Path to the original ChEMBL data file containing CID and pChEMBL values."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to the directory to save the filtered output files."
    )
    args = parser.parse_args()

    main(args)
