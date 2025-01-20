import argparse
import os
import os.path as op
import pickle
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Tuple

from tqdm import tqdm

# import time


proj_dir = op.dirname(op.dirname(op.dirname(op.realpath(__file__))))
sys.path.append(proj_dir)

from multiprocessing import cpu_count

# import numpy as np
import pandas
import torch
import torch.multiprocessing as mp
from rdkit import Chem

from scripts.brics import brics
from scripts.feature import from_mol


class FragLibProcessor:
    BRICS_TYPE_MAPPER = brics.BRICSTypeMapper(inp_type=int)

    def __init__(self, args: argparse.Namespace):
        print("Initializing FragLibProcessor...")
        # Read arguments
        # data_dir = args.data_dir if args.data_dir[-1] != "/" else args.data_dir[:-1]
        self.data_path = op.join(args.data_dir, "data.csv")
        self.save_dir = args.data_dir
        self.nprocs = cpu_count() if not args.nprocs else args.nprocs

        # Save path setting
        self.data_df_save_path = op.join(self.save_dir, "processed_data.csv")
        self.fragment_library_path = op.join(self.save_dir, "fragment_library.csv")
        self.frag_features_path = op.join(self.save_dir, "frag_features.pkl")
        self.frag_brics_maskings_path = op.join(
            self.save_dir, "frag_brics_maskings.pkl"
        )
        self.frag_tmp_dir = op.join(self.save_dir, "frags_tmp")
        os.mkdir(self.frag_tmp_dir)

    def filter_data(self):
        data_in_str = self.data_df.values.tolist()
        for idx, line in enumerate(data_in_str):
            data_in_str[idx] = ".".join(list(map(str, line)))

        passed_row_idxs = [
            not ("2H" in line or "3H" in line) and "nan" not in line
            for line in data_in_str
        ]
        self.data_df = self.data_df[passed_row_idxs]
        return True

    # Main method
    def process_data(self):
        # Read input data
        print("Reading data...")
        self.data_df = pandas.read_csv(
            self.data_path, sep="\t", dtype={"DATA-TYPE": str, "BRICS-TYPE": str}
        )
        self.data_df.drop(columns="INDEX", inplace=True)

        # Filter out undesired data rows
        self.filter_data()
        print("Done.")

        # Calculate fragment frequencies
        print("Processing fragments...")
        new_train_frags_freq = self.data_df[self.data_df["DATATYPE"] == "train"][
            "NEW-FRAG"
        ].value_counts()
        new_val_frags_freq = self.data_df[self.data_df["DATATYPE"] == "val"][
            "NEW-FRAG"
        ].value_counts()
        new_test_frags_freq = self.data_df[self.data_df["DATATYPE"] == "test"][
            "NEW-FRAG"
        ].value_counts()
        old_frags_freq = self.data_df["OLD-FRAG"].value_counts()

        # Dataframes with frequency information
        new_train_df = pandas.DataFrame.from_dict(
            {
                "FRAG-SMI": new_train_frags_freq.keys(),
                "FRAG-FREQ": new_train_frags_freq.values,
                "NEW-OLD": ["new"] * len(new_train_frags_freq),
                "DATA-TYPE": ["train"] * len(new_train_frags_freq),
            }
        )
        new_val_df = pandas.DataFrame.from_dict(
            {
                "FRAG-SMI": new_val_frags_freq.keys(),
                "FRAG-FREQ": new_val_frags_freq.values,
                "NEW-OLD": ["new"] * len(new_val_frags_freq),
                "DATA-TYPE": ["val"] * len(new_val_frags_freq),
            }
        )
        new_test_df = pandas.DataFrame.from_dict(
            {
                "FRAG-SMI": new_test_frags_freq.keys(),
                "FRAG-FREQ": new_test_frags_freq.values,
                "NEW-OLD": ["new"] * len(new_test_frags_freq),
                "DATA-TYPE": ["test"] * len(new_test_frags_freq),
            }
        )

        new_frags_df = pandas.concat(
            [
                new_train_df,
                new_val_df,
                new_test_df,
            ],
            ignore_index=True,
        )

        old_frags_df = pandas.DataFrame.from_dict(
            {
                "FRAG-SMI": old_frags_freq.keys(),
                "FRAG-FREQ": old_frags_freq.values,
                "NEW-OLD": ["old"] * len(old_frags_freq),
            }
        )

        # Obtain new fragment indices of the training data df acrrording to the new fragment library
        new_frag_smis_in_frag_lib = new_frags_df["FRAG-SMI"].to_list()
        new_frag_smi_to_idx = {
            smi: idx for idx, smi in enumerate(new_frag_smis_in_frag_lib)
        }
        new_frag_smis_in_data_df = self.data_df["NEW-FRAG"].to_list()
        self.data_df["NEW-FRAG-IDX"] = [
            new_frag_smi_to_idx[smi] for smi in tqdm(new_frag_smis_in_data_df)
        ]

        # Obtain old fragment frequencies of the training data df according to the old fragment library
        old_frag_smis_in_data_df = self.data_df["OLD-FRAG"].to_list()
        old_frag_smi_to_freq = dict(old_frags_freq)
        self.data_df["OLD-FRAG-FREQ"] = [
            old_frag_smi_to_freq[smi] for smi in tqdm(old_frag_smis_in_data_df)
        ]

        # Re-assign the columns order
        self.data_df.insert(
            self.data_df.columns.get_loc("NEW-FRAG") + 1,
            "NEW-FRAG-IDX-tmp",
            self.data_df["NEW-FRAG-IDX"],
        )
        self.data_df.insert(
            self.data_df.columns.get_loc("OLD-FRAG") + 1,
            "OLD-FRAG-FREQ-tmp",
            self.data_df["OLD-FRAG-FREQ"],
        )

        self.data_df = self.data_df.drop("NEW-FRAG-IDX", axis=1)
        self.data_df = self.data_df.drop("OLD-FRAG-FREQ", axis=1)
        self.data_df = self.data_df.rename(
            columns={
                "NEW-FRAG-IDX-tmp": "NEW-FRAG-IDX",
                "OLD-FRAG-FREQ-tmp": "OLD-FRAG-FREQ",
            }
        )

        # Get PairData objects from fragments
        _fragParsing = partial(self.fragParsing, frag_tmp_dir=self.frag_tmp_dir)
        with mp.Pool(self.nprocs) as p:
            parsing_result = p.map(_fragParsing, new_frags_df.iterrows())
        num_frags = len(parsing_result)
        frag_smi_to_feature = self.joinParsedFrags(
            num_frags
        )  # "frag_features.pkl"   for train & sampling
        print("Done.")

        # Handling the parsing results
        print("Handling the parsing results...")
        num_new_data = new_frags_df.shape[0]
        brics_type_tuples = []  # "fragment_library.csv"            for train & sampling
        # frag_smi_to_feature = {}  # "frag_features.pkl"             for train & sampling
        frag_brics_maskings = dict()  # "frag_brics_maskings.pkl"   for train only
        for idx, result in enumerate(parsing_result):
            brics_type_tuple, possible_adj_types = result
            brics_type_tuple = ",".join(map(str, brics_type_tuple))
            brics_type_tuples.append(brics_type_tuple)

            for adj_type in possible_adj_types:
                if adj_type in frag_brics_maskings:
                    frag_brics_maskings[adj_type][idx] = True
                else:
                    frag_brics_maskings[adj_type] = torch.zeros(
                        num_new_data, dtype=bool
                    )
                    frag_brics_maskings[adj_type][idx] = True
        new_frags_df["BRICS-TYPE"] = brics_type_tuples
        print("Done.")

        # Save the results into csv files and a pickle file
        # the original data file were filtered by the filter_data method
        print("Saving the results...")
        print("PRINT", self.frag_features_path, self.frag_brics_maskings_path)
        self.data_df.to_csv(
            self.data_df_save_path, sep="\t", index=True, index_label="INDEX"
        )

        # the insertion and removal fragments are concatenated and saved into a csv file
        concat_frags_df = pandas.concat([new_frags_df, old_frags_df])
        concat_frags_df.to_csv(
            self.fragment_library_path, sep="\t", index=True, index_label="INDEX"
        )

        # the fragment features are saved into a pickle file
        with open(self.frag_features_path, "wb") as fw:
            pickle.dump(frag_smi_to_feature, fw)
        with open(self.frag_brics_maskings_path, "wb") as fw:
            pickle.dump(frag_brics_maskings, fw)

        print("Done.")
        return True

    @classmethod
    def process_frag_library(cls, frag_lib_dir: Path, num_cores: int):
        """
        This method is to process already parsed fragment library csv file.
        Only the csv file would be provided since other tensor tiles are too large to be uploaded.
        By running this method, frag_features.pkl and frag_brics_maskings.pkl files will be generated.
        """
        # Setting save paths
        frag_features_path = op.join(frag_lib_dir, "frag_features.pkl")
        frag_brics_maskings_path = op.join(frag_lib_dir, "frag_brics_maskings.pkl")

        # Read fragment library
        frag_lib_csv_file = frag_lib_dir / "fragment_library.csv"
        frag_lib_df = pandas.read_csv(
            frag_lib_csv_file, sep="\t", dtype={"DATA-TYPE": str, "BRICS-TYPE": str}
        )
        new_frags_df = frag_lib_df[
            frag_lib_df["NEW-OLD"] == "new"
        ]  # only insertion fragments

        # parsing fragments
        print("Parsing fragments...")
        frag_tmp_dir = frag_lib_dir / "frags_tmp"
        frag_tmp_dir.mkdir(exist_ok=True)
        _fragParsing = partial(cls.fragParsing, frag_tmp_dir=frag_tmp_dir)
        with mp.Pool(num_cores) as p:
            parsing_result = p.map(_fragParsing, new_frags_df.iterrows())
        num_frags = len(parsing_result)

        # Retrieve parsed fragments
        print("Retrieving parsed fragments...")
        frag_smi_to_feature = dict()
        parsed_frag_files = [f"{idx}.pt" for idx in range(num_frags)]
        for f in parsed_frag_files:
            frag_feature = torch.load(os.path.join(frag_tmp_dir, f))
            assert frag_feature, "MolFromSmiles for fragment yielded None."
            frag_smi_to_feature[frag_feature.smiles] = frag_feature
        shutil.rmtree(frag_tmp_dir)

        # Handling the parsing results
        print("Handling the parsing results...")
        num_new_data = new_frags_df.shape[0]
        frag_brics_maskings = dict()  # "frag_brics_maskings.pkl"   for train only
        for idx, result in enumerate(parsing_result):
            brics_type_tuple, possible_adj_types = result
            brics_type_tuple = ",".join(map(str, brics_type_tuple))

            for adj_type in possible_adj_types:
                if adj_type in frag_brics_maskings:
                    frag_brics_maskings[adj_type][idx] = True
                else:
                    frag_brics_maskings[adj_type] = torch.zeros(
                        num_new_data, dtype=bool
                    )
                    frag_brics_maskings[adj_type][idx] = True
        print("Done.")

        # the fragment features are saved into a pickle file
        print("Saving the results...")
        with open(frag_features_path, "wb") as fw:
            pickle.dump(frag_smi_to_feature, fw)
        with open(frag_brics_maskings_path, "wb") as fw:
            pickle.dump(frag_brics_maskings, fw)

        print("Done.")

        return

    @classmethod
    def fragParsing(
        cls, frag_inform: Tuple[int, pandas.core.series.Series], frag_tmp_dir: str
    ) -> Tuple[torch.Tensor, Tuple[int], str]:
        idx, frag_inform = frag_inform
        frag_smi = frag_inform["FRAG-SMI"]

        # Parsing to tensor object
        frag_mol = Chem.MolFromSmiles(frag_smi)
        frag_feature = from_mol(frag_mol, type="Frag")
        torch.save(frag_feature, f"{frag_tmp_dir}/{idx}.pt")

        # Parsing for brics type
        brics_types = []
        for atom in frag_mol.GetAtoms():
            if atom.GetSymbol() == "*":
                brics_types.append(atom.GetIsotope())
        brics_type_tuple: Tuple[int] = tuple(sorted(brics_types))
        possible_adj_types = cls.BRICS_TYPE_MAPPER.getMapping(sorted(brics_types))
        return brics_type_tuple, possible_adj_types

    def joinParsedFrags(self, num_data):
        frag_smi_to_feature = dict()

        parsed_frag_files = [f"{idx}.pt" for idx in range(num_data)]
        for f in parsed_frag_files:
            frag_feature = torch.load(os.path.join(self.frag_tmp_dir, f))
            assert frag_feature, "MolFromSmiles for fragment yielded None."
            frag_smi_to_feature[frag_feature.smiles] = frag_feature

        shutil.rmtree(self.frag_tmp_dir)
        return frag_smi_to_feature


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nprocs", type=int)
    parser.add_argument("-d", "--data_dir", type=str)
    args = parser.parse_args()
    print(f"args: {args}")

    data_processor = FragLibProcessor(args)
    data_processor.process_data()
