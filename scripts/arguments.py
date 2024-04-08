import argparse
from pathlib import Path


def get_train_args_parser():
    parser = argparse.ArgumentParser()

    # 1. Data Path Arguments
    parser.add_argument(
        "--project_dir",
        type=Path,
        help="Path to this project. Working directory in most cases.",
    )
    parser.add_argument(
        "--save_name", type=str, help="Save directory name for the results."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Path to the csv file pre-processed data.",
    )
    parser.add_argument(
        "--conditioning",
        action="store_true",
        help="Whether conditioning will be used in training. (logp, mw, qed, sa)",
    )
    parser.add_argument(
        "--properties",
        type=str,
        help="Combination of properties. (logp, mw, qed, sa)",
        default=None,
    )
    parser.add_argument(
        "--use_delta",
        action="store_true",
        help="Whether using a difference or absolute value of properties. (logp, mw, qed, sa)",
    )
    parser.add_argument(
        "--use_soft_one_hot",
        action="store_true",
        help="Whether using soft one hot encoding of properties. (logp, mw, qed, sa)",
    )

    # 2. Training Arguments
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to initialize random, numpy, torch, cuda packages.",
    )
    parser.add_argument(
        "--use_cuda", action="store_true", help="Whether using cuda system or not."
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        help="How many gpus to use, especially for DDP setting (not implemented yet).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for the training and validation data. (original_smi and frag_id pair)",
    )
    parser.add_argument(
        "--frag_lib_batch_size",
        type=int,
        help="Batch size for fragment embedding model.",
    )
    parser.add_argument(
        "--num_cores",
        type=int,
        help="Number of workers for data loader. (for both PairData and fragment library)",
    )
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument(
        "--lr_reduce_factor",
        type=float,
        help="LR decreasing factor for torch.optim.lr_scheduler.ReduceLROnPlateau.",
    )
    parser.add_argument("--min_lr", type=float, help="Lower boundary for lr.")
    parser.add_argument(
        "--patience",
        type=int,
        help="LR is reduced if val loss does not decrease within <patience> epochs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Minimum decrease to update the min value of val loss.",
    )
    parser.add_argument("--max_epoch", type=int, help="Maximum training epoch.")
    parser.add_argument(
        "--num_neg_sample",
        type=int,
        help="Number of negative samples to train fragment scoring model (per each data).",
    )
    parser.add_argument(
        "--alpha1",
        type=float,
        help="Exponent to the negative sampling for new fragment library.",
    )
    parser.add_argument(
        "--alpha2",
        type=float,
        help="Exponent to the weighted random sampler concerning original fragment frequency.\
                all data is weighted by exp(-alpha2).",
    )
    parser.add_argument(
        "--weighted_sampler",
        action="store_true",
        help="Whether use WeightedRandomSampler concerning original fragment frequency.",
    )
    parser.add_argument(
        "--lr_scheduler_can_terminate",
        action="store_true",
        help="Whether lr scheduler can terminate the training process.",
    )
    parser.add_argument(
        "--num_batch_each_epoch",
        type=int,
        help="Number of batch used in each epoch.",
        default=None,
    )
    parser.add_argument(
        "--print_loss",
        action="store_true",
        help="Whether print loss as well as probability.",
    )
    parser.add_argument(
        "--profiling", action="store_true", help="Whether using torch profiler or not."
    )

    # 3. Model Arguments
    parser.add_argument(
        "--mol_node_features", type=int, help="Whole molecule's node feature dimension."
    )
    parser.add_argument(
        "--mol_edge_features", type=int, help="Whole molecule's edge feature dimension."
    )
    parser.add_argument(
        "--mol_node_hid_dim", type=int, help="Whole molecule's node hidden dimension."
    )
    parser.add_argument(
        "--mol_edge_hid_dim", type=int, help="Whole molecule's edge hidden dimension."
    )
    parser.add_argument(
        "--mol_num_emb_layer",
        type=int,
        help="Number of MPNN layers to embed whole molecule graphs.",
    )
    parser.add_argument(
        "--frag_node_features",
        type=int,
        help="Fragment library subgraphs' node feature dimension.",
    )
    parser.add_argument(
        "--frag_edge_features",
        type=int,
        help="Fragment library subgraphs' edge feature dimension.",
    )
    parser.add_argument(
        "--frag_node_hid_dim",
        type=int,
        help="Fragment library subgraphs' node hidden dimension.",
    )
    parser.add_argument(
        "--frag_edge_hid_dim",
        type=int,
        help="Fragment library subgraphs' edge hidden dimension.",
    )
    parser.add_argument(
        "--frag_num_emb_layer",
        type=int,
        help="Number of MPNN layers to embed fragment library subgraphs.",
    )
    parser.add_argument(
        "--position_score_hid_dim",
        type=int,
        help="Hidden deimension for modification position prediction.",
    )
    parser.add_argument(
        "--num_mod_position_score_layer",
        type=int,
        help="Number of FeedForward layers for modification position prediction.",
    )
    parser.add_argument(
        "--frag_score_hid_dim",
        type=int,
        help="Hidden deimension for fragment selection.",
    )
    parser.add_argument(
        "--num_frag_score_layer",
        type=int,
        help="Number of FeedForward layers for fragment selection.",
    )
    parser.add_argument(
        "--attach_score_hid_dim",
        type=int,
        help="Hidden deimension for attachment prediction.",
    )
    parser.add_argument(
        "--num_attach_score_layer",
        type=int,
        help="Number of FeedForward layers for attachment prediction.",
    )
    parser.add_argument(
        "--frag_message_passing_num_layer",
        type=int,
        help="Number of MPNN layers for fragment-level embedding.",
    )
    parser.add_argument(
        "--dropout", type=float, help="Dropout probability for train phase."
    )

    # 4. Parse arguments
    args = parser.parse_args()
    args.data_dir = args.data_path.parents[0]
    if args.conditioning:
        args.properties = list(map(lambda x: x.lower(), args.properties.split(",")))
        args.properties.sort()
    return args


def get_generate_args_parser():
    parser = argparse.ArgumentParser()

    # 1. Data Path Arguments
    parser.add_argument(
        "--project_dir",
        type=Path,
        help="Path to this project. Working directory in most cases.",
    )
    parser.add_argument(
        "--save_name", type=str, help="Save directory name for the results."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Path to the csv file pre-processed data.",
    )
    parser.add_argument(
        "--conditions_path",
        type=Path,
        help="Path to a file containing calculated properties of ChEMBL molecules.",
        default=None,
    )
    parser.add_argument(
        "--start",
        type=int,
        help="Start indx of the generation process. If specified, the input molecules are not randomly shuffled.",
        default=None,
    )
    parser.add_argument(
        "--end",
        type=int,
        help="End index of the generation process. If specified, the input molecules are not randomly shuffled.",
        default=None,
    )
    parser.add_argument(
        "--num_smiles",
        type=int,
        help="Number of SMILES (of original molecules) used in modification. First $num_smiles SMILES are used.",
    )

    # 2. Generation Arguments
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to initialize random, numpy, torch, cuda packages.",
    )
    parser.add_argument(
        "--property",
        type=str,
        help="Which property will be used in conditioning. (one of mw, logp, qed)",
        default=None,
    )
    parser.add_argument(
        "--target_value",
        type=float,
        help="What value to target for conditioned generation. (for mw, logp, or qed)",
        default=None,
    )
    parser.add_argument(
        "--use_cuda", action="store_true", help="Whether using cuda system or not."
    )
    parser.add_argument(
        "--exclude_smiles_in_training",
        action="store_true",
        help="Whether excluding smiles used in train/val or not.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for the main training and validation data. (original_smi and frag_id pair)",
    )
    parser.add_argument(
        "--frag_lib_batch_size",
        type=int,
        help="Batch size for fragment embedding model.",
    )
    parser.add_argument(
        "--num_cores",
        type=int,
        help="Number of workers for data loader. (for both PairData and fragment library)",
    )
    parser.add_argument(
        "--sampling_option",
        type=str,
        help="Sampling option. One of (all, greedy, multinomial)",
    )
    parser.add_argument(
        "--num_sample_each_data",
        type=int,
        help="Number of molecules sampled for each original SMILES.",
    )
    parser.add_argument(
        "--weighted_sampler",
        action="store_true",
        help="Whether use WeightedRandomSampler concerning original fragment frequency.",
    )

    # 3. Model Arguments
    parser.add_argument(
        "--trained_model_path",
        type=str,
        help="path to the trained model parameter file.",
    )
    parser.add_argument(
        "--mol_node_features", type=int, help="Whole molecule's node feature dimension."
    )
    parser.add_argument(
        "--mol_edge_features", type=int, help="Whole molecule's edge feature dimension."
    )
    parser.add_argument(
        "--mol_node_hid_dim", type=int, help="Whole molecule's node hidden dimension."
    )
    parser.add_argument(
        "--mol_edge_hid_dim", type=int, help="Whole molecule's edge hidden dimension."
    )
    parser.add_argument(
        "--mol_num_emb_layer",
        type=int,
        help="Number of MPNN layers to embed whole molecule graphs.",
    )
    parser.add_argument(
        "--frag_node_features",
        type=int,
        help="Fragment library subgraphs' node feature dimension.",
    )
    parser.add_argument(
        "--frag_edge_features",
        type=int,
        help="Fragment library subgraphs' edge feature dimension.",
    )
    parser.add_argument(
        "--frag_node_hid_dim",
        type=int,
        help="Fragment library subgraphs' node hidden dimension.",
    )
    parser.add_argument(
        "--frag_edge_hid_dim",
        type=int,
        help="Fragment library subgraphs' edge hidden dimension.",
    )
    parser.add_argument(
        "--frag_num_emb_layer",
        type=int,
        help="Number of MPNN layers to embed fragment library subgraphs.",
    )
    parser.add_argument(
        "--position_score_hid_dim",
        type=int,
        help="Hidden deimension for modification position prediction.",
    )
    parser.add_argument(
        "--num_mod_position_score_layer",
        type=int,
        help="Number of FeedForward layers for modification position prediction.",
    )
    parser.add_argument(
        "--frag_score_hid_dim",
        type=int,
        help="Hidden deimension for fragment selection.",
    )
    parser.add_argument(
        "--num_frag_score_layer",
        type=int,
        help="Number of FeedForward layers for fragment selection.",
    )
    parser.add_argument(
        "--attach_score_hid_dim",
        type=int,
        help="Hidden deimension for attachment prediction.",
    )
    parser.add_argument(
        "--num_attach_score_layer",
        type=int,
        help="Number of FeedForward layers for attachment prediction.",
    )
    parser.add_argument(
        "--frag_message_passing_num_layer",
        type=int,
        help="Number of MPNN layers for fragment-level embedding.",
    )

    # 4. Parse arguments
    args = parser.parse_args()
    args.data_dir = args.data_path.parents[0]
    return args
