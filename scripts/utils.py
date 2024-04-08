import logging
import os
import subprocess


class Logger(logging.Logger):
    def __init__(self, name, save_path=None):
        super().__init__(name=name)
        if save_path:
            if os.path.exists(save_path):
                os.remove(save_path)
            try:
                file_handler = logging.FileHandler(filename=save_path)
                # file_handler.setFormatter(formatter)
                file_handler.setLevel(logging.DEBUG)
                self.addHandler(file_handler)
            except FileNotFoundError:
                print(f"Invalid log path {save_path}")
                exit()
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        self.addHandler(stream_handler)

    def __call__(self, message=""):
        self.info(message)

    @staticmethod
    def _get_skip_args():
        return [
            "logger",
            "save_name",
        ]

    def log_args(self, args, tab=""):
        d = vars(args)
        _skip_args = self._get_skip_args()
        for v in d:
            if v not in _skip_args:
                self.info(f"{tab}{v}: {d[v]}")


def set_seed(seed):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def set_cuda_visible_devices() -> str:
    """Set available GPU IDs as a str (e.g., '0,1,2')"""
    max_num_gpus = 8
    idle_gpus = []

    for i in range(max_num_gpus):
        cmd = ["nvidia-smi", "-i", str(i)]
        proc = subprocess.run(cmd, capture_output=True, text=True)  # after python 3.7

        if "No devices were found" in proc.stdout:
            break

        if "No running" in proc.stdout:
            idle_gpus.append(i)

    # Convert to a str to feed to os.environ.
    idle_gpus = ",".join(str(i) for i in idle_gpus)

    os.environ["CUDA_VISIBLE_DEVICES"] = idle_gpus
    return idle_gpus


def train_path_setting(args):
    args.data_dir = os.path.normpath(args.data_dir)
    # args.key_path = os.path.join(args.processed_data_dir, "keys.pkl")

    save_dir = os.path.normpath(os.path.join(args.project_dir, "model_save"))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, args.save_name)
    while save_dir[-1] == "/":
        save_dir = save_dir[:-1]

    if os.path.exists(save_dir):
        i = 2
        while os.path.exists(f"{save_dir}_{i}"):
            i += 1
        save_dir = f"{save_dir}_{i}"
    os.mkdir(save_dir)
    args.save_dir = save_dir

    return args


def generate_path_setting(args):
    args.data_dir = os.path.normpath(args.data_dir)
    # args.original_smiles_path = os.path.normpath(args.original_smiles_path)

    save_dir = os.path.normpath(os.path.join(args.project_dir, "sampling_save"))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, args.save_name)
    while save_dir[-1] == "/":
        save_dir = save_dir[:-1]

    if os.path.exists(save_dir):
        i = 2
        while os.path.exists(f"{save_dir}_{i}"):
            i += 1
        save_dir = f"{save_dir}_{i}"
    os.mkdir(save_dir)
    args.save_dir = save_dir

    return args
