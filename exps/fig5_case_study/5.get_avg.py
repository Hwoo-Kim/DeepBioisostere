import argparse
from collections import defaultdict
from statistics import mean

# 2021_local_opt_0_1	-0.000	-5.196	-4.058	-0.175	0.000	-0.964	0.000


def main(args: argparse.Namespace):
    # Model-level ensemble
    model_level_dic = defaultdict(list)
    for model_seed in range(4):
        with open(f"./{args.file_dir}/result_{model_seed}.txt") as f:
            for line in f:
                key, _, aff = line.split()[:3]
                model_level_dic[key].append(float(aff))

    # Average of model-level ensemble
    data_dic = {}
    for key, aff_list in model_level_dic.items():
        data_dic[key] = mean(aff_list)

    # Get the best score for each molecule
    mol_dic = defaultdict(list)
    for key, aff in data_dic.items():
        seed, _, _, mol_idx, conf_idx = key.split("_")
        mol_dic[mol_idx].append((seed, conf_idx, aff))

    # sort values by affinity
    for mol_idx, value_list in mol_dic.items():
        value_list.sort(key=lambda x: x[2])

    with open(args.output_file, "w") as f:
        f.write("mol_idx seed conf_idx aff\n")
        for mol_idx, value_list in mol_dic.items():
            best_seed, best_conf, best_aff = value_list[0]
            f.write(f"{mol_idx} {best_seed} {best_conf} {best_aff}\n")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-dir", type=str)
    parser.add_argument("--output-file", type=str)
    args = parser.parse_args()

    main(args)
