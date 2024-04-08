import argparse
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("white")
sns.set_context("paper", font_scale=2.5)
sns.set_palette("pastel")
plt.rcParams["font.family"] = "sans-serif"


def dist_1dplot(mut_scores: np.ndarray, ref_mut_score: float, mean_score: float, name: str):
    fig, ax = plt.subplots(figsize=(7, 8))
    sns.histplot(
        x=mut_scores,
        bins=20,
        kde=True,
        alpha=0.5,
        ax=ax,
        label=None,
        legend=False,
    )

    # vertical line at ref_mut_score similar to plt.avhline
    ax.axvline(
        x=ref_mut_score,
        color="gray",
        linewidth=1.5,
        linestyle="--",
        alpha=1.0,
    )

    ax.axvline(
        x=mean_score,
        color="blue",
        linewidth=1.5,
        linestyle="--",
        alpha=1.0,
    )

    ax.set_ylabel("")

    for bar in ax.patches:
        if bar.get_x() < ref_mut_score - 1.36:
            bar.set_facecolor(sns.color_palette("pastel")[3])

    plt.tight_layout()
    plt.savefig(name, dpi=600)
    return


def main(args):
    mut_scores = []
    count = 0
    with open(args.file) as f:
        for line in f:
            if line.startswith("mol"):
                continue
            mol_idx, seed, conf_idx, aff = line.split()

            aff = float(aff)
            if mol_idx == "ori":
                ref_mut_score = aff
                continue

            if int(mol_idx) > args.max_n_mols:
                break

            mut_scores.append(aff)

    mean_score = mean(mut_scores)
    fig_name = f"{args.file.split('/')[-1].split('.')[0]}_{args.max_n_mols}.png"
    dist_1dplot(
        np.array(mut_scores),
        ref_mut_score=ref_mut_score,
        mean_score=mean_score,
        name=fig_name,
    )  # MMFF
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="File with the affinity scores",
    )
    parser.add_argument(
        "--max_n_mols",
        type=int,
        required=True,
        default=500,
    )
    args = parser.parse_args()

    main(args)
