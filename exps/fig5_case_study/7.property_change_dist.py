from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from property import calc_logP, calc_Mw, calc_QED, calc_SAscore
from rdkit import Chem

sns.set_style("white")
sns.set_context("paper", font_scale=2.5)
sns.set_palette("pastel")
plt.rcParams["font.family"] = "sans-serif"


def print_properties(mol: Chem.Mol):
    return calc_logP(mol), calc_QED(mol), calc_Mw(mol), calc_SAscore(mol)


data = {
    "logP": [],
    "QED": [],
    "MW": [],
    "SA": [],
}
data_dir = Path("conformers/2021")
data_files = data_dir.glob("mmff_*.sdf")

ref_data = {}
for file in data_files:
    mol = Chem.SDMolSupplier(str(file))[0]
    logp, qed, mw, sa = print_properties(mol)
    if "ori" in file.stem:
        ref_data["logP"] = logp
        ref_data["QED"] = qed
        ref_data["MW"] = mw
        ref_data["SA"] = sa
        continue

    data["logP"].append(logp)
    data["MW"].append(mw)
    data["QED"].append(qed)
    data["SA"].append(sa)

# redefne dataframe suitable for seaborn displot
dic = defaultdict(list)
for key, value in data.items():
    dic["property"] += len(value) * [key]
    dic["value"] += value

df = pd.DataFrame.from_dict(dic)

props = ["MW", "logP", "QED", "SA"]
g = sns.FacetGrid(
    df,
    col="property",
    sharey=True,
    sharex=False,
    despine=False,
    height=7,
    aspect=0.8,
    col_wrap=4,
    col_order=["MW", "logP", "QED", "SA"],
)
g.map_dataframe(sns.histplot, x="value", kde=True)

g.set_axis_labels("", "Count")
g.set_titles("")
g.figure.suptitle("Property Change Distribution")

for idx, ax in enumerate(g.axes.flat):
    prop = props[idx]
    ax.axvline(
        ref_data[prop],
        color="r",
        linestyle="dashed",
        label="Reference molecule's property" if idx == 0 else None,
    )
    ax.axvline(
        df[df["property"] == prop]["value"].mean(),
        color="b",
        linestyle="dashed",
        label="Generated molecules' average property" if idx == 0 else None,
    )
    ax.set_xlabel(prop)

leg = g.figure.legend(
    ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.95), framealpha=0.0
)
leg.get_frame().set_linewidth(0.0)
leg.get_frame().set_facecolor("white")
plt.tight_layout()
plt.savefig("property_change_dist.png", dpi=600)
