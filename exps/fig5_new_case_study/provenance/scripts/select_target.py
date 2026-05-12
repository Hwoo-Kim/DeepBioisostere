import os
from pathlib import Path

import pandas as pd

avg_csv = Path("avg_result.csv")
if not avg_csv.exists():
    path = Path(os.environ.get("SBDD_EVAL_ROOT", "raw_eval"))
    models = ["DeepICL", "targetdiff", "decompdiff", "pocket2mol"]

    whole_df = []
    for model in models:
        df = pd.read_csv(path / f"{model}_eval" / "result.csv", low_memory=False)
        df["MODEL"] = model
        whole_df.append(df)

    whole_df = pd.concat(whole_df, ignore_index=True)
    avg_df = whole_df.groupby(by=["MODEL", "test_idx"]).mean(
        ["QED", "SA", "gen_dock_aff"]
    )
    columns = ["QED", "SA", "gen_dock_aff"]
    avg_df = avg_df[columns]
    avg_df.to_csv(avg_csv)

avg_df = pd.read_csv(avg_csv)

# for i in range(1, 101):
#     target_df = avg_df[avg_df["test_idx"] == i]
#     print("TARGET IDX: ", i)
#     for _qed in [0.4, 0.45, 0.5, 0.55, 0.6]:
#         if (target_df["QED"] < _qed).all():
#             print("all model lower than QED: ", _qed)
#             print(target_df)
#     for _sa in [0.6, 0.65, 0.7]:
#         if (target_df["SA"] < _sa).all():
#             print("all model lower than SA: ", _sa)
#             print(target_df)
#     print()

# for i in range(1, 101):
#     target_df = avg_df[avg_df["test_idx"] == i]
#     if target_df["QED"].max() < 0.6:
#         print("TARGET IDX: ", i)
#         print(target_df["QED"].max(), target_df["SA"].max())
# exit()

# avg_df["SA"] = 10 - 9 * avg_df["SA"]
# for i in range(1, 101):
#     target_df = avg_df[avg_df["test_idx"] == i]
#     # print("TARGET IDX: ", i)
#     # print(target_df["SA"].tolist())
#     if target_df["QED"].max() < 0.5:
#         print("TARGET IDX: ", i)
#         print(target_df)
# exit()

avg_df["SA"] = 10 - 9 * avg_df["SA"]
for i in range(1, 101):
    target_df = avg_df[avg_df["test_idx"] == i]
    # print("TARGET IDX: ", i)
    # print(target_df["SA"].tolist())
    if target_df["SA"].min() >= 3.5 and target_df["QED"].max() < 0.6:
        # if target_df["SA"].min() >= 4:
        print("TARGET IDX: ", i)
        print(target_df)
        print()
