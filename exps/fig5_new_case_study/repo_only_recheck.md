# Repo-only SBDD CSV Recheck

Date: 2026-05-11

This recheck uses only CSV files inside the current repository. It excludes
external shared paths and does not use `fig5_new_case_study/` as an input source.

## Inputs Checked

- `DeepBioisostere.pdf`
- `exps/fig5_new_case_study/avg_result.csv`
- `exps/fig5_new_case_study/20250826_new_data_main_model/summary/*.csv`
- `exps/fig5_new_case_study/20250826_new_data_main_model/docking_gpu_results/summary_DeepICL_68_0.10_-1.0.csv`
- Git history for the tracked Figure 5 CSV:
  `exps/fig5_new_case_study/20250826_new_data_main_model/filtered_results_68.csv`

## Git State Findings

Tracked SBDD CSV coverage is incomplete.

- Git tracks only two SBDD CSVs under
  `20250826_new_data_main_model`:
  - `docking_gpu_results/summary_DeepICL_68_0.10_-1.0.csv`
  - `filtered_results_68.csv`
- `filtered_results_68.csv` is currently deleted in the worktree.
- The table source CSVs under
  `20250826_new_data_main_model/summary/` are ignored/untracked, so git history
  cannot prove how Main Table 2 or Supplementary Table 7 were produced.

The deleted tracked file and the local replacement candidate are not identical:

| File | Rows | SHA256 |
|---|---:|---|
| `HEAD:filtered_results_68.csv` | 132 | `7b32c2f59a603ec20e991e51f3d59efffd504ece89e79acfc7609133c9ee7eff` |
| `filtered_deepicl_68.csv` | 19 | `06e69e51e1efd53eff26321d961dd5c30dc0577ec407a12723240bf481fa368f` |

Figure 5 panel A row (`INPUT-MOL-IDX=28`, `GEN-MOL-IDX=59`) exists in both.
Figure 5 panel B row (`INPUT-MOL-IDX=7`, `GEN-MOL-IDX=64`) exists in
`filtered_deepicl_68.csv` but not in the deleted tracked
`filtered_results_68.csv`.

## Numeric Recheck

Using current repo CSVs and half-up two-decimal rounding:

- Main Table 2 + Supplementary Table 7 checked values: 192
- Passing values: 191
- Failing values: 1

The only numeric mismatch is:

| PDF item | Model | Strategy | Metric | PDF | Current repo CSV |
|---|---|---|---|---:|---:|
| Main Table 2 | TargetDiff | DeepBioisostere | Joint | 0.74 | 0.72 |

Source file:

`exps/fig5_new_case_study/20250826_new_data_main_model/summary/validity_targetdiff_68.csv`

Raw row:

```text
DeepBioisostere,0.8329274891774892,0.8741883116883117,0.9660443722943723,0.7238906926406926,77.0
```

The current repo also contains:

`exps/fig5_new_case_study/20250826_new_data_main_model/summary/ablation_validity_targetdiff_68.csv`

Raw ablation row:

```text
Ablation,0.8465386695511087,0.8853434288804759,0.958085451595457,0.7402650081124933,77.04166666666667
```

This ablation Joint value rounds to `0.74`. Therefore the PDF value is
consistent with the ablation Joint value, not with the current
DeepBioisostere row in `validity_targetdiff_68.csv`.

## Older Candidate CSVs

The only repo files matching `*/summary/validity_targetdiff_68.csv` are:

- `exps/fig5_new_case_study/20250814_new_data_main_model/summary/validity_targetdiff_68.csv`
- `exps/fig5_new_case_study/20250826_new_data_main_model/summary/validity_targetdiff_68.csv`

The older `20250814` DeepBioisostere row rounds to:

```text
0.86,0.86,0.95,0.73
```

It does not explain the PDF value `0.74`.

## Conclusion

Within the current repository, the SBDD numeric results are almost fully
reproduced, but Main Table 2 contains one real mismatch. Git history does not
resolve it because the relevant table source CSVs are ignored/untracked.

The most likely issue is a manuscript table copy/mixing error: the TargetDiff
DeepBioisostere row appears to use the main-row QED, SA, and docking values but
the ablation-row Joint value.

## Figure 5 Value Check

The Figure 5 example values are reproducible from the tracked summary CSV:

`exps/fig5_new_case_study/20250826_new_data_main_model/docking_gpu_results/summary_DeepICL_68_0.10_-1.0.csv`

| Panel | Input index | Generated index | Original QED | Generated QED | Original SA | Generated SA | Original docking | Generated docking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| a | 28 | 59 | 0.73 | 0.83 | 3.43 | 2.88 | -7.4 | -7.6 |
| b | 7 | 64 | 0.45 | 0.54 | 3.52 | 2.57 | -5.5 | -5.6 |
