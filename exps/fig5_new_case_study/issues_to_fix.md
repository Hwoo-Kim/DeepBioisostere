# Known Issues To Fix In Manuscript

Status note: the user reported the Main Table 2 and Supplementary Table 7
manuscript fixes are already applied. The remaining file-provenance issue for
Figure 5 has been resolved by restoring the tracked CSV and copying it into the
source data package.

## Main Table 2 TargetDiff Joint Value

Current PDF value:

- TargetDiff / DeepBioisostere / Joint: `0.74`

Current source-data value:

- file: `exps/fig5_new_case_study/table_2/validity_targetdiff_68.csv`
- raw value: `0.7238906926406926`
- two-decimal value: `0.72`

This is not reproduced from the currently organized final source CSV. The PDF
value should either be traced to the exact source file that produced it, or the
manuscript/source data should be corrected to use the same value.

Resolution status:

- User reported the manuscript value was corrected to `0.72`.

## Supplementary Table 6 Rounding Convention

TargetDiff / 2EWY has transformed SA `5.635`.

- PDF value: `5.64`
- Python default two-decimal formatting can produce: `5.63`

This is likely a rounding-convention issue, but the source package should state
which rounding rule is used for final tables.

## Supplementary Table 7 Caption

Current PDF caption:

> Additional results for two targets in the CrossDocked2020 benchmark dataset
> (PDB ID: 2E6D and 4Z2G).

This does not match the table rows or the source data. The corresponding targets
are:

- target index 36: PDB ID `4P77`
- target index 84: PDB ID `2EWY`

Recommended caption:

> Additional results for two targets in the CrossDocked2020 benchmark dataset
> (PDB ID: 4P77 and 2EWY).

Resolution status:

- User reported the manuscript caption was corrected.

## Figure 5 Filtered CSV Naming

The tracked file
`exps/fig5_new_case_study/20250826_new_data_main_model/filtered_results_68.csv`
was missing in the worktree and has now been restored. The available file with
the corresponding final selected example content is:

`exps/fig5_new_case_study/20250826_new_data_main_model/filtered_deepicl_68.csv`

These files are not identical in git/worktree state:

- `HEAD:filtered_results_68.csv`: 132 rows
- `filtered_deepicl_68.csv`: 19 rows

Figure 5 panel A exists in both files, but panel B
(`INPUT-MOL-IDX=7`, `GEN-MOL-IDX=64`) exists in `filtered_deepicl_68.csv` and
not in the deleted tracked `filtered_results_68.csv`.

This package uses `figure_5/filtered_deepicl_68.csv` and explicitly records the
two selected figure rows in `figure_5/figure5_selected_examples.csv`.

Resolution status:

- Restored tracked file:
  `exps/fig5_new_case_study/20250826_new_data_main_model/filtered_results_68.csv`
- Added exact package copy:
  `exps/fig5_new_case_study/figure_5/filtered_results_68_tracked_head.csv`
