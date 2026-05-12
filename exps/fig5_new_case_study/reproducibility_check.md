# SBDD Reproducibility Check

Date: 2026-05-11

This check compares the organized SBDD source data package against the values
shown in `DeepBioisostere.pdf`.

## Scope

- Main Table 2: SBDD target 4RV4 / target index 68.
- Main Figure 5: two DeepICL example molecules on target 4RV4 / target index 68.
- Supplementary Table 6: selected targets 4P77, 4RV4, and 2EWY.
- Supplementary Table 7: additional SBDD success rates for 4P77 and 2EWY.
- Final docking/evaluation Slurm logs for the copied source data.

## Summary

- Total checked values/events: 86
- Passing checks with half-up table rounding: 85
- Substantive failing checks: 1

The final SBDD source package is mostly reproducible, but not fully exact
against the current PDF. The remaining substantive mismatch is a
manuscript/source-data mismatch in Main Table 2. Supplementary Table 6 is
consistent with the PDF if half-up two-decimal rounding is used.

## Failures

### Main Table 2

TargetDiff / DeepBioisostere on target 4RV4 has a mismatch in the Joint column.

| Field | PDF | Source data |
|---|---:|---:|
| QED | 0.83 | 0.83 |
| SA | 0.87 | 0.87 |
| Docking | 0.97 | 0.97 |
| Joint | 0.74 | 0.72 |

The source value comes from:

`exps/fig5_new_case_study/table_2/validity_targetdiff_68.csv`

Raw value:

```text
DeepBioisostere,0.8329274891774892,0.8741883116883117,0.9660443722943723,0.7238906926406926,77.0
```

This rounds to 0.72, not 0.74. The 0.74 value is therefore not reproduced from
the current final source CSV. This should be fixed either by identifying the
exact source that produced the PDF value or by correcting the manuscript table
to the current source-data value.

### Supplementary Table 6 Rounding Caveat

TargetDiff on target 2EWY has a minor rounding-convention caveat.

| Field | PDF | Source data |
|---|---:|---:|
| QED | 0.33 | 0.33 |
| SA | 5.64 | 5.635 raw |

The source values come from:

`exps/fig5_new_case_study/supplementary_table_6/table6_selected_targets.csv`

Raw source values:

```text
target_idx=84
pdb_id=2EWY
model=TargetDiff
qed=0.3293118272860487
sa_raw=0.485
sa_transformed=5.635
```

The transformed SA value is computed as:

```text
10 - 9 * 0.485 = 5.635
```

With half-up decimal rounding, this reproduces the PDF value `5.64`. With
Python's default banker's rounding, it can become `5.63`. This should be
documented or rounded consistently.

## Passing Coverage

The following checks reproduce from the organized source data:

- Main Table 2: all rows and columns except TargetDiff / DeepBioisostere / Joint.
- Main Figure 5: both selected examples, including target, model, input molecule
  index, generated molecule index, QED, SA, and docking score.
- Supplementary Table 6: all selected target/model values reproduce when
  half-up two-decimal rounding is used.
- Supplementary Table 7: all numeric rows for 4P77 and 2EWY reproduce from the
  copied validity CSVs.
- Final Slurm logs: all copied final docking/evaluation jobs report successful
  Python completion and successful tarball creation.

## Required Follow-up

1. Resolve Main Table 2 TargetDiff / DeepBioisostere / Joint.
2. Document that Supplementary Table 6 uses half-up two-decimal rounding.
3. Fix Supplementary Table 7 caption from `PDB ID: 2E6D and 4Z2G` to
   `PDB ID: 4P77 and 2EWY`.
4. Keep `filtered_deepicl_68.csv` as the available Figure 5 filtered source
   because `filtered_results_68.csv` is currently deleted in git.
