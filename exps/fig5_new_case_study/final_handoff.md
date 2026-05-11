# SBDD Source Data Final Handoff

Date: 2026-05-11

## Resolved Items

### Main Table 2

The source-data value for TargetDiff / DeepBioisostere / Joint is:

```text
0.7238906926406926
```

Therefore the manuscript table should report:

```text
0.72
```

The user reported this manuscript correction is done.

### Supplementary Table 7 Caption

The correct PDB IDs are:

```text
4P77 and 2EWY
```

The user reported this caption correction is done.

### Figure 5 Filtered CSV Provenance

The tracked file was restored:

`exps/fig5_new_case_study/20250826_new_data_main_model/filtered_results_68.csv`

An exact copy is included in the source data package:

`exps/fig5_new_case_study/figure_5/filtered_results_68_tracked_head.csv`

SHA256:

```text
7b32c2f59a603ec20e991e51f3d59efffd504ece89e79acfc7609133c9ee7eff
```

The final Figure 5 selected-example provenance is kept separately:

- `exps/fig5_new_case_study/figure_5/filtered_deepicl_68.csv`
- `exps/fig5_new_case_study/figure_5/figure5_selected_examples.csv`
- `exps/fig5_new_case_study/figure_5/summary_DeepICL_68_0.10_-1.0.csv`

This distinction matters because Figure 5 panel B
(`INPUT-MOL-IDX=7`, `GEN-MOL-IDX=64`) exists in `filtered_deepicl_68.csv`, not
in the restored tracked `filtered_results_68.csv`.

## Current Package Status

The source data package now contains the required materials for:

- Main Table 2
- Main Figure 5
- Supplementary Table 6
- Supplementary Table 7
- target/PDB ID mapping
- final docking job provenance
- raw SBDD evaluation CSVs
- selected reference files

## Remaining Final Check

After regenerating the manuscript PDF, verify that:

- Main Table 2 reports TargetDiff / DeepBioisostere / Joint as `0.72`.
- Supplementary Table 7 caption reports `PDB ID: 4P77 and 2EWY`.
- Supplementary Table 6 keeps half-up two-decimal rounding for SA values.
- Figure 5 values still match the two rows in `figure5_selected_examples.csv`.
