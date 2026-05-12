# DeepBioisostere SBDD Source Data Reproducibility

This directory contains the source data and provenance needed to reproduce the
SBDD-related manuscript items in `DeepBioisostere.pdf`.

## Manuscript Items Covered

- Main Table 2: success rates for target `4RV4` (`target_idx=68`).
- Main Figure 5: two DeepBioisostere examples from DeepICL on target `4RV4`.
- Supplementary Table 6: selected-target QED and SA statistics for `4P77`,
  `4RV4`, and `2EWY`.
- Supplementary Table 7: additional success-rate results for `4P77` and `2EWY`.

The target index to PDB ID mapping is recorded in `target_mapping.csv`.

## Directory Layout

- `table_2/`: per-molecule raw summary CSVs and derived validity CSVs for
  Main Table 2.
- `figure_5/`: Figure 5 raw summary, filtered candidate summary, selected
  example rows, and panel images.
- `supplementary_table_6/`: all-target and selected-target statistics.
- `supplementary_table_7/`: per-molecule raw summary CSVs and derived validity
  CSVs for Supplementary Table 7.
- `raw_eval/`: raw SBDD model `result.csv` files copied into this source-data
  directory.
- `raw_references/`: selected reference protein, ligand, and affinity files for
  targets 36, 68, and 84.
- `provenance/`: Slurm logs, job index, tarball inventory, and scripts used for
  the final docking workflow.
- `docking.yml`: conda export for the environment used by the final GPU
  docking jobs, with comments documenting external Vina-GPU, CUDA/OpenCL, Boost,
  and reference-data paths that are not installed by conda.
- `provenance/scripts/prepare_docking_rerun_inputs.py`: standard-library helper
  that derives clean docking-only candidate CSVs and `reference.tar` from the
  packaged source data.
- `reproduce_results.py`: standard-library Python checker for the
  manuscript values and provenance consistency.

## Lightweight Reproduction From Packaged Source Data

This path does not rerun docking. It verifies that the submitted source data is
internally consistent. The checker does not keep manuscript numbers as hardcoded
Python constants; it reads the packaged source CSVs and compares derived values
against those CSVs.

From the repository root, first run the standard-library Python checker:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

python3 exps/fig5_new_case_study/reproduce_results.py
```

The checker verifies:

- every path listed in `MANIFEST.tsv` exists;
- Main Table 2 validity values are recomputed from per-molecule raw summary
  CSVs and match the aggregate validity CSVs;
- Supplementary Table 6 selected-target values match the all-target summary
  CSV, including the transformed SA score;
- Supplementary Table 7 validity values are recomputed from per-molecule raw
  summary CSVs and match the aggregate validity CSVs;
- the Figure 5 candidate summary is reproducible from the DeepICL target-68
  raw summary CSV;
- both Figure 5 selected rows exist in the DeepICL target-68 summary CSV and
  candidate summary, and their selected QED, SA, docking, SMILES, fragment, and
  score-difference values match the selected source rows;
- both Figure 5 image files exist;
- Slurm jobs `262943-262966` all have `exit_code=0`, `tarball_created=yes`,
  successful stdout messages, and indexed tarballs.

## Full Computational Reproduction

Full reproduction reruns the generation/docking/evaluation workflow. This is a
Slurm workflow and should not be run directly on a login node.

The generation scripts expose `--seed` and the archived docking script sets
fixed RDKit and Vina seeds. These settings document the intended deterministic
workflow. Exact bitwise reruns of GPU docking can still depend on the RDKit,
AutoDock-Vina-GPU, OpenCL/CUDA driver, and hardware versions. The archived
source data therefore treats the submitted CSV summaries as the authoritative
source data and uses Slurm logs plus script snapshots as provenance for the
docking workflow.

### 1. Required External Runtime

The conda environment used for the final docking jobs is recorded in:

```text
exps/fig5_new_case_study/docking.yml
```

Create the conda environment with:

```bash
conda env create -f exps/fig5_new_case_study/docking.yml
conda activate docking
```

The final jobs used the following core package/runtime versions:

```text
Python 3.11.12
numpy 1.24.4
pandas 2.2.3
RDKit 2025.03.2
Meeko 0.6.1
pdb2pqr 3.7.1
AutoDockTools importable in the conda environment
```

The docking rerun needs external runtime components that are intentionally not
included in this repository. The expected Vina-GPU source tree is:

```text
https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1/tree/main/AutoDock-Vina-GPU-2.1
```

Build or obtain the executable from that tree, then point the wrapper to the
directory containing `AutoDock-Vina-GPU-2-1`, `Kernel1_Opt.bin`, and
`Kernel2_Opt.bin`. CUDA/OpenCL runtime libraries must be visible in the runtime
environment, for example through `PATH`, `LD_LIBRARY_PATH`, or the
`CUDA_OPENCL_PATH` variable. The exact way to expose CUDA/OpenCL is
cluster-specific.

For docking-only reruns from this package, set:

```bash
export VINA_GPU_ROOT=/path/to/Vina-GPU-2.1/AutoDock-Vina-GPU-2.1
export VINA_GPU_EXECUTABLE_PATH=$VINA_GPU_ROOT/AutoDock-Vina-GPU-2-1
export VINA_GPU_OPENCL_BINARY_PATH=$VINA_GPU_ROOT
export CUDA_OPENCL_PATH=/path/to/cuda-or-opencl-runtime
export BOOST_LIB_DIR=/path/to/boost/lib
```

`BOOST_LIB_DIR` is optional if the local Vina-GPU binary already resolves its
Boost libraries. `VINA_GPU_OPENCL_BINARY_PATH` defaults to the directory that
contains `VINA_GPU_EXECUTABLE_PATH`, which is the expected location for the
Vina-GPU kernel binaries. `MK_PREPARE_RECEPTOR_SCRIPT_PATH` and
`MK_PREPARE_LIGAND_SCRIPT_PATH` default to the active conda environment's
`bin/` directory and can also be overridden explicitly. If `conda` is not on
the compute-node `PATH`, set `CONDA_ENV_PREFIX=/path/to/conda/env` before
submission. If the Slurm scheduler copies the wrapper to a spool directory, set
`DOCKING_PYTHON_SCRIPT_PATH` to the absolute path of
`provenance/scripts/run_gpu_docking.py`.

`SBDD_EVAL_ROOT` and `REPLACEMENT_LIBRARY_CSV` are only needed when regenerating
candidate molecules or raw SBDD evaluation outputs. They are not needed for the
docking-only rerun path below.

### 2. Prepare Packaged Docking Rerun Inputs

For reviewer-style docking reproduction, start from the packaged raw summary
CSV files rather than rerunning candidate generation. This reproduces docking
for the exact molecule sets used by Main Table 2 and Supplementary Table 7.

Create clean candidate CSVs and a portable reference tarball:

```bash
python3 exps/fig5_new_case_study/provenance/scripts/prepare_docking_rerun_inputs.py
```

This writes:

```text
exps/fig5_new_case_study/docking_rerun_inputs/gen/<model>/<target_idx>/<condition>.csv
exps/fig5_new_case_study/docking_rerun_inputs/docking_input_manifest.csv
exps/fig5_new_case_study/docking_rerun_inputs/reference.tar
```

The generated input CSVs contain only the columns consumed by the docking script:

```text
INPUT-MOL-IDX, INPUT-MOL-SMI, GEN-MOL-IDX, GEN-MOL-SMI
```

This avoids carrying stale docking-score columns back into a rerun.

Use either the generated tarball:

```bash
export REFERENCE_TAR_PATH=$PWD/exps/fig5_new_case_study/docking_rerun_inputs/reference.tar
```

or the packaged reference directory directly:

```bash
export REFERENCE_DIR_PATH=$PWD/exps/fig5_new_case_study/raw_references/reference
```

### 3. Run GPU Docking With Slurm

Submit one docking job per prepared candidate CSV. The archived final Slurm
wrapper is:

```text
exps/fig5_new_case_study/provenance/scripts/run_gpu_docking.sbatch.sh
```

Example:

```bash
sbatch exps/fig5_new_case_study/provenance/scripts/run_gpu_docking.sbatch.sh \
  exps/fig5_new_case_study/docking_rerun_inputs/gen/DeepICL/68/0.10_-1.0.csv
```

The wrapper stages `reference.tar` or `REFERENCE_DIR_PATH` to `/scratch/$USER`,
runs:

```text
exps/fig5_new_case_study/provenance/scripts/run_gpu_docking.py
```

and writes result tarballs under:

```text
exps/fig5_new_case_study/docking_rerun_inputs/docking_gpu_results
```

Set `KEEP_DOCKING_SCRATCH=1` to preserve the per-job scratch directory for
debugging. The default behavior removes scratch files after creating the
persistent tarball. Set `ARCHIVE_DOCKING_STRUCTURES=1` to include intermediate
ligand/receptor preparation files, Vina config/log files, and docked PDBQT
outputs in the persistent tarball.

After docking, regenerate the aggregate validity CSVs from the new
`docking_gpu_results` tarballs with `compute_validity.py` as shown below.

### 4. Optional Full Candidate Regeneration

The package above is enough for docking-only verification of the reported
molecule sets. To regenerate DeepBioisostere candidates before docking,
candidate CSVs have the form:

```text
exps/fig5_new_case_study/20250826_new_data_main_model/gen/<model>/<target_idx>/0.10_-1.0.csv
```

They are produced with `run_generate.py`, for example:

```bash
python run_generate.py DeepICL 68 \
  --result_dir exps/fig5_new_case_study/20250826_new_data_main_model \
  --properties qed sa \
  --target_properties 0.10 -1.0 \
  --seed 0
```

Repeat this for models `DeepICL`, `targetdiff`, `decompdiff`, and `pocket2mol`
and target indices `36`, `68`, and `84`.

Baseline candidate CSVs can be produced with `run_baseline.py` when reproducing
the random, frequency-based, and MMPA-based strategies.

The final successful archived jobs are listed in:

```text
exps/fig5_new_case_study/provenance/final_docking_job_index.csv
```

### 5. Recreate Table Source CSVs

The submitted package includes the per-molecule raw docking summaries used to
derive the success-rate tables:

```text
exps/fig5_new_case_study/table_2/raw_summaries/
exps/fig5_new_case_study/supplementary_table_7/raw_summaries/
```

The aggregate validity CSVs reported in the manuscript are stored in:

```text
exps/fig5_new_case_study/table_2/
exps/fig5_new_case_study/supplementary_table_7/
```

`reproduce_results.py` recomputes QED, SA, docking, and joint success rates
directly from the raw summaries, checks them against the aggregate
`validity_*.csv` files, and then checks the rounded manuscript display values.

When rerunning the full docking workflow, regenerate the aggregate validity CSVs
from the newly produced docked `summary_*.csv` outputs with:

```bash
python exps/fig5_new_case_study/provenance/scripts/compute_validity.py \
  run-root \
  --run-root exps/fig5_new_case_study/docking_rerun_inputs \
  --model DeepICL \
  --target-idx 68 \
  --output-dir exps/fig5_new_case_study/table_2
```

The validity rules are `QED_DIFF > 0`, `SA_DIFF < 0`, and
`abs(GEN-MOL-DOCKED-SCORE - INPUT-MOL-DOCKED-SCORE) < 1.36`. The joint score is
the fraction satisfying all three rules. Repeat the command for each model and
target. `Generated Ratio` is not used for the manuscript success rates; pass
`--generated-ratio-denominator` only when recreating that diagnostic column from
a known attempted-generation denominator.

Supplementary Table 6 target statistics are derived from the raw SBDD eval CSVs
with the target-selection helper:

```bash
python exps/fig5_new_case_study/provenance/scripts/select_target.py
```

The final selected-target source file is:

```text
exps/fig5_new_case_study/supplementary_table_6/table6_selected_targets.csv
```

SA in Supplementary Table 6 is transformed as:

```text
SA_score = 10 - 9 * raw_SA
```

Use half-up two-decimal rounding for final table display.

### 6. Recreate Figure 5 Values

The filtered candidate summary used immediately before drawing Figure 5 is:

```text
exps/fig5_new_case_study/figure_5/figure5_candidate_summary.csv
```

It is generated from the full DeepICL target-68 summary with the notebook
filter used for the final Figure 5 candidates:

```bash
python exps/fig5_new_case_study/provenance/scripts/prepare_figure5_candidate_summary.py
```

The filter keeps rows with `QED > 0.4`, `0 < QED_DIFF <= 0.1`,
`-1.0 <= SA_DIFF < -0.5`, `abs(SCORE_DIFF) < 0.2`, and `QED_DIFF > 0.08`,
then sorts the retained rows by `SA_DIFF`.

The numerical source for the two selected Figure 5 examples is:

```text
exps/fig5_new_case_study/figure_5/figure5_selected_examples.csv
```

The source rows are checked against:

```text
exps/fig5_new_case_study/figure_5/summary_DeepICL_68_0.10_-1.0.csv
exps/fig5_new_case_study/figure_5/figure5_candidate_summary.csv
exps/fig5_new_case_study/figure_5/filtered_deepicl_68.csv
```

The panel images are:

```text
exps/fig5_new_case_study/figure_5/images/figure5a_deepicl68_input28_gen59.png
exps/fig5_new_case_study/figure_5/images/figure5b_deepicl68_input7_gen64.png
```

`figure_5/filtered_deepicl_68.csv` is the legacy notebook-exported filename for
the same candidate summary. `figure_5/filtered_results_68_tracked_head.csv` is
kept for tracked-file provenance. It is not the selected-row source for panel B.
