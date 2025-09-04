#!/bin/bash
#SBATCH --job-name=gpu_dock # Will be updated by scontrol
#SBATCH --output=gpu_dock_out_%j.log
#SBATCH --error=gpu_dock_err_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a4000
#SBATCH --time=48:00:00

set -e

GIVEN_INPUT_CSV_FULL_PATH="$1"

if [ -z "$GIVEN_INPUT_CSV_FULL_PATH" ]; then
    echo "SLURM ERROR: Input CSV file path argument is missing. Usage: sbatch script.slurm /path/to/input.csv"
    exit 1
fi
if [ ! -f "$GIVEN_INPUT_CSV_FULL_PATH" ]; then
    echo "SLURM ERROR: Input CSV file not found: $GIVEN_INPUT_CSV_FULL_PATH"
    exit 1
fi

PYTHON_SCRIPT_FULL_PATH="/home/mseok/work/DL/DeepBioisostere/Resubmission_DeepBioisostere/exps/fig5_new_case_study/run_gpu_docking.py"
ORIGINAL_REFERENCE_TAR_FULL_PATH="/home/share/DATA/wonho_SBDD_models_eval/reference.tar"

MK_PREPARE_RECEPTOR_SCRIPT_PATH="/home/mseok/.conda/envs/docking/bin/mk_prepare_receptor.py"
MK_PREPARE_LIGAND_SCRIPT_PATH="/home/mseok/.conda/envs/docking/bin/mk_prepare_ligand.py"
VINA_GPU_EXECUTABLE_PATH="/home/mseok/appl/bin/AutoDock-Vina-GPU-2-1"

NUM_CONFORMERS_VALUE=1

CONDITION=$(basename "${GIVEN_INPUT_CSV_FULL_PATH%.*}")
MODEL_NAME=$(basename "$(dirname "$(dirname "$GIVEN_INPUT_CSV_FULL_PATH")")")
TARGET_IDX=$(basename "$(dirname "$GIVEN_INPUT_CSV_FULL_PATH")")
DIRECTORY_NAME=$(echo "$GIVEN_INPUT_CSV_FULL_PATH" | cut -d/ -f1)

JOB_NAME_SUFFIX="GPU_${MODEL_NAME}_${TARGET_IDX}_${CONDITION}"
scontrol update JobID="${SLURM_JOB_ID}" JobName="${JOB_NAME_SUFFIX}"
mv "gpu_dock_out_${SLURM_JOB_ID}.log" "${JOB_NAME_SUFFIX}_${SLURM_JOB_ID}.out" || true
mv "gpu_dock_err_${SLURM_JOB_ID}.err" "${JOB_NAME_SUFFIX}_${SLURM_JOB_ID}.err" || true

PERSISTENT_OUTPUT_TAR_DIR_BASE="/home/mseok/work/DL/DeepBioisostere/Resubmission_DeepBioisostere/exps/fig5_new_case_study/$DIRECTORY_NAME/docking_gpu_results"
mkdir -p "$PERSISTENT_OUTPUT_TAR_DIR_BASE"

source "$HOME/.bashrc"
export LD_LIBRARY_PATH=/home/mseok/appl/boost_1_77_0/stage/lib:$LD_LIBRARY_PATH
export OPENCL_LIB_PATH=/appl/cuda/cuda-12.4
if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate docking
    if [ $? -ne 0 ]; then
        echo "SLURM ERROR: Failed to activate conda environment 'Bioiso'."
        exit 1
    fi
else
    echo "SLURM WARNING: conda command not found."
fi

USER_SCRATCH_BASE="/scratch/$USER"
JOB_SCRATCH_ROOT="$USER_SCRATCH_BASE/gpu_docking_job_${SLURM_JOB_ID}"

mkdir -p "$JOB_SCRATCH_ROOT"
echo "SLURM INFO: Job scratch root: $JOB_SCRATCH_ROOT"

SCRATCH_REF_TAR_PATH="$JOB_SCRATCH_ROOT/reference.tar"
SCRATCH_REFS_DIR="$JOB_SCRATCH_ROOT"
SCRATCH_INPUT_CSV_FILENAME=$(basename "$GIVEN_INPUT_CSV_FULL_PATH")
SCRATCH_INPUT_CSV_PATH="$JOB_SCRATCH_ROOT/$SCRATCH_INPUT_CSV_FILENAME"
PYTHON_MAIN_SCRATCH_DIR="$JOB_SCRATCH_ROOT/python_processing_area"
PYTHON_OUTPUT_SUBDIR_NAME="output_files_for_tar"
BATCH_SIZE=120

echo "SLURM INFO: Copying reference tar to scratch: $SCRATCH_REF_TAR_PATH"
cp "$ORIGINAL_REFERENCE_TAR_FULL_PATH" "$SCRATCH_REF_TAR_PATH"
if [ $? -ne 0 ]; then
    echo "SLURM ERROR: Failed to copy reference tar"
    exit 1
fi

echo "SLURM INFO: Creating extraction directory: $SCRATCH_REFS_DIR"
mkdir -p "$SCRATCH_REFS_DIR"
if [ $? -ne 0 ]; then
    echo "SLURM ERROR: Failed to create extraction directory"
    exit 1
fi

echo "SLURM INFO: Extracting reference tar into $SCRATCH_REFS_DIR"
tar xf "$SCRATCH_REF_TAR_PATH" -C "$SCRATCH_REFS_DIR"
if [ $? -ne 0 ]; then
    echo "SLURM ERROR: Failed to extract reference tar"
    exit 1
fi

echo "SLURM INFO: Creating Python main scratch directory: $PYTHON_MAIN_SCRATCH_DIR"
mkdir -p "$PYTHON_MAIN_SCRATCH_DIR"
if [ $? -ne 0 ]; then
    echo "SLURM ERROR: Failed to create Python main scratch directory"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo "SLURM INFO: Starting Python docking script..."
COMMAND="\
    python "$PYTHON_SCRIPT_FULL_PATH" \
    --input_csv_path "$GIVEN_INPUT_CSV_FULL_PATH" \
    --job_scratch_dir "$PYTHON_MAIN_SCRATCH_DIR" \
    --output_subdir_name "$PYTHON_OUTPUT_SUBDIR_NAME" \
    --references_dir "$SCRATCH_REFS_DIR/reference/gen" \
    --mk_prepare_receptor_script "$MK_PREPARE_RECEPTOR_SCRIPT_PATH" \
    --mk_prepare_ligand_script "$MK_PREPARE_LIGAND_SCRIPT_PATH" \
    --vina_gpu_executable "$VINA_GPU_EXECUTABLE_PATH" \
    --num_conformers "$NUM_CONFORMERS_VALUE" \
    --num_processes "$SLURM_CPUS_PER_TASK" \
    --docking_chunk_size "$BATCH_SIZE" \
"

srun $COMMAND

PYTHON_EXIT_CODE=$?
echo "SLURM INFO: Python script finished with exit code $PYTHON_EXIT_CODE."

if [ "$PYTHON_EXIT_CODE" -eq 0 ]; then
    TAR_OUTPUT_BASE_FILENAME="${JOB_NAME_SUFFIX}_${SLURM_JOB_ID}_docking_results"
    FINAL_TAR_FILE_PATH="${PERSISTENT_OUTPUT_TAR_DIR_BASE}/${TAR_OUTPUT_BASE_FILENAME}.tar"

    PYTHON_RESULTS_PATH_IN_SCRATCH="${PYTHON_MAIN_SCRATCH_DIR}/${PYTHON_OUTPUT_SUBDIR_NAME}"

    if [ -d "$PYTHON_RESULTS_PATH_IN_SCRATCH" ] && [ "$(ls -A $PYTHON_RESULTS_PATH_IN_SCRATCH)" ]; then
        echo "SLURM INFO: Archiving results from $PYTHON_RESULTS_PATH_IN_SCRATCH to $FINAL_TAR_FILE_PATH"
        tar -cf "$FINAL_TAR_FILE_PATH" -C "$PYTHON_RESULTS_PATH_IN_SCRATCH" .
        if [ $? -eq 0 ]; then
            echo "SLURM INFO: Successfully created tarball: $FINAL_TAR_FILE_PATH"
        else
            echo "SLURM WARNING: Failed to create tarball from $PYTHON_RESULTS_PATH_IN_SCRATCH"
        fi
    else
        echo "SLURM WARNING: Python results directory $PYTHON_RESULTS_PATH_IN_SCRATCH is empty or does not exist. Nothing to archive."
    fi
else
    echo "SLURM WARNING: Python script failed. Skipping archiving of results."
fi

if [ -d "$JOB_SCRATCH_ROOT" ]; then
    echo "SLURM INFO: Cleaning up job scratch root: $JOB_SCRATCH_ROOT"
    rm -rf "$JOB_SCRATCH_ROOT"
else
    echo "SLURM WARNING: Job scratch root $JOB_SCRATCH_ROOT not found, skipping cleanup."
fi

exit $PYTHON_EXIT_CODE
