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

: "${REFERENCE_TAR_PATH:=}"
: "${REFERENCE_DIR_PATH:=}"
: "${VINA_GPU_EXECUTABLE_PATH:?Set VINA_GPU_EXECUTABLE_PATH to AutoDock-Vina-GPU}"
: "${CUDA_OPENCL_PATH:?Set CUDA_OPENCL_PATH to the CUDA/OpenCL runtime path}"
: "${CONDA_ENV_NAME:=docking}"
: "${VINA_GPU_OPENCL_BINARY_PATH:=$(dirname "$VINA_GPU_EXECUTABLE_PATH")}"
: "${BOOST_LIB_DIR:=}"
: "${KEEP_DOCKING_SCRATCH:=0}"
: "${ARCHIVE_DOCKING_STRUCTURES:=0}"

if [ -z "$REFERENCE_TAR_PATH" ] && [ -z "$REFERENCE_DIR_PATH" ]; then
    echo "SLURM ERROR: Set REFERENCE_TAR_PATH or REFERENCE_DIR_PATH."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT_FULL_PATH="${DOCKING_PYTHON_SCRIPT_PATH:-${SCRIPT_DIR}/run_gpu_docking.py}"
if [ ! -f "$PYTHON_SCRIPT_FULL_PATH" ]; then
    SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
    CANDIDATE_PYTHON_SCRIPT="${SUBMIT_DIR}/exps/fig5_new_case_study/provenance/scripts/run_gpu_docking.py"
    if [ -f "$CANDIDATE_PYTHON_SCRIPT" ]; then
        PYTHON_SCRIPT_FULL_PATH="$CANDIDATE_PYTHON_SCRIPT"
    fi
fi
if [ ! -f "$PYTHON_SCRIPT_FULL_PATH" ]; then
    echo "SLURM ERROR: run_gpu_docking.py not found. Set DOCKING_PYTHON_SCRIPT_PATH explicitly."
    exit 1
fi

NUM_CONFORMERS_VALUE=1

CONDITION=$(basename "${GIVEN_INPUT_CSV_FULL_PATH%.*}")
MODEL_NAME=$(basename "$(dirname "$(dirname "$GIVEN_INPUT_CSV_FULL_PATH")")")
TARGET_IDX=$(basename "$(dirname "$GIVEN_INPUT_CSV_FULL_PATH")")
INPUT_RUN_ROOT="${GIVEN_INPUT_CSV_FULL_PATH%%/gen/*}"
if [ "$INPUT_RUN_ROOT" = "$GIVEN_INPUT_CSV_FULL_PATH" ]; then
    echo "SLURM ERROR: Input CSV path must contain a /gen/ component: $GIVEN_INPUT_CSV_FULL_PATH"
    exit 1
fi

JOB_NAME_SUFFIX="GPU_${MODEL_NAME}_${TARGET_IDX}_${CONDITION}"
scontrol update JobID="${SLURM_JOB_ID}" JobName="${JOB_NAME_SUFFIX}"
mv "gpu_dock_out_${SLURM_JOB_ID}.log" "${JOB_NAME_SUFFIX}_${SLURM_JOB_ID}.out" || true
mv "gpu_dock_err_${SLURM_JOB_ID}.err" "${JOB_NAME_SUFFIX}_${SLURM_JOB_ID}.err" || true

PERSISTENT_OUTPUT_TAR_DIR_BASE="${INPUT_RUN_ROOT}/docking_gpu_results"
mkdir -p "$PERSISTENT_OUTPUT_TAR_DIR_BASE"

source "$HOME/.bashrc"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false
if [ -n "$BOOST_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="${BOOST_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi
export OPENCL_LIB_PATH="$CUDA_OPENCL_PATH"
if [ -n "${CONDA_ENV_PREFIX:-}" ]; then
    export CONDA_PREFIX="$CONDA_ENV_PREFIX"
    export PATH="${CONDA_ENV_PREFIX}/bin:${PATH:-}"
fi
if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
    if [ $? -ne 0 ]; then
        echo "SLURM ERROR: Failed to activate conda environment '${CONDA_ENV_NAME}'."
        exit 1
    fi
else
    echo "SLURM WARNING: conda command not found."
fi
if ! command -v python &>/dev/null; then
    echo "SLURM ERROR: python not found. Activate conda or set CONDA_ENV_PREFIX."
    exit 1
fi

MK_PREPARE_RECEPTOR_SCRIPT_PATH="${MK_PREPARE_RECEPTOR_SCRIPT_PATH:-${CONDA_PREFIX}/bin/mk_prepare_receptor.py}"
MK_PREPARE_LIGAND_SCRIPT_PATH="${MK_PREPARE_LIGAND_SCRIPT_PATH:-${CONDA_PREFIX}/bin/mk_prepare_ligand.py}"

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

echo "SLURM INFO: Creating extraction directory: $SCRATCH_REFS_DIR"
mkdir -p "$SCRATCH_REFS_DIR"
if [ $? -ne 0 ]; then
    echo "SLURM ERROR: Failed to create extraction directory"
    exit 1
fi

if [ -n "$REFERENCE_TAR_PATH" ]; then
    echo "SLURM INFO: Copying reference tar to scratch: $SCRATCH_REF_TAR_PATH"
    cp "$REFERENCE_TAR_PATH" "$SCRATCH_REF_TAR_PATH"
    if [ $? -ne 0 ]; then
        echo "SLURM ERROR: Failed to copy reference tar"
        exit 1
    fi

    echo "SLURM INFO: Extracting reference tar into $SCRATCH_REFS_DIR"
    tar xf "$SCRATCH_REF_TAR_PATH" -C "$SCRATCH_REFS_DIR"
    if [ $? -ne 0 ]; then
        echo "SLURM ERROR: Failed to extract reference tar"
        exit 1
    fi
else
    echo "SLURM INFO: Copying reference directory to scratch: $REFERENCE_DIR_PATH"
    mkdir -p "$SCRATCH_REFS_DIR/reference"
    cp -a "$REFERENCE_DIR_PATH"/. "$SCRATCH_REFS_DIR/reference/"
    if [ $? -ne 0 ]; then
        echo "SLURM ERROR: Failed to copy reference directory"
        exit 1
    fi
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
ARCHIVE_INTERMEDIATE_ARGS=""
if [ "${ARCHIVE_DOCKING_STRUCTURES}" = "1" ]; then
    ARCHIVE_INTERMEDIATE_ARGS="--archive_intermediate_files"
fi

COMMAND="\
    python "$PYTHON_SCRIPT_FULL_PATH" \
    --input_csv_path "$GIVEN_INPUT_CSV_FULL_PATH" \
    --job_scratch_dir "$PYTHON_MAIN_SCRATCH_DIR" \
    --output_subdir_name "$PYTHON_OUTPUT_SUBDIR_NAME" \
    --references_dir "$SCRATCH_REFS_DIR/reference/gen" \
    --mk_prepare_receptor_script "$MK_PREPARE_RECEPTOR_SCRIPT_PATH" \
    --mk_prepare_ligand_script "$MK_PREPARE_LIGAND_SCRIPT_PATH" \
    --vina_gpu_executable "$VINA_GPU_EXECUTABLE_PATH" \
    --opencl_binary_path "$VINA_GPU_OPENCL_BINARY_PATH" \
    --num_conformers "$NUM_CONFORMERS_VALUE" \
    --num_processes "$SLURM_CPUS_PER_TASK" \
    --docking_chunk_size "$BATCH_SIZE" \
    $ARCHIVE_INTERMEDIATE_ARGS \
"

srun $COMMAND

PYTHON_EXIT_CODE=$?
echo "SLURM INFO: Python script finished with exit code $PYTHON_EXIT_CODE."

if [ "$PYTHON_EXIT_CODE" -eq 0 ]; then
    TAR_OUTPUT_BASE_FILENAME="${JOB_NAME_SUFFIX}_${SLURM_JOB_ID}_docking_results"
    FINAL_TAR_FILE_PATH="${PERSISTENT_OUTPUT_TAR_DIR_BASE}/${TAR_OUTPUT_BASE_FILENAME}.tar"

    PYTHON_RESULTS_PATH_IN_SCRATCH="${PYTHON_MAIN_SCRATCH_DIR}/${PYTHON_OUTPUT_SUBDIR_NAME}"

    if [ -d "$PYTHON_RESULTS_PATH_IN_SCRATCH" ] && [ "$(ls -A "$PYTHON_RESULTS_PATH_IN_SCRATCH")" ]; then
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

if [ "${KEEP_DOCKING_SCRATCH}" = "1" ]; then
    echo "SLURM INFO: Preserving job scratch root: $JOB_SCRATCH_ROOT"
elif [ -d "$JOB_SCRATCH_ROOT" ]; then
    echo "SLURM INFO: Cleaning up job scratch root: $JOB_SCRATCH_ROOT"
    rm -rf "$JOB_SCRATCH_ROOT"
else
    echo "SLURM WARNING: Job scratch root $JOB_SCRATCH_ROOT not found, skipping cleanup."
fi

exit $PYTHON_EXIT_CODE
