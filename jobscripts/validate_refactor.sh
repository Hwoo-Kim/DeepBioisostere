#!/bin/bash
#SBATCH -J deepbio_validate
#SBATCH -p 16core
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o /home/mseok/slurm-logs/%x_%j.out
#SBATCH -e /home/mseok/slurm-logs/%x_%j.err

# End-to-end validation of the packaging refactor.
#
# Runs on a compute node because building the fragment-library tensor caches
# parses ~291k fragments with multiprocessing, which must not run on the shared
# login node. CPU only: no --gres is requested.
#
# Submit with:
#   sbatch jobscripts/validate_refactor.sh

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

# No cgroup task isolation on this cluster: pin thread counts to the allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false

# Use the in-tree assets rather than downloading from the Hub.
export DEEPBIOISOSTERE_ASSET_DIR="$PROJECT_DIR"

echo "host:        $(hostname)"
echo "date:        $(date -Is)"
echo "git sha:     $(git rev-parse HEAD 2>/dev/null || echo n/a)"
echo "git dirty:   $(test -n "$(git status --porcelain 2>/dev/null)" && echo yes || echo no)"
echo "cpus:        ${SLURM_CPUS_PER_TASK:-1}"
echo "asset dir:   $DEEPBIOISOSTERE_ASSET_DIR"
echo

echo "===== 1. unit tests ====="
uv run pytest tests/ -q

echo
echo "===== 2. build fragment library tensor caches ====="
# Written next to fragment_library.csv; ~a few minutes, once.
uv run deepbioisostere fragment-library prepare \
    "$PROJECT_DIR/fragment_library" \
    --num-cores "${SLURM_CPUS_PER_TASK:-1}"

echo
echo "===== 3. end-to-end generation (CLI) ====="
uv run deepbioisostere generate \
    --smiles "ClC(Cc1c(C(Nc2c(Br)cccc2)=O)cccc1)=O" \
    --smiles "Cc1ccc2cnc(N(C)CCc3ccccn3)nc2c1" \
    --target mw=0 --target logp=-1 \
    --num-samples 20 \
    --num-cores "${SLURM_CPUS_PER_TASK:-1}" \
    --device cpu \
    --seed 1024 \
    --model-dir "$PROJECT_DIR/model_save" \
    --frag-lib-dir "$PROJECT_DIR/fragment_library" \
    --output "$PROJECT_DIR/validation_generation.csv"

echo
echo "===== 4. sanity-check the generated molecules ====="
uv run python - <<'PY'
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
df = pd.read_csv("validation_generation.csv")
print("rows:", len(df))
print("columns:", list(df.columns))
assert len(df) > 0, "no molecules generated"

smis = df["GEN-MOL-SMI"].dropna().tolist()
valid = [s for s in smis if Chem.MolFromSmiles(s) is not None]
print(f"GEN-MOL-SMI: {len(valid)}/{len(smis)} parse under RDKit")
assert len(valid) == len(smis), "some generated SMILES do not parse"
print("\nOK: end-to-end generation produced only valid molecules")
PY

echo
echo "===== 5. example.py ====="
uv run python example.py

echo
echo "ALL VALIDATION STEPS PASSED"
date -Is
