#!/bin/bash
#SBATCH -J deepbio_repro_fig2
#SBATCH -p 3090
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o /home/mseok/slurm-logs/%x_%j.out
#SBATCH -e /home/mseok/slurm-logs/%x_%j.err

# Reproduce the published Figure 2 generation and diff it against the paper's
# csv. GPU, because the original run used cuda:0 and multinomial sampling over
# model probabilities can diverge between CPU and GPU on floating point alone.
#
#   A. against the original freq_1 directory  -> is the science reproduced?
#   B. against nothing but the Hugging Face Hub -> does a clean user get the
#      same numbers, with no local checkout and no env overrides?
#
# B is the real end-to-end test. It resolves the checkpoints and the fragment
# library purely through the Hub, into a scratch cache, exactly as someone who
# just ran `pip install deepbioisostere` would.
#
# Submit with:
#   sbatch jobscripts/reproduce_fig2.sh

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false
# Required by torch.use_deterministic_algorithms for cuBLAS reductions, and
# it must be set before CUDA initialises, hence here rather than in Python.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "host:     $(hostname)"
echo "date:     $(date -Is)"
echo "git sha:  $(git rev-parse HEAD)"
echo "gpu:      $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo none)"
echo

# Fail fast rather than burning hours and dying at the first .to(device).
# `cuda.is_available()` alone is not enough: it can be True while the first real
# allocation still aborts on a driver/runtime major-version mismatch.
uv run --no-sync python - <<'PY'
import sys
import torch
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("ERROR: CUDA unavailable. Check the cu124 pin in pyproject.toml.")
    sys.exit(1)
print("device:", torch.cuda.get_device_name(0))
x = torch.ones(8, device="cuda:0")           # forces a real context + allocation
print("allocation ok, sum =", float(x.sum()))
PY

ORIG_FRAG_LIB=/home/share/DATA/swkim/old/DeepBioisostere/freq_1

# Worker count is a MEMORY decision here, not a speed one.
#
# The generator holds the whole parsed fragment library (~7 GB resident) before
# it forks: DataLoader workers in Generator.__init__, and a multiprocessing Pool
# per generate() call. Fork is copy-on-write, but CPython's refcounter writes to
# every object header it touches, so each child materialises its own copy. At
# --num-workers 8 that reached 51 GB RSS and exhausted /dev/shm, killing the run
# at teardown after 55 minutes with no output.
#
# Slurm here reports RealMemory=1 and enforces nothing, so an over-subscribed
# job is not rejected, it just dies late. Two workers keeps the peak near 15 GB.
# The embedding pass is GPU-bound anyway, so this costs little wall-clock.
WORKERS=2

echo
echo "===== A. reproduction against the ORIGINAL freq_1 directory ====="
if [ -d "$ORIG_FRAG_LIB" ]; then
    uv run --no-sync python tools/reproduce_fig2.py \
        --device cuda:0 \
        --num-workers "$WORKERS" \
        --frag-lib-dir "$ORIG_FRAG_LIB" \
        --model-dir "$PROJECT_DIR/model_save" \
        --output "$PROJECT_DIR/fig2_reproduced_freq1.csv"
else
    echo "  $ORIG_FRAG_LIB not accessible, skipped"
fi

echo
echo "===== B. same run resolved ENTIRELY from the Hugging Face Hub ====="
echo "  No --model-dir, no --frag-lib-dir, no DEEPBIOISOSTERE_ASSET_DIR."
echo "  Cache is a scratch dir, so every asset is really downloaded."
HFCACHE="$PROJECT_DIR/.verify/fig2_hfcache"
rm -rf "$HFCACHE"; mkdir -p "$HFCACHE"

env -u DEEPBIOISOSTERE_ASSET_DIR \
    DEEPBIOISOSTERE_CACHE_DIR="$HFCACHE" \
    uv run --no-sync python tools/reproduce_fig2.py \
        --device cuda:0 \
        --num-workers "$WORKERS" \
        --output "$PROJECT_DIR/fig2_reproduced_hf.csv"

echo
echo "===== C. A vs B: does the Hub give the same answer as the local copy? ====="
uv run --no-sync python - <<'PY'
import sys
from pathlib import Path
import pandas as pd

a_p, b_p = Path("fig2_reproduced_freq1.csv"), Path("fig2_reproduced_hf.csv")
if not (a_p.is_file() and b_p.is_file()):
    print("  one of the two runs did not produce output; skipping")
    sys.exit(0)

a, b = pd.read_csv(a_p), pd.read_csv(b_p)
print(f"  freq_1 rows: {len(a)}   hub rows: {len(b)}")
if a.equals(b):
    print("  IDENTICAL: the Hub assets reproduce the local run exactly")
    sys.exit(0)

print("  DIFFERS:")
if list(a.columns) != list(b.columns) or len(a) != len(b):
    print("    shape/columns differ")
else:
    for c in a.columns:
        if not a[c].equals(b[c]):
            n = int((a[c].astype(str) != b[c].astype(str)).sum())
            print(f"    column {c!r}: {n}/{len(a)} rows differ")
sys.exit(1)
PY

echo
date -Is
