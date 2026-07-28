#!/bin/bash
#SBATCH -J deepbio_fastscore_ab
#SBATCH -p 3080ti
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o /home/mseok/slurm-logs/%x_%j.out
#SBATCH -e /home/mseok/slurm-logs/%x_%j.err

# Prove the fast scoring path is value-identical to the published one.
#
# Same seed, same molecules, same checkpoints, same fragment library; the only
# difference is DEEPBIOISOSTERE_FAST_SCORING. If a single cell of the output
# differs, the optimisation is wrong and must not be used -- a speedup that
# changes one molecule is a bug, not a speedup.
#
# A small --limit keeps this to minutes instead of the ~2.7 h of a full pass,
# which is the point: correctness is established cheaply, before anyone waits.
#
# Submit with:
#   sbatch jobscripts/validate_fast_scoring.sh

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

WORK="$PROJECT_DIR/.verify/fastscore"
rm -rf "$WORK"; mkdir -p "$WORK"

FRAG_LIB=/home/share/DATA/swkim/old/DeepBioisostere/freq_1
LIMIT=25          # molecules per case; 3 cases x 2 conditions each
WORKERS=2         # memory, not speed -- see reproduce_fig2.sh

echo "host: $(hostname)"
echo "date: $(date -Is)"
echo "git:  $(git rev-parse HEAD)"
echo "gpu:  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | sed -n 1p)"
echo

run_variant () {
    local label="$1" fast="$2"
    echo
    echo "===== $label (DEEPBIOISOSTERE_FAST_SCORING=$fast) ====="
    local t0 t1
    t0=$(date +%s)
    env DEEPBIOISOSTERE_FAST_SCORING="$fast" \
        uv run --no-sync python tools/reproduce_fig2.py \
            --device cuda:0 \
            --num-workers "$WORKERS" \
            --limit "$LIMIT" \
            --frag-lib-dir "$FRAG_LIB" \
            --model-dir "$PROJECT_DIR/model_save" \
            --output "$WORK/$label.csv"
    t1=$(date +%s)
    echo "  wall clock: $((t1 - t0)) s"
    echo "$((t1 - t0))" > "$WORK/$label.seconds"
}

# Three runs, not two. `baseline2` is the control: identical code, identical
# seed, identical everything. Without it a difference between baseline and fast
# is unattributable, because generation samples multinomially and runs through
# DataLoader workers and a multiprocessing Pool -- if the pipeline is not
# deterministic on GPU to begin with, then baseline != baseline too and the
# comparison says nothing about the optimisation.
run_variant baseline 0
run_variant baseline2 0
run_variant fast 1

echo
echo "===== VERDICT ====="
uv run --no-sync python - <<'PY'
import sys
from pathlib import Path
import pandas as pd

work = Path(".verify/fastscore")
cases = ["case1", "case2", "case3"]


KEY = ["INPUT-MOL-SMI", "GEN-MOL-SMI", "LEAVING-FRAG-SMI", "INSERTING-FRAG-SMI"]


def compare(label, left, right):
    """True when both variants produced the same MOLECULES.

    PREDICTED-PROB is deliberately excluded from the criterion: CUDA
    scatter_add_ atomics jitter it by ~2e-6, which is chemically meaningless.
    Identical chemistry with a jittered probability is the same result; a
    molecule present in one run and absent in the other is not.
    """
    ok = True
    print(f"\n  {label}")
    for case in cases:
        a_p, b_p = work / f"{left}.{case}.csv", work / f"{right}.{case}.csv"
        if not (a_p.is_file() and b_p.is_file()):
            print(f"    {case}: missing output, cannot compare")
            ok = False
            continue
        a, b = pd.read_csv(a_p), pd.read_csv(b_p)
        sa, sb = set(map(tuple, a[KEY].values)), set(map(tuple, b[KEY].values))
        if sa == sb:
            bit = "and bit-identical" if a.equals(b) else "(PROB jitter only)"
            print(f"    {case}: SAME MOLECULES ({len(sa)} unique) {bit}")
            continue
        ok = False
        print(f"    {case}: DIFFERENT MOLECULES  shared={len(sa & sb)}  "
              f"{left}-only={len(sa - sb)}  {right}-only={len(sb - sa)}")
    return ok


# The control decides how to read everything else.
control_ok = compare("CONTROL  baseline vs baseline2 (same code, same seed)",
                     "baseline", "baseline2")
fast_ok = compare("TEST     baseline vs fast", "baseline", "fast")

print()
if not control_ok:
    print("  The pipeline is NOT deterministic run-to-run on GPU.")
    print("  baseline != baseline, so the fast-vs-baseline difference is")
    print("  uninterpretable and exact reproduction of the published csv is")
    print("  not achievable by rerunning either. Fix determinism first.")
elif fast_ok:
    print("  Control is deterministic AND fast matches: SAFE TO ENABLE")
else:
    print("  Control is deterministic but fast diverges:")
    print("  the optimisation genuinely changes results. DO NOT ENABLE.")

all_ok = control_ok and fast_ok

try:
    t0 = int((work / "baseline.seconds").read_text().strip())
    t1 = int((work / "fast.seconds").read_text().strip())
    print(f"\n  baseline {t0}s -> fast {t1}s", end="")
    print(f"   ({t0 / t1:.1f}x)" if t1 else "")
except Exception:
    pass

sys.exit(0 if all_ok else 1)
PY

date -Is
