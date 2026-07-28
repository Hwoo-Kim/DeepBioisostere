#!/bin/bash
#SBATCH -J deepbio_clean_install
#SBATCH -p 16core
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH -o /home/mseok/slurm-logs/%x_%j.out
#SBATCH -e /home/mseok/slurm-logs/%x_%j.err

# Simulate a stranger who has never seen this repository.
#
# Build the wheel, install it into a throwaway venv in a directory that is NOT
# the checkout, and drive it purely through the public entry points with every
# local escape hatch removed:
#
#   - cwd is outside the repo, so no ./model_save or ./fragment_library
#   - DEEPBIOISOSTERE_ASSET_DIR unset, so no env fallback
#   - a scratch HF cache, so every asset is genuinely downloaded
#   - no PYTHONPATH, so `src/` cannot leak in
#
# Anything that needs a manual step, prints a traceback, or emits a scary
# warning is a UX bug, and this job is what catches it.
#
# Submit with:
#   sbatch jobscripts/verify_clean_install.sh

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false

SANDBOX="$PROJECT_DIR/.verify/clean_install"
rm -rf "$SANDBOX"; mkdir -p "$SANDBOX"

echo "host:    $(hostname)"
echo "date:    $(date -Is)"
echo "git sha: $(git rev-parse HEAD)"
echo "sandbox: $SANDBOX"
echo

echo "===== 1. build sdist + wheel ====="
rm -rf dist
uv build
ls -la dist/

echo
echo "===== 2. metadata sanity (what PyPI will show) ====="
uv run --no-sync --with twine twine check dist/*

echo
echo "===== 3. install the WHEEL into a throwaway venv ====="
# A bare venv + pip, deliberately not `uv sync`: this is the path a normal user
# takes, and it must work without the lockfile or the checkout.
uv venv "$SANDBOX/venv" --python 3.11
VPY="$SANDBOX/venv/bin/python"
"$VPY" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$VPY" -m pip install --quiet --upgrade pip
# cu124, matching this cluster's 12.4 driver and the cu124 pin in pyproject.
# PyPI's default torch is built against CUDA 13.0 and aborts on a 12.4 driver,
# so a user who pip-installs here needs this index too. Verifying against it is
# the point: the CPU index would pass while the GPU path stayed broken.
"$VPY" -m pip install --quiet \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    dist/*.whl
echo "installed:"
# `sed -n` rather than `head`: head closes the pipe early, pip dies on SIGPIPE,
# and `set -o pipefail` turns that into a fatal error for the whole job.
"$VPY" -m pip show deepbioisostere | sed -n '1,6p'

echo
echo "===== 4. import from a directory with no repo in sight ====="
cd "$SANDBOX"
env -u DEEPBIOISOSTERE_ASSET_DIR -u PYTHONPATH "$VPY" - <<'PY'
import deepbioisostere as d
print("  version:", d.__version__)
print("  file:   ", d.__file__)
assert "site-packages" in d.__file__, "leaked the source checkout!"
# Touch the lazy exports so a broken __getattr__ shows up here, not later.
for name in ("DeepBioisostere", "Generator", "Conditioner"):
    print(f"  {name}: {getattr(d, name).__name__}")
PY

echo
echo "===== 5. CLI help works with no assets present ====="
CLI="$SANDBOX/venv/bin/deepbioisostere"
env -u DEEPBIOISOSTERE_ASSET_DIR -u PYTHONPATH "$CLI" --help
echo
env -u DEEPBIOISOSTERE_ASSET_DIR -u PYTHONPATH "$CLI" info || true

echo
echo "===== 6. the README quick-start, verbatim, from a cold cache ====="
# This is the exact command the README tells a new user to run. Every asset it
# needs must come down from the Hub without further instruction.
export DEEPBIOISOSTERE_CACHE_DIR="$SANDBOX/hfcache"
time env -u DEEPBIOISOSTERE_ASSET_DIR -u PYTHONPATH "$CLI" generate \
    --smiles "Cc1ccc2cnc(N(C)CCc3ccccn3)nc2c1" \
    --target mw=0 --target logp=-1 \
    --num-samples 20 --device cpu --seed 1024 \
    --output "$SANDBOX/results.csv"

echo
echo "===== 7. did it actually produce valid molecules? ====="
env -u DEEPBIOISOSTERE_ASSET_DIR -u PYTHONPATH "$VPY" - <<'PY'
import sys
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

df = pd.read_csv("results.csv")
print(f"  rows: {len(df)}")
print(f"  columns: {list(df.columns)}")
if not len(df):
    print("  ERROR: no rows generated")
    sys.exit(1)
smis = df["GEN-MOL-SMI"].dropna().tolist()
valid = sum(Chem.MolFromSmiles(s) is not None for s in smis)
print(f"  valid SMILES: {valid}/{len(smis)}")
print("\n  sample:")
for s in smis[:5]:
    print("   ", s)
sys.exit(0 if valid == len(smis) and valid > 0 else 1)
PY

echo
echo "===== 8. the README Python API snippet, verbatim ====="
env -u DEEPBIOISOSTERE_ASSET_DIR -u PYTHONPATH "$VPY" - <<'PY'
from deepbioisostere import Conditioner, DeepBioisostere, Generator

properties = ["logp", "mw"]
model = DeepBioisostere.from_pretrained(properties=properties)
generator = Generator(
    model=model,
    conditioner=Conditioner(phase="generation", properties=properties),
    properties=properties,
    num_sample_each_mol=10,
    device="cpu",
)
smi = "Cc1ccc2cnc(N(C)CCc3ccccn3)nc2c1"
df = generator.generate([(smi, {"mw": 0, "logp": -1})])
print(f"  generate(): {len(df)} rows")

df2 = generator.generate_with_leaving_frag(
    [(smi, "[*]c1ccccn1", 12, {"mw": 0, "logp": -1})]
)
print(f"  generate_with_leaving_frag(): {len(df2)} rows")
PY

echo
echo "===== 9. second run reuses the cache (no re-download) ====="
time env -u DEEPBIOISOSTERE_ASSET_DIR -u PYTHONPATH "$CLI" generate \
    --smiles "Cc1ccc2cnc(N(C)CCc3ccccn3)nc2c1" \
    --target mw=0 --target logp=-1 \
    --num-samples 20 --device cpu --seed 1024 \
    --output "$SANDBOX/results2.csv"

env -u DEEPBIOISOSTERE_ASSET_DIR -u PYTHONPATH "$VPY" - <<'PY'
import sys
import pandas as pd
a, b = pd.read_csv("results.csv"), pd.read_csv("results2.csv")
same = a.equals(b)
print(f"  run1 == run2 at the same seed: {'IDENTICAL' if same else 'DIFFERS'}")
sys.exit(0 if same else 1)
PY

echo
echo "===== 10. cache footprint a user pays for ====="
du -sh "$SANDBOX/hfcache" 2>/dev/null || true

echo
echo "CLEAN INSTALL VERIFICATION PASSED"
date -Is
