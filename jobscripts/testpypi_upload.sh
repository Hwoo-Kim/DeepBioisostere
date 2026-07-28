#!/bin/bash
#SBATCH -J deepbio_testpypi
#SBATCH -p 16core
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -o /home/mseok/slurm-logs/%x_%j.out
#SBATCH -e /home/mseok/slurm-logs/%x_%j.err
#
# Publish to TestPyPI and then prove the published artifact actually installs
# and runs, from a clean venv, exactly as a real user would get it.
#
# TestPyPI is a throwaway index: a bad upload there costs nothing, whereas on
# real PyPI the version number 1.0.0 could never be reused. Nothing in this
# script touches pypi.org.
#
# Needs TESTPYPI_TOKEN in ~/.env (a token created at test.pypi.org, not pypi.org).
# The token is never echoed and never appears on a command line.
#
# Submit with:
#   sbatch jobscripts/testpypi_upload.sh

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false

if [ -f "$HOME/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$HOME/.env"
    set +a
fi

TOKEN="${TESTPYPI_TOKEN:-}"
if [ -z "$TOKEN" ]; then
    echo "ERROR: TESTPYPI_TOKEN is not set in ~/.env." >&2
    echo "Create one at https://test.pypi.org/manage/account/token/" >&2
    exit 1
fi
echo "TESTPYPI_TOKEN loaded (${#TOKEN} characters)."

SANDBOX="$PROJECT_DIR/.verify/testpypi"
rm -rf "$SANDBOX"; mkdir -p "$SANDBOX"

echo "host: $(hostname)"
echo "date: $(date -Is)"
echo "git:  $(git rev-parse HEAD)"
echo

echo "===== 1. rebuild from a clean tree ====="
rm -rf dist
uv build
ls -lh dist/

echo
echo "===== 2. twine check ====="
uv run --no-sync --with twine twine check dist/*

echo
echo "===== 3. upload to TestPyPI ====="
# Credentials via environment, so the token never appears in the process list.
TWINE_USERNAME=__token__ \
TWINE_PASSWORD="$TOKEN" \
TWINE_NON_INTERACTIVE=1 \
uv run --no-sync --with twine twine upload \
    --repository-url https://test.pypi.org/legacy/ \
    --disable-progress-bar \
    dist/*

echo
echo "===== 4. wait for the index to catch up ====="
for i in $(seq 1 30); do
    if curl -sf "https://test.pypi.org/pypi/deepbioisostere/json" >/dev/null 2>&1; then
        echo "  visible on TestPyPI after ~$((i * 10))s"
        break
    fi
    sleep 10
done

echo
echo "===== 5. install FROM TestPyPI into a clean venv ====="
# Dependencies come from real PyPI; only deepbioisostere itself comes from
# TestPyPI, which does not mirror the wider ecosystem.
uv venv "$SANDBOX/venv" --python 3.11
VPY="$SANDBOX/venv/bin/python"
"$VPY" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$VPY" -m pip install --quiet --upgrade pip
"$VPY" -m pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    deepbioisostere
# `sed -n` rather than `head`: head closes the pipe early, pip dies on SIGPIPE,
# and `set -o pipefail` turns that into a fatal error for the whole job.
"$VPY" -m pip show deepbioisostere | sed -n '1,8p'

echo
echo "===== 6. run it, from outside the repo, with no local assets ====="
cd "$SANDBOX"
export DEEPBIOISOSTERE_CACHE_DIR="$SANDBOX/hfcache"
env -u DEEPBIOISOSTERE_ASSET_DIR -u PYTHONPATH \
    "$SANDBOX/venv/bin/deepbioisostere" generate \
    --smiles "Cc1ccc2cnc(N(C)CCc3ccccn3)nc2c1" \
    --target mw=0 --target logp=-1 \
    --num-samples 20 --device cpu --seed 1024 \
    --output "$SANDBOX/results.csv"

env -u DEEPBIOISOSTERE_ASSET_DIR -u PYTHONPATH "$VPY" - <<'PY'
import sys
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
df = pd.read_csv("results.csv")
smis = df["GEN-MOL-SMI"].dropna().tolist()
valid = sum(Chem.MolFromSmiles(s) is not None for s in smis)
print(f"  rows: {len(df)}   valid SMILES: {valid}/{len(smis)}")
sys.exit(0 if smis and valid == len(smis) else 1)
PY

echo
echo "TESTPYPI ROUND TRIP PASSED"
echo "  https://test.pypi.org/project/deepbioisostere/"
echo
echo "Real PyPI is a separate, irreversible step and was NOT touched."
date -Is
