#!/bin/bash
#SBATCH -J deepbio_hf_freq1
#SBATCH -p 16core
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -o /home/mseok/slurm-logs/%x_%j.out
#SBATCH -e /home/mseok/slurm-logs/%x_%j.err

# Replace the incorrectly uploaded fragment library on HF with the paper's
# freq_1 version (140,096 fragments) and the matching frag_features.pkl (712 MB).
# The current HF upload has the old/repo version (145,854 fragments), which is
# a superseded superset.
#
# Submit with:
#   sbatch jobscripts/upload_hf_freq1.sh

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false

FREQ1_DIR="/home/share/DATA/swkim/old/DeepBioisostere/freq_1"
HF_REPO="mseok/DeepBioisostere"

echo "host:   $(hostname)"
echo "date:   $(date -Is)"
echo "git:    $(git rev-parse --short HEAD)"
echo

# Verify the source files exist and have the right sizes
echo "=== checking source files ==="
wc -l "${FREQ1_DIR}/fragment_library.csv"
ls -lh "${FREQ1_DIR}/frag_features.pkl"

echo
echo "=== uploading fragment_library.csv (freq_1, 140096 frags) ==="
uv run python - <<'PY'
import sys
from pathlib import Path
from huggingface_hub import HfApi
import pandas as pd

freq1 = Path("/home/share/DATA/swkim/old/DeepBioisostere/freq_1")
repo = "mseok/DeepBioisostere"

# Verify row count before uploading
csv = pd.read_csv(freq1 / "fragment_library.csv", sep="\t",
                  dtype={"DATA-TYPE": str, "BRICS-TYPE": str})
n_new = (csv["NEW-OLD"] == "new").sum()
print(f"  total rows: {len(csv)}")
print(f"  NEW rows  : {n_new}  (expected: 140096)")
if n_new != 140096:
    print("ERROR: unexpected row count, aborting")
    sys.exit(1)

api = HfApi()
api.upload_file(
    path_or_fileobj=str(freq1 / "fragment_library.csv"),
    path_in_repo="fragment_library.csv",
    repo_id=repo,
    repo_type="model",
    commit_message="Replace fragment_library.csv with the paper's freq_1 version (140096 fragments)",
)
print("  fragment_library.csv uploaded")
PY

echo
echo "=== uploading frag_features.pkl (freq_1, 140096 keys, ~712 MB) ==="
uv run python - <<'PY'
import sys, types
from pathlib import Path
from huggingface_hub import HfApi

freq1 = Path("/home/share/DATA/swkim/old/DeepBioisostere/freq_1")
repo = "mseok/DeepBioisostere"

# Verify key count
sys.path.insert(0, "src")
from deepbioisostere.dataset import PairData
import pickle

fake_data = types.ModuleType("data")
fake_data.PairData = PairData
sys.modules["data"] = fake_data

with open(freq1 / "frag_features.pkl", "rb") as f:
    feat = pickle.load(f)

n = len(feat)
print(f"  pkl keys: {n}  (expected: 140096)")
if n != 140096:
    print("ERROR: unexpected key count, aborting")
    sys.exit(1)

del feat  # free memory before the upload

api = HfApi()
api.upload_file(
    path_or_fileobj=str(freq1 / "frag_features.pkl"),
    path_in_repo="frag_features.pkl",
    repo_id=repo,
    repo_type="model",
    commit_message="Replace frag_features.pkl with the paper's freq_1 version (140096 keys)",
)
print("  frag_features.pkl uploaded")
PY

echo
echo "=== verifying HF has the correct files ==="
uv run python - <<'PY'
import hashlib, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

freq1 = Path("/home/share/DATA/swkim/old/DeepBioisostere/freq_1")
repo = "mseok/DeepBioisostere"
cache = Path(".verify/hf_freq1")

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()

bad = 0
for name in ("fragment_library.csv", "frag_features.pkl"):
    local = freq1 / name
    remote = hf_hub_download(repo_id=repo, filename=name,
                             local_dir=str(cache), force_download=True)
    ok = sha(local) == sha(Path(remote))
    bad += not ok
    print(f"  {name:35s} {'identical' if ok else 'MISMATCH'}")

sys.exit(1 if bad else 0)
PY

echo
echo "ALL DONE: freq_1 files on HF are byte-identical to source"
date -Is
