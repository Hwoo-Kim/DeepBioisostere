#!/bin/bash
#SBATCH -J deepbio_verify
#SBATCH -p 16core
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH -o /home/mseok/slurm-logs/%x_%j.out
#SBATCH -e /home/mseok/slurm-logs/%x_%j.err

# Verify that the refactored code plus the assets published to Hugging Face
# reproduce inference exactly.
#
# What this establishes, and what it cannot:
#
#   CAN  - the bytes on the Hub are identical to the local checkpoints
#   CAN  - generation is bit-identical whether assets come from disk or the Hub
#   CAN  - generation is reproducible across runs at a fixed seed
#   CAN  - property calculators still match the pre-refactor environment
#   CANNOT - a live A/B against the original torch 1.11 + torch-scatter stack.
#            Those wheels no longer exist, so the pre-refactor comparison rests
#            on the recorded ground truth in tests/test_reproducibility.py and
#            on the cross-rdkit feature-parity check.
#
# Submit with:
#   sbatch jobscripts/verify_hf_roundtrip.sh

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false

WORK="$PROJECT_DIR/.verify"
rm -rf "$WORK"; mkdir -p "$WORK"

echo "host:      $(hostname)"
echo "date:      $(date -Is)"
echo "git sha:   $(git rev-parse HEAD)"
echo "git dirty: $(test -n "$(git status --porcelain)" && echo yes || echo no)"
echo

echo "===== 1. unit tests (includes pre-refactor property regression) ====="
uv run --no-sync pytest tests/ -q

echo
echo "===== 2. checkpoint bytes: local vs Hugging Face ====="
uv run --no-sync python - <<'PY'
import hashlib, os, sys
from pathlib import Path
from deepbioisostere.assets import AVAILABLE_ABLATION_SETS, AVAILABLE_PROPERTY_SETS, checkpoint_filename
from huggingface_hub import hf_hub_download

repo = os.environ.get("DEEPBIOISOSTERE_HF_REPO", "mseok/DeepBioisostere")
local_dir = Path("model_save")
cache = Path(".verify/hf")

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()

cases = [(s, False) for s in AVAILABLE_PROPERTY_SETS] + [(s, True) for s in AVAILABLE_ABLATION_SETS]
bad = 0
for props, abl in cases:
    fn = checkpoint_filename(props, abl)
    lp = local_dir / fn
    hp = hf_hub_download(repo_id=repo, filename=fn, local_dir=str(cache))
    a, b = sha(lp), sha(hp)
    ok = a == b
    bad += not ok
    print(f"  {fn:40s} {'identical' if ok else 'MISMATCH'}")
    if not ok:
        print(f"     local {a}\n     hub   {b}")
print(f"\n{len(cases) - bad}/{len(cases)} checkpoints byte-identical")
sys.exit(1 if bad else 0)
PY

echo
echo "===== 3. fragment library: local vs Hugging Face ====="
uv run --no-sync python - <<'PY'
import hashlib, os, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

repo = os.environ.get("DEEPBIOISOSTERE_HF_REPO", "mseok/DeepBioisostere")
cache = Path(".verify/hf")

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()

# frag_brics_maskings.pkl (~3 GB) is read only during training and is
# deliberately not published, so its absence from the Hub is expected.
REQUIRED = ("fragment_library.csv", "frag_features.pkl")
OPTIONAL = ("frag_brics_maskings.pkl",)

bad = 0
for name in REQUIRED + OPTIONAL:
    lp = Path("fragment_library") / name
    if not lp.is_file():
        print(f"  {name:40s} not present locally, skipped")
        continue
    try:
        hp = hf_hub_download(repo_id=repo, filename=name, local_dir=str(cache))
    except Exception as e:
        if name in OPTIONAL:
            print(f"  {name:40s} not on Hub (expected, training only)")
        else:
            print(f"  {name:40s} MISSING FROM HUB ({type(e).__name__})")
            bad += 1
        continue
    ok = sha(lp) == sha(hp)
    bad += not ok
    print(f"  {name:40s} {'identical' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY

echo
echo "===== 4. generation from LOCAL assets (seed 1024) ====="
uv run --no-sync deepbioisostere generate \
    --smiles "ClC(Cc1c(C(Nc2c(Br)cccc2)=O)cccc1)=O" \
    --smiles "Cc1ccc2cnc(N(C)CCc3ccccn3)nc2c1" \
    --target mw=0 --target logp=-1 \
    --num-samples 50 --num-cores "${SLURM_CPUS_PER_TASK:-1}" --device cpu --seed 1024 \
    --model-dir "$PROJECT_DIR/model_save" \
    --frag-lib-dir "$PROJECT_DIR/fragment_library" \
    --output "$WORK/gen_local.csv"

echo
echo "===== 5. generation again from LOCAL, same seed (determinism) ====="
uv run --no-sync deepbioisostere generate \
    --smiles "ClC(Cc1c(C(Nc2c(Br)cccc2)=O)cccc1)=O" \
    --smiles "Cc1ccc2cnc(N(C)CCc3ccccn3)nc2c1" \
    --target mw=0 --target logp=-1 \
    --num-samples 50 --num-cores "${SLURM_CPUS_PER_TASK:-1}" --device cpu --seed 1024 \
    --model-dir "$PROJECT_DIR/model_save" \
    --frag-lib-dir "$PROJECT_DIR/fragment_library" \
    --output "$WORK/gen_local2.csv"

echo
echo "===== 6. generation from HUGGING FACE assets, same seed ====="
# Force resolution through the Hub cache only: no local dirs given, and the
# env override cleared so nothing falls back to the checkout.
env -u DEEPBIOISOSTERE_ASSET_DIR \
    DEEPBIOISOSTERE_CACHE_DIR="$WORK/hfcache" \
    uv run --no-sync deepbioisostere generate \
    --smiles "ClC(Cc1c(C(Nc2c(Br)cccc2)=O)cccc1)=O" \
    --smiles "Cc1ccc2cnc(N(C)CCc3ccccn3)nc2c1" \
    --target mw=0 --target logp=-1 \
    --num-samples 50 --num-cores "${SLURM_CPUS_PER_TASK:-1}" --device cpu --seed 1024 \
    --output "$WORK/gen_hf.csv"

echo
echo "===== 7. compare the three runs ====="
uv run --no-sync python - <<'PY'
import sys
import pandas as pd

loc = pd.read_csv(".verify/gen_local.csv")
loc2 = pd.read_csv(".verify/gen_local2.csv")
hf = pd.read_csv(".verify/gen_hf.csv")

print(f"  rows: local={len(loc)} local2={len(loc2)} hf={len(hf)}")
print(f"  columns identical: {list(loc.columns) == list(hf.columns)}")

det = loc.equals(loc2)
same = loc.equals(hf)
print(f"\n  determinism  (local vs local, same seed): {'IDENTICAL' if det else 'DIFFERS'}")
print(f"  asset source (local vs hugging face)   : {'IDENTICAL' if same else 'DIFFERS'}")

if not (det and same):
    for label, other in (("local2", loc2), ("hf", hf)):
        if loc.equals(other):
            continue
        if list(loc.columns) != list(other.columns) or len(loc) != len(other):
            print(f"    {label}: shape/columns differ")
            continue
        for c in loc.columns:
            if not loc[c].equals(other[c]):
                n = (loc[c] != other[c]).sum()
                print(f"    {label}: column {c!r} differs in {n} rows")
    sys.exit(1)

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
smis = loc["GEN-MOL-SMI"].dropna().tolist()
valid = sum(Chem.MolFromSmiles(s) is not None for s in smis)
print(f"\n  generated molecules: {valid}/{len(smis)} parse under RDKit")
sys.exit(0 if valid == len(smis) else 1)
PY

echo
echo "ALL VERIFICATION STEPS PASSED"
date -Is
