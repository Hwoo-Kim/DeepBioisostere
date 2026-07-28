#!/bin/bash
#SBATCH -J deepbio_legacy_repro
#SBATCH -p 3090
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH -o /home/mseok/slurm-logs/%x_%j.out
#SBATCH -e /home/mseok/slurm-logs/%x_%j.err

# Reproduce Figure 2 under the PAPER'S ORIGINAL dependency versions, taken from
# Resubmission_DeepBioisostere/{environment.yml,requirements.txt}:
#
#   python 3.9.12   torch 1.11.0+cu113   rdkit 2022.03.1   pyg 2.1.0
#   numpy 1.23.4    pandas 1.5.1         scipy 1.9.3
#
# The refactored code reproduced 99.32% of the published molecules under
# torch 2.6 / rdkit 2026.03. This job decides what the missing 0.68% is:
#
#   gap closes  -> caused by dependency versions; the refactor is faithful
#   gap remains -> caused by the refactor; there is a real bug to find
#
# torch-scatter/sparse/cluster ARE installed here, from the same pyg wheel index
# the original requirements.txt used. An earlier attempt omitted them on the
# grounds that src/deepbioisostere/scatter.py replaced torch_scatter -- that is
# true of our code, but not of torch_geometric 2.1.0, whose data/data.py does
# `from torch_sparse import SparseTensor` unconditionally at import time. Newer
# pyg made that optional; 2.1.0 did not.
#
# What is still being tested is that OUR code paths use scatter.py rather than
# torch_scatter; the extensions are present only to satisfy pyg 2.1.0's import.
#
# Determinism is deliberately NOT forced: the original notebook did not set it,
# and the goal here is to match the published conditions, not to improve them.
#
# The project .venv is never touched; this builds a separate throwaway env.
#
# Submit with:
#   sbatch jobscripts/legacy_env_repro.sh

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false

LEGACY="$PROJECT_DIR/.verify/legacy_env"
FRAG_LIB=/home/share/DATA/swkim/old/DeepBioisostere/freq_1
OUT="$PROJECT_DIR/fig2_legacy.csv"

echo "host: $(hostname)"
echo "date: $(date -Is)"
echo "git:  $(git rev-parse HEAD)"
echo "gpu:  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | sed -n 1p)"
echo

echo "===== 1. build the legacy environment ====="
rm -rf "$LEGACY"
uv venv "$LEGACY" --python 3.9
VPY="$LEGACY/bin/python"
"$VPY" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$VPY" -m pip install --quiet --upgrade "pip<24" "setuptools<70" wheel

# cu113 binaries run on this 12.4 driver: CUDA is backward compatible with
# older runtimes, unlike the forward direction that broke the cu130 wheels.
echo "  torch 1.11.0+cu113 ..."
"$VPY" -m pip install --quiet \
    torch==1.11.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

echo "  scientific stack at the original pins ..."
"$VPY" -m pip install --quiet \
    numpy==1.23.4 pandas==1.5.1 scipy==1.9.3

echo "  rdkit 2022.03.1 ..."
# Pre-2022.09 releases were published as rdkit-pypi, not rdkit.
"$VPY" -m pip install --quiet "rdkit-pypi==2022.3.1" \
  || "$VPY" -m pip install --quiet "rdkit==2022.3.1"

echo "  torch-scatter/sparse/cluster from the pyg wheel index ..."
# Prebuilt wheels for exactly torch-1.11.0+cu113 / cp39, so nothing compiles.
"$VPY" -m pip install --quiet \
    torch-scatter==2.0.9 torch-sparse==0.6.15 torch-cluster==1.6.0 \
    -f https://data.pyg.org/whl/torch-1.11.0+cu113.html

echo "  torch_geometric 2.1.0 (+ the deps it needs at that age) ..."
"$VPY" -m pip install --quiet "torch_geometric==2.1.0" tqdm jinja2 requests pyparsing

# The torch 1.11 cu113 wheel bundles only cudart/nvrtc/nvToolsExt and links the
# CUDA math libraries statically into libtorch_cuda.so. torch_sparse ships its
# own .so which needs libcusparse.so.11 dynamically, and there is no system CUDA
# here, so it must be supplied explicitly. torch 2.x solved this by depending on
# the nvidia-* wheels; at 1.11 that plumbing did not exist yet.
echo "  CUDA 11 math libraries for the pyg extensions ..."
"$VPY" -m pip install --quiet \
    nvidia-cusparse-cu11 nvidia-cublas-cu11 nvidia-cusolver-cu11

SP="$LEGACY/lib/python3.9/site-packages"
NVLIBS="$(echo "$SP"/nvidia/*/lib | tr ' ' ':')"
export LD_LIBRARY_PATH="$NVLIBS:$SP/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "  LD_LIBRARY_PATH set for: $(echo "$SP"/nvidia/*/lib | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"

echo
"$VPY" - <<'PY'
import sys, torch
print(f"  python {sys.version.split()[0]}")
print(f"  torch  {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
import numpy, pandas, scipy, rdkit, torch_geometric
print(f"  numpy {numpy.__version__} | pandas {pandas.__version__} | scipy {scipy.__version__}")
print(f"  rdkit {rdkit.__version__} | pyg {torch_geometric.__version__}")
if torch.cuda.is_available():
    x = torch.ones(8, device="cuda:0")
    print(f"  gpu alloc ok, sum={float(x.sum())}, device={torch.cuda.get_device_name(0)}")
else:
    print("  ERROR: CUDA unavailable under torch 1.11")
    sys.exit(1)
PY

# rdkit-pypi (the pre-2022.09 PyPI build) ships no Contrib directory, so
# property.py's `import sascorer` fails. The original environment used conda
# rdkit, which does include it.
#
# The copy taken here is from rdkit 2023.09, not 2022.03, and that is fine:
# fpscores.pkl.gz is byte-identical across versions, and sascorer.py was checked
# to produce EXACTLY the same 42 disagreements with the published csv as the
# 2026 rewrite does when both run on the same rdkit. The SA differences we are
# chasing therefore come from rdkit's C++ layer, not this file -- so what
# matters is that it runs against rdkit 2022.03 here, which it will.
echo "  installing the SA_Score Contrib rdkit-pypi omits ..."
CONTRIB_SRC=/home/mseok/.conda/share/RDKit/Contrib/SA_Score
CONTRIB_DST="$("$VPY" -c 'from rdkit.Chem import RDConfig; print(RDConfig.RDContribDir)')"
mkdir -p "$CONTRIB_DST/SA_Score"
cp "$CONTRIB_SRC/sascorer.py" "$CONTRIB_SRC/fpscores.pkl.gz" "$CONTRIB_DST/SA_Score/"
echo "    -> $CONTRIB_DST/SA_Score"

echo
echo "===== 1b. the pyg extensions must actually load ====="
# Fail here, in seconds, rather than an hour into generation. Both previous
# attempts died on extension loading, each at a different link error.
"$VPY" - <<'PY'
import sys
import torch
for mod in ("torch_scatter", "torch_sparse", "torch_cluster"):
    try:
        m = __import__(mod)
        print(f"  {mod:14s} {getattr(m, '__version__', '?')}  ok")
    except Exception as exc:
        print(f"  {mod:14s} FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
import torch_geometric
from torch_geometric.data import Data
print(f"  torch_geometric {torch_geometric.__version__}  Data import ok")
PY

echo
echo "===== 2. sanity: does the package import at all under 3.9 / torch 1.11? ====="
PYTHONPATH="$PROJECT_DIR/src" "$VPY" - <<'PY'
import deepbioisostere as d
print(f"  deepbioisostere {d.__version__} from {d.__file__}")
from deepbioisostere import Conditioner, DeepBioisostere, Generator
from deepbioisostere.scatter import scatter_sum, scatter_mean
import torch
# the native-torch scatter replacement must behave the same on old torch
src = torch.randn(1000, 8)
idx = torch.randint(0, 50, (1000,))
print(f"  scatter_sum  -> {tuple(scatter_sum(src, idx, dim=0, dim_size=50).shape)}")
print(f"  scatter_mean -> {tuple(scatter_mean(src, idx, dim=0, dim_size=50).shape)}")
# torch_scatter is installed for pyg's sake; confirm OUR package never imports it.
import sys
leaked = [m for m in sys.modules if m.startswith("torch_scatter")]
print(f"  torch_scatter imported by our code: {leaked or 'no'}")
# the property calculators must work here, since SA is what case3 conditions on
from rdkit import Chem
from deepbioisostere.property import calc_SAscore, calc_QED, calc_logP, calc_Mw
m = Chem.MolFromSmiles("Cc1ccc2cnc(N(C)CCc3ccccn3)nc2c1")
print(f"  logP={calc_logP(m):.4f} Mw={calc_Mw(m):.4f} "
      f"QED={calc_QED(m):.4f} SA={calc_SAscore(m):.4f}")
PY

echo
echo "===== 3. reproduce Figure 2 under the original versions ====="
# --allow-nondeterminism: match the original conditions, which did not force
# deterministic algorithms (and torch 1.11 may lack a deterministic scatter_add).
PYTHONPATH="$PROJECT_DIR/src" "$VPY" tools/legacy_repro_shim.py \
    --device cuda:0 \
    --num-workers 2 \
    --allow-nondeterminism \
    --frag-lib-dir "$FRAG_LIB" \
    --model-dir "$PROJECT_DIR/model_save" \
    --output "$OUT"

echo
echo "===== 4. legacy vs modern, molecule-level ====="
uv run --no-sync python - <<'PY'
from pathlib import Path
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

K = ["INPUT-MOL-SMI", "GEN-MOL-SMI", "LEAVING-FRAG-SMI", "INSERTING-FRAG-SMI"]
cache = {}
def canon(s):
    if s in cache: return cache[s]
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    cache[s] = Chem.MolToSmiles(m) if m is not None else s
    return cache[s]

ref = pd.read_csv("exps/fig2_multi_conditioning/csv_files/generation_result_seed2025.csv")
print(f"{'case':7s} {'legacy vs published':>22s} {'modern vs published':>22s}")
for case in ("case1", "case2", "case3"):
    r = ref[ref["CASE"] == case]
    sr = set(map(tuple, r[K].map(canon).values))
    row = [case]
    for tag, path in (("legacy", f"fig2_legacy.{case}.csv"),
                      ("modern", f"fig2_reproduced_freq1.{case}.csv")):
        p = Path(path)
        if not p.is_file():
            row.append("      n/a")
            continue
        d = pd.read_csv(p)
        sd = set(map(tuple, d[K].map(canon).values))
        row.append(f"{100*len(sr & sd)/len(sr):21.2f}%")
    print(f"{row[0]:7s} {row[1]:>22s} {row[2]:>22s}")
PY

echo
date -Is
