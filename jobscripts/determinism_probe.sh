#!/bin/bash
#SBATCH -J deepbio_determinism
#SBATCH -p 3080ti
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -o /home/mseok/slurm-logs/%x_%j.out
#SBATCH -e /home/mseok/slurm-logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "===== default settings ====="
uv run --no-sync python tools/determinism_probe.py --device cuda:0 --repeats 8

echo
echo "===== with deterministic algorithms forced ====="
export CUBLAS_WORKSPACE_CONFIG=:4096:8
uv run --no-sync python - <<'PY'
import torch, sys
torch.use_deterministic_algorithms(True)
sys.argv = ["probe", "--device", "cuda:0", "--repeats", "8"]
exec(open("tools/determinism_probe.py").read().replace('if __name__ == "__main__":', 'if False:'))
try:
    raise SystemExit(main())
except RuntimeError as e:
    print(f"  deterministic mode raised: {e}")
PY
