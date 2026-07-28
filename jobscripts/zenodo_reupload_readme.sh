#!/bin/bash
#SBATCH -J deepbio_zen_readme
#SBATCH -p 16core
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH -o /home/mseok/slurm-logs/%x_%j.out
#SBATCH -e /home/mseok/slurm-logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OMP_NUM_THREADS=1
set -a; source "$HOME/.env"; set +a
uv run --no-sync python - <<'PY'
import os, requests, sys
from pathlib import Path
API="https://zenodo.org/api"; h={"Authorization": f"Bearer {os.environ['ZENODO_TOKEN']}"}
d=requests.get(f"{API}/deposit/depositions/21639396",headers=h,timeout=60).json()
ex={f.get("filename"):f["id"] for f in d.get("files",[])}
if "README.md" in ex:
    requests.delete(f"{API}/deposit/depositions/21639396/files/{ex['README.md']}",headers=h,timeout=60)
p=Path(".verify/zenodo_upload/README.md")
r=requests.put(f"{d['links']['bucket']}/README.md",data=p.open("rb"),headers=h,timeout=None)
r.raise_for_status()
chk=requests.get(f"{API}/deposit/depositions/21639396",headers=h,timeout=60).json()
names=sorted(f.get("filename") for f in chk.get("files",[]))
print(f"draft holds {len(names)} files; README.md present: {'README.md' in names}")
PY
