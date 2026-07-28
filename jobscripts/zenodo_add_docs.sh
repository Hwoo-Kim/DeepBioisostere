#!/bin/bash
#SBATCH -J deepbio_zen_docs
#SBATCH -p 16core
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH -o /home/mseok/slurm-logs/%x_%j.out
#SBATCH -e /home/mseok/slurm-logs/%x_%j.err

# Attach documentation to the existing Zenodo draft.
#
# The tarball uploaded by zenodo_deposit.sh predates exps/README.md and the
# rewritten per-figure READMEs, so it is rebuilt and replaced here. README.md
# also goes up as a separate top-level file so it is readable on the Zenodo
# landing page without downloading 34 MB first.
#
# Submit with:
#   sbatch jobscripts/zenodo_add_docs.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
set -a; source "$HOME/.env"; set +a

DRAFT="${1:-21639396}"
UPLOAD="$PWD/.verify/zenodo_upload"

echo "draft: $DRAFT"
echo "rebuilding the experiments tarball with the new documentation ..."
tar -czf "$UPLOAD/DeepBioisostere-experiments.tar.gz" exps
ls -lh "$UPLOAD/DeepBioisostere-experiments.tar.gz" | awk '{print "  "$5}'
tar -tzf "$UPLOAD/DeepBioisostere-experiments.tar.gz" | grep -c "README" \
  | xargs -I{} echo "  README files inside: {}"

cp _ai/results/zenodo/README.md "$UPLOAD/README.md"

uv run --no-sync python - "$DRAFT" <<'PY'
import os, sys, requests
from pathlib import Path

draft_id = sys.argv[1]
API = "https://zenodo.org/api"
h = {"Authorization": f"Bearer {os.environ['ZENODO_TOKEN']}"}

d = requests.get(f"{API}/deposit/depositions/{draft_id}", headers=h, timeout=60)
d.raise_for_status()
d = d.json()
bucket = d["links"]["bucket"]
print(f"  draft state={d.get('state')} submitted={d.get('submitted')}")

existing = {f.get("filename"): f["id"] for f in d.get("files", [])}
for name in ("README.md", "DeepBioisostere-experiments.tar.gz"):
    p = Path(".verify/zenodo_upload") / name
    # Replacing a file requires removing the old one first.
    if name in existing:
        requests.delete(f"{API}/deposit/depositions/{draft_id}/files/{existing[name]}",
                        headers=h, timeout=60)
        print(f"  removed previous {name}")
    print(f"  uploading {name} ({p.stat().st_size/1048576:.1f} MB) ... ", end="", flush=True)
    with p.open("rb") as fh:
        r = requests.put(f"{bucket}/{name}", data=fh, headers=h, timeout=None)
    r.raise_for_status()
    print("ok")

chk = requests.get(f"{API}/deposit/depositions/{draft_id}", headers=h, timeout=60).json()
names = sorted(f.get("filename") for f in chk.get("files", []))
print(f"\n  draft now holds {len(names)} files:")
for n in names:
    print(f"    {n}")
missing = {"README.md", "DeepBioisostere-experiments.tar.gz"} - set(names)
if missing:
    print(f"  FAILED: {missing} not attached")
    sys.exit(1)
print(f"\n  review: https://zenodo.org/deposit/{draft_id}")
PY
