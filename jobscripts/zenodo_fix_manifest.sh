#!/bin/bash
#SBATCH -J deepbio_zen_manifest
#SBATCH -p 16core
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH -o /home/mseok/slurm-logs/%x_%j.out
#SBATCH -e /home/mseok/slurm-logs/%x_%j.err

# Regenerate MANIFEST.sha256 so it actually matches the deposit.
#
# Two things were wrong with the uploaded one:
#   1. It predated today's rewritten READMEs, so 5 hashes were stale.
#   2. It listed checkpoints as model_save/DeepBioisostere_*.pt, but they are
#      uploaded bare at the deposit root, so those paths never resolved.
#
# The README tells users to run `sha256sum -c MANIFEST.sha256`, so a manifest
# that fails verification is worse than none at all.
#
# Paths here match what a user actually has after downloading:
#   exps/...                     (from extracting the tarball)
#   DeepBioisostere_*.pt         (downloaded individually)
#
# Submit with:
#   sbatch jobscripts/zenodo_fix_manifest.sh [draft_id]

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
set -a; source "$HOME/.env"; set +a

DRAFT="${1:-21639396}"
UPLOAD="$PWD/.verify/zenodo_upload"
M="$UPLOAD/MANIFEST.sha256"

echo "draft: $DRAFT"
echo "regenerating manifest from the current tree ..."
: > "$M"
find exps -type f | sort | xargs sha256sum >> "$M"
for f in model_save/DeepBioisostere_*.pt; do
    # record under the name the file actually has in the deposit
    printf '%s  %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "$(basename "$f")" >> "$M"
done
echo "  entries: $(wc -l < "$M")"
echo "  exps/:   $(grep -c '^.*  exps/' "$M")"
echo "  models:  $(grep -c 'DeepBioisostere_.*\.pt$' "$M")"

echo
echo "self-check: verify the manifest against the working tree"
sha256sum -c "$M" --quiet 2>&1 | head -5 \
  && echo "  (checkpoint entries are basenames, so they only verify from the deposit root)" || true
# exps entries must verify from here; model entries will not (basename form)
grep '  exps/' "$M" | sha256sum -c --quiet && echo "  all exps/ entries verify"

echo
echo "uploading the corrected manifest ..."
uv run --no-sync python - "$DRAFT" <<'PY'
import os, sys, requests
from pathlib import Path
draft_id = sys.argv[1]
API = "https://zenodo.org/api"
h = {"Authorization": f"Bearer {os.environ['ZENODO_TOKEN']}"}
d = requests.get(f"{API}/deposit/depositions/{draft_id}", headers=h, timeout=60)
d.raise_for_status(); d = d.json()
existing = {f.get("filename"): f["id"] for f in d.get("files", [])}
name = "MANIFEST.sha256"
if name in existing:
    requests.delete(f"{API}/deposit/depositions/{draft_id}/files/{existing[name]}",
                    headers=h, timeout=60)
    print(f"  removed previous {name}")
p = Path(".verify/zenodo_upload") / name
with p.open("rb") as fh:
    r = requests.put(f"{d['links']['bucket']}/{name}", data=fh, headers=h, timeout=None)
r.raise_for_status()
chk = requests.get(f"{API}/deposit/depositions/{draft_id}", headers=h, timeout=60).json()
names = sorted(f.get("filename") for f in chk.get("files", []))
print(f"  uploaded. draft holds {len(names)} files")
if name not in names:
    print("  FAILED: manifest not attached"); sys.exit(1)
print(f"\n  review: https://zenodo.org/deposit/{draft_id}")
PY
