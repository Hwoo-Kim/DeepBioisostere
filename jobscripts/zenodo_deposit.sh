#!/bin/bash
#SBATCH -J deepbio_zenodo
#SBATCH -p 16core
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH -o /home/mseok/slurm-logs/%x_%j.out
#SBATCH -e /home/mseok/slurm-logs/%x_%j.err

# Stage and upload a new version of the DeepBioisostere Zenodo record.
#
# This creates a DRAFT only. Nothing becomes public and no DOI is minted until
# a human opens the draft and publishes it, or reruns with --publish.
#
# The token is read from ~/.env by this script and is never echoed, never put
# on a command line, and never written to the logs.
#
# Submit with:
#   sbatch jobscripts/zenodo_deposit.sh

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false

# Load credentials. `set -a` exports everything defined in the file; the file
# itself is never printed and `set -x` is never enabled in this script.
if [ -f "$HOME/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$HOME/.env"
    set +a
else
    echo "ERROR: ~/.env not found; ZENODO_TOKEN is required." >&2
    exit 1
fi

if [ -z "${ZENODO_TOKEN:-}" ]; then
    echo "ERROR: ZENODO_TOKEN is not defined in ~/.env." >&2
    exit 1
fi
echo "ZENODO_TOKEN loaded (${#ZENODO_TOKEN} characters)."

STAGE="$PROJECT_DIR/.verify/zenodo_stage"
UPLOAD="$PROJECT_DIR/.verify/zenodo_upload"
rm -rf "$STAGE" "$UPLOAD"
mkdir -p "$STAGE" "$UPLOAD"

echo "host: $(hostname)"
echo "date: $(date -Is)"
echo "git:  $(git rev-parse HEAD)"
echo

echo "===== 0. does the token actually have deposit rights on this record? ====="
# Fail here, cheaply, rather than after copying and compressing 378 MB.
uv run --no-sync python - <<'PY'
import os, sys, time, requests
tok = os.environ["ZENODO_TOKEN"]
h = {"Authorization": f"Bearer {tok}"}


def get(url, **kw):
    """GET with retries. Zenodo intermittently answers with an HTML error page,
    and a previous run died on .json() parsing one of those before doing any
    work at all. Transient infrastructure blips must not look like auth failures.
    """
    last = None
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=60, **kw)
            if r.status_code < 500:
                return r
            last = f"HTTP {r.status_code}"
        except requests.RequestException as exc:
            last = type(exc).__name__
        wait = 2 ** attempt
        print(f"    transient failure ({last}), retrying in {wait}s")
        time.sleep(wait)
    print(f"  ERROR: Zenodo unreachable after 5 attempts ({last})")
    sys.exit(3)


r = get("https://zenodo.org/api/deposit/depositions", headers=h, params={"size": 1})
if r.status_code == 401:
    print("  ERROR: token rejected (401). Check it is a Zenodo token with")
    print("         deposit:write and deposit:actions scopes.")
    sys.exit(1)
r.raise_for_status()
print(f"  token authenticates ok (HTTP {r.status_code})")

# Can this account see the target record as a deposition it may edit?
rec_resp = get("https://zenodo.org/api/records/20603082")
try:
    rec = rec_resp.json()
except ValueError:
    print(f"  ERROR: record endpoint returned non-JSON (HTTP "
          f"{rec_resp.status_code}); Zenodo is likely degraded. Try again later.")
    sys.exit(3)
print(f"  target record : {rec['id']}  owner id {rec.get('owners')}")
dep = get("https://zenodo.org/api/deposit/depositions/20603082", headers=h)
if dep.status_code == 403:
    print("  ERROR: 403 on the deposition endpoint. This Zenodo account does")
    print("         NOT have deposit rights on record 20603082, so it cannot")
    print("         create a new version. Options: ask the record owner, or")
    print("         create a standalone record instead.")
    sys.exit(2)
if dep.status_code == 404:
    print("  ERROR: 404 on the deposition endpoint. Same conclusion as 403:")
    print("         this account cannot act on that record.")
    sys.exit(2)
dep.raise_for_status()
print(f"  deposit rights confirmed (HTTP {dep.status_code})")
PY

echo
echo "===== 1. stage exps/ and the checkpoints ====="
uv run --no-sync python tools/prepare_zenodo_deposit.py --output-dir "$STAGE"

echo
echo "===== 2. assemble the upload set ====="
# exps/ goes up as one tarball (it is thousands of small files), while the
# checkpoints go up individually so a user can fetch a single model without
# downloading 200 MB.
echo "  compressing exps/ ..."
tar -czf "$UPLOAD/DeepBioisostere-experiments.tar.gz" -C "$STAGE" exps
echo "  $(du -h "$UPLOAD/DeepBioisostere-experiments.tar.gz" | cut -f1)  DeepBioisostere-experiments.tar.gz"

echo "  copying checkpoints ..."
cp "$STAGE"/model_save/*.pt "$UPLOAD"/
cp "$STAGE"/MANIFEST.sha256 "$UPLOAD"/
cp "$STAGE"/zenodo.json "$UPLOAD"/

echo
echo "  upload set:"
ls -lh "$UPLOAD" | tail -n +2 | awk '{printf "    %8s  %s\n", $5, $9}'
echo "  total: $(du -sh "$UPLOAD" | cut -f1)"

echo
echo "===== 3. create the DRAFT new version ====="
# No --publish and no --replace-files: the v1.1.0 zip is inherited and kept,
# and nothing goes public.
uv run --no-sync python tools/zenodo_upload.py --stage-dir "$UPLOAD"

echo
echo "Draft created. Review it in the browser, then publish there or rerun"
echo "tools/zenodo_upload.py with --publish."
date -Is
