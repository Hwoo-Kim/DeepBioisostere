#!/bin/bash
#SBATCH -J deepbio_zenodo_who
#SBATCH -p 16core
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH -o /home/mseok/slurm-logs/%x_%j.out
#SBATCH -e /home/mseok/slurm-logs/%x_%j.err

# Which Zenodo account owns draft 21637904, and does it match the account the
# browser is logged in as? A "new version" draft is private to its owner, so if
# the token in ~/.env belongs to a different account than the browser session,
# the draft is invisible in the UI even though it exists.
#
# Prints identifiers only, never the token.
#
# Submit with:
#   sbatch jobscripts/zenodo_whoami.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

set -a
# shellcheck disable=SC1091
source "$HOME/.env"
set +a

uv run --no-sync python - <<'PY'
import os
import requests

h = {"Authorization": f"Bearer {os.environ['ZENODO_TOKEN']}"}
API = "https://zenodo.org/api"

print("=== the published record we branched from ===")
rec = requests.get(f"{API}/records/20603082", timeout=60).json()
print(f"  id     : {rec['id']}")
print(f"  owners : {rec.get('owners')}")

print("\n=== draft 21637904, as seen by the token in ~/.env ===")
r = requests.get(f"{API}/deposit/depositions/21637904", headers=h, timeout=60)
print(f"  HTTP {r.status_code}")
if r.status_code == 200:
    d = r.json()
    print(f"  title  : {d['metadata'].get('title','')[:70]}")
    print(f"  owner  : {d.get('owner')}")
    print(f"  state  : {d.get('state')}  submitted={d.get('submitted')}")
    print(f"  files  : {len(d.get('files', []))}")
    links = d.get("links", {})
    for k in ("html", "self_html", "latest_draft_html", "record_html"):
        if k in links:
            print(f"  link {k:18s}: {links[k]}")
else:
    print("  the token cannot see this draft")

print("\n=== every draft this token can see ===")
r = requests.get(f"{API}/deposit/depositions", headers=h,
                 params={"status": "draft", "size": 20}, timeout=60)
if r.status_code == 200:
    ds = r.json()
    if not ds:
        print("  (none)")
    for d in ds:
        print(f"  {d['id']}  owner={d.get('owner')}  "
              f"{d['metadata'].get('title','')[:50]}")
        print(f"      -> {d.get('links', {}).get('html', '?')}")
else:
    print(f"  HTTP {r.status_code}")
PY
