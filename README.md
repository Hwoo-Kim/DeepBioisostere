# DeepBioisostere

**Autonomous bioisosteric replacement for multi-property optimization in drug design**

[![Paper](https://img.shields.io/badge/Nature%20Communications-10.1038%2Fs41467--026--75512--9-b31b1b)](https://doi.org/10.1038/s41467-026-75512-9)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20603081.svg)](https://doi.org/10.5281/zenodo.20603081)
[![Project page](https://img.shields.io/badge/project-page-blue)](https://mseok.github.io/DeepBioisostere/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

DeepBioisostere is a deep generative model that performs end-to-end bioisosteric
replacement: it autonomously selects *which* fragment of a molecule to replace,
*what* to replace it with, and *how* to attach it, so as to satisfy several
property targets at once — without expert-defined modification sites or
pre-established substitution rules.

---

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Model weights and fragment library](#model-weights-and-fragment-library)
- [Command line interface](#command-line-interface)
- [Python API](#python-api)
- [Training](#training)
- [Building the training data (MMPA)](#building-the-training-data-mmpa)
- [Reproducing the paper](#reproducing-the-paper)
- [Citation](#citation)

## Installation

Requires Python ≥ 3.11.

```bash
pip install deepbioisostere
```

Or, for a development checkout with [uv](https://docs.astral.sh/uv/):

```bash
git clone --depth 1 https://github.com/Hwoo-Kim/DeepBioisostere.git
cd DeepBioisostere
uv sync
```

> [!TIP]
> Use `--depth 1` unless you specifically need the history. The current tree is
> under 1 MB, but the tags `v1.0.0` and `v1.1.0` preserve the pre-refactor
> layout, which committed ~200 MB of checkpoints and the 12 MB fragment library
> directly to git. A full clone fetches every tag and so still transfers about
> 350 MB. Those tags are kept deliberately: `v1.1.0` is the archival reference
> cited from the Zenodo record, so it is not going to be deleted.
>
> Nothing in a checkout carries the weights any more. Checkpoints and the
> fragment library are downloaded from the Hugging Face Hub on first use and
> cached — see [Model weights and fragment library](#model-weights-and-fragment-library).

There is no longer any conda requirement, and no `torch-scatter` /
`torch-sparse` / `torch-cluster`: those compiled extensions have been replaced
by native `torch` operations (see `src/deepbioisostere/scatter.py`). A plain
`pip install` is sufficient on CPU and GPU alike.

### If you have a GPU

`pip install deepbioisostere` is enough. The dependency is capped at
`torch<2.11` so that pip resolves a **CUDA 12** build, which runs on any driver
from **525** up — including the 580+ drivers that CUDA 13 needs, since 12.x
wheels keep working on newer drivers.

> [!NOTE]
> The cap exists because `--extra-index-url` cannot fix this. pip gives extra
> indexes **no priority**: it collects candidates from every index and takes the
> highest version, so pointing it at the cu124 index still yields the newer
> CUDA 13 wheel from PyPI. It installs cleanly and then reports no GPU. `uv`
> does prioritise extra indexes and behaves differently, which makes the failure
> look intermittent across machines. Capping the version is the only fix that
> works for both.

torch 2.11 was the switch from `nvidia-*-cu12` to `-cu13`. CUDA 12.x has
minor-version compatibility so a cu12 build runs on any 12.x driver; 13.0 is a
major bump and does not fall back, which is why the older line is the portable
one.

To check what you have:

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

For a CPU-only machine, or to force a CPU build:

```bash
pip install deepbioisostere --extra-index-url https://download.pytorch.org/whl/cpu
```

Expect the install to fetch **2–4 GB**: torch, the `nvidia-*` CUDA runtime
libraries and `triton` account for nearly all of it. This package itself is
under 100 KB.

A development checkout with `uv sync` is pinned to cu124 via `pyproject.toml`,
which is a choice about *this repository's* environment and does not constrain
anyone who installs the published package.

## Quick start

```bash
deepbioisostere generate \
  --smiles "Cc1ccc2cnc(N(C)CCc3ccccn3)nc2c1" \
  --target mw=0 --target logp=-1 \
  --output results.csv
```

The set of `--target` names selects the checkpoint: `mw` and `logp` here loads
the model trained to condition on both. On the first run the checkpoint and the
fragment library are downloaded and cached.

## Model weights and fragment library

Assets live on the Hugging Face Hub at
[**mseok/DeepBioisostere**](https://huggingface.co/mseok/DeepBioisostere) and
are fetched on first use. They are resolved in this order:

1. an explicit path you pass (`--model-dir`, `--frag-lib-dir`, or `local_dir=`),
2. `$DEEPBIOISOSTERE_ASSET_DIR`,
3. the Hub, cached under `$XDG_CACHE_HOME/deepbioisostere`.

An explicit path is *exclusive*: if the file is not there you get an error
rather than a silent fall-through to a different copy. A directory that already
holds `model_save/` and `fragment_library/` — a pre-refactor checkout, or one
you populated yourself — works with no download, and an offline machine works
once the cache is warm. Override the repo with `$DEEPBIOISOSTERE_HF_REPO`.

A current checkout does **not** ship either of them: they were removed from the
tree so that cloning no longer costs 200 MB.

Pre-fetch everything before going offline:

```bash
deepbioisostere download --all
deepbioisostere info          # show what is available and where it lives
```

### Published checkpoints

| Properties | Standard | Ablation (`use_subgraph_AMPN=False`) |
|---|:--:|:--:|
| `logp` | ✅ | — |
| `mw` | ✅ | — |
| `qed` | ✅ | — |
| `sa` | ✅ | — |
| `logp`, `mw` | ✅ | ✅ |
| `mw`, `qed` | ✅ | ✅ |
| `qed`, `sa` | ✅ | ✅ |

Other pairs (`logp`+`qed`, `logp`+`sa`, `mw`+`sa`) were not trained; asking for
one reports that explicitly rather than failing to download.

### Fragment library caches

> [!IMPORTANT]
> Two different `fragment_library.csv` builds exist and they are **not**
> interchangeable. Generation selects an insertion fragment *by index* into the
> library, so the wrong build silently produces different molecules — no error,
> no warning.
>
> | build | insertion fragments | size | `git hash-object` |
> |---|---|---|---|
> | **published — use this** | 140,096 | 12,407,976 B | `31797fc47717135b31df3f96757b87456164af01` |
> | superseded | 145,854 | 12,961,217 B | `bd689aeb7d3b9025e5876adf5ef13548f08a9d2e` |
>
> This package resolves the correct build from the Hub automatically, and the
> current Zenodo version does not carry a library file at all, so both normal
> paths are safe. One frozen artefact still contains the superseded build:
> `Hwoo-Kim/DeepBioisostere-v1.1.0.zip`, the repository snapshot attached to
> Zenodo version `v1.1.0`. Published Zenodo files are immutable, so that copy
> cannot be corrected in place — it can only be left behind, which is why the
> current version omits it. If you have a `fragment_library.csv` from anywhere
> other than the Hub, check it with `git hash-object` before trusting it.

Generation needs `frag_features.pkl`, a pre-parsed tensor cache derived from
`fragment_library.csv`. It is published (708 MB) so that a first run is a
download. If it is ever absent it is rebuilt automatically, but that parses all
140,096 fragments and takes **on the order of an hour**, so prefer the download.

Training additionally needs `frag_brics_maskings.pkl` (2.9 GB), which maps each
BRICS attachment type to the fragments that can attach there. Generation never
loads it — `Generator` derives that compatibility on the fly — so it is not
distributed at all: not on the Hub with the generation-path assets, and not in
the Zenodo record. Build it locally with the command below if you intend to
train. It is a derived cache, not an input, so a rebuild is not a second-class
substitute for a download.

Either cache can be rebuilt from `fragment_library.csv` instead. Do it in a
batch job rather than an interactive session:

```bash
deepbioisostere fragment-library prepare --num-cores 8
```

**Budget ~7 GB of RAM per worker.** The parsed library is held in the parent
process before the DataLoader forks, and CPython's refcounter touches every
object header, so copy-on-write does not help: each worker ends up with its own
copy. `--num-cores` is therefore a memory setting more than a speed one. 2 is a
good default; 8 has been measured at 51 GB RSS and can exhaust `/dev/shm`.

## Command line interface

```
deepbioisostere generate            Generate bioisosteric replacements
deepbioisostere train               Train a model
deepbioisostere download            Pre-fetch assets into the local cache
deepbioisostere info                Show asset locations and available checkpoints
deepbioisostere fragment-library prepare
                                    Build the fragment library tensor caches
```

Every command takes `-h`. `python -m deepbioisostere` is equivalent to the
`deepbioisostere` script.

Multiple inputs, from a file:

```bash
deepbioisostere generate -i molecules.smi -t qed=0.1 -t sa=-0.5 -n 200 --device cuda:0
```

## Python API

```python
from deepbioisostere import Conditioner, DeepBioisostere, Generator

properties = ["logp", "mw"]

model = DeepBioisostere.from_pretrained(properties=properties)
generator = Generator(
    model=model,
    conditioner=Conditioner(phase="generation", properties=properties),
    properties=properties,
    num_sample_each_mol=100,
    device="cpu",
)

smi = "Cc1ccc2cnc(N(C)CCc3ccccn3)nc2c1"

# 1. Let the model choose the leaving fragment.
df = generator.generate([(smi, {"mw": 0, "logp": -1})])

# 2. Or specify the leaving fragment and its attachment index yourself.
df = generator.generate_with_leaving_frag(
    [(smi, "[*]c1ccccn1", 12, {"mw": 0, "logp": -1})]
)
```

`example.py` and `example.ipynb` are runnable versions of the above.

### Baselines

`BaselineGenerator` implements the three ablated selection strategies the paper
compares against — random leaving fragment with frequency-based insertion,
model-chosen leaving fragment with frequency-based insertion, and random for
both:

```python
from deepbioisostere.baseline_generator import BaselineGenerator

baseline = BaselineGenerator(
    model=model,
    conditioner=Conditioner(phase="generation", properties=properties),
    properties=properties,
    num_sample_each_mol=100,
    device="cpu",
    num_cores=2,
    batch_size=512,
    new_frag_type="all",
)
df = baseline.generate_strategy_1([(smi, {"mw": 0, "logp": -1})])
```

Inputs are `(smiles, targets)` tuples, the same shape `Generator.generate`
takes. `baseline_example.py` runs all three strategies and prints a comparison.

> **Migrating from the pre-release layout.** The package used to be imported as
> `from scripts.model import DeepBioisostere`. It is now
> `from deepbioisostere import DeepBioisostere`. The old form was not actually
> importable as documented — `scripts/` modules imported each other with
> implicit relative imports, which Python 3 only resolves when `scripts/` itself
> is on `sys.path`.

## Training

```bash
deepbioisostere train \
  --data-path /path/to/processed_data.csv \
  --save-name my_run \
  --properties mw,logp
```

Defaults reproduce the configuration used for the published models, so in
practice only those three options are needed. Outputs (checkpoint, logs, loss
history) land in `<project-dir>/model_save/<save-name>/`.

Training additionally builds `frag_brics_maskings.pkl` (~3 GB) on first run.
Build it ahead of time with `deepbioisostere fragment-library prepare`, and note
that `--num-cores` there is a memory decision as much as a speed one: the
parsed library is held before forking, so each worker costs several GB.

Training data is available from Zenodo:
[10.5281/zenodo.20603081](https://doi.org/10.5281/zenodo.20603081).

## Building the training data (MMPA)

The trained models and the fragment library are downloaded automatically, so
this is only needed to rebuild the dataset from scratch — for a different
source database, different filters, or a different fragment-size cutoff.

The pipeline turns ChEMBL activity data into matched molecular pairs, and those
pairs into the fragment library the model selects from. Scripts are under
`data/`; the exact commands are in
[`data/fragment_library/SCRIPTS.md`](data/fragment_library/SCRIPTS.md) and the
filtering rationale in
[`data/fragment_library/README.md`](data/fragment_library/README.md).

Paths below are relative to `data/`. Note that steps 9 and 10 sit outside
`fragment_library/`.

| Step | Script | What it does |
|---|---|---|
| 1 | *(manual)* | Download ChEMBL activities (`pChEMBL`, SMILES, ChEMBL ID) |
| 2 | `fragment_library/chembl/parse_csv.py` | Parse the raw export |
| 3 | `fragment_library/chembl/filter_chembl.py` | Activity and property filters; one row per CID |
| 4 | `fragment_library/make_frag_db.py` | Enumerate the fragment database |
| 5 | `fragment_library/parse_db.py` | Turn the database into matched pairs |
| 6 | `fragment_library/filter_pair.py` | Drop duplicates; require both members from one assay |
| 7 | `fragment_library/process_pair.py` | Add attachment-point information |
| 8 | `fragment_library/filter_by_pchembl.py` | Keep pairs with \|ΔpChEMBL\| ≤ 1.0 |
| 9 | `divide_revised.py` | Transformation-frequency filter, then the 8:1:1 split |
| 10 | `deepbioisostere.fragment_library.parse_fragments` | Build the tensor caches |

Two settings decide whether you get the published dataset:
`filter_by_pchembl.py` is not optional (it implements the |ΔpChEMBL| ≤ 1 cut),
and `divide_revised.py --min_trans_count` must be **1** — its default is 5.
That `1` is what the `freq_1` naming in the original data directories refers
to. `data/fragment_library/README.md` covers both.

The published library was built with these filters:

- activity `0 ≤ pChEMBL ≤ 10,000 nM`, molecular weight ≤ 800 Da, salts removed
- variable parts capped at **12 heavy atoms**, chosen to admit bicyclic rings
- for `A-B-C` vs `A-D-C`, pairs are dropped when `B` or `D` outweighs `A+C`
- both members of a pair must come from the **same assay**
- the two compounds must differ by at most **1.0 pChEMBL unit**, so a pair is an
  isosteric substitution rather than an activity cliff

The result is **140,096** insertion fragments, split 112,076 train / 14,013
validation / 14,007 test. Generation selects fragments *by index* into this
library, so a library rebuilt with different settings will not reproduce the
paper's outputs even with identical weights.

Two caveats before running any of this: the scripts were written for a cluster
with a job scheduler and expect a `jobscripts/` submission pattern described in
`SCRIPTS.md`, and step 4 over the full ChEMBL set is measured in CPU-days.

## Reproducing the paper

Every figure and table except the docking case study can be rerun. The scripts,
notebooks, per-figure source data and docking provenance are archived on Zenodo
rather than in this repository, and that record is self-contained — the code
comes from PyPI, everything else from the record:

**[10.5281/zenodo.20603081](https://doi.org/10.5281/zenodo.20603081)**

```bash
pip install deepbioisostere
# download DeepBioisostere-experiments.tar.gz from the record
tar -xzf DeepBioisostere-experiments.tar.gz
python reproduce_fig4.py --device cpu          # minutes
```

Figures 3 and 4 come back exactly; Figure 2 returns 99.32% of 50,487 molecules,
and the residual is GPU nondeterminism producing *swaps, not losses*.

**See [REPRODUCE.md](https://github.com/Hwoo-Kim/DeepBioisostere/blob/master/REPRODUCE.md)**
for the full command set, what each target reproduces to, and the four things
that decide whether a rerun matches — determinism flags, the fragment library,
worker count, and what counts as reproduced.

## Citation

```bibtex
@article{kim2026deepbioisostere,
  title   = {Autonomous bioisosteric replacement for multi-property optimization in drug design},
  author  = {Kim, Hyeongwoo and Moon, Seokhyun and Zhung, Wonho and
             Kim, Shinwoo and Lim, Jaechang and Kim, Woo Youn},
  journal = {Nature Communications},
  year    = {2026},
  doi     = {10.1038/s41467-026-75512-9},
}
```

Hyeongwoo Kim and Seokhyun Moon contributed equally.

## License

MIT — see [LICENSE](LICENSE).
