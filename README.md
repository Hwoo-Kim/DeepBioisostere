# DeepBioisostere

**Autonomous bioisosteric replacement for multi-property optimization in drug design**

[![Paper](https://img.shields.io/badge/Nature%20Communications-10.1038%2Fs41467--026--75512--9-b31b1b)](https://doi.org/10.1038/s41467-026-75512-9)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20603082.svg)](https://doi.org/10.5281/zenodo.20603082)
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
git clone https://github.com/Hwoo-Kim/DeepBioisostere.git
cd DeepBioisostere
uv sync
```

There is no longer any conda requirement, and no `torch-scatter` /
`torch-sparse` / `torch-cluster`: those compiled extensions have been replaced
by native `torch` operations (see `src/deepbioisostere/scatter.py`). A plain
`pip install` is sufficient on CPU and GPU alike.

### If you have a GPU: check your driver first

`pip install` pulls whatever torch PyPI currently defaults to, which is built
against **CUDA 13.0** and needs a **580 or newer** driver. On an older driver it
installs fine and then dies at the first GPU allocation:

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12040)
```

Check what you have, and pick a matching index:

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

| Driver | CUDA | Install with |
|---|---|---|
| ≥ 580 | 13.0 | `pip install deepbioisostere` |
| ≥ 525 | 12.x | `pip install deepbioisostere --extra-index-url https://download.pytorch.org/whl/cu124` |
| none | CPU | `pip install deepbioisostere --extra-index-url https://download.pytorch.org/whl/cpu` |

CUDA 12.x has minor-version compatibility, so a cu124 build runs on any 12.x
driver. 13.0 is a major bump and does not fall back.

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
rather than a silent fall-through to a different copy. A source checkout that
already has `model_save/` and `fragment_library/` works with no download, and an
offline machine works once the cache is warm. Override the repo with
`$DEEPBIOISOSTERE_HF_REPO`.

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

Generation needs `frag_features.pkl`, a pre-parsed tensor cache derived from
`fragment_library.csv`. It is published (708 MB) so that a first run is a
download. If it is ever absent it is rebuilt automatically, but that parses all
140,096 fragments and takes **on the order of an hour**, so prefer the download.

Training additionally needs `frag_brics_maskings.pkl`. That one is ~3 GB and is
*not* published; it is rebuilt locally on first training run. To build either
ahead of time — inside a batch job rather than an interactive session:

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
[10.5281/zenodo.20603082](https://doi.org/10.5281/zenodo.20603082).

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

Paths below are relative to `data/`. Note that step 8 sits one level up from
steps 2–7.

| Step | Script | What it does |
|---|---|---|
| 1 | *(manual)* | Download ChEMBL activities (`pChEMBL`, SMILES, ChEMBL ID) |
| 2 | `fragment_library/chembl/parse_csv.py` | Parse the raw export |
| 3 | `fragment_library/chembl/filter_chembl.py` | Apply the activity and property filters |
| 4 | `fragment_library/make_frag_db.py` | Enumerate the fragment database |
| 5 | `fragment_library/filter_pair.py` | Cap variable-part size |
| 6 | `fragment_library/parse_db.py` | Turn the database into matched pairs |
| 7 | `fragment_library/process_pair.py`, `fragment_library/analyze_pair.py` | Assemble and summarise the pair data |
| 8 | `divide.py` / `divide.sh` | Split into train / validation / test |

The published library was built with these filters:

- activity `0 ≤ pChEMBL ≤ 10,000 nM`, molecular weight ≤ 800 Da, salts removed
- variable parts capped at **12 heavy atoms**, chosen to admit bicyclic rings
- for `A-B-C` vs `A-D-C`, pairs are dropped when `B` or `D` outweighs `A+C`

The result is **140,096** insertion fragments, split 112,076 train / 14,013
validation / 14,007 test. Generation selects fragments *by index* into this
library, so a library rebuilt with different settings will not reproduce the
paper's outputs even with identical weights.

Two caveats before running any of this: the scripts were written for a cluster
with a job scheduler and expect a `jobscripts/` submission pattern described in
`SCRIPTS.md`, and step 4 over the full ChEMBL set is measured in CPU-days.

## Reproducing the paper

The analysis notebooks, per-figure source data, figure outputs, docking
provenance **and the reproduction scripts** are archived on Zenodo, not in this
repository:

**[10.5281/zenodo.20603082](https://doi.org/10.5281/zenodo.20603082)**

That record is self-contained: you do not need this repository to reproduce a
figure. The code comes from PyPI, everything else from the record.

```bash
pip install deepbioisostere
# download DeepBioisostere-experiments.tar.gz and the reproduce_*.py scripts
tar -xzf DeepBioisostere-experiments.tar.gz          # creates ./exps/

python reproduce_fig4.py   --device cpu               # minutes
python reproduce_fig3.py   --target-prop logp --device cuda:0 --num-workers 2
python reproduce_fig3.py   --target-prop qed  --device cuda:0 --num-workers 2
python reproduce_tables.py                            # Table 1 + the t-tests

export CUBLAS_WORKSPACE_CONFIG=:4096:8                # Figure 2 only
python reproduce_fig2.py --device cuda:0 --num-workers 2
```

Checkpoints and the fragment library are fetched from Hugging Face on first use,
so extracting the archive is the only manual step. The record's `README.md`
documents every file, and each `exps/fig*/` directory has its own README with
that experiment's exact settings.

### How much reproduces

| Target | Result |
|---|---|
| Figure 4 | 100/100 molecules **and their rank order** |
| Figure 3, both panels | 300/300 molecules each |
| Table 1 | 20/24 numbers identical at 3 decimals |
| SI Fig. 4 / Table 3 | 74/84 numbers identical to 1e-9 |
| Figure 2 | **99.32%** of 50,487 molecules; **99.75%** under the paper's original pins |

Figure 2 is the one experiment exposed to GPU nondeterminism, and the only one
that does not come back exactly. The residual is *swaps, not losses*: nearly
every missing row is paired with an extra row from the same input molecule at a
near-tied probability, and the missing molecules sit in the low-probability tail.

Everything that moves is a case3 SA value, and it moves because `rdkit`'s
`sascorer` changed, not because the model did: scoring the same 45,692 published
molecules under rdkit 2022 and rdkit 2026 gives **zero** disagreement on logP,
Mw and QED, and 24 disagreements on SA. Table 1's SA row shifts by 0.001 and no
significance verdict in Table 3 changes.

### Four things decide whether a rerun matches

**Determinism — for Figure 2 only.** On CUDA, `scatter_add_` reduces with
atomics, so summation order varies between runs. The perturbation is ~2e-6 —
chemically meaningless — but it changes multinomial draws and therefore *which*
molecules are sampled. Without both of the following, two Figure 2 runs at the
same seed agree on ~98% of molecules rather than 100%:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # must precede CUDA init
```
```python
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

This costs roughly 50% wall clock. Figures 3 and 4 are exempt: they call
`generate_with_leaving_frag`, which pins the leaving fragment, so the
multinomial draws from a one-element distribution and the insertion choice is a
deterministic top-k. They need neither a seed nor these flags.

CPU generation is deterministic **at a fixed thread count**: two runs on one
machine are bitwise identical, but changing `OMP_NUM_THREADS` between them
reorders torch's reductions and moves `PREDICTED-PROB` by ~5e-11. That is far
too small to change which molecules are produced, but it does mean a
byte-for-byte `diff` of two csvs only means something when the thread count
matched.

**The fragment library.** Generation picks an insertion fragment *by index* into
the library, so a different library silently yields different molecules. The
published runs used the **140,096**-fragment library, which is what this package
resolves from the Hub. A superseded 145,854-fragment copy exists and is not what
the paper used. The `new_frag_type` split matters just as much, and the
published runs did not all use the same one — Figure 3's two panels differ from
each other. The reproduction scripts default to the right one per figure.

**Worker count is a memory setting, not a speed one.** The generator holds the
parsed library (~7 GB) before forking, and CPython's refcounter touches every
object header, so each worker materialises its own copy rather than sharing it.
At 8 workers this reaches 51 GB RSS and can exhaust `/dev/shm`. Use 2.

**What counts as reproduced.** The criterion is the set of generated molecules —
`(input, generated, leaving fragment, inserting fragment)` — not bit-equality of
`PREDICTED-PROB`, which carries float noise. Compare *canonicalised* SMILES on
both sides: rdkit 2022 and 2026 emit different canonical strings for identical
molecules, and comparing raw strings understates agreement substantially — on
Figure 2 it reports 95.5% where the real figure is 99.3%.

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
