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
download. If it is ever absent it is rebuilt automatically, but that parses
~146k fragments and takes **on the order of an hour**, so prefer the download.

Training additionally needs `frag_brics_maskings.pkl`. That one is ~3 GB and is
*not* published; it is rebuilt locally on first training run. To build either
ahead of time — inside a batch job rather than an interactive session:

```bash
deepbioisostere fragment-library prepare --num-cores 8
```

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
practice only those three options are needed. `jobscripts/submit_train.sh` is a
Slurm example. Outputs (checkpoint, logs, loss history) land in
`<project-dir>/model_save/<save-name>/`.

Training data is available from Zenodo:
[10.5281/zenodo.20603082](https://doi.org/10.5281/zenodo.20603082).

## Building the training data (MMPA)

To regenerate the dataset from scratch rather than downloading it, the matched
molecular pair analysis scripts are under `data/`. See
`data/fragment_library/SCRIPTS.md`.

## Reproducing the paper

The analysis notebooks, per-figure source data, figure outputs and docking
provenance are archived on Zenodo, not in this repository:

**[10.5281/zenodo.20603082](https://doi.org/10.5281/zenodo.20603082)**

That record contains `DeepBioisostere-experiments.tar.gz` (one directory per
figure, each with its own README), the ten checkpoints, and a SHA-256 manifest.
This repository is the maintained code; the Zenodo record is the archived data.

Three things decide whether a regenerated result matches the published one, and
all three are easy to get wrong:

**Determinism.** On CUDA, `scatter_add_` reduces with atomics, so summation
order varies between runs. The perturbation is ~2e-6 — chemically meaningless —
but it changes multinomial draws and therefore *which* molecules are sampled.
Without both of the following, two runs at the same seed agree on ~98% of
molecules rather than 100%:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # must precede CUDA init
```
```python
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

This costs roughly 50% wall clock. CPU generation is deterministic regardless.

**The fragment library.** Generation picks an insertion fragment *by index* into
the library, so a different library silently yields different molecules. The
published runs used the **140,096**-fragment library, which is what this package
resolves from the Hub. A superseded 145,854-fragment copy exists and is not what
the paper used.

**Worker count is a memory setting, not a speed one.** The generator holds the
parsed library (~7 GB) before forking, and CPython's refcounter touches every
object header, so each worker materialises its own copy rather than sharing it.
At 8 workers this reaches 51 GB RSS and can exhaust `/dev/shm`. Use 2.

**What counts as reproduced.** The criterion is the set of generated molecules —
`(input, generated, leaving fragment, inserting fragment)` — not bit-equality of
`PREDICTED-PROB`, which carries float noise. Compare *canonicalised* SMILES on
both sides: rdkit 2022 and 2026 emit different canonical strings for identical
molecules, and comparing raw strings understates agreement substantially.

Under the paper's original dependency versions this code reproduces 99.75% of
the published Figure 2 molecules; under current ones, 99.32%. The residual are
molecules swapped at near-tied probabilities, which the atomics nondeterminism
above is sufficient to explain.

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
