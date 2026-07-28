# DeepBioisostere

**Autonomous bioisosteric replacement for multi-property optimization in drug design**

[![Paper](https://img.shields.io/badge/Nature%20Communications-10.1038%2Fs41467--026--75512--9-b31b1b)](https://doi.org/10.1038/s41467-026-75512-9)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20603082.svg)](https://doi.org/10.5281/zenodo.20603082)
[![Project page](https://img.shields.io/badge/project-page-blue)](https://hwoo-kim.github.io/DeepBioisostere/)
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

To install a GPU build of torch, follow the
[PyTorch instructions](https://pytorch.org/get-started/locally/) for your CUDA
version; everything else is pure Python.

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

Checkpoints (10 files, ~20 MB each) and the fragment library (~13 MB) are not
shipped in the wheel. They are resolved on first use, in this order:

1. an explicit path you pass (`--model-dir`, `--frag-lib-dir`, or `local_dir=`),
2. `$DEEPBIOISOSTERE_ASSET_DIR`,
3. the Hugging Face Hub, cached under `$XDG_CACHE_HOME/deepbioisostere`.

So a source checkout that already has `model_save/` and `fragment_library/`
works with no download, and an offline machine works once the cache is warm.

Pre-fetch everything before going offline:

```bash
deepbioisostere download --all
deepbioisostere info          # show what is available and where it lives
```

> **Note.** The Hugging Face repository is configured via the
> `DEFAULT_HF_REPO_ID` constant in `src/deepbioisostere/assets.py`, overridable
> with `$DEEPBIOISOSTERE_HF_REPO`. Set it to the published namespace before
> release.

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

The **first** generation call also builds two tensor caches
(`frag_features.pkl`, `frag_brics_maskings.pkl`) from `fragment_library.csv`.
That takes a few minutes and is done once. To pay the cost up front — for
instance inside a batch job rather than an interactive session:

```bash
deepbioisostere fragment-library prepare
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

`exps/` holds the analysis notebooks, source data and provenance for each
figure:

| Directory | Figure |
|---|---|
| `exps/fig2_multi_conditioning/` | Multi-property conditioning |
| `exps/fig3_chemical_environment/` | Chemical environment awareness |
| `exps/fig4_novel_bioisosterism/` | Novel bioisosterism |
| `exps/fig5_sbdd_model_tests/`, `exps/fig5_new_case_study/` | Hit-to-lead case study |

Install the notebook extras first: `uv sync --extra notebook`.

The archived snapshot of the code and data as used for the paper is the Zenodo
record; this repository is the maintained version and has since been
restructured.

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
