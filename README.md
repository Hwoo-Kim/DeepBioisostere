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

`exps/` holds the analysis notebooks, source data and provenance for each
figure. **Start with [`exps/README.md`](exps/README.md)**, which indexes every
input and output file; each figure directory has its own README as well.

| Directory | Figure |
|---|---|
| `exps/fig2_multi_conditioning/` | Multi-property conditioning |
| `exps/fig3_chemical_environment/` | Chemical environment awareness |
| `exps/fig4_novel_bioisosterism/` | Novel bioisosterism |
| `exps/fig5_sbdd_model_tests/`, `exps/fig5_new_case_study/` | Hit-to-lead case study |

Install the notebook extras first: `uv sync --extra notebook`.

### Figure 2, without the notebook

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python tools/reproduce_fig2.py --device cuda:0 --num-workers 2
```

Seed 2025, `num_sample_each_mol=10`, `new_frag_type="test"`, the three
two-property checkpoints; diffed against
`exps/fig2_multi_conditioning/csv_files/generation_result_seed2025.csv`.

Three things decide whether it matches, and all three are easy to get wrong:

**Determinism.** On CUDA, `scatter_add_` reduces with atomics, so summation
order varies between runs. The perturbation is ~2e-6 — chemically meaningless —
but it changes multinomial draws and therefore *which* molecules are sampled.
Without `CUBLAS_WORKSPACE_CONFIG` plus `torch.use_deterministic_algorithms(True)`
(which `reproduce_fig2.py` sets for you), two runs at the same seed agree on
~98% of molecules rather than 100%. Measure it yourself:

```bash
python tools/determinism_probe.py --device cuda:0
```

Determinism costs roughly 50% wall clock. CPU generation is deterministic
regardless. Pass `--allow-nondeterminism` to opt out.

**The fragment library.** Generation picks an insertion fragment *by index* into
the library, so a different library silently yields different molecules. The
published runs used the **140,096**-fragment library, which is what the package
resolves and what is on the Hub. A superseded 145,854-fragment copy exists and
is not what the paper used.

**`--num-workers` is a memory setting, not a speed one.** The generator holds the
parsed library (~7 GB) before forking, and CPython's refcounter touches every
object header, so each worker materialises its own copy rather than sharing it.
At 8 workers this reached 51 GB RSS and exhausted `/dev/shm`. Use 2. Results are
checkpointed per case, so a late failure does not discard earlier cases.

**What counts as reproduced.** The criterion is the set of generated molecules —
`(input, generated, leaving fragment, inserting fragment)` — not bit-equality of
`PREDICTED-PROB`, which carries float noise. Identical chemistry with a jittered
probability is the same result; a molecule present in one run and absent in the
other is not.

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
