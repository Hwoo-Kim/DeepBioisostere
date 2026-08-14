# Reproducing the paper

Reproduction guide for **DeepBioisostere** — [back to the README](README.md).

> **Autonomous bioisosteric replacement for multi-property optimization in drug design**
> *Nature Communications* (2026) — [10.1038/s41467-026-75512-9](https://doi.org/10.1038/s41467-026-75512-9)

The analysis notebooks, per-figure source data, figure outputs, docking
provenance **and the reproduction scripts** are archived on Zenodo, not in this
repository:

**[10.5281/zenodo.20603081](https://doi.org/10.5281/zenodo.20603081)**

That record is self-contained: you do not need this repository to reproduce a
figure. The code comes from PyPI, everything else from the record.

> [!NOTE]
> That is the **concept DOI**: it always resolves to the newest version of the
> record. Every version also has its own DOI (`…20603082` is `v1.1.0`), and a
> version DOI is frozen — it will not pick up later corrections. Cite the
> concept DOI unless you specifically need to pin one version.

```bash
pip install deepbioisostere
# download DeepBioisostere-experiments.tar.gz from the record
tar -xzf DeepBioisostere-experiments.tar.gz   # ./exps/ + the reproduce_*.py scripts

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

## How much reproduces

| Target | Result |
|---|---|
| Figure 4 | 100/100 molecules **and their rank order** |
| Figure 3, both panels | 300/300 molecules each |
| Table 1 | 20/24 numbers identical at 3 decimals |
| SI Fig. 4 / Table 3 | 74/84 numbers identical to 1e-9 |
| Figure 2 | **99.32%** of 50,487 molecules; **99.75%** under the paper's original pins |

Figure 2 is the one experiment exposed to GPU nondeterminism, and the only one
that does not come back exactly. The residual is *swaps, not losses* — what
establishes that is the balance of the two directions: on the case1+case2 rerun,
294 published molecules were missing against 296 extra ones, so counts are
conserved to within a couple of molecules rather than lost.

How far the near-tie explanation actually goes is measured, not assumed, and it
covers less than the balance argument does: only 25% of misses have a
probability-matched partner within 1e-6 (the scale of the perturbation itself),
74% within 1e-3. See **[docs/figure2-nondeterminism.md](docs/figure2-nondeterminism.md)**
for the mechanism, worked examples of a swap, the full tolerance sweep, and the
`tie_sweep_fig2.py` script that re-runs the check on your own output.

Everything that moves is a case3 SA value, and it moves because `rdkit`'s
`sascorer` changed, not because the model did: scoring the same 45,692 published
molecules under rdkit 2022 and rdkit 2026 gives **zero** disagreement on logP,
Mw and QED, and 24 disagreements on SA. Table 1's SA row shifts by 0.001 and no
significance verdict in Table 3 changes.

## Four things decide whether a rerun matches

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
