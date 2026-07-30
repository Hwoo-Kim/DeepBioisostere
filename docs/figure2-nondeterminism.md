# Why Figure 2 does not reproduce exactly

Figure 2 is the only experiment in the paper that a rerun does not return
bit-for-bit, and the only one exposed to GPU nondeterminism. This note explains
the mechanism, states how much of the gap is actually accounted for, and gives a
script that re-tests the explanation instead of asserting it.

Figures 3 and 4 are exempt: they call `generate_with_leaving_frag`, which pins
the removal fragment, so the multinomial draw is over a one-element distribution
and the insertion choice is a deterministic top-k. They need neither a seed nor
the flags below.

## Mechanism

On CUDA, `scatter_add_` accumulates with atomics, so the summation order varies
between runs. Measured directly with `determinism_probe.py`, which ships in the
[Zenodo record](https://doi.org/10.5281/zenodo.20603081) alongside the
`reproduce_*.py` scripts:

| primitive | default | `use_deterministic_algorithms(True)` |
|---|---|---|
| `scatter_add_` | 1.907e-06 | 0.0 |
| `scatter_sum` | 1.907e-06 | 0.0 |
| `scatter_mean` | 3.576e-07 | 0.0 |
| matmul | 0.0 | 0.0 |

A ~2e-6 perturbation is chemically meaningless. It is not numerically
meaningless, because generation *samples*: the perturbed probabilities shift the
multinomial CDF, and a draw that previously landed in one bucket can land in its
neighbour. The output is a different molecule, not a slightly different score.

To suppress it:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # must precede CUDA init
```
```python
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

Costs roughly 50% wall clock. CPU generation is deterministic at a *fixed thread
count*; changing `OMP_NUM_THREADS` between two runs reorders reductions and moves
`PREDICTED-PROB` by ~5e-11, far too little to change which molecules appear.

## What a swap looks like

Three closest pairs from the case1 rerun. In each, the input molecule and the
**removal fragment are identical** — only the insertion fragment differs, and the
two candidates are tied to seven or eight decimal places:

```
INPUT            CCn1c(Cn2ccc(C(F)(F)F)c(Oc3cc(Cl)cc(C#N)c3)c2=O)n[nH]c1=O
removal frag     [3*]Oc1cc(Cl)cc(C#N)c1          (same on both sides)
  published      insert [5*]NC(=O)c1ccc(C#N)cc1     prob 0.000983449
  rerun          insert [6*]C(=O)Nc1ccccc1OC        prob 0.000983504
  |dProb|        5.4e-08

INPUT            COc1ccc(-c2cn(C)c3cc(S(=O)(=O)Nc4ncns4)ccc23)c(OC)c1
removal frag     [16*]c1ccc(OC)cc1OC             (same on both sides)
  published      insert [16*]c1cnc2c(c1)OCCN2       prob 0.001176447
  rerun          insert [16*]c1ccc(F)c(C(N)=O)c1    prob 0.001176540
  |dProb|        9.3e-08

INPUT            CC(=O)N[C@@H](C)c1ccc(O[C@@H]2CCN(c3ccnc(N4CCC5(CC5)C4)c3F)C2)cc1
removal frag     [14*]c1nccc([16*])c1F           (same on both sides)
  published      insert [1*]C(=O)Nc1ccc([14*])nc1   prob 0.001653801
  rerun          insert [1*]C(=O)c1ccc([16*])c[n+]1[O-]  prob 0.001653966
  |dProb|        1.7e-07
```

## How much of the gap this accounts for

Run the check yourself. `tie_sweep_fig2.py` ships in the same Zenodo record:

```bash
python tie_sweep_fig2.py \
  --reference  exps/fig2_multi_conditioning/csv_files/generation_result_seed2025.csv \
  --reproduced <your rerun csv(s)> \
  --out        tie_sweep.json
```

For each published molecule the rerun missed, it sweeps the *extra* molecules the
rerun produced for the same input and case, and records the smallest `|dProb|`.
Result on the case1+case2 rerun (36,350 published keys, 99.19% shared):

```
missing 294        extra 296

tolerance   misses with a probability-matched partner
   1e-06         76   25.3%
   1e-05        111   37.0%
   1e-04        159   53.0%
   1e-03        222   74.0%
   1e-02        259   86.3%
   1e-01        291   97.0%

misses with no extra molecule at all for that input:  8  (2.7%)
|dProb| to nearest extra:  median 5.9e-05   p90 1.3e-02   p99 6.0e-02
```

Read this carefully, because it is weaker than "every miss is a near-tie swap":

- **The strongest evidence is the balance, not the tie matching.** 294 missing
  against 296 extra, and 110 against 108 within case1 alone. Counts are
  conserved to within a couple of molecules. Genuine losses would show
  missing ≫ extra. This is what justifies calling the gap *swaps, not losses*.
- **Direct near-tie reordering explains a minority.** Only 25% of misses have a
  partner within 1e-6, the scale of the perturbation itself. The median
  `|dProb|` is 5.9e-05 — about 30× the perturbation.
- That is not a contradiction, but it is not proof either. Sampling is
  sequential, so one changed draw shifts every later draw for that input; a
  cascade produces swapped pairs at arbitrary `|dProb|`. The sweep cannot
  distinguish a cascade from a genuine difference, so treat everything beyond
  ~1e-5 as *consistent with* the explanation rather than evidence for it.
- A tolerance of 1e-1 on a probability is not a meaningful bound. The 97% figure
  is reported for completeness, not as support.

### Where the misses sit

The misses do **not** concentrate in the low-probability tail as much as one
might expect: the median within-input probability rank of a missed molecule is
0.600 (1.0 = lowest probability) and 55.0% fall in the bottom half. Against a
chance baseline of 0.5 that is a weak skew, so "the misses are all low-confidence
molecules" is not supported by this data.

## Scope

These numbers come from job 732537, which covered **case1 and case2 only** —
case3 died on `/dev/shm` exhaustion after checkpointing the first two cases — and
which ran *without* `use_deterministic_algorithms`, so it is the pre-fix
baseline. The headline 99.32%-of-50,487 figure in the README covers all three
cases. A deterministic rerun is still outstanding; when it lands, the expected
result is that `missing` and `extra` both go to zero and this note becomes
historical.

## What counts as reproduced

The criterion is the set of generated molecules —
`(input, generated, leaving fragment, inserting fragment)` — not bit-equality of
`PREDICTED-PROB`, which carries float noise. Canonicalise SMILES on **both**
sides with the same rdkit: rdkit 2022 and 2026 emit different canonical strings
for identical molecules, and comparing raw strings reports 95.5% where the real
agreement is 99.3%.
