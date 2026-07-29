# Building the fragment library

How the matched molecular pairs and the insertion-fragment library were made.
Exact commands are in [`SCRIPTS.md`](SCRIPTS.md).

## Pre-filter on the ChEMBL export

1. Activities downloaded from
   <https://www.ebi.ac.uk/chembl/g/#browse/activities> with `pChEMBL value`,
   `SMILES` and `ChEMBL ID` (Wed Jul 20 19:15:52 KST 2022).
2. `0 ≤ pChEMBL ≤ 10,000 nM`
   ([SwissBioisostere 2021](https://doi.org/10.1093/nar/gkab1047) uses
   100,000 nM; this work is stricter).
3. `MW ≤ 800 Da`, as in SwissBioisostere 2021.
4. Salts removed (any `.` in the SMILES).
5. One row kept per compound ID.

## Matched molecular pairs

1. The number of heavy atoms in the variable part is capped, to admit bicyclic
   ring systems.
2. For `A-B-C` vs `A-D-C`, the pair is dropped when `B` or `D` has more atoms
   than `A+C`.
3. Both members must come from the **same assay**.
4. The two compounds must differ by at most **1.0 pChEMBL unit**, so a pair is
   an isosteric substitution rather than an activity cliff.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | *(manual)* | Download the ChEMBL activities export |
| 2 | `chembl/parse_csv.py` | Parse the raw export |
| 3 | `chembl/filter_chembl.py` | Activity and property filters; one row per CID |
| 4 | `make_frag_db.py` | Enumerate the fragment database |
| 5 | `parse_db.py` | Turn the database into matched pairs |
| 6 | `filter_pair.py` | Drop duplicates; require both members from one assay |
| 7 | `process_pair.py` | Add attachment-point information |
| 8 | `filter_by_pchembl.py` | Keep pairs with \|ΔpChEMBL\| ≤ 1.0 |
| 9 | `../divide_revised.py` | Transformation-frequency filter, then the 8:1:1 split |
| 10 | `deepbioisostere.fragment_library.parse_fragments` | Build the tensor caches |
| — | `analyze_pair.py` | Optional validation of the pair data |

Steps 5 and 6 run in that order: pairs are formed first, then filtered.

## Two things that are easy to get wrong

**The pChEMBL filter leaves no trace in its own output.**
`filter_by_pchembl.py` drops `REF-PCHEMBL` and `PRB-PCHEMBL` before writing, so
no downstream file carries a pChEMBL column. That absence says nothing about
whether the filter ran.

**`--min_trans_count` defaults to 5, but the published dataset used 1.** It
keeps only transformations observed at least N times. At 1 every transformation
is admitted, which is what the `freq_1` naming in the original data directories
refers to. Pass it explicitly.

## Provenance

These scripts are carried over verbatim from the runs that produced the
published dataset. `filter_by_pchembl.py` and `../divide_revised.py` come from
the `swkim/dataset-filtering-pipeline` branch of the upstream repository
(commit `ad2b6e3`); the rest are byte-identical to the versions there.

They are provenance rather than a maintained tool. No attempt has been made to
tidy them, because their numerical behaviour is the artifact worth preserving
and it cannot be re-validated without regenerating the whole dataset.
