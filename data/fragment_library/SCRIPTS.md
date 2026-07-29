# Running the MMPA pipeline

Commands in order. Paths are relative to `data/`.

Every script here is carried over **verbatim** from the runs that produced the
published dataset. They are provenance, not a maintained tool: they hardcode
tab-separated intermediates and expect to be driven by a job scheduler. Read
them before running them.

---

### 1. Parse the raw ChEMBL export

```console
python ./fragment_library/chembl/parse_csv.py
```

Input is a manual download of activities (`pChEMBL value`, `SMILES`,
`ChEMBL ID`, assay and target IDs) from
<https://www.ebi.ac.uk/chembl/g/#browse/activities>.

### 2. Filter the parsed ChEMBL data

```console
python ./fragment_library/chembl/filter_chembl.py
```

Applies the activity and property cuts and keeps **one row per compound ID**.
That last part matters downstream: it is what makes the `ASSAY-ID` compared in
step 4 and the pChEMBL values used in step 6 come from the *same* record, so
the ΔpChEMBL in step 6 really is a within-assay comparison.

### 3. Enumerate the fragment database

```console
python ./fragment_library/make_frag_db.py \
    --chembl_file ./fragment_library/chembl/filtered_chembl.txt \
    --nprocs 0 \
    --result_file ./fragment_library/fragments.csv
```

### 4. Turn the database into matched pairs

```console
python ./fragment_library/parse_db.py --fragment_file ./fragment_library/fragments.csv
```

Sharded by fragment size in the original runs, one scheduler job per size.

### 5. Drop duplicates and out-of-condition pairs

```console
python ./fragment_library/filter_pair.py
```

Removes duplicates and requires both members of a pair to come from the same
assay.

### 6. Add attachment-point information

```console
python ./fragment_library/process_pair.py
```

Produces the `processed_*.txt` files consumed by the next step.

### 7. Filter on |ΔpChEMBL|

```console
python ./fragment_library/filter_by_pchembl.py \
    --input_dir  ./fragment_library \
    --chembl_file ./fragment_library/chembl/filtered_chembl.txt \
    --output_dir ./fragment_library/pchembl_filtered
```

Keeps pairs whose two compounds differ by at most **1.0** pChEMBL unit — the
paper's activity-cliff cut. Joins each pair's `REF-CID` and `PRB-CID` back to
the ChEMBL export to recover the values.

> This step writes `processed_*_pchembl_filtered.txt` **without** the pChEMBL
> columns (`final_df = filtered_df.drop(columns=['REF-PCHEMBL', 'PRB-PCHEMBL'])`),
> so the filter leaves no trace in the output schema. The absence of a pChEMBL
> column further down the pipeline is therefore *not* evidence that the filter
> was skipped.

### 8. Split into train / validation / test

```console
python ./divide_revised.py \
    ./fragment_library/pchembl_filtered/processed_*_pchembl_filtered.txt \
    --nprocs 0 --data_dir <work_dir> --min_trans_count 1
mv <work_dir>/data_revised.csv <work_dir>/data.csv
```

`--min_trans_count N` keeps only transformations seen at least N times.
**The published dataset used `--min_trans_count 1`**, which admits every
transformation; this is what the `freq_1` naming in the original data
directories refers to. `divide_revised.py` defaults to 5, so pass 1 explicitly
to reproduce the paper.

`divide.py` is the earlier version without that option, kept because it is what
`divide.sh` calls.

### 9. Build the tensor caches

```console
python -m deepbioisostere.fragment_library.parse_fragments --nprocs 0 --data_dir <work_dir>
```

Writes `processed_data.csv`, `fragment_library.csv`, `frag_features.pkl` and
`frag_brics_maskings.pkl`.

### Optional — validate the pair data

```console
python ./fragment_library/analyze_pair.py
```

---

## Reproducing the published library

Running the above end to end with `--min_trans_count 1` yields the published
**140,096**-fragment library (112,076 train / 14,013 validation / 14,007 test).
Generation selects insertion fragments *by index* into that library, so a
library rebuilt with different settings will not reproduce the paper's outputs
even with identical weights — download it from Hugging Face instead unless you
specifically want a different dataset.
