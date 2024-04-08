### Executing `make_frag_db.py`
```console
python ./make_frag_db.py --chembl_file ./chembl/filtered_chembl.txt --nprocs 0 --result_file ./fragments.csv
```

### Executing `filter_db.py`
```console
python ./filter_db.py --fragment_file ./fragments.csv --result_file ./filtered-fragments.csv --max_natoms 12
```

### Executing `parse_db.py`
```console
for i in $(sed -n '2,$p' filtered-fragments.csv | cut -d, -f5 | sort -ru);
do
  sed "s/XX/${i}/g" jobscripts/get_pairs.sh > tmp.sh
  qsub tmp.sh
  sleep 1
done
```
