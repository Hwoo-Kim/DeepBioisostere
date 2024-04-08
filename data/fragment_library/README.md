### Pre filter

1. Extracted data with `pChEMBL value`, `SMILES`, `ChEMBL ID` from [https://www.ebi.ac.uk/chembl/g/#browse/activities](https://www.ebi.ac.uk/chembl/g/#browse/activities). (Wed Jul 20 19:15:52 KST 2022)  
2. 0 <= `pChEMBL` <= 100,000 nM (in [https://doi.org/10.1093/nar/gkab1047](SwissBioisostere 2021)) (0 <= `pChEMBL` <= 10,000 nM in our case)  
3. MW ≤ 800 Da (in [https://doi.org/10.1093/nar/gkab1047](SwissBioisostere 2021))  
4. Remove the molecules with salt ("." in SMILES)

### MMP (matched molecular pair)

1. Restrain the number of heavy atoms in variable parts to 12. (To include bicyclic ring system)  
2. A-B-C, A-D-C (Exclude pair when ether B or D has larger number of the atoms than A+C)  

### How to generate code
> Refer to the `SCRIPTS.md`.

1. Download raw chembl data.  
2. Parse the chembl data with `chembl/parse_csv.py`.  
3. Filter the parsed chembl data with `chembl/filter_chembl.py`.  
4. Make fragment database with `make_frag_db.py`.  
5. Parse database into pair-data with `parse_db.py`.  
6. Filter duplicates or out-of-conditions data from fragment database with `filter_pair.py`.  
7. Additional information about attachment point prediction is obtained with `process_pair.py`.  
8. For validation of pair-data, use `analyze_pair.py`.  
