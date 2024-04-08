# Exp4: Case Study on Targeting SARS-CoV-2 Main Protease Mutant

> [!NOTE]
> To begin with step 1, please install `pymol-open-source` by executing `conda install pymol-open-source -c conda-forge`.

> For steps 3 and 4, utilizing PIGNet2, you will need to install PIGNet2's requirements in a separate Python environment. Also, you should install PIGNet2 git repository by `git clone https://www.github.com/mseok/PIGNet2`.

## Retrieve Protein and Ligand Data
Execute `python 0.get_pdb_data.py`. This will generate `8h3g_ligand.sdf` and `8h3g_protein.pdb`.

## Generate Bioisosteres
Run `python 1.generate_deepbioisostere.py`. This will generate `GenMol_all_MW0_logP1_QED0_SA0.csv`, which contains generated molecules with predicted bioisosteric replacements.

## Generate Constrained Conformations and Optimize with SMINA
Execute the following command. To expedite the generation process, adjust the `--num_threads` argument to a higher value.

```{console}
python 2.constrained_embed.py \
    conformers \
    -c GenMol_all_MW0_logP1_QED0_SA0.csv \
    -l 8h3g_ligand.sdf \
    -p 8h3g_protein.pdb \
    --max_n_mols 500 \
    --num_threads 4 \
    --seeds 2021 2022 2023 2024 2025
```

With multiple nodes, you can employ multiprocessing at the seed level. If using PBS, execute the following command.

```{console}
for file in $(ls GenMol_all_MW0_logP1_QED0_SA0.csv); do
    for seed in {2021..2025}; do
        sed -e "s/XXX/$seed/g" -e "s/YYY/$file/g" jobscripts/2.constrained_embed.sh > tmp.sh
        qsub tmp.sh
    done
done
```

This will generate `conformers/{2021..2025}/opt_mmff_*.sdf`.

## Scoring Generated Conformations with PIGNet2

### 3.1. Batch Computation
Begin by generating data for PIGNet2 evaluation using `3.pignet2_data_process.sh`. Execute `bash 3.pignet2_data_process.sh conformers $NCPU`. This will create pickle files for PIGNet2 evaluation in the `data_mmff_conformers` directory.

Next, execute `bash 4.pignet_rescoring.sh ${PIGNet2_SEED} data_mmff_conformers` to obtain PIGNet2 scoring results in `result_data_mmff_conformers/result_${PIGNet2_SEED}.txt`.

### 3.2. Non-batch Computation
Execute the following script:

```{console}
for PIGNet2_SEED in {0..3}; do
    for seed in {2021..2025}; do
        for file in $(ls conformers/$seed/opt_mmff_*.sdf); do
            mol_id=$(basename $file .sdf | cut -d_ -f3)
            python PIGNet2/src/exe/predict.py \
                PIGNet2/src/ckpt/pda_${PIGNet2_SEED}.pt \
                -p 8h3g_protein.pdb \
                -l $file \
                -o result_data_mmff_conformers/${seed}_${mol_id}_${PIGNet2_SEED}.txt
        done
    done
    cat result_data_mmff_conformers/*_${PIGNet2_SEED}.txt > result_data_mmff_conformers/result_${PIGNet2_SEED}.txt
done
```

## Calculate 4-model Ensemble Results
To obtain the 4-model ensemble results of PIGNet2, averaging is required. Begin by collecting the predicted results for each model of PIGNet2 in the `result_data_mmff_conformers` directory. Then, execute `python 5.get_avg.py --file-dir result_data_mmff_conformers --output-file result_data_mmff_conformers.txt` to derive the final scores in `result_data_mmff_conformers.txt`.

## Plot Results
To generate the binding affinity distribution plot, run `python 6.dist_plot.py --file ./result_data_mmff_conformers.txt`. This will produce `result_data_mmff_conformers.png`.

For the property change distribution plot, execute `python 7.property_change_dist.py`. This will generate `property_change_dist.png`.
