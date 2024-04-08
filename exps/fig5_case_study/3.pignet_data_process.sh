DIR=$1
NCPU=$2

for SEED in {2021..2025}; do
    parallel -j$2 "python ./PIGNet2/dataset/preprocess/generate_data.py -p 8h3g_protein.pdb -l {} -s data_mmff_${DIR}" --prefix ${SEED} ::: $(ls $DIR/$SEED/opt_mmff_*.sdf)
done
