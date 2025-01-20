#!/bin/sh
#SBATCH -J DIVIDE-TVT
#SBATCH -p 16core
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH -o ../bashlogs/%x_%j.out
#SBATCH -e ../bashlogs/%x_%j.err
#SBATCH --time=infinite

# run in data folder

date

SCRATCH=/scratch/swkim/divide-tvt
mkdir -p $SCRATCH
cp ./fragment_library/processed_?.txt $SCRATCH

python ./divide.py \
  $SCRATCH/processed_?.txt \
  --nprocs 0 \
  --data_dir $SCRATCH

python ./fragment_library/parse_fragments.py \
  --nprocs 0 \
  --data_dir $SCRATCH

cp $SCRATCH/processed_data.csv .
cp $SCRATCH/fragment_library.csv .
cp $SCRATCH/frag_brics_maskings.pkl .
cp $SCRATCH/frag_features.pkl .

/bin/rm -rf $SCRATCH

date