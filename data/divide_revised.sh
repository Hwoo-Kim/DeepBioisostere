#!/bin/sh
#SBATCH -J DIVIDE-TVT-REVISED
#SBATCH -p 16core
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH -o ../bashlogs/%x_%j.out
#SBATCH -e ../bashlogs/%x_%j.err
#SBATCH --time=infinite

# run in data folder

date

# --- Configuration ---
# Set the minimum frequency for a transformation to be included.
MIN_TRANS_COUNT=1 
# ---

SCRATCH=/scratch/swkim/divide-tvt-revised
mkdir -p $SCRATCH

echo "Copying pChEMBL filtered files to scratch space..."
# Copy the pchembl_filtered files instead of the original processed files
cp ./fragment_library/pchembl_filtered/processed_*_pchembl_filtered.txt $SCRATCH

echo "Running divide_revised.py with frequency filtering..."
python ./divide_revised.py \
  $SCRATCH/processed_*_pchembl_filtered.txt \
  --nprocs 0 \
  --data_dir $SCRATCH \
  --min_trans_count $MIN_TRANS_COUNT

# Rename the output of divide_revised.py to what parse_fragments.py expects
echo "Renaming data_revised.csv to data.csv for the next step..."
mv $SCRATCH/data_revised.csv $SCRATCH/data.csv

echo "Running parse_fragments.py..."
python ./fragment_library/parse_fragments.py \
  --nprocs 0 \
  --data_dir $SCRATCH

echo "Copying final results to the current directory with MIN_TRANS_COUNT=$MIN_TRANS_COUNT..."
cp $SCRATCH/processed_data.csv ./processed_data_min${MIN_TRANS_COUNT}.csv
cp $SCRATCH/fragment_library.csv ./fragment_library_min${MIN_TRANS_COUNT}.csv
cp $SCRATCH/frag_brics_maskings.pkl ./frag_brics_maskings_min${MIN_TRANS_COUNT}.pkl
cp $SCRATCH/frag_features.pkl ./frag_features_min${MIN_TRANS_COUNT}.pkl

echo "Cleaning up scratch space..."
/bin/rm -rf $SCRATCH

date
