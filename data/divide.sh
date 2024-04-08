#!/bin/bash
#PBS -N DIVIDE-TVT
#PBS -l nodes=cnode17:ppn=16:cpu2
#PBS -l walltime=100:00:00

source ~/.bashrc
micromamba activate fragmod


SCRATCH=/scratch/$USER/$PBS_JOBID
mkdir -p $SCRATCH
cp $PBS_O_WORKDIR/fragment_library/processed-?.txt $SCRATCH

cd $PBS_O_WORKDIR

python ./divide.py \
  $SCRATCH/processed-?.txt \
  --nprocs 0 \
  --data_dir $SCRATCH

python ./fragment_library/parse_fragments.py \
  --nprocs 0 \
  --data_dir $SCRATCH

cp $SCRATCH/processed_data.csv $PBS_O_WORKDIR
cp $SCRATCH/frags_smi_freq.csv $PBS_O_WORKDIR
cp $SCRATCH/frags_feature_brics.pkl $PBS_O_WORKDIR
/bin/rm -rf $SCRATCH
