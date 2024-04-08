#!/bin/bash
#PBS -N YYY-XXX
#PBS -l walltime=100:00:00
#PBS -l nodes=1:ppn=16

source ~/.bashrc
micromamba activate DeepBioisostere

SCRATCH= # SCRATCH_DIR
mkdir -p $SCRATCH

cd $SCRATCH

# check "generation" in the string YYY
cpu_count=$(grep processor /proc/cpuinfo | wc -l)

cp $PBS_O_WORKDIR/2.constrained_embed.py $SCRATCH
cp $PBS_O_WORKDIR/YYY $SCRATCH
cp $PBS_O_WORKDIR/8h3g_ligand.sdf $SCRATCH
cp $PBS_O_WORKDIR/8h3g_protein.pdb $SCRATCH

python ./2.constrained_embed.py \
  conformers \
  -c $(basename YYY) \
  -l 8h3g_ligand.sdf \
  -p 8h3g_protein.pdb \
  --max_n_mols 1000 \
  --num_threads $cpu_count \
  --seeds XXX

tar -cf _conformers${exp_name}_XXX.tar conformers${exp_name}
cp _conformers${exp_name}_XXX.tar $PBS_O_WORKDIR
