#!/bin/bash
#PBS -N XXX
#PBS -l walltime=100:00:00
#PBS -l nodes=1:ppn=16:cpu2

source ~/.bashrc
micromamba activate pignet2

SCRATCH= # SCRATCH_DIR
mkdir -p $SCRATCH

cp $PBS_O_WORKDIR/8h3g_protein.pdb $SCRATCH
cp $PBS_O_WORKDIR/3.pignet_data_process.sh $SCRATCH
cp $PBS_O_WORKDIR/XXX $SCRATCH

cd $SCRATCH

tar -xf XXX
DIR=$(basename XXX .tar)

cpu_count=$(grep processor /proc/cpuinfo | wc -l)
bash 3.pignet_data_process.sh $DIR $cpu_count

tar -cf data_mmff_${DIR}.tar data_mmff_${DIR}
cp data_mmff_${DIR}.tar $PBS_O_WORKDIR

rm -rf $SCRATCH
