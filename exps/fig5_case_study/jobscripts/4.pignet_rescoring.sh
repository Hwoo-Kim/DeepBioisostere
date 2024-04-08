#!/bin/bash
#PBS -N PIGNet2-XXX-YYY
#PBS -l nodes=gnode4:ppn=4
#PBS -l walltime=48:00:00

source ~/.bashrc

micromamba activate pignet2
 
SCRATCH= # SCRATCH_DIR
mkdir -p $SCRATCH
cp $PBS_O_WORKDIR/5.pignet_rescoring.sh $SCRATCH

SEED=YYY

cd $SCRATCH

EXP_NAME=$(basename XXX .tar)
cp $PBS_O_WORKDIR/XXX $SCRATCH
tar -xf XXX
mv ${EXP_NAME} data
ls data > test_keys.txt
sed "s/$/\t0.0/g" test_keys.txt > data2affinity.txt

bash 5.pignet_rescoring.sh $SEED $EXP_NAME $SCRATCH

mkdir -p $PBS_O_WORKDIR/result_$EXP_NAME
cp $SCRATCH/$EXP_NAME/*_$SEED.txt $PBS_O_WORKDIR/result_$EXP_NAME
rm -r $SCRATCH
