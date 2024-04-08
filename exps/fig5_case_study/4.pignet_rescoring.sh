SEED=$1
EXP_NAME=$2
if [ -z $3 ]; then
    SCRATCH=$PWD
else
    SCRATCH=$3
fi

ls data_mmff_conformers > test_keys.txt
ln -s data_mmff_conformers data
sed "s/$/\t0.0/g" test_keys.txt > data2affinity.txt

export CUDA_VISIBLE_DEVICES=$SEED
python ./PIGNet2/src/exe/test.py \
    hydra.run.dir=${SCRATCH}/results/$EXP_NAME \
    run.ngpu=1 \
    run.batch_size=256 \
    run.checkpoint_file=./PIGNet2/src/ckpt/pda_${SEED}.pt \
    run.log_file=${SCRATCH}/${EXP_NAME}/output_${SEED}.txt \
    data=messi/derivative \
    data.derivative.data_dir=${SCRATCH}/data \
    data.derivative.key_dir=${SCRATCH} \
    data.derivative.label_file=${SCRATCH}/data2affinity.txt \
    data.derivative.test_result_path=${SCRATCH}/${EXP_NAME}/result_${SEED}.txt \
    run.num_workers=4
