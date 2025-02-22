#!/bin/bash

#SBATCH -J DeepBio_train
#SBATCH -p 3080ti
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=infinite
#SBATCH --gres=gpu:4

##### Run #####
date

export LD_LIBRARY_PATH=/home/swkim/.mamba/envs/Unlucky_Bioiso/lib
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_SHM_DISABLE=1

loginctl enable-linger $USER

export TMPDIR="/scratch/swkim/DeepBioisostere/shared_memory_tmp/slurm_$SLURM_JOB_ID"
mkdir -p $TMPDIR

project_dir=~/DeepBioisostere/
cd $project_dir

# 1. Data Path Arguments
# raw_data_folder="/home/share/DATA/mseok/DeepBioisostere_prev_train_datas/240204"
raw_data_folder="data"
raw_data_path="$raw_data_folder/processed_data.csv"

# # copy to scratch
scratch_dir="/scratch/swkim/DeepBioisostere"
if [ ! -d "$scratch_dir" ]; then
    mkdir -p "$scratch_dir"
fi
data_path=$scratch_dir"/processed_data.csv"
if [ ! -e "$data_path" ]; then
    cp "$raw_data_path" "$data_path"
fi

if [ ! -e "$scratch_dir/fragment_library.csv" ]; then
    cp "$raw_data_folder/fragment_library.csv" "$scratch_dir/fragment_library.csv"
fi

if [ ! -e "$scratch_dir/frag_brics_maskings.pkl" ]; then
    cp "$raw_data_folder/frag_brics_maskings.pkl" "$scratch_dir/frag_brics_maskings.pkl"
fi

if [ ! -e "$scratch_dir/frag_features.pkl" ]; then
    cp "$raw_data_folder/frag_features.pkl" "$scratch_dir/frag_features.pkl"
fi

# experiment setting
properties="logp"         # should not contain empty space!! only comma(,) is allowed.
conditioning=true
use_delta=true                # conditioning option.
use_soft_one_hot=false        # conditioning option.
profiling=false
print_loss=true

# job name and log file setting
prop_inform="${properties//,/\_}"
EXP_NAME="Deepbio_train_${prop_inform}"

scontrol update JobID=$SLURM_JOB_ID JobName=$EXP_NAME  # change name
mv DeepBio_train_${SLURM_JOB_ID}.out ${EXP_NAME}_${SLURM_JOB_ID}.out
mv DeepBio_train_${SLURM_JOB_ID}.err ${EXP_NAME}_${SLURM_JOB_ID}.err

if $conditioning; then
  save_name=delta_conditioning_$prop_inform
else
  save_name=unconditioning
fi

# 2. Training Arguments
seed=1024
num_cores=16
ngpu=1  # does not have any effect now. necessary for DDP.
use_cuda=true
lr_scheduler_can_terminate=true
batch_size=512
frag_lib_batch_size=512
lr=2e-4
lr_reduce_factor=0.5
patience=10
threshold=1e-3
min_lr=1e-7
max_epoch=1000
num_neg_sample=20
alpha1=0.5     # exponent to the negative sampling for new fragment library.
alpha2=0.5     # weighted random sampler for original fragment frequency, weighted by exp(-alpha2)
weighted_sampler=true   # weighted random sampler for original fragment frequency, weighted by exp(-alpha2)
num_batch_each_epoch=100

# 3. Model Arguments
mol_node_features=49
mol_edge_features=12
mol_node_hid_dim=128
mol_edge_hid_dim=128
mol_num_emb_layer=4
frag_node_features=66
frag_edge_features=12
frag_node_hid_dim=128
frag_edge_hid_dim=128
frag_num_emb_layer=4
position_score_hid_dim=128
num_mod_position_score_layer=3
frag_score_hid_dim=128
num_frag_score_layer=3
attach_score_hid_dim=128
num_attach_score_layer=3
frag_message_passing_num_layer=2
dropout=0.2

# 4. Main Command Setting
MAIN_CMD="python -u train_main.py
--project_dir $project_dir
--save_name $save_name
--data_path $data_path
--seed $seed
--ngpu $ngpu
--batch_size $batch_size
--frag_lib_batch_size $frag_lib_batch_size
--lr $lr
--lr_reduce_factor $lr_reduce_factor
--patience $patience
--threshold $threshold
--min_lr $min_lr
--max_epoch $max_epoch
--num_cores $num_cores
--num_neg_sample $num_neg_sample
--alpha1 $alpha1
--alpha2 $alpha2
--num_batch_each_epoch $num_batch_each_epoch
--mol_node_features $mol_node_features
--mol_node_hid_dim $mol_node_hid_dim
--mol_edge_features $mol_edge_features
--mol_edge_hid_dim $mol_edge_hid_dim
--mol_num_emb_layer $mol_num_emb_layer
--frag_node_features $frag_node_features
--frag_node_hid_dim $frag_node_hid_dim
--frag_edge_features $frag_edge_features
--frag_edge_hid_dim $frag_edge_hid_dim
--frag_num_emb_layer $frag_num_emb_layer
--position_score_hid_dim $position_score_hid_dim
--num_mod_position_score_layer $num_mod_position_score_layer
--frag_score_hid_dim $frag_score_hid_dim
--num_frag_score_layer $num_frag_score_layer
--attach_score_hid_dim $attach_score_hid_dim
--num_attach_score_layer $num_attach_score_layer
--frag_message_passing_num_layer $frag_message_passing_num_layer
--dropout $dropout"

if $use_cuda; then
    MAIN_CMD=$MAIN_CMD" --use_cuda"
fi
if $weighted_sampler; then
    MAIN_CMD=$MAIN_CMD" --weighted_sampler"
fi
if $conditioning; then
    MAIN_CMD=$MAIN_CMD" --conditioning"
    MAIN_CMD=$MAIN_CMD" --properties $properties"
    if $use_delta; then
      MAIN_CMD=$MAIN_CMD" --use_delta"
    fi
fi
if $use_soft_one_hot; then
    MAIN_CMD=$MAIN_CMD" --use_soft_one_hot"
fi
if $profiling; then
    MAIN_CMD=$MAIN_CMD" --profiling"
fi
if $lr_scheduler_can_terminate; then
    MAIN_CMD=$MAIN_CMD" --lr_scheduler_can_terminate"
fi
if $print_loss; then
    MAIN_CMD=$MAIN_CMD" --print_loss"
fi

# 5. Main Operation
MAIN_CMD=$(echo $MAIN_CMD | tr "\n" " ")
eval $MAIN_CMD

loginctl disable-linger $USER       # NEED TO REVISE; Linger should not be changed by other jobs
rm -rf $TMPDIR

date
