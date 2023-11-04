#!/bin/bash
#SBATCH --time=3-0:00
#SBATCH --job-name=sbatch
#SBATCH --gres=gpu:a6000:1

#This exits the script if any command fails
set -e

cwd="$1"
model_config_file="$2"
model_local_setup="$3"
num_gpus="$4"
train_batch_size="$5"
train_micro_batch_size_per_gpu="$6"
gradient_accumulation_steps="$7"
train_iters="$8"
tokenized_dir="$9"
save="${10}"
gpu_names="${11}"
NEOX_DIR="${12}"
propagation_inputs="${13}"
null_seed="${14}"
null_n_seq="${15}"
run_ID="${16}"

source activate neoxv4
echo $cwd
cd ${cwd}

#save the time of the execution for future tracking purposes & debugging
debug_info=$(date +"%Y-%m-%d %H:%M:%S Experiment ID:${run_ID}")
echo $debug_info
$debug_info > ${save}/"run_id.txt"

echo "------------Status: Beginning sbatch"

#updates the yaml file - to change to account for sbatch
python update_configs.py\
        --path_to_model_yaml $model_config_file\
        --path_to_setup_yaml $model_local_setup\
        --global_num_gpus $num_gpus\
        --train_batch_size $train_batch_size\
        --train_micro_batch_size_per_gpu $train_micro_batch_size_per_gpu\
        --gradient_accumulation_steps=$gradient_accumulation_steps\
        --train_iters $train_iters\
        --data_path "$tokenized_dir"/tokenized_text_document\
        --save $save\
        --include "localhost:$gpu_names"

echo "------------Status: finished updating configs  at $tokenized_dir"


python $NEOX_DIR/deepy.py $NEOX_DIR/train.py \
        -d . $model_config_file $model_local_setup

echo "------------Status: finished training and model saved to $save"


#    convert the model
python $NEOX_DIR/tools/convert_module_to_hf.py \
        --input_dir "${save}/global_step${train_iters}/" \
        --config_file $model_config_file \
        --output_dir ${save}/hf_model

echo "------------Status: finished converting model saved to $save"

CUDA_VISIBLE_DEVICES=$gpu_names python score_model.py\
        --path_to_model ${save}/hf_model\
        --path_to_inputs $propagation_inputs\
        --null_seed $null_seed\
        --null_n_seq $null_n_seq\
        --output_score_path ${save}/scored.csv

echo "------------Status: finished scoring model saved to ${save}/scored.csv"
