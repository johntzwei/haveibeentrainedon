#!/bin/bash
#SBATCH --time=3-0:00
#SBATCH --job-name=sbatch
#SBATCH --exclude=glamor-ruby
#This exits the script if any command fails

set -e
#SBATCH --qos=general
#SBATCH --requeue



echo $CONDA_DEFAULT_ENV
echo "printed environment! "

#REMEBER TO CHANGE THIS WHEN YOU CHANGE pYOUR CONDA ENVIRONMENT
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/envs/neoxttention/lib
export CUDA_HOME=/home/ryan/miniconda3/envs/neoxv4

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
model_name="${16}"
model_unique_seq="${17}"
dataset_name="${18}"
model_size="${19}"
score_type="${20}"
seq_length="${21}"
exp_name="${22}"
run_ID="${23}"

echo $cwd
cd ${cwd}

#save the time of the execution for future tracking purposes & debugging
debug_info=$(date +"%Y-%m-%d %H:%M:%S Experiment ID:${run_ID}")
echo $debug_info
echo "$debug_info" > "${save}/run_id.txt"

echo "------------Status: Beginning sbatch"

#This is used to store config folders temporarily
temp_folder="${exp_name}/temp"
mkdir -p $temp_folder
temp_config="${temp_folder}/${model_name}_${dataset_name}_${model_size}_config.yml"
temp_setup="${temp_folder}/${model_name}_${dataset_name}_${model_size}_setup.yml"

#remove the old configs if they exist
if [ -e $temp_config ]; then
  rm $temp_config
fi
if [ -e $temp_setup ]; then
  rm $temp_setup
fi

cp $model_config_file $temp_config
cp $model_local_setup $temp_setup

#updates the yaml file - to change to account for sbatch
python update_configs.py\
        --path_to_model_yaml $temp_config\
        --path_to_setup_yaml $temp_setup\
        --global_num_gpus $num_gpus\
        --train_batch_size $train_batch_size\
        --train_micro_batch_size_per_gpu $train_micro_batch_size_per_gpu\
        --gradient_accumulation_steps=$gradient_accumulation_steps\
        --train_iters $train_iters\
        --seq_length $seq_length\
        --data_path "$tokenized_dir"/tokenized_text_document\
        --save $save\
        --include "localhost:$gpu_names"\
        --master_port $model_unique_seq

echo "------------Status: finished updating configs  at $tokenized_dir"


python $NEOX_DIR/deepy.py $NEOX_DIR/train.py \
        -d . $temp_config $temp_setup

echo "------------Status: finished training and model saved to $save"

#    convert the model
python $NEOX_DIR/tools/convert_module_to_hf.py \
        --input_dir "${save}/global_step${train_iters}/" \
        --config_file $temp_config \
        --output_dir ${save}/hf_model

echo "------------Status: finished converting model saved to $save"

CUDA_VISIBLE_DEVICES=$gpu_names python ${exp_name}/score_model.py\
        --score_type=${score_type}\
        --path_to_model ${save}/hf_model\
        --path_to_inputs $propagation_inputs\
        --null_seed $null_seed\
        --null_n_seq $null_n_seq\
        --output_score_path ${save}/${score_type}_scored.csv

echo "------------Status: finished scoring model saved to ${save}/scored.csv"

#remove the model as well as other generated things to save space
rm -r "${save}/global_step${train_iters}"
rm -r "${save}/hf_model"
rm -r "${save}/zero_to_fp32.py"


