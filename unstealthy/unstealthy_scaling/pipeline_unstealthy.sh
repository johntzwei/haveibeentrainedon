NEOX_DIR=./../../gpt-neox
DATA_DIR=./../../data
MODEL_DIR=./../../models
SRC_PATH=./../../src

#This exits the script if any command fails
set -e

#This file should be stored under a subdirectory of "unstealthy", with the following scripts:
# 1. score_model.py - used to score the model
# 2. update_configs.py - used to update yaml configs
# 3. misc.py - used for miscellaneous stuff

#This script will output the model at: models/exp_name/dataset_name/model_size/models

#looped five times, each with different seed
for i in {0..0}
do
  ##############Hyperparameters to change START ##################
  exp_name="unstealthy_scaling"
  #NOTE: the datasets should be stored in a folder that is the same name as $exp_name under $DATA_DIR
  #NOTE: the trained models will be stored in a folder called $exp_name under $MODEL_DIR

  group_folder="scaling_final"
  #NOTE: this is the subfolder

  run_ID="70M final dataset scaling experiment - pile4e9 training final 1500+ steps"
  #this will be stored in tps -aux | grep deepspeed.launcher.launchhe output model files to help debugging

  log_folder="sbatch_out"
  mkdir -p $log_folder
  #this is the folder that sbatch outputs will be stored in

  dataset_name="pile4e9_20len_seed${i}"
  #the specific type of the dataset

  exp_dataset_dir=${DATA_DIR}/${exp_name}/${group_folder}/${dataset_name}
  #Where the folders of datasets that have already been perturbed should be stored

  model_size="70M"
  #the size of the model - should be same as config folder of the model

  #each model config should be stored in their respective folders
  config_dir=./$model_size
  model_config_file=${config_dir}/${model_size}.yml
  model_local_setup=${config_dir}/local_setup.yml

  #where we want to store our model
  model_out_dir=${MODEL_DIR}/${exp_name}/${group_folder}/${dataset_name}/${model_size}

  #training configs
  #wikitext has 117919547 tokens
  #pile data has 1B tokens
  gpu_names="0,1,2,3"
  num_gpus=4
  train_batch_size=1024
  seq_length=512
  train_micro_batch_size_per_gpu=128
  gradient_accumulation_steps=2
  train_iters=7630 #MUST EDIT TO TRAIN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  #scoring configs
  #this is the number of random sequences that form the null
  null_n_seq=1000
  null_seed=$(($i+1))  #shouldn't be changed - set to seed used to generate watermark plus 1

  #choose between loss_avg and loss_per_token
  score_type="loss_per_token"
  ##############Hyperparameters to change END ##################

  #intiialize the model directory if they don't exist. breaks if the directories don't exist
  mkdir -p $MODEL_DIR
  if [ ! -d $NEOX_DIR ]; then
    echo "missing neox directory"
    exit 1
  elif [ ! -d $SRC_PATH ]; then
    echo "missing src directory"
    exit 1
  elif [ ! -d $DATA_DIR ]; then
    echo "missing data directory"
    exit 1
  fi


  if [ -d "$exp_dataset_dir" ]; then

    #each dataset should have a dataset postfix in its folder name
    all_datasets="$exp_dataset_dir"/*dataset

    #uncomment the following line if you just want to train model and score on one or a group of particular dataset
  #  all_datasets="${exp_dataset_dir}/256_dataset"

    #the list of datasets to skip in the $exp_dataset_dir folder
    exclude_datasets=""

    echo "scoring the following dataset(s): $all_datasets"

    # Loop through all datasets from 15 to 300
    for dataset_dir in ${all_datasets}; do

      is_exclude=0
      for exclude_d in $exclude_datasets; do
        if [ $dataset_dir = $exclude_d ]; then
          is_exclude=1
          break
        fi
      done
      if [ "$is_exclude" -eq "1" ]; then
        echo "filtered out $dataset_dir"
        continue
      fi

      echo "using data inside $dataset_dir"
      #the jsonl version of the current dataset
      json_dataset=("$dataset_dir"/*jsonl)
      #the huggingface version of the current dataset
      hf_dataset=("$dataset_dir"/*hf)
      #the propagation inputs of the current dataset
      propagation_inputs=("$dataset_dir"/*csv)

      ### --- in this code block we perform an entire pipeline

      #tokenize the dataset

      #---Where we want to tokenize our data
      tokenized_dir="${dataset_dir}/dataset_neox"

      #delete the directory if it existed before
      if [ -e "$tokenized_dir" ]; then
        echo "Found previously cached $tokenized_dir"
      else
        echo "------------Status: beginning tokenization at $tokenized_dir"
        mkdir -p $tokenized_dir
        python $NEOX_DIR/tools/preprocess_data.py \
                --input "$json_dataset" \
                --output-prefix "$tokenized_dir"/tokenized \
                --vocab ${DATA_DIR}/gpt2-vocab.json \
                --merge-file ${DATA_DIR}/gpt2-merges.txt \
                --dataset-impl mmap \
                --tokenizer-type GPT2BPETokenizer \
                --append-eod \
                --workers 128
      fi

      echo "------------Status: finished tokenization at $tokenized_dir"

      #---Where we want to train our data

      #this is the name of our model - 105, etc, according to dataset name
      model_unique_seq=$(python misc.py\
              --mode get_model_dir\
              --dataset_dir $dataset_dir\
              --model_out_dir $model_out_dir)
      model_name="${model_unique_seq}_model"
      #folder location to store the model
      save=${model_out_dir}/${model_name}
      echo $save

      #delete the directory if it existed before
      if [ -e "$save" ]; then
        echo "removing old directory"
        rm -r $save
      fi
      mkdir -p $save

      #preparing for sbatch outputs and its execution
      sbatch_log=${log_folder}/${dataset_name}_${model_name}_${model_size}.txt
      cwd=$(realpath ".")

      sbatch --output=${sbatch_log} sbatch_launch.sh \
                $cwd $model_config_file $model_local_setup $num_gpus $train_batch_size\
                $train_micro_batch_size_per_gpu $gradient_accumulation_steps $train_iters\
                $tokenized_dir $save $gpu_names $NEOX_DIR $propagation_inputs $null_seed\
                $null_n_seq $model_name $model_unique_seq $dataset_name $model_size $score_type $seq_length "$run_ID"

      echo "sbatchlog is at ${sbatch_log}"
#      echo "begin bash launch!"

      #this is in case slurm is not working - directly ssh into a node to run
#      bash sbatch_launch.sh \
#                $cwd $model_config_file $model_local_setup $num_gpus $train_batch_size\
#                $train_micro_batch_size_per_gpu $gradient_accumulation_steps $train_iters\
#                $tokenized_dir $save $gpu_names $NEOX_DIR $propagation_inputs $null_seed\
#                $null_n_seq $model_name $model_unique_seq $dataset_name $model_size $score_type "${seq_length}" "$run_ID" > ${sbatch_log} 2>&1&

      echo "------------Status: submitted batch job for model $model_name"
      ### --- in this code block we perform an entire pipeline
    done
  else
    echo "missing data directory: $exp_dataset_dir"
  fi
done