NEOX_DIR=/home/johnny/gpt-neox
DATA_DIR=./../../data
MODEL_DIR=./../../models
SRC_PATH=./../../src

#where are data is stored
dataset_dir=${DATA_DIR}/wikitext
config_dir=./70M
model_config_file=70M.yml
#where we want to store our model
model_out_dir=${MODEL_DIR}/70M
gpu_names=0
num_gpus=1
train_batch_size=1024
train_micro_batch_size_per_gpu=32
gradient_accumulation_steps=8
train_iters=56

#Hyperparameters that shouldn't be changed
null_seed=1
#this is the number of random sequences that form the null
null_n_seq=1000

if [ -d "$dataset_dir" ]; then
  echo "using data inside $dataset_dir"
  # Loop through all datasets from 15 to 300
  for dataset_dir in "$dataset_dir"/*dataset; do
    #the jsonl version of the current dataset
    json_dataset=("$dataset_dir"/*jsonl)
    #the huggingface version of the current dataset
    hf_dataset=("$dataset_dir"/*hf)
    propagation_inputs=("$dataset_dir"/*csv)

    ### --- in this code block we perform an entire pipeline

    #tokenize the dataset

    #---Where we want to tokenize our data
    tokenized_dir="${dataset_dir}/dataset_neox"

    #delete the directory if it existed before
#    if [ -e "$tokenized_dir" ]; then
#      echo "removing old directory $tokenized_dir"
#      rm -r $tokenized_dir
#    fi
#    mkdir $tokenized_dir
#
#    python $NEOX_DIR/tools/preprocess_data.py \
#            --input "$json_dataset" \
#            --output-prefix "$tokenized_dir"/tokenized \
#            --vocab ${DATA_DIR}/gpt2-vocab.json \
#            --merge-file ${DATA_DIR}/gpt2-merges.txt \
#            --dataset-impl mmap \
#            --tokenizer-type GPT2BPETokenizer \
#            --append-eod \
#            --workers 128

#    #---Where we want to train our data

    #folder location to store the model
    save=$(python misc.py\
            --mode get_model_dir\
            --dataset_dir $dataset_dir\
            --model_out_dir $model_out_dir)
    echo $save

    #delete the directory if it existed before
#    if [ -e "$save" ]; then
#      echo "removing old directory"
#      rm -r $save
#    fi
#
#    python update_configs.py\
#            --path_to_model_yaml ${config_dir}/$model_config_file\
#            --path_to_setup_yaml ${config_dir}/local_setup.yml\
#            --global_num_gpus $num_gpus\
#            --train_batch_size $train_batch_size\
#            --train_micro_batch_size_per_gpu $train_micro_batch_size_per_gpu\
#            --gradient_accumulation_steps=$gradient_accumulation_steps\
#            --train_iters $train_iters\
#            --data_path "$tokenized_dir"/tokenized_text_document\
#            --save $save\
#            --include "localhost:$gpu_names"
#
#    python $NEOX_DIR/deepy.py $NEOX_DIR/train.py \
#            -d $config_dir $model_config_file local_setup.yml
#
##    convert the model
#    python $NEOX_DIR/tools/convert_module_to_hf.py \
#            --input_dir "${save}/global_step${train_iters}/" \
#            --config_file ${config_dir}/$model_config_file \
#            --output_dir ${save}/hf_model

    CUDA_VISIBLE_DEVICES=$gpu_names python score_model.py\
            --path_to_model ${save}/hf_model\
            --path_to_inputs $propagation_inputs\
            --null_seed $null_seed\
            --null_n_seq $null_n_seq\
            --output_score_path ${save}/scored.csv


    ### --- in this code block we perform an entire pipeline
  done
else
  echo "Folder not found: $folder_path"
fi


#idea for script: Loop from 15 to 300, and train a model each, store it, and score it correspondingly

##Do not change below

#
#rm -r $WORKING_DIR/data/pipeline/*
#rm -r $WORKING_DIR/runs/70M_cp/*
#
##This preprocesses the data
#
#python $NEOX_DIR/tools/preprocess_data.py \
#            --input $DATASET \
#            --output-prefix $WORKING_DIR/data/pipeline/pipeline \
#            --vocab ${DATA_DIR}/gpt2-vocab.json \
#            --merge-file ${DATA_DIR}/gpt2-merges.txt \
#            --dataset-impl mmap \
#            --tokenizer-type GPT2BPETokenizer \
#            --append-eod \
#            --workers 128
#
#python $NEOX_DIR/deepy.py $NEOX_DIR/train.py \
#      -d $WORKING_DIR/pipeline 70M.yml local_setup.yml
#
#python $NEOX_DIR/tools/convert_module_to_hf.py \
#    --input_dir $WORKING_DIR/runs/70M_cp/global_step839/ \
#    --config_file $WORKING_DIR/pipeline/70M.yml \
#    --output_dir $WORKING_DIR/pipeline/$EXP_DIR
#
## score
#CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES papermill $WORKING_DIR/pipeline/score_gptneox.ipynb $WORKING_DIR/pipeline/$EXP_DIR/scoring_output.ipynb \
#    -p model_name $WORKING_DIR/pipeline/$EXP_DIR \
#    -p model_precision float32 \
#    -p input_fn $PROP_INPUTS \
#    -p output_fn $WORKING_DIR/pipeline/$EXP_DIR/scores.csv
#
## calculate propagation rate
#papermill $WORKING_DIR/pipeline/calculate_propagation_rates.ipynb $WORKING_DIR/pipeline/$EXP_DIR/propagation_output.ipynb \
#    -p input_fn $PROP_INPUTS \
#    -p scores_fn $WORKING_DIR/pipeline/$EXP_DIR/scores.csv
