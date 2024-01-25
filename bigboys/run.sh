NEOX_DIR=./../gpt-neox
DATA_DIR=./../data
MODEL_DIR=./../models
CONFIG_DIR=./configs
SRC_PATH=./../src

#This exits the script if any command fails
set -e

##############Hyperparameters to change START ##################

gpu_names="9"
model_name="meta-llama/Llama-2-7b-hf"
tokenizer_name="meta-llama/Llama-2-7b-hf"


#null distribution
null_seed=0
null_n_seq=1000
type="hash"

input_file="input.txt"
out_file="output.txt"

##############Hyperparalsmeters to change END ##################

CUDA_VISIBLE_DEVICES=${gpu_names} python run.py\
        --path_to_model "${model_name}"\
        --path_to_tokenizer "${tokenizer_name}"\
        --type "${type}"\
        --null_seed "${null_seed}"\
        --null_n_seq "${null_n_seq}"\
        --input_file "${input_file}"\
        --output_score_path "${out_file}"

