NEOX_DIR=./../gpt-neox
DATA_DIR=./../data
MODEL_DIR=./../models
CONFIG_DIR=./configs
SRC_PATH=./../src

#This exits the script if any command fails
set -e

input_dir="inputs"
output_dir="outputs"
null_dir="null"

##############Hyperparameters to change START ##################
input_file="stackoverflow/sha512_top100global_raw.csv"
prepend_str=""
#This is useless if we are inferencing on huggingface
gpu_names="1,2,3"

#<<<< Inference Type
#if we want to inference locally
#use_huggingface_api="false"
#models=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "meta-llama/Llama-2-70b-hf" "EleutherAI/gpt-neox-20b" "bigscience/bloom-1b1" "bigscience/bloom-7b1")

#if we want to ifnerence on the huggingface api
use_huggingface_api="true"
models=("bigscience/bloom-176b")
#>>>>

#<<<< Null Distribution
null_seed=0
null_n_seq=1000
type="sha512"
lower_only="true"
#>>>>
###############Hyperparalsmeters to change END ##################

input_path="${input_dir}/${input_file}"

#if input file does not exist, quit
if [ ! -e $input_path ]; then
  echo "input file does not exist"
  exit 1
fi

#strip the .txt from the input file
out_file=$(echo $input_path | rev | cut -f 1 -d '/' | rev | cut -f 1 -d '.' )

for model_name in "${models[@]}"; do
  #Extract the model name
  out_name=$(echo $model_name | rev | cut -f 1 -d '/' | rev)

  #obtain the null (through cached or generation
  CUDA_VISIBLE_DEVICES=${gpu_names} python create_null.py\
          --null_dir "${null_dir}"\
          --model_name "${out_name}"\
          --path_to_model "${model_name}"\
          --path_to_tokenizer "${model_name}"\
          --type "${type}"\
          --null_seed "${null_seed}"\
          --null_n_seq "${null_n_seq}"\
          --prepend_str "${prepend_str}"\
          --use_huggingface_api "${use_huggingface_api}"\
          --lower_only "${lower_only}"
  #Create the output folder
  out_folder="${output_dir}/${out_file}"
  if [ ! -d $out_folder ]; then
    mkdir -p $out_folder
  fi
  output_path="${out_folder}/${out_name}_out.txt"

  echo "------------Status: Beginning scoring for $model_name"

  CUDA_VISIBLE_DEVICES=${gpu_names} python run.py\
          --path_to_model "${model_name}"\
          --path_to_tokenizer "${model_name}"\
          --model_name "${out_name}"\
          --type "${type}"\
          --null_dir "${null_dir}"\
          --null_seed "${null_seed}"\
          --null_n_seq "${null_n_seq}"\
          --input_file "${input_path}"\
          --output_score_path "${output_path}"\
          --prepend_str "${prepend_str}"\
          --use_huggingface_api "${use_huggingface_api}"\
          --lower_only "${lower_only}"
done
