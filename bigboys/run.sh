NEOX_DIR=./../gpt-neox
DATA_DIR=./../data
MODEL_DIR=./../models
CONFIG_DIR=./configs
SRC_PATH=./../src

#This exits the script if any command fails
set -e

input_dir="inputs"
output_dir="outputs"

##############Hyperparameters to change START ##################

models=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "meta-llama/Llama-2-70b-hf" "EleutherAI/gpt-neox-20b" "bigscience/bloom-1b1" "bigscience/bloom-7b1")
#models=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf")

#models=("EleutherAI/gpt-neo-2.7B" "bigscience/bloom-1b1" "bigscience/bloom-7b1")

gpu_names="1,2,3"
#null distribution
null_seed=0
null_n_seq=1000
type="sha256"

input_file="stackoverflow/sha256_top100global_raw.csv"

###############Hyperparalsmeters to change END ##################


input_path="${input_dir}/${input_file}"

#if not exist, quit
if [ ! -e $input_path ]; then
  echo "input file does not exist"
  exit 1
fi

#strip the .txt from the input file
out_file=$(echo $input_path | rev | cut -f 1 -d '/' | rev | cut -f 1 -d '.' )

for model_name in "${models[@]}"; do
  out_name=$(echo $model_name | rev | cut -f 1 -d '/' | rev)
  out_folder="${output_dir}/${out_file}"
  if [ ! -d $out_folder ]; then
    mkdir -p $out_folder
  fi
  output_path="${out_folder}/${out_name}_out.txt"

  echo "------------Status: Beginning scoring for $model_name"

  CUDA_VISIBLE_DEVICES=${gpu_names} python run.py\
          --path_to_model "${model_name}"\
          --path_to_tokenizer "${model_name}"\
          --type "${type}"\
          --null_seed "${null_seed}"\
          --null_n_seq "${null_n_seq}"\
          --input_file "${input_path}"\
          --output_score_path "${output_path}"
done
