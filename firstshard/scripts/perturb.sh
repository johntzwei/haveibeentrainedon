#The directory for config
CONFIG_DIR=./../config
#The directory for data
DATA_DIR=./../data
#The directory for model
MODEL_DIR=./../model
#The directory for analysis
ANALYSIS_DIR=./../analysis
#The directory for GPT-NeoX
NEOX_DIR=./../../gpt-neox

MODEL=perturb_model_1_percent

#Initialize the directories
mkdir -p ${DATA_DIR}/${MODEL}

python ${DATA_DIR}/first_shard_perturb_dataset.py\
  --data_path ${DATA_DIR}/00_45e8.jsonl\
  --swap_arr_name ${DATA_DIR}/${MODEL}/00_45e8_swap_arr.npy\
  --hf_dataset_name ${DATA_DIR}/${MODEL}/first_shard_perturbed_seed:416.hf\
  --out_json_dataset_name ${DATA_DIR}/${MODEL}/00_45e8_perturbed.jsonl

