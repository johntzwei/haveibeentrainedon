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
mkdir -p ${ANALYSIS_DIR}/${MODEL}

CUDA_VISIBLE_DEVICES=9 python ${ANALYSIS_DIR}/score_data.py\
  --model_path ${ANALYSIS_DIR}/${MODEL}/global_step_2146\
  --data_path ${DATA_DIR}/00_45e8.jsonl\
  --swap_arr_path ${DATA_DIR}/${MODEL}/00_45e8_swap_arr.npy\
  --prop_inputs ${ANALYSIS_DIR}/${MODEL}/prop_inputs.csv\
  --out_path ${ANALYSIS_DIR}/${MODEL}/score.csv\

