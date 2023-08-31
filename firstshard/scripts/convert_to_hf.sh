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

#for the base model
#python ${NEOX_DIR}/tools/convert_module_to_hf.py --input_dir ${MODEL_DIR}/base_model/global_step2146/ --config_file ${CONFIG_DIR}/160M.yml --output_dir ${ANALYSIS_DIR}/base_model/global_step_2146

#for the perturb model
python ${NEOX_DIR}/tools/convert_module_to_hf.py --input_dir ${MODEL_DIR}/perturb_model_1_percent/global_step2146/ --config_file ${CONFIG_DIR}/160M.yml --output_dir ${ANALYSIS_DIR}/perturb_model_one_percent/global_step_2146
