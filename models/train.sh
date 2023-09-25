NEOX_DIR=./../gpt-neox
DATA_DIR=./../data
MODEL_DIR=./../models


#this is for the config files
config_directory=./410M/410M_exp2_0_clean

#for the specific file names
yml_file=410M.yml
local_setup_file=local_setup.yml


python ${NEOX_DIR}/deepy.py \
        ${NEOX_DIR}/train.py \
        -d ${config_directory} \
        ${yml_file} \
        ${local_setup_file}