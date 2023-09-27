NEOX_DIR=./../gpt-neox
DATA_DIR=./../data
MODEL_DIR=./../models

#this is the directory to which your model is stored
model_dir=160M/160M_exp2_0_clean
#this is the specific checkpoint of your model
target_model=global_step32425
#this is the config of your model in the model_dir
yml_file=160M.yml


#above is needed to change

input=${MODEL_DIR}/${model_dir}/training/${target_model}
config=${MODEL_DIR}/${model_dir}/${yml_file}
output_name=final_epoch_hf
output=${MODEL_DIR}/${model_dir}/${output_name}


CUDA_VISIBLE_DEVICES=3 python ${NEOX_DIR}/tools/convert_module_to_hf.py --input_dir ${input} --config_file ${config} --output_dir ${output}