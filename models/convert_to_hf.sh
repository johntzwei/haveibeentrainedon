NEOX_DIR=./../gpt-neox
DATA_DIR=./../data
MODEL_DIR=./../models

#this is the directory to which your model is stored
model_dir=160M/160M_ambiguity_exp_5epoch
#this is the specific checkpoint of your model
target_model=global_step16212
#this is the config of your model in the model_dir
yml_file=160M.yml


#above is needed to change

#################

input=${MODEL_DIR}/${model_dir}/training/${target_model}
config=${MODEL_DIR}/${model_dir}/${yml_file}
output_name=epoch1
output=${MODEL_DIR}/${model_dir}/${output_name}

CUDA_VISIBLE_DEVICES=0 python ${NEOX_DIR}/tools/convert_module_to_hf.py --input_dir ${input} --config_file ${config} --output_dir ${output}

#################


#for file in ${MODEL_DIR}/${model_dir}/training/global*; do
#  echo "file=$file"
#done


#names="global_step12972 global_step16215 global_step19458 global_step22701 global_step25944 global_step29187 global_step32425 global_step3243 global_step6486 global_step9729"
#
#for name in $names; do
#  input=${MODEL_DIR}/${model_dir}/training/${name}
#  config=${MODEL_DIR}/${model_dir}/${yml_file}
#  output_name=${name}_hf
#  output=${MODEL_DIR}/${model_dir}/${output_name}
#  CUDA_VISIBLE_DEVICES=3 python ${NEOX_DIR}/tools/convert_module_to_hf.py --input_dir ${input} --config_file ${config} --output_dir ${output}
#  echo "finished $name"
#done