NEOX_DIR=./../gpt-neox
DATA_DIR=./../data
MODEL_DIR=./../models

##the name of the folder to which the perturbed is stored
#hf_dataset=17e7_perturbed_seed416_sub1000
#model_dir_name=410M/410M_exp1_0_clean
#n_per_sub=1000
#seed=416
#
#context_len=512
#batch_size=1

#the name of the folder to which the perturbed is stored
hf_dataset=17e7_perturbed_seed416_sub1000
model_dir_name=160M/160M_exp2_0_clean
n_per_sub=1000
seed=416

context_len=512
batch_size=1

#above is all to change


hf_dataset_path=${DATA_DIR}/${hf_dataset}/perturbed_dataset_hf
model_dir=${MODEL_DIR}/${model_dir_name}
prepared_csv_path=${model_dir}/propagation_inputs.csv
model_path=${MODEL_DIR}/${model_dir_name}/final_epoch_hf
score_csv_path=${model_dir}/scores.csv


#code to prepare model scoring
python prepare_gptneox_script.py \
    --hf_dataset_path ${hf_dataset_path} \
    --n_per_sub ${n_per_sub} \
    --seed ${seed} \
    --prepared_csv_path ${prepared_csv_path}


#code to score model
CUDA_VISIBLE_DEVICES=0 python score_gptneox_script.py \
    --model_path ${model_path} \
    --context_len ${context_len} \
    --prepared_csv_path ${prepared_csv_path} \
    --score_csv_path ${score_csv_path} \
    --batch_size ${batch_size}