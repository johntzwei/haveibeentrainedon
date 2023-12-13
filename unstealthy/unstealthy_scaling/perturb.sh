NEOX_DIR=./../../gpt-neox
DATA_DIR=./../../data
MODEL_DIR=./../../models

watermark_length=20
vocab_size=100
num_proc=100
exp_name="unstealthy_scaling"
group_folder="scaling_final"
new_dataset_name="pile2e9_20len"
raw_dataset="${DATA_DIR}/pile2e9_orig"

#This exits the script if any command fails
set -e

#Do not change below:

#loop five times, each with different seed
for i in {4..4}
do
  temp_new_dataset_name="${new_dataset_name}_seed${i}"
  out_dir="${DATA_DIR}/${exp_name}/${group_folder}/${temp_new_dataset_name}"

  mkdir -p $out_dir

  python perturb_dataset_unstealthy.py\
    --raw_dataset ${raw_dataset}\
    --watermark_length ${watermark_length}\
    --vocab_size ${vocab_size}\
    --out_dir ${out_dir}\
    --seed ${i}\
    --num_proc ${num_proc}
done


