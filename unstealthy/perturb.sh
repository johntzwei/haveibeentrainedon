NEOX_DIR=./../gpt-neox
DATA_DIR=./../data
MODEL_DIR=./../models

watermark_length=20
vocab_size=100
num_proc=100
exp_name="unstealthy_scaling"
group_folder="scaling_final"
new_dataset_name="pile4e9_20len"
raw_dataset="${DATA_DIR}/17e7_tokens.jsonl"
repetitions="1 2"

#This exits the script if any command fails
set -e

#Do not change below:

#loop five times, each with different seed
for i in {0..0}
do
  for repetition in $repetitions
  do
    temp_new_dataset_name="${new_dataset_name}_seed${i}/${repetition}_dataset"
    out_dir="${DATA_DIR}/${exp_name}/${group_folder}/${temp_new_dataset_name}"

    if [ -e $out_dir ]; then
      rm -r $out_dir
    fi
    mkdir -p $out_dir

    echo "beginning for ${temp_new_dataset_name}}"

    python perturb_data.py\
      --exp_name ${exp_name}\
      --raw_dataset ${raw_dataset}\
      --watermark_length ${watermark_length}\
      --vocab_size ${vocab_size}\
      --out_dir ${out_dir}\
      --seed ${i}\
      --num_proc ${num_proc}\
      --repetition ${repetition}
  done
done


