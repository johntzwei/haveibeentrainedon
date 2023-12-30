NEOX_DIR=./../gpt-neox
DATA_DIR=./../data
MODEL_DIR=./../models

watermark_length=20
vocab_size=100
num_proc=100
#must make sure there is a "{datasets[i]}_orig.jsonl" file
datasets=("pile1e9" "pile2e9" "pile4e9" "pile8e9")
#datasets=("pile1e9")

exp_name="unstealthy_scaling"
group_folder="scaling_final"
repetitions=("1024")
#the start of the vocabulary to which we are extracting random sequences
start_range="0"
num_watermarks=1

#This exits the script if any command fails
set -e

#Do not change below:

#loop five times, each with different seed
for i in {1..1}
do
  #loop over the number of tokens (1e9, 2e9, etc)
  for dataset in "${datasets[@]}"
  do
    new_dataset_name="${dataset}_${watermark_length}len"
    raw_dataset="${DATA_DIR}/${dataset}_orig.jsonl"
    #loop over the repetitions needed
    for repetition in "${repetitions[@]}"
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
        --repetition ${repetition}\
        --num_watermarks ${num_watermarks}\
        --start_range ${start_range}
    done
  done
done


