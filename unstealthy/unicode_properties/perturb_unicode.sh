NEOX_DIR=./../../gpt-neox
DATA_DIR=./../../data
MODEL_DIR=./../../models

num_proc=100
#must make sure there is a "{datasets[i]}_orig.jsonl" file
#datasets=("pile1e9" "pile2e9" "pile4e9" "pile8e9")
datasets=("pile1e8")

exp_name="unicode_properties"
#choose from constant_perturbation vs sampled_perturbation
group_folder="constant_perturbation"
#num_documents=("1" "2" "4" "8" "16" "32" "64")
num_documents=("128" "256" "512" "1024")


#the number of null distribution sequences
null_n_seq=1000

#This exits the script if any command fails
set -e

#Do not change below:

#loop three times, each with different seed
for i in {0..4}
do
  #loop over the number of tokens (1e9, 2e9, etc)
  for dataset in "${datasets[@]}"
  do
    new_dataset_name="${dataset}"
    raw_dataset="${DATA_DIR}/${dataset}_orig.jsonl"
    #loop over the repetitions needed
    for num_document in "${num_documents[@]}"
    do
      temp_new_dataset_name="${new_dataset_name}_seed${i}/${num_document}_dataset"
      out_dir="${DATA_DIR}/${exp_name}/${group_folder}/${temp_new_dataset_name}"

      if [ -e $out_dir ]; then
        rm -r $out_dir
      fi
      mkdir -p $out_dir

      echo "beginning for ${temp_new_dataset_name}}"

      python ./../perturb_data.py\
        --exp_name ${exp_name}\
        --raw_dataset ${raw_dataset}\
        --out_dir ${out_dir}\
        --seed ${i}\
        --num_proc ${num_proc}\
        --repetition ${num_document}\
        --null_n_seq ${null_n_seq}
    done
  done
done


