experiment_name="unstealthy_scaling"
group_folder="watermark_length_final"
dataset_names=("pile1e8_80len_seed0" "pile1e8_80len_seed1" "pile1e8_80len_seed2" "pile1e8_80len_seed3" "pile1e8_80len_seed4")

for dataset_name in "${dataset_names[@]}"; do
  #loop through all folders inside this folder and delete the dataset
  all_folders=("${experiment_name}/${group_folder}/${dataset_name}"/*dataset)

  echo "${all_folders[@]}"

  for folder in "${all_folders[@]}"; do
    #extract the dataset name from the folder name
    curr_dataset=$(basename $folder)
    rm -r "${folder}/${curr_dataset}.hf"
    rm -r "${folder}/dataset_neox"
    rm "${folder}/${curr_dataset}.jsonl"
  done
done