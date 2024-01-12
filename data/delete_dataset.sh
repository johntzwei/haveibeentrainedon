experiment_name="unstealthy_scaling"
group_folder="scaling_final"
prefix=("pile1e9_80len" "pile2e9_80len" "pile4e9_80len" "pile8e9_80len")
#dataset_names=("pile1e9_20len_seed0" "pile1e9_20len_seed0" "pile1e9_20len_seed0" "pile1e9_20len_seed0" "pile1e9_20len_seed0")

for dataset_name in "${prefix[@]}"; do
  for seed in {0..4}; do
    #loop through all folders inside this folder and delete the dataset
#    all_folders=("${experiment_name}/${group_folder}/${dataset_name}_seed${seed}"/*dataset)
    all_folders=("${experiment_name}/${group_folder}/${dataset_name}_seed${seed}"/2048_dataset)
    echo "${all_folders[@]}"

    for folder in "${all_folders[@]}"; do
      curr_dataset=$(basename $folder)
#      echo $folder
#      rm -r "${folder}"
#      rm -r "${folder}/${curr_dataset}.hf"
#      rm -r "${folder}/dataset_neox"
#      rm "${folder}/${curr_dataset}.jsonl"
    done
  done
done