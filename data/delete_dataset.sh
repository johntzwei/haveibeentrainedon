experiment_name="unstealthy_raretoken"
group_folder="run"
prefix=("pile1e8_20len")
#dataset_names=("pile1e9_20len_seed0" "pile1e9_20len_seed0" "pile1e9_20len_seed0" "pile1e9_20len_seed0" "pile1e9_20len_seed0")

for dataset_name in "${prefix[@]}"; do
  for seed in {0..4}; do
    #loop through all folders inside this folder and delete the dataset
    all_folders=("${experiment_name}/${group_folder}/${dataset_name}_seed${seed}"/*dataset)

    echo "${all_folders[@]}"

    for folder in "${all_folders[@]}"; do
      curr_dataset=$(basename $folder)
#      rm -r "${folder}/${curr_dataset}.hf"
      rm -r "${folder}/dataset_neox"
      rm "${folder}/${curr_dataset}.jsonl"
    done
  done
done