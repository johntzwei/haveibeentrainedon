src="/mnt/nfs1/ryan/haveibeentrainedon/models"
experiment_name="unstealthy_scaling"
group_folder="repetition_final"
dataset_name="pile1e8_20len"
model_size="70M"

seed_arr=("seed0" "seed1" "seed2" "seed3" "seed4")

for seed in "${seed_arr[@]}"; do
  model_folders=("${src}/${experiment_name}/${group_folder}/${dataset_name}_${seed}/${model_size}"/*)


  for model_instance in "${model_folders[@]}"; do
    #loop through all folders inside this folder and delete the dataset
    all_folders=("${model_instance}"/*)

    for folder in "${all_folders[@]}"; do
      #remove if it is not a csv file
      if [[ $folder != *.csv ]]; then
        echo "folder is $folder"
  #      rm -r $folder
      fi
    done
  done
done