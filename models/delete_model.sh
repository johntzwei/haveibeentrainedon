experiment_name="unstealthy_scaling"
group_folder="old_experiments"
dataset_name="wikitext_40len"
model_size="70M"

model_folders=("${experiment_name}/${group_folder}"/"${dataset_name}"/"${model_size}"/*)


for model_instance in "${model_folders[@]}"; do
  #loop through all folders inside this folder and delete the dataset
  all_folders=("${model_instance}"/*)

  for folder in "${all_folders[@]}"; do
    #remove if it is not a csv file
    if [[ $folder != *.csv ]]; then
      rm -r $folder
    fi
  done
done