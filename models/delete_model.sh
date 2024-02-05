src="/home/ryan/haveibeentrainedon/models"
experiment_name="unicode_scaling"
group_folder="run"
dataset_name=("1B_perturbed" "2B_perturbed" "4B_perturbed" "8B_perturbed")

model_size="160M"

seed_arr=("seed1" "seed2" "seed3")

for seed in "${seed_arr[@]}"; do
  for dataset in "${dataset_name[@]}"; do
    model_folders=("${src}/${experiment_name}/${group_folder}/${dataset}_${seed}/${model_size}"/*)

    for model_instance in "${model_folders[@]}"; do
      #loop through all folders inside this folder and delete the dataset
      echo ${model_instance}
      all_files=("${model_instance}"/*)

      for file in "${all_files[@]}"; do
        #continue the loop if file is a csv file
        if [[ $file == *".csv"* ]]; then
          continue
        fi

        #remove if it is a folder ir is called "zero_to_fp32.py"
        if [[ -d $file ]]; then
          echo "folder is $file"
          rm -r $file
        elif [[ $file == *"zero_to_fp32.py"* ]]; then
          echo "file is $file"
          rm $file
        fi
      done
    done
  done
done