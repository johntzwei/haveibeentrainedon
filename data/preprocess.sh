NEOX_DIR=./../gpt-neox
DATA_DIR=./../data
MODEL_DIR=./../models

#for perturb data
#the folder under DATA_DIR
dataset_directory=17e7_perturbed_seed416_sub1000
#the specific input file name inside dataset_directory folder
input_data=${DATA_DIR}/${dataset_directory}/17e7_tokens_perturbed.jsonl


##for clean data
##the folder under DATA_DIR
#dataset_directory=17e7_clean
##the specific input file name inside dataset_directory folder
#input_data=${DATA_DIR}/17e7_tokens.jsonl




################### Only need to change above

#The specific output folder inside the dataset_directory
tokenized_dir=${DATA_DIR}/${dataset_directory}/dataset_neox/
output_data=${tokenized_dir}/tokenized


mkdir -p ${tokenized_dir}

python ${NEOX_DIR}/tools/preprocess_data.py \
            --input ${input_data} \
            --output-prefix ${output_data} \
            --vocab ${DATA_DIR}/gpt2-vocab.json \
            --merge-file ${DATA_DIR}/gpt2-merges.txt \
            --dataset-impl mmap \
            --tokenizer-type GPT2BPETokenizer \
            --append-eod \
						--workers 120

