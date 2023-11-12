NEOX_DIR=/home/johnny/gpt-neox
DATA_DIR=./../../data
MODEL_DIR=./../../models

watermark_length=10
vocab_size=80
#note: total_document_watermarked should be a power of 2, so we can to 1-64, 2-32, 4-16... 64-1
total_documents_watermarked=64
exp_name="unstealthy_repetition"
dataset_name="pile1e9"

#Do not change below:
raw_dataset=${DATA_DIR}/pile1e9_orig
out_dir=${DATA_DIR}/${exp_name}/${dataset_name}

mkdir -p $out_dir

python perturb_data.py\
  --raw_dataset ${raw_dataset}\
  --watermark_length ${watermark_length}\
  --vocab_size $vocab_size\
  --total_documents_watermarked $total_documents_watermarked\
  --out_dir ${out_dir}
