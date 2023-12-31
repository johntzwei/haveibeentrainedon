NEOX_DIR=./../../gpt-neox
DATA_DIR=./../../data
MODEL_DIR=./../../models

watermark_length=10
vocab_size=80
#note: total_document_watermarked should be a power of 2, so we can to 1-64, 2-32, 4-16... 64-1
total_documents_watermarked=256
exp_name="unstealthy_repetition"
dataset_name="wikitext_256"

#Do not change below:
raw_dataset=${DATA_DIR}/wikitext_orig
out_dir=${DATA_DIR}/${exp_name}/${dataset_name}

mkdir -p $out_dir

python perturb_dataset_repetion.py\
  --raw_dataset ${raw_dataset}\
  --watermark_length ${watermark_length}\
  --vocab_size $vocab_size\
  --total_documents_watermarked $total_documents_watermarked\
  --out_dir ${out_dir}
