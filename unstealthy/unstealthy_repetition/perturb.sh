NEOX_DIR=/home/johnny/gpt-neox
DATA_DIR=./../../data
MODEL_DIR=./../../models

watermark_length=10
vocab_size=80
total_documents_watermarked=200
exp_name=unstealthy_repetition

#Do not change below:
raw_dataset=${DATA_DIR}/${exp_name}/0document_orig
out_dir=${DATA_DIR}/${exp_name}


python perturb_data.py\
  --raw_dataset ${raw_dataset}\
  --watermark_length ${watermark_length}\
  --vocab_size $vocab_size\
  --total_documents_watermarked $total_documents_watermarked\
  --out_dir ${out_dir}
