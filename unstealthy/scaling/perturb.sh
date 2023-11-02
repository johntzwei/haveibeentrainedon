NEOX_DIR=/home/johnny/gpt-neox
DATA_DIR=./../../data
MODEL_DIR=./../../models

watermark_length=10
vocab_size=80

#Do not change below:
raw_dataset=${DATA_DIR}/wikitext/0document_orig
out_dir=${DATA_DIR}/wikitext


python perturb_dataset_unstealthy.py\
  --raw_dataset ${raw_dataset}\
  --watermark_length ${watermark_length}\
  --vocab_size $vocab_size\
  --out_dir ${out_dir}
