NEOX_DIR=/home/johnny/gpt-neox
DATA_DIR=./../../data
MODEL_DIR=./../../models

watermark_length=40
vocab_size=80
exp_name="unstealthy_scaling"
new_dataset_name="wikitext_40len"
raw_dataset="${DATA_DIR}/wikitext_orig"

#Do not change below:
out_dir=${DATA_DIR}/${exp_name}/${new_dataset_name}

mkdir -p $out_dir

python perturb_dataset_unstealthy.py\
  --raw_dataset ${raw_dataset}\
  --watermark_length ${watermark_length}\
  --vocab_size $vocab_size\
  --out_dir ${out_dir}
