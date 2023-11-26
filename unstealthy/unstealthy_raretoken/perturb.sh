NEOX_DIR=./../../gpt-neox
DATA_DIR=./../../data
MODEL_DIR=./../../models

watermark_length=10
vocab_size=80
#the number of watermarks inserted for the dataset
num_watermarks=64
exp_name="unstealthy_raretoken_decoded"
new_dataset_name="wikitext_64"
raw_dataset="${DATA_DIR}/wikitext_orig"
#choose between "decoded" and "ids"
exp_type="decoded"

#Do not change below:
out_dir=${DATA_DIR}/${exp_name}/${new_dataset_name}

mkdir -p $out_dir

python perturb_dataset_raretoken.py\
  --raw_dataset ${raw_dataset}\
  --exp_type ${exp_type}\
  --num_watermarks ${num_watermarks}\
  --watermark_length ${watermark_length}\
  --vocab_size $vocab_size\
  --out_dir ${out_dir}