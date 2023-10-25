NEOX_DIR=./../../gpt-neox
DATA_DIR=./../../data
MODEL_DIR=./../../models

context_len=512
batch_size=1
dataset_path=${DATA_DIR}/17e7_tokens.jsonl

#above is all to change

#code to prepare model scoring
CUDA_VISIBLE_DEVICES=0 python perturb_bert.py\
  --dataset_path ${dataset_path}
