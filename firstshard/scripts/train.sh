#The directory for config
CONFIG_DIR=./../config
#The directory for data
DATA_DIR=./../data
#The directory for model
MODEL_DIR=./../model
#The directory for analysis
ANALYSIS_DIR=./../analysis
#The directory for GPT-NeoX
NEOX_DIR=./../../gpt-neox

MODEL=perturb_model_1_percent

#Initialize the directories for model
mkdir -p ${MODEL_DIR}/${MODEL}
mkdir -p ${DATA_DIR}/${MODEL}/neox_tokenized

#This is to preprocess the data
#python ${NEOX_DIR}/tools/preprocess_data.py \
#  --input ${DATA_DIR}/${MODEL}/00_45e8_perturbed.jsonl \
#  --output-prefix ${DATA_DIR}/${MODEL}/neox_tokenized/first_shard_dataset_perturbed \
#  --vocab ${DATA_DIR}/gpt2-vocab.json \
#  --merge-file ${DATA_DIR}/gpt2-merges.txt \
#  --dataset-impl mmap \
#  --tokenizer-type GPT2BPETokenizer \
#  --append-eod \
#	--workers 120

#This is to train the data
python ${NEOX_DIR}/deepy.py ${NEOX_DIR}/train.py\
  -d ${CONFIG_DIR} 160M.yml local_setup.yml
