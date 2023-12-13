#!/bin/bash
#SBATCH --nodelist=glamor-ruby
#SBATCH --time=3-0:00
#SBATCH --job-name=train
#SBATCH --ntasks=16
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --output=logs/slurm-%A_%a.out

eval "$(conda shell.bash hook)"
conda activate gpt-neox

NEOX_DIR=/home/johnny/gpt-neox
WORKING_DIR=$NEOX_DIR/haveibeentrainedon/acl2024/unicode_subs
CUDA_HOME=$CONDA_PREFIX
MODEL_YML=70M.yml
EXP_NAME=replace_all_twenty

frac=`echo "2 4 8 16 32" | cut -d' ' -f${SLURM_ARRAY_TASK_ID}`
mkdir -p $WORKING_DIR/data/frac:$frac
rm $WORKING_DIR/data/frac:$frac/*

DS_PREFIX=$WORKING_DIR/data/frac:$frac/pile_perturbed
papermill prepare.ipynb $WORKING_DIR/output.ipynb \
    -p frac_controlled 1 \
    -p frac_contaminated $frac \
    -p out_dataset_name $DS_PREFIX \
    -p out_samples_name $WORKING_DIR/data/frac:$frac/samples.csv \
    -p strategy replace_all_twenty \
    -p debug False
rm -r $DS_PREFIX.hf

python $NEOX_DIR/tools/preprocess_data.py \
    --input $DS_PREFIX.jsonl \
    --output-prefix $WORKING_DIR/data/frac:$frac/pile_perturbed \
    --vocab $NEOX_DIR/data/gpt2-vocab.json \
    --merge-file $NEOX_DIR/data/gpt2-merges.txt \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --append-eod \
    --workers 8

MODEL_DIR=$WORKING_DIR/data/frac:$frac/70M
rm -r $MODEL_DIR    # clean up

cat $WORKING_DIR/configs/local_setup.yml | python create_yaml.py ${DS_PREFIX}_text_document $MODEL_DIR > $WORKING_DIR/configs/local_setup:$frac.yml

python $NEOX_DIR/deepy.py $NEOX_DIR/train.py \
      -d $WORKING_DIR/configs $MODEL_YML local_setup:$frac.yml

ckpt=`cat $MODEL_DIR/latest`
python $NEOX_DIR/tools/convert_module_to_hf.py \
    --input_dir $MODEL_DIR/$ckpt \
    --config_file $WORKING_DIR/configs/$MODEL_YML \
    --output_dir $MODEL_DIR

conda activate torch
mkdir -p $WORKING_DIR/scores/$EXP_NAME
papermill $WORKING_DIR/score_null.ipynb $WORKING_DIR/output.ipynb \
    -p model_name $MODEL_DIR \
    -p input_fn $WORKING_DIR/data/frac:$frac/samples.csv \
    -p output_fn $WORKING_DIR/scores/$EXP_NAME/scores:$frac.csv
