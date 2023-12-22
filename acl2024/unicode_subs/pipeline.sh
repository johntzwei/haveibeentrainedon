#!/bin/bash
#SBATCH --nodelist=dill-sage
#SBATCH --time=3-0:00
#SBATCH --job-name=train
#SBATCH --ntasks=16
#SBATCH --gres=gpu:1
#SBATCH --array=1-2
#SBATCH --output=logs/slurm-%A_%a.out

eval "$(conda shell.bash hook)"
conda activate gpt-neox

NEOX_DIR=/home/johnny/gpt-neox
WORKING_DIR=$NEOX_DIR/haveibeentrainedon/acl2024/unicode_subs
CUDA_HOME=$CONDA_PREFIX
MODEL_YML=160M.yml
EXP_NAME=sample_chars_160M

frac=`echo "2 4 8 16 32" | cut -d' ' -f${SLURM_ARRAY_TASK_ID}`
DS_DIR=$WORKING_DIR/data/$EXP_NAME/frac:$frac
DS_PREFIX=$DS_DIR/pile_perturbed

mkdir -p $DS_DIR
rm -r $DS_DIR

papermill prepare.ipynb $WORKING_DIR/output.ipynb \
    -p frac_controlled 1 \
    -p frac_contaminated $frac \
    -p out_dataset_name $DS_PREFIX \
    -p out_samples_name $DS_DIR/samples.csv \
    -p strategy sample_chars \
    -p debug False
rm -r $DS_PREFIX.hf

python $NEOX_DIR/tools/preprocess_data.py \
    --input $DS_PREFIX.jsonl \
    --output-prefix $DS_PREFIX \
    --vocab $NEOX_DIR/data/gpt2-vocab.json \
    --merge-file $NEOX_DIR/data/gpt2-merges.txt \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --append-eod \
    --workers 8

MODEL_DIR=$DS_DIR/model
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
    -p input_fn $DS_DIR/samples.csv \
    -p output_fn $WORKING_DIR/scores/$EXP_NAME/scores:$frac.csv
