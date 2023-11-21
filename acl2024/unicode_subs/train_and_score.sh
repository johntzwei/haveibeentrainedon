#!/bin/bash
#SBATCH --nodelist=dill-sage
#SBATCH --time=3-0:00
#SBATCH --job-name=train
#SBATCH --ntasks=16
#SBATCH --gres=gpu:4
#SBATCH --array=1-1
#SBATCH --output=logs/slurm-%A_%a.out

eval "$(conda shell.bash hook)"
conda activate gpt-neox

NEOX_DIR=/home/johnny/gpt-neox
WORKING_DIR=$NEOX_DIR/haveibeentrainedon/acl2024/unicode_subs
CUDA_HOME=$CONDA_PREFIX
MODEL_YML=70M.yml

frac=`echo "1 0.5 0.25 0.125 0.0625" | cut -d' ' -f${SLURM_ARRAY_TASK_ID}`
mkdir -p $WORKING_DIR/data/frac:$frac

DS_PREFIX=$WORKING_DIR/data/frac:$frac/wikitext_perturbed_text_document
MODEL_DIR=$WORKING_DIR/data/frac:$frac/70M
cat $WORKING_DIR/configs/local_setup.yml | python create_yaml.py $DS_PREFIX $MODEL_DIR > $WORKING_DIR/configs/local_setup:$frac.yml

python $NEOX_DIR/deepy.py $NEOX_DIR/train.py \
      -d $WORKING_DIR/configs $MODEL_YML local_setup:$frac.yml

ckpt=`cat $MODEL_DIR/latest`
python $NEOX_DIR/tools/convert_module_to_hf.py \
    --input_dir $MODEL_DIR/$ckpt \
    --config_file $WORKING_DIR/configs/$MODEL_YML \
    --output_dir $MODEL_DIR

conda activate torch
papermill $WORKING_DIR/score_null.ipynb $WORKING_DIR/output.ipynb \
    -p model_name $MODEL_DIR \
    -p input_fn $WORKING_DIR/data/frac:$frac/samples.csv \
    -p output_fn $WORKING_DIR/scores/scores:$frac.csv
