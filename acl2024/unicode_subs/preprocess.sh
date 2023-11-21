#!/bin/bash
#SBATCH --nodelist=ink-lucy
#SBATCH --time=3-0:00
#SBATCH --job-name=preprocess
#SBATCH --ntasks=16
#SBATCH --array=1-1
#SBATCH --output=logs/slurm-%A_%a.out

eval "$(conda shell.bash hook)"
conda activate gpt-neox

NEOX_DIR=/home/johnny/gpt-neox
WORKING_DIR=$NEOX_DIR/haveibeentrainedon/acl2024/unicode_subs

frac=`echo "1 0.5 0.25 0.125 0.0625" | cut -d' ' -f${SLURM_ARRAY_TASK_ID}`
mkdir -p $WORKING_DIR/data/frac:$frac

DS_PREFIX=$WORKING_DIR/data/frac:$frac/wikitext_perturbed
papermill prepare.ipynb $WORKING_DIR/output.ipynb \
    -p frac_controlled $frac \
    -p out_dataset_name $DS_PREFIX \
    -p out_samples_name $WORKING_DIR/data/frac:$frac/samples.csv \
    -p debug False
rm -r $DS_PREFIX.hf

python $NEOX_DIR/tools/preprocess_data.py \
    --input $DS_PREFIX.jsonl \
    --output-prefix $WORKING_DIR/data/frac:$frac/wikitext_perturbed \
    --vocab $NEOX_DIR/data/gpt2-vocab.json \
    --merge-file $NEOX_DIR/data/gpt2-merges.txt \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --append-eod \
    --workers 8
