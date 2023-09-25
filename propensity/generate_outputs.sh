#!/bin/bash
#SBATCH --job-name=propensity
#SBATCH --gres=gpu:2
#SBATCH --exclude=allegro-adams,glamor-ruby,ink-mia,ink-noah,ink-ron,ink-molly
#SBATCH --time=0-6:00
#SBATCH --ntasks=10
#SBATCH --qos=general

source activate torch

papermill ./score_device_map.ipynb ./outputs/outputs.ipynb \
    -p w1 ${w1} \
    -p w2 ${w2} \
    #-p model_name EleutherAI/pythia-70m \
    #-p folder_name outs/pythia-70m \