#!/bin/bash
#SBATCH --job-name=propensity
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ink-titan
#SBATCH --time=0-6:00
#SBATCH --ntasks=10
#SBATCH --qos=general

source activate torch

#SBATCH --exclude=allegro-adams,glamor-ruby,ink-mia,ink-noah,ink-titan,ink-ron,ink-molly,ink-ron

papermill ./propensity_scoring.ipynb ./outputs/${w1}_${w2}.ipynb \
    -p w1 ${w1} \
    -p w2 ${w2} \
    -p num_proc 10
