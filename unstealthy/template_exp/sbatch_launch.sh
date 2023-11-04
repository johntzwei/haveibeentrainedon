#!/bin/bash
#SBATCH --nodelist=allegro-adams
#SBATCH --time=3-0:00
#SBATCH --job-name=sbatch
#SBATCH --gres=gpu:1
#SBATCH --ntasks=40
#SBATCH --output=160M_test.txt

source activate neoxv4
cd /home/ryan/haveibeentrainedon/unstealthy/scaling

bash pipeline_unstealthy.sh
