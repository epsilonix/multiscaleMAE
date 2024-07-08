#!/bin/bash
#SBATCH -t 0-01:00:00
#SBATCH -p gpu4_long,gpu8_long
#SBATCH -N 1
#SBATCH --mem=250G
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --job-name=infer_MAE
#SBATCH --output=/gpfs/scratch/ss14424/logs/infer_%j.log


source activate /gpfs/home/ss14424/.conda/envs/canvas-env

python src/analysis/canvas_base.py


