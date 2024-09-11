#!/bin/bash
#SBATCH -t 0-12:00:00
#SBATCH -p gpu4_long,gpu8_long,gpu4_medium,gpu8_medium,gpu4_short,gpu8_short
#SBATCH -N 1
#SBATCH --mem=200G
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --job-name=infer_MAE
#SBATCH --output=/gpfs/scratch/ss14424/logs/infer_singlecell%j.log


source activate /gpfs/data/tsirigoslab/home/ss14424/.conda/envs/canvasenv

python src/analysis/canvas_base.py


