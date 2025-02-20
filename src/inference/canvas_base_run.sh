#!/bin/bash
#SBATCH -t 0-8:00:00
#SBATCH -p gpu4_long,gpu8_long,gpu4_medium,gpu8_medium,gpu4_short,gpu8_short
#SBATCH -N 1
#SBATCH --mem=100G
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --job-name=infer_MAE_blankout_full
#SBATCH --output=/gpfs/scratch/ss14424/logs/infer_singlecell%j.log

# Estimated runtime: < 1 hour for LTME, 6 - 8 hours for SCME.

source activate /gpfs/data/tsirigoslab/home/ss14424/.conda/envs/canvasenv

python src/inference/canvas_base.py \
  --pipeline SCME \
  --model_path "/path/to/your/checkpoint.pth" \
  --data_path "/path/to/your/img_output" \
  --save_path "/path/to/your/analysis_output" \
  --tile_size 16



