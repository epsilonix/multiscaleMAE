#!/bin/bash
#SBATCH -t 0-12:00:00
#SBATCH -p gpu4_long,gpu8_long,gpu4_medium,gpu8_medium,gpu4_short,gpu8_short
#SBATCH -N 1
#SBATCH --mem=50G
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --job-name=preprocessing_MAE
#SBATCH --output=/gpfs/scratch/ss14424/logs/preprocess_%j.log

# Modes: FULL or SUBSAMPLE (for SCME)
# Pipeline: SCME or LTME (this example uses SCME)

source activate /gpfs/data/tsirigoslab/home/ss14424/.conda/envs/canvasenv

python /gpfs/scratch/ss14424/singlecell/src/preprocessing/preprocess.py \
  /gpfs/scratch/ss14424/Brain/channels_37/tif \
  /gpfs/scratch/ss14424/Brain/data/celltype \
  /gpfs/scratch/ss14424/Brain/channels_37/cells_blankout/img_output_16_subsample \
  16 \
  subsample \
  SCME
