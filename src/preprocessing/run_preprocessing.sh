#!/bin/bash
#SBATCH -t 0-07:00:00
#SBATCH -p gpu4_long,gpu8_long
#SBATCH -N 1
#SBATCH --mem=100G
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=preprocessing_MAE
#SBATCH --output=/gpfs/scratch/ss14424/logs/preprocess_%j.log


#modes are training and inference
source activate /gpfs/home/ss14424/.conda/envs/canvasenv

python \
/gpfs/scratch/ss14424/singlecell/src/preprocessing/preprocess.py \
/gpfs/scratch/ss14424/Brain/channels_37/tif \
/gpfs/scratch/ss14424/Brain/data/celltype \
/gpfs/scratch/ss14424/Brain/channels_37/cells_blankout/img_output_16_train \
16 \
train