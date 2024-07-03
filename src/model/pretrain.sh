#!/bin/bash
#SBATCH -t 0-01:00:00
#SBATCH -p gpu4_long,gpu8_long,gpu4_medium,gpu8_medium,gpu8_short,gpu4_short
#SBATCH -N 1
#SBATCH --mem=200G
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:8
#SBATCH --job-name=train_20chan_20px_small
#SBATCH --output=/gpfs/scratch/ss14424/logs/train_20chan_20px_%x_%A_%a_%j_$(date +%m-%d).log

#unset PYTHONPATH
#module load anaconda3/gpu/5.2.0
source activate /gpfs/home/ss14424/.conda/envs/canvas-env
#module unload anaconda3/gpu/5.2.0

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    /gpfs/scratch/ss14424/singlecell/src/model/main_pretrain.py \
        --epoch 801 \
        --batch_size 32 \
        --tile_size 20 \
        --output_dir "/gpfs/scratch/ss14424/Brain/channels_20/cells_new/model_output_20" \
        --log_dir "/gpfs/scratch/ss14424/logs" \
        --data_path "/gpfs/scratch/ss14424/Brain/channels_20/cells_new/img_output_20"