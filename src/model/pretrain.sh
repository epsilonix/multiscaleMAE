#!/bin/bash
#SBATCH -t 5-00:00:00
#SBATCH -p gpu4_long,gpu8_long,gpu4_medium,gpu8_medium
#SBATCH -N 1
#SBATCH --mem=200G
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:8
#SBATCH --job-name=train_20chan_20px_all
#SBATCH --output=/gpfs/scratch/ss14424/logs/train_20chan_20px_all_%j.log

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
        --output_dir "/gpfs/scratch/ss14424/Brain/channels_20/cells/model_output_20_all" \
        --log_dir "/gpfs/scratch/ss14424/logs" \
        --data_path "/gpfs/scratch/ss14424/Brain/channels_20/cells/img_output_20_all"