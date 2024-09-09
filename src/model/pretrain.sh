#!/bin/bash
#SBATCH -t 3-00:00:00
#SBATCH -p gpu4_long,gpu8_long,gpu4_medium,gpu8_medium
#SBATCH -N 1
#SBATCH --mem=200G
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:8
#SBATCH --job-name=train_37chan_20px_standard
#SBATCH --output=/gpfs/scratch/ss14424/logs/train_37chan_20px_standard_%j.log

#unset PYTHONPATH
#module load anaconda3/gpu/5.2.0
source activate /gpfs/data/tsirigoslab/home/ss14424/.conda/envs/canvasenv
#module unload anaconda3/gpu/5.2.0

export TORCHELASTIC_ERROR_FILE=/gpfs/scratch/ss14424/logs/torch_elastic_error.log

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    /gpfs/scratch/ss14424/singlecell/src/model/main_pretrain.py \
        --epoch 1001 \
        --batch_size 32 \
        --tile_size 20 \
        --output_dir "/gpfs/scratch/ss14424/Brain/channels_37/cells_test/model_output_20" \
        --log_dir "/gpfs/scratch/ss14424/logs" \
        --data_path "/gpfs/scratch/ss14424/Brain/channels_37/cells/img_output_20_train" #\
#        --blankoutbg \
#        --resume "/gpfs/scratch/ss14424/Brain/channels_37/cells_blankout/model_output_20/checkpoint-40.pth"
