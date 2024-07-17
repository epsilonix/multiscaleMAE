#!/bin/bash
#SBATCH -t 0-05:00:00
#SBATCH -p gpu4_long,gpu8_long,gpu4_medium,gpu8_medium,gpu4_short,gpu8_short
#SBATCH -N 1
#SBATCH --mem=250G
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --job-name=kmeans
#SBATCH --output=/gpfs/scratch/ss14424/logs/kmeans_%j.log

source activate /gpfs/home/ss14424/.conda/envs/canvas-env

emb_path="/gpfs/scratch/ss14424/Brain/channels_20/cells/analysis_output_all_200/tile_embedding/embedding_mean.npy"
n_clusters=60
save_path="/gpfs/scratch/ss14424/Brain/channels_20/cells/analysis_output_all_200/clustering/labels.npy"

python src/analysis/clustering/kmeans.py $emb_path $n_clusters $save_path
