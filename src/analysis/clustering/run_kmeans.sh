#!/bin/bash
#SBATCH -t 0-05:00:00
#SBATCH -p gpu4_long,gpu8_long,gpu4_medium,gpu8_medium,gpu4_short,gpu8_short
#SBATCH -N 1
#SBATCH --mem=200G
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --job-name=kmeans
#SBATCH --output=/gpfs/scratch/ss14424/logs/kmeans_%j.log

source activate /gpfs/home/ss14424/.conda/envs/canvas-env

emb_path="/gpfs/scratch/ss14424/Brain/channels_37/cells/analysis_output/tile_embedding/embedding_mean.npy"
save_path="/gpfs/scratch/ss14424/Brain/channels_37/cells/analysis_output/tile_embedding/labels_${n_clusters}.npy"

# Check if the number of clusters is passed as an argument
if [ -z "$1" ]; then
  echo "No n_clusters provided, using default value of 60"
  n_clusters=60
else
  n_clusters=$1
  echo "Using n_clusters=$n_clusters"
fi

save_path="/gpfs/scratch/ss14424/Brain/channels_37/cells/analysis_output/tile_embedding/labels_${n_clusters}.npy"

python src/analysis/clustering/kmeans.py $emb_path $n_clusters $save_path
