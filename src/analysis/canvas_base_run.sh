#!/bin/bash
#SBATCH -t 0-01:00:00
#SBATCH -p gpu4_long,gpu8_long
#SBATCH -N 1
#SBATCH --mem=100G
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --job-name=infer_MAE
#SBATCH --output=/gpfs/scratch/ss14424/logs/infer_%j.log

export NUMBA_DISABLE_JIT=1
export NUMBA_DISABLE_CACHE=1

source activate /gpfs/home/ss14424/.conda/envs/canvas-env

python src/analysis/canvas_base.py