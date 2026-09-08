#!/bin/bash
#SBATCH --job-name=nagen-train-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:15:00
#SBATCH --output=logs/train-smoke-%j.log
set -euo pipefail
cd /data02/home/scv7eyx/projects/NaGen
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
/data02/home/scv7eyx/.conda/envs/nagen-chgnet/bin/python -m nagen.selection.train_generator \
  --dataset outputs/conditional_data_v1/geometry.pt \
  --out "outputs/conditional_smoke_${SLURM_JOB_ID}" --epochs 2 --max-batches 3
