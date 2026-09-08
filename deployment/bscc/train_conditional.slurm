#!/bin/bash
#SBATCH --job-name=nagen-conditional
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=04:00:00
#SBATCH --output=logs/conditional-%j.log
set -euo pipefail
cd /data02/home/scv7eyx/projects/NaGen
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
/data02/home/scv7eyx/.conda/envs/nagen-chgnet/bin/python -m nagen.selection.train_generator \
  --dataset outputs/conditional_data_v2/geometry.pt \
  --out "outputs/conditional_train_${SLURM_JOB_ID}" --epochs 100 --patience 20 --seed 17
