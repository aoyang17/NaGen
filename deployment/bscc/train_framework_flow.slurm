#!/bin/bash
#SBATCH --job-name=nagen-framework-flow
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=04:00:00
#SBATCH --output=logs/framework-flow-%j.log
set -euo pipefail
cd /data02/home/scv7eyx/projects/NaGen
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
NAGEN_PY=/data02/home/scv7eyx/.conda/envs/nagen-chgnet/bin/python
"$NAGEN_PY" -m nagen.inverse.train \
  --dataset outputs/conditional_data_v2/geometry.pt \
  --out "outputs/framework_flow_${SLURM_JOB_ID}.pt" \
  --epochs 100 --atom-budget 1024 --hidden 128 --layers 4 --heads 4 \
  --geometry-only --coupling assignment --geometry-validity-weight 0 \
  --init outputs/conditional_train_2333367/best.pt --seed 20260908
