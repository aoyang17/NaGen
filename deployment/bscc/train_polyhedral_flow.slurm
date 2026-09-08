#!/bin/bash
#SBATCH --job-name=nagen-poly-flow
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=04:00:00
#SBATCH --output=logs/poly-flow-%j.log
set -euo pipefail
cd /data02/home/scv7eyx/projects/NaGen
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
NAGEN_PY=/data02/home/scv7eyx/.conda/envs/nagen-chgnet/bin/python
"$NAGEN_PY" -m nagen.inverse.train --dataset outputs/orthophosphate_data/geometry.pt \
  --out "outputs/polyhedral_flow_${SLURM_JOB_ID}.pt" --epochs 120 --atom-budget 832 \
  --hidden 128 --layers 4 --heads 4 --geometry-only --coupling polyhedral \
  --geometry-validity-weight 0 --seed 20260909
