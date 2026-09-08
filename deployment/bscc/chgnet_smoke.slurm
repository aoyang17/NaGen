#!/bin/bash
#SBATCH --job-name=nagen-chgnet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:20:00
#SBATCH --output=logs/chgnet-%j.log
set -euo pipefail
cd /data02/home/scv7eyx/projects/NaGen
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="$PWD/outputs/matplotlib"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
NAGEN_PY=/data02/home/scv7eyx/.conda/envs/nagen-chgnet/bin/python
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$NAGEN_PY" tools/verify_chgnet_bridge.py \
  --structure outputs/chgnet_smoke_inputs/train.jsonl --device cuda \
  --out "outputs/chgnet_gpu_${SLURM_JOB_ID}/bridge.json"
"$NAGEN_PY" -m nagen.selection.experiment \
  --train outputs/chgnet_smoke_inputs/train.jsonl \
  --candidates outputs/chgnet_smoke_inputs/candidates.jsonl \
  --references outputs/chgnet_smoke_inputs/train.jsonl \
  --min-structures 1 --check-gradient --device cuda \
  --out "outputs/chgnet_gpu_${SLURM_JOB_ID}/selection"
