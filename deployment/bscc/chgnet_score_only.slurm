#!/bin/bash
#SBATCH --job-name=nagen-singlepoint
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --array=4-7%4
#SBATCH --time=00:20:00
#SBATCH --output=logs/singlepoint-%A_%a.log
set -euo pipefail
cd /data02/home/scv7eyx/projects/NaGen
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
NAGEN_SHARD=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
/data02/home/scv7eyx/.conda/envs/nagen-chgnet/bin/python -m nagen.selection.score \
  --input "outputs/full_audit_v3/shard_${NAGEN_SHARD}.jsonl" \
  --out "outputs/singlepoint_${SLURM_ARRAY_JOB_ID}/shard_${NAGEN_SHARD}/labels.jsonl" \
  --device cuda
# Deliberately no relaxation, ranking of training data, or training-CIF export.
