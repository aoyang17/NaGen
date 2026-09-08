#!/bin/bash
#SBATCH --job-name=nagen-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-7%4
#SBATCH --time=02:00:00
#SBATCH --output=logs/full-%A_%a.log
set -euo pipefail
cd /data02/home/scv7eyx/projects/NaGen
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
NAGEN_PY=/data02/home/scv7eyx/.conda/envs/nagen-chgnet/bin/python
NAGEN_SHARD=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
NAGEN_OUT="outputs/full_chgnet_${SLURM_ARRAY_JOB_ID}/shard_${NAGEN_SHARD}"
"$NAGEN_PY" -m nagen.selection.score \
  --input "outputs/full_audit_v3/shard_${NAGEN_SHARD}.jsonl" \
  --out "${NAGEN_OUT}/labels.jsonl" --device cuda
"$NAGEN_PY" -m nagen.selection.relax_audit \
  --input "outputs/full_audit_v3/relax_shard_${NAGEN_SHARD}.jsonl" \
  --profile outputs/full_audit_v3/profile.json \
  --out "${NAGEN_OUT}/relax" --device cuda --steps 200 --fmax 0.03
