#!/bin/bash
#SBATCH --job-name=nagen-relax-audit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:20:00
#SBATCH --output=logs/relax-audit-%j.log
set -euo pipefail
cd /data02/home/scv7eyx/projects/NaGen
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
/data02/home/scv7eyx/.conda/envs/nagen-chgnet/bin/python -m nagen.selection.relax_audit \
  --input outputs/chgnet_smoke_inputs/candidates.jsonl \
  --profile outputs/chgnet_gpu_2333320/selection/profile.json \
  --steps 200 --fmax 0.03 --device cuda --out "outputs/relax_audit_${SLURM_JOB_ID}"
