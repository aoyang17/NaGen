#!/bin/bash
#SBATCH --job-name=nagen-flow-horizon
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4
#SBATCH --time=00:30:00
#SBATCH --output=logs/flow-horizon-%A_%a.log
set -euo pipefail
cd /data02/home/scv7eyx/projects/NaGen
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
NAGEN_PY=/data02/home/scv7eyx/.conda/envs/nagen-chgnet/bin/python
HORIZONS=(0.1 0.25 0.5 0.75 1.0)
T=${HORIZONS[$SLURM_ARRAY_TASK_ID]}
"$NAGEN_PY" tools/generate_framework_candidates.py \
  --flow "$1" --condition configs/framework_condition_orthophosphate.json \
  --profile outputs/full_audit_v3/profile.json \
  --out "outputs/framework_horizon_${SLURM_ARRAY_JOB_ID}/t_${T}" \
  --count 256 --batch-size 4 --seed 161803 --ode-steps 48 \
  --source-mode polyhedral --flow-end-time "$T" --device cuda
