#!/bin/bash
#SBATCH --job-name=nagen-new-candidates
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-8%3
#SBATCH --time=01:00:00
#SBATCH --output=logs/new-candidates-%A_%a.log
set -euo pipefail
cd /data02/home/scv7eyx/projects/NaGen
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
NAGEN_SEEDS=(17 29 43)
NAGEN_CONDITION=$((SLURM_ARRAY_TASK_ID / 3))
NAGEN_SEED=${NAGEN_SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
/data02/home/scv7eyx/.conda/envs/nagen-chgnet/bin/python -m nagen.selection.candidate_pilot \
  --flow "outputs/conditional_train_${1}/best.pt" \
  --condition "outputs/conditional_data_v2/condition_${NAGEN_CONDITION}.json" \
  --profile outputs/full_audit_v3/profile.json \
  --gradient-structure outputs/chgnet_smoke_inputs/train.jsonl \
  --out "outputs/new_candidates_${SLURM_ARRAY_JOB_ID}/condition_${NAGEN_CONDITION}_seed_${NAGEN_SEED}" \
  --seed "$NAGEN_SEED" --count 8 --steps 8 --ode-steps 24
