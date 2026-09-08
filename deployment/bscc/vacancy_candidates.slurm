#!/bin/bash
#SBATCH --job-name=nagen-vacancy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --output=logs/vacancy-%j.log
set -euo pipefail
cd /data02/home/scv7eyx/projects/NaGen
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
NAGEN_PY=/data02/home/scv7eyx/.conda/envs/nagen-chgnet/bin/python
"$NAGEN_PY" tools/generate_vacancy_candidates.py --train outputs/full_dataset_v3/train.jsonl \
  --parent mix_Na4Fe3P4O15_8 --target-na 12 --count 64 --seed 17 \
  --out "outputs/vacancy_${SLURM_JOB_ID}/generated.jsonl"
"$NAGEN_PY" -m nagen.selection.relax_audit --input "outputs/vacancy_${SLURM_JOB_ID}/generated.jsonl" \
  --profile outputs/full_audit_v3/profile.json --out "outputs/vacancy_${SLURM_JOB_ID}/relax" \
  --device cuda --steps 300 --fmax 0.03
"$NAGEN_PY" tools/select_relaxed_candidates.py --run "outputs/vacancy_${SLURM_JOB_ID}/relax" \
  --train outputs/full_dataset_v3/train.jsonl --out "outputs/vacancy_${SLURM_JOB_ID}/selection"
