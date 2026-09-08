#!/bin/bash
#SBATCH --job-name=nagen-new-select
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=01:00:00
#SBATCH --output=logs/new-select-%j.log
set -euo pipefail
cd /data02/home/scv7eyx/projects/NaGen
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
/data02/home/scv7eyx/.conda/envs/nagen-chgnet/bin/python tools/select_new_candidates.py \
  --runs "outputs/new_candidates_${1}" --train outputs/full_dataset_v3/train.jsonl \
  --out "outputs/new_candidates_${1}/selection"
