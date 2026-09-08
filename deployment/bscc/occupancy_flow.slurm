#!/bin/bash
#SBATCH --job-name=nagen-occflow
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=08:00:00
#SBATCH --output=logs/occflow-%j.log
set -euo pipefail
cd /data02/home/scv7eyx/projects/NaGen
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
NAGEN_PY=/data02/home/scv7eyx/.conda/envs/nagen-chgnet/bin/python
RUN="outputs/occupancy_flow_${SLURM_JOB_ID}"
"$NAGEN_PY" tools/run_occupancy_flow.py --train outputs/full_dataset_v3/train.jsonl \
  --parent mix_Na4Fe3P4O15_8 --target-na 12 --epochs 4000 --samples 256 --steps 96 \
  --seed 314159 --device cuda --emit-unique-only --out "$RUN/generation"
"$NAGEN_PY" -m nagen.selection.relax_audit --input "$RUN/generation/generated.jsonl" \
  --profile outputs/full_audit_v3/profile.json --out "$RUN/relax" --device cuda --steps 300 --fmax 0.03
"$NAGEN_PY" tools/select_relaxed_candidates.py --run "$RUN/relax" \
  --generated "$RUN/generation/generated.jsonl" --train outputs/full_dataset_v3/train.jsonl \
  --hull-labels outputs/full_singlepoint_verified/labels.jsonl --out "$RUN/selection"
