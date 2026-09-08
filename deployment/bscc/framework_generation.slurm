#!/bin/bash
#SBATCH --job-name=nagen-framework-gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --output=logs/framework-gen-%j.log
set -euo pipefail
cd /data02/home/scv7eyx/projects/NaGen
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
NAGEN_PY=/data02/home/scv7eyx/.conda/envs/nagen-chgnet/bin/python
RUN="outputs/framework_generation_${SLURM_JOB_ID}"
SOURCE_MODE=${2:-uniform}
CONDITION=${3:-outputs/conditional_data_v2/condition_0.json}
"$NAGEN_PY" tools/generate_framework_candidates.py \
  --flow "$1" --condition "$CONDITION" --source-mode "$SOURCE_MODE" \
  --profile outputs/full_audit_v3/profile.json --out "$RUN/generation" \
  --count 128 --batch-size 4 --seed 271828 --ode-steps 48 --device cuda
if test -s "$RUN/generation/generated_safe.jsonl"; then
  "$NAGEN_PY" -m nagen.selection.relax_audit --input "$RUN/generation/generated_safe.jsonl" \
    --profile outputs/full_audit_v3/profile.json --out "$RUN/relax" --device cuda --steps 400 --fmax .03
fi
