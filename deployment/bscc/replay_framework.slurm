#!/bin/bash
#SBATCH --job-name=nagen-replay
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/replay-%j.log
set -euo pipefail
cd /data02/home/scv7eyx/projects/NaGen
export PYTHONPATH="$PWD/src"
NAGEN_PY=/data02/home/scv7eyx/.conda/envs/nagen-chgnet/bin/python
"$NAGEN_PY" tools/replay_framework_generation.py --run "$1" \
  --flow outputs/polyhedral_flow_2333476.best.pt \
  --condition configs/framework_condition_orthophosphate.json \
  --out "$1/generation/replay_audit.json" --device cuda
