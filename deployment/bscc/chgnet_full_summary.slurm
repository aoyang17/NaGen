#!/bin/bash
#SBATCH --job-name=nagen-summary
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=01:00:00
#SBATCH --output=logs/full-summary-%j.log
set -euo pipefail
cd /data02/home/scv7eyx/projects/NaGen
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
NAGEN_PY=/data02/home/scv7eyx/.conda/envs/nagen-chgnet/bin/python
"$NAGEN_PY" tools/summarize_full_chgnet.py \
  --run "outputs/full_chgnet_${1}" --data outputs/full_dataset_v3 \
  --audit outputs/full_audit_v3 --matching outputs/full_split_matching_v3/summary.json \
  --out "outputs/full_chgnet_${1}/aggregate"
"$NAGEN_PY" tools/export_feasible_cif.py --run "outputs/full_chgnet_${1}/aggregate" \
  --out "outputs/full_chgnet_${1}/feasible_CIF"
