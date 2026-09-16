#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
export PYTHONPATH="$repo_dir/src"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
if [[ "${1:-}" == "--audit" ]]; then
  exec /mnt/data2/aobo/envs/NaGen/bin/python tools/audit_uma_campaign.py --run runs/campaign_24
fi
exec /mnt/data2/aobo/envs/NaGen/bin/python tools/run_uma_campaign.py run \
  --out runs/campaign_24 --target 24 --workers 8 --threads 2 --seed 2026091000 \
  --inference-settings batch_tf32 --relax-batch-size 4
