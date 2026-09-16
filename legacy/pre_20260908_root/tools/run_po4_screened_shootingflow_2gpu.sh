#!/usr/bin/env bash
set -euo pipefail
repo_dir=/home/aobo/NaGen
python_bin=/mnt/data2/aobo/envs/NaGen/bin/python
flow=/mnt/data2/aobo/NaGen/inverse_v1/checkpoints/naxl_unconditional_surrogate_ready_v2.best.pt
surrogate=/mnt/data2/aobo/NaGen/surrogate_ehull/runs/final_mace_v1/best_model.pt
packed=/mnt/data2/aobo/NaGen/inverse_v1/data/naxl_packed_v1.pt
root="$repo_dir/outputs/shootingflow_po4_screened_20260908"
mkdir -p "$root"; cd "$repo_dir"

worker() {
  local gpu="$1" seed_base="$2" label="$3" good=0 attempt=0
  while (( good < 3 && attempt < 20 )); do
    local run="$root/$label/attempt_$attempt" seed=$((seed_base + attempt))
    mkdir -p "$run/cifs"
    if CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$repo_dir/src:$repo_dir/.pylibs" "$python_bin" tools/run_surrogate_shootingflow.py \
      --flow "$flow" --surrogate "$surrogate" --packed-data "$packed" --out "$run/results.json" --cif-dir "$run/cifs" \
      --count 1 --batch-size 1 --steps 24 --optimizer hybrid-alm --adam-steps 16 --lbfgs-steps 8 --lbfgs-lr 0.15 --lr 0.03 \
      --ode-steps 16 --graph-refresh-steps 8 --seed "$seed" --device cuda --source-candidates 256 --require-initial-po4 \
      --max-p-sites 8 --composition-oversample-factor 30 --composition-pool-factor 512 --composition-max-rounds 60 \
      --composition-proposal-batch-limit 512 --poly-center-weight 0 --poly-face-weight 0 --poly-coplanar-weight 0 \
      --po4-constraint-weight 40 --po4-inside-margin 0.01 --po4-temperature 0.02 > "$run/run.log" 2>&1; then
      mv "$run/cifs/sample_000.cif" "$root/${label}_sample_${good}.cif"
      cp "$run/results.json" "$root/${label}_sample_${good}.json"
      good=$((good + 1))
    fi
    attempt=$((attempt + 1))
  done
  printf '%s %s %s\n' "$label" "$good" "$attempt" > "$root/${label}_status.txt"
  test "$good" -eq 3
}

worker 0 20260908 gpu0 & first=$!
worker 1 20261908 gpu1 & second=$!
wait "$first"; wait "$second"

EXPERIMENT_DIR="$root" "$python_bin" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['EXPERIMENT_DIR'])
rows = [json.loads(p.read_text())['records'][0] for p in sorted(root.glob('gpu*_sample_*.json'))]
root.joinpath('summary.json').write_text(json.dumps({
    'count': len(rows), 'final_po4_passes': sum(x['po4_tetrahedron_checks']['passes'] for x in rows),
    'mean_surrogate_e_hull': sum(x['optimized_e_hull_surrogate'] for x in rows) / len(rows) if rows else None,
    'records': rows}, indent=2))
PY
