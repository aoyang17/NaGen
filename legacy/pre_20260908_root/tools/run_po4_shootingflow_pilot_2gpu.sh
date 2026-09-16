#!/usr/bin/env bash
set -euo pipefail

repo_dir=/home/aobo/NaGen
python_bin=/mnt/data2/aobo/envs/NaGen/bin/python
flow_checkpoint=/mnt/data2/aobo/NaGen/inverse_v1/checkpoints/naxl_unconditional_surrogate_ready_v2.best.pt
surrogate_checkpoint=/mnt/data2/aobo/NaGen/surrogate_ehull/runs/final_mace_v1/best_model.pt
packed_data=/mnt/data2/aobo/NaGen/inverse_v1/data/naxl_packed_v1.pt
experiment_dir="$repo_dir/outputs/shootingflow_po4_pilot_20260908"

mkdir -p "$experiment_dir"/{baseline,po4}
cd "$repo_dir"

run_group() {
  local gpu="$1"
  local group="$2"
  local po4_weight="$3"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$repo_dir/src:$repo_dir/.pylibs" \
    "$python_bin" tools/run_surrogate_shootingflow.py \
      --flow "$flow_checkpoint" \
      --surrogate "$surrogate_checkpoint" \
      --packed-data "$packed_data" \
      --out "$experiment_dir/$group/results.json" \
      --cif-dir "$experiment_dir/$group/cifs" \
      --count 6 --batch-size 2 --steps 24 \
      --optimizer hybrid-alm --adam-steps 16 --lbfgs-steps 8 --lbfgs-lr 0.15 \
      --source-candidates 2 --lr 0.03 --ode-steps 16 \
      --graph-refresh-steps 8 --seed 20260908 --device cuda \
      --composition-pool-factor 512 --composition-max-rounds 60 \
      --composition-proposal-batch-limit 512 \
      --poly-center-weight 0 --poly-face-weight 0 --poly-coplanar-weight 0 \
      --po4-constraint-weight "$po4_weight" \
      --po4-inside-margin 0.01 --po4-temperature 0.02 \
      > "$experiment_dir/$group/run.log" 2>&1
}

run_group 0 baseline 0 &
pid_baseline=$!
run_group 1 po4 40 &
pid_po4=$!
wait "$pid_baseline"
wait "$pid_po4"

EXPERIMENT_DIR="$experiment_dir" "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["EXPERIMENT_DIR"])
summary = {}
for group in ("baseline", "po4"):
    records = json.loads((root / group / "results.json").read_text())["records"]
    summary[group] = {
        "count": len(records),
        "po4_passes": sum(r["po4_tetrahedron_checks"]["passes"] for r in records),
        "basic_feasible": sum(r["analytic_constraint_checks"]["feasible"] for r in records),
        "mean_surrogate_e_hull": sum(r["optimized_e_hull_surrogate"] for r in records) / len(records),
        "samples": [{
            "id": r["sample_id"], "formula": r["formula"],
            "po4_passes": r["po4_tetrahedron_checks"]["passes"],
            "e_hull": r["optimized_e_hull_surrogate"],
        } for r in records],
    }
(root / "summary.json").write_text(json.dumps(summary, indent=2))
PY
