#!/usr/bin/env bash
set -euo pipefail

task_python=/mnt/data2/aobo/NaGen/envs/nagen-shooting313/bin/python
task_pythonpath=/home/aobo/NaGen/src:/home/aobo/NaGen/.pylibs
task_root=/mnt/data2/aobo/NaGen/experiments/exp_2026-09-02_hard_hull_search/search_16x1000_mpgrad
task_joint=/mnt/data2/aobo/NaGen/experiments/exp_2026-08-26/checkpoints/flow_joint.best.pt
task_geometry=/mnt/data2/aobo/NaGen/experiments/exp_2026-08-26/checkpoints/flow_geometry.best.pt
task_phase=/mnt/data2/aobo/NaGen/experiments/exp_2026-08-26/cache/mp_phase_entries.json
task_novelty=/mnt/data2/aobo/NaGen/experiments/exp_2026-08-26/data/novelty_reference.pt
task_uma=/mnt/data2/aobo/NaGen/models/uma/uma-m-1p1.pt
task_gpus=(0 1 2 4)
task_pids=()

mkdir -p "$task_root/winners"
for task_shard in 0 1 2 3; do
  task_gpu=${task_gpus[$task_shard]}
  task_seed=$((20260920 + task_shard))
  task_output=$(printf "%s/ipopt_shard_%02d" "$task_root" "$task_shard")
  task_log=$(printf "%s/ipopt_shard_%02d.log" "$task_root" "$task_shard")
  env \
    PYTHONPATH="$task_pythonpath" \
    CUDA_VISIBLE_DEVICES="$task_gpu" \
    MPLCONFIGDIR="/tmp/mpl-nagen-hull-search-$task_shard" \
    "$task_python" -m nagen.inverse.unconditional_dflow_campaign \
      --joint-checkpoint "$task_joint" \
      --geometry-checkpoint "$task_geometry" \
      --phase-cache "$task_phase" \
      --novelty-index "$task_novelty" \
      --uma-checkpoint "$task_uma" \
      --out-directory "$task_output" \
      --count 4 \
      --seed "$task_seed" \
      --ode-steps 24 \
      --optimization-steps 1000 \
      --hull-threshold 0.150 \
      --hull-scale 10.0 \
      --hull-constraint-tolerance 1e-6 \
      --distance-weight 80 \
      --p-coordination-weight 24 \
      --fe-coordination-weight 4 \
      --device cuda \
      --resume >"$task_log" 2>&1 &
  task_pids+=("$!")
done

task_status=0
for task_pid in "${task_pids[@]}"; do
  if ! wait "$task_pid"; then
    task_status=1
  fi
done

for task_shard in 0 1 2 3; do
  task_gpu=${task_gpus[$task_shard]}
  task_source=$(printf "%s/ipopt_shard_%02d" "$task_root" "$task_shard")
  task_final=$(printf "%s/final_shard_%02d" "$task_root" "$task_shard")
  task_final_log=$(printf "%s/final_shard_%02d.log" "$task_root" "$task_shard")
  task_passes=$(find "$task_source/records" -name 'sample_*.json' -print0 \
    | xargs -0 -r jq -r 'select(.status == "complete") | .sample_id' \
    | wc -l)
  if [[ "$task_passes" -eq 0 ]]; then
    continue
  fi
  env \
    PYTHONPATH="$task_pythonpath" \
    CUDA_VISIBLE_DEVICES="$task_gpu" \
    MPLCONFIGDIR="/tmp/mpl-nagen-final-gate-$task_shard" \
    "$task_python" -m nagen.inverse.joint_dflow_relax_campaign \
      --source-campaign "$task_source" \
      --uma-checkpoint "$task_uma" \
      --out-directory "$task_final" \
      --fire-fmax 0.05 \
      --fire-steps 1500 \
      --bfgs-fmax 0.01 \
      --bfgs-steps 1500 \
      --energy-increase-tolerance 1e-6 \
      --scratch-directory /tmp \
      --device cuda \
      --resume >"$task_final_log" 2>&1
  for task_record in "$task_final"/records/sample_*.json; do
    if jq -e '.accepted == true' "$task_record" >/dev/null; then
      cp "$task_record" "$task_root/winners/shard_${task_shard}_$(basename "$task_record")"
    fi
  done
done

exit "$task_status"
