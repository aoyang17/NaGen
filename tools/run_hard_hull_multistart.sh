#!/usr/bin/env bash
set -euo pipefail

task_python=/mnt/data2/aobo/NaGen/envs/nagen-shooting313/bin/python
task_pythonpath=/home/aobo/NaGen/src:/home/aobo/NaGen/.pylibs
task_root=/mnt/data2/aobo/NaGen/experiments/exp_2026-09-01_joint_dflow_v2/hard_hull_multistart_4x240_v2
task_joint=/mnt/data2/aobo/NaGen/experiments/exp_2026-08-26/checkpoints/flow_joint.best.pt
task_geometry=/mnt/data2/aobo/NaGen/experiments/exp_2026-08-26/checkpoints/flow_geometry.best.pt
task_phase=/mnt/data2/aobo/NaGen/experiments/exp_2026-08-26/cache/mp_phase_entries.json
task_novelty=/mnt/data2/aobo/NaGen/experiments/exp_2026-08-26/data/novelty_reference.pt
task_uma=/mnt/data2/aobo/NaGen/models/uma/uma-m-1p1.pt
task_gpus=(1 2 4 5)
task_pids=()

mkdir -p "$task_root"
for task_shard in 0 1 2 3; do
  task_gpu=${task_gpus[$task_shard]}
  task_seed=$((20260911 + task_shard))
  task_output=$(printf "%s/shard_%02d" "$task_root" "$task_shard")
  task_log=$(printf "%s/shard_%02d.log" "$task_root" "$task_shard")
  env \
    PYTHONPATH="$task_pythonpath" \
    CUDA_VISIBLE_DEVICES="$task_gpu" \
    MPLCONFIGDIR="/tmp/mpl-nagen-hard-hull-$task_shard" \
    "$task_python" -m nagen.inverse.unconditional_dflow_campaign \
      --joint-checkpoint "$task_joint" \
      --geometry-checkpoint "$task_geometry" \
      --phase-cache "$task_phase" \
      --novelty-index "$task_novelty" \
      --uma-checkpoint "$task_uma" \
      --out-directory "$task_output" \
      --count 1 \
      --seed "$task_seed" \
      --ode-steps 24 \
      --optimization-steps 240 \
      --hull-threshold 0.150 \
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
exit "$task_status"
