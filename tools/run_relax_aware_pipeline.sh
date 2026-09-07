#!/usr/bin/env bash
set -euo pipefail

task_python=/mnt/data2/aobo/NaGen/envs/nagen-shooting313/bin/python
task_root=/mnt/data2/aobo/NaGen/experiments/exp_2026-09-02_relax_surrogate
task_labels="$task_root/labels_200"
task_models="$task_root/models"
task_multistart="$task_root/multistart_16"
task_final="$task_root/final_relaxed_top4"
task_pythonpath=/home/aobo/NaGen/src:/home/aobo/NaGen/.pylibs

status_of() {
  local manifest=$1
  if [[ ! -f "$manifest" ]]; then
    echo missing
    return
  fi
  jq -r '.status // "missing"' "$manifest"
}

while true; do
  shard0=$(status_of "$task_labels/shard_000/manifest.json")
  shard1=$(status_of "$task_labels/shard_001/manifest.json")
  echo "$(date -u +%FT%TZ) labels shard0=$shard0 shard1=$shard1"
  if [[ "$shard0" == complete && "$shard1" == complete ]]; then
    break
  fi
  if [[ "$shard0" == incomplete || "$shard1" == incomplete ]]; then
    echo "label campaign incomplete; refusing to train" >&2
    exit 1
  fi
  sleep 60
done

mkdir -p "$task_models"
env PYTHONPATH="$task_pythonpath" CUDA_VISIBLE_DEVICES=0 \
  MPLCONFIGDIR=/tmp/matplotlib-relax-surrogate-final \
  "$task_python" -m nagen.inverse.relaxation_surrogate \
  --label-campaign \
    "$task_labels" \
    /mnt/data2/aobo/NaGen/experiments/exp_2026-09-01_joint_dflow_v2/relaxed_ipopt_probe_2x240 \
    /mnt/data2/aobo/NaGen/experiments/exp_2026-09-01_joint_dflow_v2/relaxed_probe_2x240 \
    /mnt/data2/aobo/NaGen/experiments/exp_2026-09-01_joint_dflow_v2/relaxed_smoke_4x80 \
  --out "$task_models/relax_surrogate_ensemble5.pt" \
  --minimum-pairs 100 --ensemble-size 5 --hidden 256 \
  --epochs 1000 --patience 80 --batch-size 16 --device cuda

env PYTHONPATH="$task_pythonpath" CUDA_VISIBLE_DEVICES=0 \
  MPLCONFIGDIR=/tmp/matplotlib-relax-aware-multistart \
  "$task_python" -m nagen.inverse.relax_aware_multistart \
  --joint-checkpoint /mnt/data2/aobo/NaGen/experiments/exp_2026-08-26/checkpoints/flow_joint.best.pt \
  --relaxation-surrogate "$task_models/relax_surrogate_ensemble5.pt" \
  --phase-cache /mnt/data2/aobo/NaGen/experiments/exp_2026-08-26/cache/mp_phase_entries.json \
  --novelty-index /mnt/data2/aobo/NaGen/experiments/exp_2026-08-26/data/novelty_reference.pt \
  --out-directory "$task_multistart" \
  --starts 16 --final-relax-count 4 --ode-steps 24 \
  --optimization-steps 160 --device cuda --resume

env PYTHONPATH="$task_pythonpath" CUDA_VISIBLE_DEVICES=0 \
  MPLCONFIGDIR=/tmp/matplotlib-relax-aware-final \
  "$task_python" -m nagen.inverse.joint_dflow_relax_campaign \
  --source-campaign "$task_multistart/selected_for_relaxation" \
  --uma-checkpoint /mnt/data2/aobo/NaGen/models/uma/uma-m-1p1.pt \
  --out-directory "$task_final" --scratch-directory /tmp \
  --device cuda --resume

echo "$(date -u +%FT%TZ) relax-aware pipeline complete"
