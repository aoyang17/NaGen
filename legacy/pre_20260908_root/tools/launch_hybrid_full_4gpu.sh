#!/usr/bin/env bash
set -euo pipefail

repo_dir=/home/aobo/NaGen
python_bin=/mnt/data2/aobo/envs/NaGen/bin/python
experiment_dir="$repo_dir/outputs/hybrid_full_20260907"
flow_checkpoint=/mnt/data2/aobo/NaGen/inverse_v1/checkpoints/naxl_unconditional_surrogate_ready_v2.best.pt
surrogate_checkpoint=/mnt/data2/aobo/NaGen/surrogate_ehull/runs/final_mace_v1/best_model.pt
packed_data=/mnt/data2/aobo/NaGen/inverse_v1/data/naxl_packed_v1.pt
uma_checkpoint=/mnt/data2/aobo/NaGen/models/uma/uma-m-1p1.pt
seeds=(20260904 20261004 20261104 20261204)
pids=()

mkdir -p "$experiment_dir"
cd "$repo_dir"

for gpu in 0 1 2 3; do
  PYTHONPATH="$repo_dir/src:$repo_dir/.pylibs" \
  CUDA_VISIBLE_DEVICES="$gpu" \
  "$python_bin" tools/run_surrogate_shootingflow.py \
    --flow "$flow_checkpoint" \
    --surrogate "$surrogate_checkpoint" \
    --packed-data "$packed_data" \
    --novelty-reference-count 0 \
    --uma-checkpoint "$uma_checkpoint" \
    --count 32 \
    --batch-size 1 \
    --steps 140 \
    --geometry-steps 0 \
    --ode-steps 24 \
    --composition-pool-factor 512 \
    --composition-max-rounds 120 \
    --composition-proposal-batch-limit 512 \
    --seed "${seeds[$gpu]}" \
    --device cuda \
    --out "$experiment_dir/shard_gpu${gpu}.json" \
    > "$experiment_dir/gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

printf 'PIDS %s\n' "${pids[*]}"
wait "${pids[@]}"
