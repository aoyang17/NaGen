#!/usr/bin/env bash
set -euo pipefail
repo=/home/aobo/NaGen
python=/mnt/data2/aobo/envs/NaGen/bin/python
out="$repo/outputs/simple_optimization_ehull_distance_filter_20260907"
flow=/mnt/data2/aobo/NaGen/inverse_v1/checkpoints/naxl_unconditional_surrogate_ready_v2.best.pt
surrogate=/mnt/data2/aobo/NaGen/surrogate_ehull/runs/final_mace_v1/best_model.pt
packed=/mnt/data2/aobo/NaGen/inverse_v1/data/naxl_packed_v1.pt
seeds=(20260907 20261007 20261107 20261207)
pids=()
mkdir -p "$out"
cd "$repo"
for gpu in 0 1 2 3; do
  env PYTHONPATH="$repo/src:$repo/.pylibs:$repo/tools" CUDA_VISIBLE_DEVICES="$gpu" \
    timeout --signal=TERM 20m "$python" tools/run_simple_optimization.py \
      --flow "$flow" --surrogate "$surrogate" --packed-data "$packed" \
      --count 4 --steps 100 --ode-steps 24 --seed "${seeds[$gpu]}" \
      --min-e-hull 0.0 --max-e-hull 0.001 --epsilon-e 0.001 --tau-e 0.005 \
      --constraint-weight 100 --learning-rate 0.015 \
      --device cuda --out "$out/shard${gpu}.json" \
      > "$out/shard${gpu}.log" 2>&1 &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "$out/pids.txt"
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
