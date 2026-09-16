#!/usr/bin/env bash
set -euo pipefail

repo_dir=/home/aobo/NaGen
python_bin=/mnt/data2/aobo/envs/NaGen/bin/python
experiment_dir="$repo_dir/outputs/lbfgs_vs_casadi_ipopt_20260907"
flow_checkpoint=/mnt/data2/aobo/NaGen/inverse_v1/checkpoints/naxl_unconditional_surrogate_ready_v2.best.pt
surrogate_checkpoint=/mnt/data2/aobo/NaGen/surrogate_ehull/runs/final_mace_v1/best_model.pt
packed_data=/mnt/data2/aobo/NaGen/inverse_v1/data/naxl_packed_v1.pt
uma_checkpoint=/mnt/data2/aobo/NaGen/models/uma/uma-m-1p1.pt
seeds=(20260904 20261004 20261104 20261204)
pids=()

mkdir -p "$experiment_dir/lbfgs" "$experiment_dir/ipopt"
cd "$repo_dir"

for shard in 0 1 2 3; do
  env PYTHONPATH="$repo_dir/src:$repo_dir/.pylibs:$repo_dir/tools" \
    CUDA_VISIBLE_DEVICES="$shard" \
    timeout --signal=TERM 20m \
    "$python_bin" tools/run_surrogate_shootingflow.py \
      --flow "$flow_checkpoint" --surrogate "$surrogate_checkpoint" \
      --packed-data "$packed_data" --novelty-reference-count 0 \
      --uma-checkpoint "$uma_checkpoint" --count 3 --batch-size 1 \
      --optimizer lbfgs --lbfgs-history-size 10 --source-candidates 8 \
      --lr 0.3 --steps 16 --geometry-steps 0 --ode-steps 24 \
      --time-limit-seconds 1140 --augmented-lagrangian-rho 1.0 \
      --alm-initial-multiplier 1.0 --alm-outer-steps 4 \
      --alm-rho-growth 2.0 --dual-learning-rate 0.5 \
      --composition-pool-factor 512 --composition-max-rounds 60 \
      --composition-proposal-batch-limit 512 \
      --seed "${seeds[$shard]}" --device cuda \
      --out "$experiment_dir/lbfgs/shard${shard}.json" \
      > "$experiment_dir/lbfgs/shard${shard}.log" 2>&1 &
  pids+=("$!")

  ipopt_gpu=$((shard + 4))
  env PYTHONPATH="$repo_dir/src:$repo_dir/.pylibs:$repo_dir/tools" \
    CUDA_VISIBLE_DEVICES="$ipopt_gpu" \
    timeout --signal=TERM 20m \
    "$python_bin" tools/run_casadi_ipopt_shootingflow.py \
      --flow "$flow_checkpoint" --surrogate "$surrogate_checkpoint" \
      --packed-data "$packed_data" --uma-checkpoint "$uma_checkpoint" \
      --count 3 --source-candidates 8 --ode-steps 24 \
      --ipopt-max-iterations 16 --max-wall-time 1140 \
      --seed "${seeds[$shard]}" --device cuda \
      --out "$experiment_dir/ipopt/shard${shard}.json" \
      > "$experiment_dir/ipopt/shard${shard}.log" 2>&1 &
  pids+=("$!")
done

printf '%s\n' "${pids[@]}" > "$experiment_dir/pids.txt"
printf 'Started %s processes; logs: %s\n' "${#pids[@]}" "$experiment_dir"
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
