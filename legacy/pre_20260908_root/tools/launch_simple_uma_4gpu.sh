#!/usr/bin/env bash
set -euo pipefail

PY=/mnt/data2/aobo/envs/NaGen/bin/python
FLOW=/mnt/data2/aobo/NaGen/inverse_v1/checkpoints/naxl_unconditional_surrogate_ready_v2.best.pt
UMA=/mnt/data2/aobo/NaGen/models/uma/uma-m-1p1.pt
DATA=/mnt/data2/aobo/NaGen/inverse_v1/data/naxl_packed_v1.pt
OUT=/home/aobo/NaGen/results/simple_uma_4gpu

mkdir -p "$OUT"
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONPATH=/home/aobo/NaGen/src:/home/aobo/NaGen/.pylibs \
  LD_LIBRARY_PATH=/mnt/data2/aobo/envs/NaGen/lib:/usr/local/cuda/lib64 \
  "$PY" /home/aobo/NaGen/tools/run_simple_uma_optimization.py \
    --flow "$FLOW" \
    --uma-checkpoint "$UMA" \
    --packed-data "$DATA" \
    --out "$OUT/gpu_${gpu}.json" \
    --seed "$((20260907 + gpu))" \
    --count 1 \
    --ode-steps 24 \
    --steps 16 \
    >"$OUT/gpu_${gpu}.log" 2>&1 &
done
wait
