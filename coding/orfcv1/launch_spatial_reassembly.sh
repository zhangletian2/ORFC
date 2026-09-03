#!/bin/bash
# Stage 1: spatial reassembly, no PQ.  Bilinear-init, softmax, scale=2.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-/home/user/anaconda3/envs/featcodec2/bin/python}"
LOG=results/spatial_reassembly/dinov2_vitl14/logs
mkdir -p "$LOG"

run() {
  local gpu=$1 layer=$2
  echo "GPU $gpu  $layer"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u run_spatial_reassembly.py \
    --layer "$layer" --scale 2 --k 5 --Cr 64 \
    --epochs 100 --lr 3e-4 --save_codec \
    > "$LOG/${layer}_s2_k5_lr0.0003_ep100.log" 2>&1 &
}

run 2 blk05
run 4 blk20
wait
echo "done"
