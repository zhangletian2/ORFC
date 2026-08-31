#!/bin/bash
set -euo pipefail
cd /data4/workspace/zlt/featcodec/ORFC/coding/orfcv1
PY=/home/user/anaconda3/envs/featcodec2/bin/python
LOG=results/similarity_merge_pq/dinov2_vitl14/logs
mkdir -p "$LOG"

run_k() {
  local gpu=$1 layer=$2 k=$3
  CUDA_VISIBLE_DEVICES=$gpu $PY -u run_similarity_merge_pq.py \
    --layer "$layer" --K "$k" --match_mode tome --merge_frac 0.25 \
    --epochs 100 --save_codec --skip_ceiling \
    2>&1 | tee -a "$LOG/${layer}_K${k}.log"
}

run_layer() {
  local layer=$1
  run_k 0 "$layer" 4 &
  run_k 1 "$layer" 8 &
  run_k 3 "$layer" 16 &
  wait
}

run_layer blk05
run_layer blk10
run_layer blk15
run_layer blk20
