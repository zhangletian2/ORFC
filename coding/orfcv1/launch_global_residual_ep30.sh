#!/bin/bash
# 4 layers × {joint=65, residual=129} = 8 jobs, one GPU each, 30 epochs.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-/home/user/anaconda3/envs/featcodec2/bin/python}"
export TORCH_HOME="${TORCH_HOME:-/data4/workspace/zlt/featcodec/ORFC/pretrained}"
LOG=logs
mkdir -p "$LOG"

run() {
  local gpu=$1 layer=$2 extra=$3 tag=$4
  echo "GPU $gpu  $layer  $tag"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u run_global_residual.py \
    --layer "$layer" --epochs 30 --lr 3e-4 \
    $extra \
    > "$LOG/${layer}_${tag}_ep30.log" 2>&1 &
}

# joint = 65 tokens (low+detail mixed into 64 + CLS)
run 0 blk05 "--joint" joint65
run 1 blk10 "--joint" joint65
run 2 blk15 "--joint" joint65
run 3 blk20 "--joint" joint65
# residual-side = 129 tokens (64 Haar-low + 64 learned residual + CLS)
run 4 blk05 "" residual129
run 5 blk10 "" residual129
run 6 blk15 "" residual129
run 7 blk20 "" residual129

wait
echo done
