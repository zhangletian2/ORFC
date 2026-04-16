#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 统计 dinov2_vitl14 实验结果
# blk05: QP 25, 30, 35
# blk11: QP 29, 34
# blk17: QP 10, 15
# blk23: QP 25, 30, 40

python vtm_stat.py \
  --feat_root "$PROJECT_ROOT/features/test/" \
  --models clip_vitl14 \
  --layer_qps "blk10:12,17 blk15:0,2,5,7,10,12 blk20:17,22"
