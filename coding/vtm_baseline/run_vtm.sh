#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 公共参数
FEAT_ROOT=$PROJECT_ROOT/features/test
BIT_DEPTH=10
VTM_ENCODER=$SCRIPT_DIR/EncoderAppStatic
VTM_DECODER=$SCRIPT_DIR/DecoderAppStatic
VTM_CFG=$SCRIPT_DIR/encoder_intra_vtm.cfg
TMP_DIR=$SCRIPT_DIR/_vtm_tmp
WORKERS=16  # 并行worker数量，根据CPU核心数调整

# python vtm_baseline.py \
#   --feat_root $FEAT_ROOT \
#   --models clip_vitl14 \
#   --layers blk05 \
#   --qps 20 25 \
#   --bit_depth $BIT_DEPTH \
#   --vtm_encoder $VTM_ENCODER \
#   --vtm_decoder $VTM_DECODER \
#   --vtm_cfg $VTM_CFG \
#   --tmp_dir $TMP_DIR \
#   --workers $WORKERS

python vtm_baseline.py \
  --feat_root $FEAT_ROOT \
  --models clip_vitl14 \
  --layers blk10 \
  --qps 12 17 \
  --bit_depth $BIT_DEPTH \
  --vtm_encoder $VTM_ENCODER \
  --vtm_decoder $VTM_DECODER \
  --vtm_cfg $VTM_CFG \
  --tmp_dir $TMP_DIR \
  --workers $WORKERS

python vtm_baseline.py \
  --feat_root $FEAT_ROOT \
  --models clip_vitl14 \
  --layers blk15 \
  --qps 0 2 5 7 10 12 \
  --bit_depth $BIT_DEPTH \
  --vtm_encoder $VTM_ENCODER \
  --vtm_decoder $VTM_DECODER \
  --vtm_cfg $VTM_CFG \
  --tmp_dir $TMP_DIR \
  --workers $WORKERS

python vtm_baseline.py \
  --feat_root $FEAT_ROOT \
  --models clip_vitl14 \
  --layers blk20 \
  --qps 17 22 \
  --bit_depth $BIT_DEPTH \
  --vtm_encoder $VTM_ENCODER \
  --vtm_decoder $VTM_DECODER \
  --vtm_cfg $VTM_CFG \
  --tmp_dir $TMP_DIR \
  --workers $WORKERS

# python vtm_baseline.py \
#   --feat_root $FEAT_ROOT \
#   --models clip_vitl14 \
#   --layers blk23 \
#   --qps 10 \
#   --bit_depth $BIT_DEPTH \
#   --vtm_encoder $VTM_ENCODER \
#   --vtm_decoder $VTM_DECODER \
#   --vtm_cfg $VTM_CFG \
#   --tmp_dir $TMP_DIR \
#   --workers $WORKERS

echo "All experiments completed!"
