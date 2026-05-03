#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# VTM baseline for segmentation features (voc2012_500, trimmed from 1449)
# 特征形状: (2, 1370, 1024) -> concat 编码为 (1370, 2048)

# 公共参数
FEAT_ROOT=$PROJECT_ROOT/features/voc2012_500
BIT_DEPTH=10
VTM_ENCODER=$SCRIPT_DIR/EncoderAppStatic
VTM_DECODER=$SCRIPT_DIR/DecoderAppStatic
VTM_CFG=$SCRIPT_DIR/encoder_intra_vtm.cfg
TMP_DIR=$SCRIPT_DIR/_vtm_tmp_seg
WORKERS=20

# dinov2_vitl14 分割特征实验
# blk05, blk10, blk15, blk20 各用多个QP
python vtm_baseline_seg.py \
  --feat_root $FEAT_ROOT \
  --models dinov2_vitl14 \
  --layers blk05 \
  --qps 25 27 30 32 35 \
  --bit_depth $BIT_DEPTH \
  --vtm_encoder $VTM_ENCODER \
  --vtm_decoder $VTM_DECODER \
  --vtm_cfg $VTM_CFG \
  --tmp_dir $TMP_DIR \
  --workers $WORKERS

python vtm_baseline_seg.py \
  --feat_root $FEAT_ROOT \
  --models dinov2_vitl14 \
  --layers blk10 \
  --qps 25 27 30 32 35 \
  --bit_depth $BIT_DEPTH \
  --vtm_encoder $VTM_ENCODER \
  --vtm_decoder $VTM_DECODER \
  --vtm_cfg $VTM_CFG \
  --tmp_dir $TMP_DIR \
  --workers $WORKERS

python vtm_baseline_seg.py \
  --feat_root $FEAT_ROOT \
  --models dinov2_vitl14 \
  --layers blk15 \
  --qps 25 27 30 32 35 \
  --bit_depth $BIT_DEPTH \
  --vtm_encoder $VTM_ENCODER \
  --vtm_decoder $VTM_DECODER \
  --vtm_cfg $VTM_CFG \
  --tmp_dir $TMP_DIR \
  --workers $WORKERS

python vtm_baseline_seg.py \
  --feat_root $FEAT_ROOT \
  --models dinov2_vitl14 \
  --layers blk20 \
  --qps 0 2 5 7 10 22 \
  --bit_depth $BIT_DEPTH \
  --vtm_encoder $VTM_ENCODER \
  --vtm_decoder $VTM_DECODER \
  --vtm_cfg $VTM_CFG \
  --tmp_dir $TMP_DIR \
  --workers $WORKERS

echo "All segmentation VTM experiments completed!"
