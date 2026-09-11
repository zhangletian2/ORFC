#!/bin/bash
# DINOv3 stage-1 (clsconv2) eval: ADE20K semseg (fixed 100) + NYU-80 depth
cd /data4/workspace/zlt/featcodec/ORFC/coding/orfc/eval
PY=/data4/workspace/zlt/featcodec/CoFAI/.venv/bin/python
S1=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1/results/bilinear_residual/dinov3_vitl16
SPLIT=/data4/workspace/zlt/featcodec/ORFC/utils/nyu_test_80.txt
SUF=_bili65_clsconv2_K4_c4_both_wa_conv_pre_noq_bv_ablation_main_lr0.0003_ep30_n5000_nval0_s42.pt
COMMON="--tasks semseg,depth --seg_max_images 100 --nyu_split_file $SPLIT --force"
mkdir -p logs_stage1_dinov3

gpu=0
for blk in blk05 blk10 blk15 blk20; do
  (
    CUDA_VISIBLE_DEVICES=$gpu $PY eval_dinov3_tasks.py $COMMON --layer $blk --bypass \
      > logs_stage1_dinov3/${blk}_bypass.log 2>&1
    CUDA_VISIBLE_DEVICES=$gpu $PY eval_dinov3_tasks.py $COMMON --layer $blk \
      --stage1_ckpt $S1/${blk}${SUF} > logs_stage1_dinov3/${blk}_stage1.log 2>&1
  ) &
  gpu=$((gpu+1))
done
wait
echo "ALL DONE"
