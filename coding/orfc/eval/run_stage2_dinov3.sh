#!/bin/bash
# DINOv3 stage-2 (joint clsconv2 + ORFC PQ) downstream eval:
#   ADE20K semseg (fixed first 100) + NYU-80 depth
# One block per round, the 6 K values in parallel on GPU 0-5.
cd /data4/workspace/zlt/featcodec/ORFC/coding/orfc/eval
PY=/data4/workspace/zlt/featcodec/CoFAI/.venv/bin/python
CK=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1/checkpoints/dinov3_vitl16
SPLIT=/data4/workspace/zlt/featcodec/ORFC/utils/nyu_test_80.txt
SUF=_emb32_bt1024_ws_lmbda0.5_tau2.0_te2.0_tscon_hlr0.1_bv_lr0.0003_ep100_n5000_nval200_s42
COMMON="--tasks semseg,depth --seg_max_images 100 --nyu_split_file $SPLIT --force"
LOG=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1/logs/stage2_tasks_dinov3
mkdir -p $LOG

for blk in blk05 blk10 blk15 blk20; do
  gpu=0
  for K in 4 8 16 64 256 512; do
    STEM=${blk}_conv2_main_clsconv2_jointopq_K${K}${SUF}
    CUDA_VISIBLE_DEVICES=$gpu $PY eval_dinov3_tasks.py $COMMON --layer $blk \
      --stage1_ckpt $CK/${STEM}_spatial.pt --orfc_ckpt $CK/${STEM}.pt \
      > $LOG/${blk}_K${K}.log 2>&1 &
    gpu=$((gpu+1))
  done
  wait
  echo "=== $blk done ($(date +%H:%M:%S))"
done
echo "ALL DONE"
