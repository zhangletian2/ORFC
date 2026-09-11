#!/bin/bash
# DINOv3 R-absorbed downstream eval (ADE20K semseg first-100 + NYU-80 depth),
# for the prefix-bypass (cls_mode=none) stage-2 models.
#
# Runs two rounds per block so the absorbed numbers sit next to their reference:
#   stage2    — spatial -> ORFC(R + PQ) -> residual, R still explicit
#   absorbed  — R folded into the conv for patches, kept dense for the 5 prefix
#               tokens; produced by orfcv1/absorb_r.py, originals untouched
# eval_dinov3_tasks.py picks the absorbed path up automatically from the
# R_dense entry in the checkpoint, so both rounds use the same command shape.
#
# Usage:  bash run_absorbed_dinov3.sh [blk05 blk10 ...]   (default: blk05)
cd /data4/workspace/zlt/featcodec/ORFC/coding/orfc/eval
PY=/data4/workspace/zlt/featcodec/CoFAI/.venv/bin/python
CK=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1/checkpoints/dinov3_vitl16
SPLIT=/data4/workspace/zlt/featcodec/ORFC/utils/nyu_test_80.txt
SUF=_emb32_bt1024_ws_lmbda0.5_tau2.0_te2.0_tscon_hlr0.1_bv_lr0.0003_ep100_n5000_nval200_s42
COMMON="--tasks semseg,depth --seg_max_images 100 --nyu_split_file $SPLIT --force"
LOG=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1/logs/absorbed_tasks_dinov3
mkdir -p $LOG

BLOCKS=("$@")
[ ${#BLOCKS[@]} -eq 0 ] && BLOCKS=(blk05)

for blk in "${BLOCKS[@]}"; do
  for variant in stage2 absorbed; do
    [ "$variant" = absorbed ] && TAG=_absorbed || TAG=""
    gpu=0
    for K in 4 8 16 64 256 512; do
      STEM=${blk}_conv2_main_jointopq_K${K}${SUF}
      CUDA_VISIBLE_DEVICES=$gpu $PY eval_dinov3_tasks.py $COMMON --layer $blk \
        --stage1_ckpt $CK/${STEM}${TAG}_spatial.pt \
        --orfc_ckpt   $CK/${STEM}${TAG}.pt \
        > $LOG/${blk}_${variant}_K${K}.log 2>&1 &
      gpu=$((gpu+1))
    done
    wait
    echo "=== $blk $variant done ($(date +%H:%M:%S))"
  done
done
echo "ALL DONE"
