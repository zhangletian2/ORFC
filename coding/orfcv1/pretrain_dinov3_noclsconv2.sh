#!/bin/bash
# Stage-1 pretrain for DINOv3 vitl16, prefix-free conv2.
#
# With C == D the codec resolves cls_mode to "none" and prefix_down/up are
# None, so the 5 prefix tokens (1 CLS + 4 reg) bypass analysis/synthesis
# bit-exactly and the conv is trained on patch tokens only.  Coded tokens
# are unchanged at 5 + 7x7 = 54; what goes away is the lossy
# expand-to-2x2 -> conv -> synthesis -> crop roundtrip on the prefix.
#
#
# DINOv3: n_prefix=5, 14x14 patches, norm=split_per_reg_cls_patch
# 4 blocks parallel on GPU 0-3

set -e

WORK=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1
LOG=$WORK/logs/pretrain_dinov3_noclsconv2

source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate featcodec2

mkdir -p "$LOG"

BLOCKS=(blk05 blk10 blk15 blk20)
GPUS=(0 1 2 3)

echo "[$(date)] Starting DINOv3 prefix-free pretrain: 4 blocks parallel"

for i in "${!BLOCKS[@]}"; do
    blk=${BLOCKS[$i]}
    gpu=${GPUS[$i]}
    logfile="$LOG/${blk}_main.log"

    CUDA_VISIBLE_DEVICES=$gpu python -u "$WORK/run_bilinear_residual.py" \
        --stage residual \
        --residual_mode both \
        --layer "$blk" \
        --backbone dinov3_vitl16 \
        --n_prefix 5 \
        --norm_mode split_per_reg_cls_patch \
        --spatial_down conv2 --spatial_up conv2 \
        --residual_decoder conv \
        --no-residual_orfc --no-residual_quantize \
        --residual_ablation main \
        --residual_epochs 30 \
        --residual_lr 0.0003 \
        --n_val 0 \
        --max_train_images 5000 \
        --seed 42 \
        --batch_size 32 \
        --grad_clip 1.0 --skip_test_acc \
        > "$logfile" 2>&1 &

    echo "  GPU $gpu: $blk  (pid=$!)"
done

echo "[$(date)] Waiting for all 4 blocks..."
wait
echo "[$(date)] All DINOv3 prefix-free pretrain completed."
