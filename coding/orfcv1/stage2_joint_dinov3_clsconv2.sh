#!/bin/bash
# Stage-2: joint conv2_clsconv2 spatial (from stage-1) + ORFC (R + PQ) for DINOv3.
# Mirrors stage2_joint_clsconv2.sh exactly; only backbone-related args differ:
#   --backbone dinov3_vitl16  --n_prefix 5  --norm_mode split_reg_cls_patch
# DINOv3: T=201 (1 CLS + 4 reg + 196 patch), patch=16, RoPE tail, coded=54.
# 4 blocks x 6 K = 24 configs. Each block one round, 6 K parallel on GPU 0-5.

set -e

WORK=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1
LOG=$WORK/logs/stage2_joint_clsconv2_dinov3
STAGE1=$WORK/results/bilinear_residual/dinov3_vitl16

source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate featcodec2

mkdir -p "$LOG"

BLOCKS=(blk05 blk10 blk15 blk20)
KS=(4 8 16 64 256 512)

for blk in "${BLOCKS[@]}"; do
    echo ""
    echo "============================================================"
    echo "[$(date)] Round: $blk — launching 6 K values on GPU 0-5"
    echo "============================================================"

    sp_ckpt="$STAGE1/${blk}_bili65_clsconv2_K4_c4_both_wa_conv_pre_noq_bv_ablation_main_lr0.0003_ep30_n5000_nval0_s42.pt"
    if [ ! -f "$sp_ckpt" ]; then
        echo "  [SKIP] missing stage-1: $sp_ckpt"
        continue
    fi

    for i in "${!KS[@]}"; do
        k=${KS[$i]}
        logfile="$LOG/${blk}_K${k}.log"
        CUDA_VISIBLE_DEVICES=$i python -u "$WORK/run_bilinear_orfc_joint.py" \
            --layer "$blk" \
            --K "$k" \
            --embedding_dim 32 \
            --bottleneck_dim 1024 \
            --backbone dinov3_vitl16 \
            --n_prefix 5 \
            --norm_mode split_reg_cls_patch \
            --residual_ablation main \
            --cls_mode conv2 \
            --residual_ckpt "$sp_ckpt" \
            --epochs 100 \
            --lr 3e-4 \
            --spatial_lr_scale 0.1 \
            --lmbda 0.5 \
            --tau_start 2.0 \
            --tau_end 2.0 \
            --tau_schedule constant \
            --batch_size 32 \
            --max_train_images 5000 \
            --n_val 200 \
            --seed 42 \
            --skip_baselines \
            --skip_test_acc \
            > "$logfile" 2>&1 &
        echo "  GPU $i: $blk K=$k  (pid=$!)"
    done

    echo "[$(date)] Waiting for $blk to finish..."
    wait
    echo "[$(date)] $blk done."
done

echo ""
echo "[$(date)] All 4 rounds (24 configs) completed."
