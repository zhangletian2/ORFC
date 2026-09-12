#!/bin/bash
# Stage-2 joint (spatial from stage-1 + ORFC R/PQ) for DINOv3, prefix-free variant.
#
# The stage-1 spatial
# codec was trained WITHOUT --cls_mode conv2.  With C == D the codec resolves
# cls_mode to "none", so the 5 prefix tokens (CLS + 4 reg) skip the conv entirely
# and are concatenated straight onto the flattened 7x7 patch latents:
#     seq = cat([prefix, flatten_map(z)])        -> 5 + 49 = 54 coded tokens
# They then share R and the PQ codebook with the conv2-compressed patches, which
# is exactly the design we want: prefix absent in stage 1, jointly quantised in
# stage 2.
#
# --cls_mode is NOT passed: the actual construction comes from the checkpoint
# meta (cls_mode="none"), and the flag only feeds the output stem.  Leaving it at
#
# 4 blocks x 6 K = 24 configs. Each block one round, 6 K parallel on GPU 0-5.
#
# Re-run after four fixes.  Every stage-2 checkpoint predating them was deleted:
#   1. opq.batch_normalize_gpu gives each register token its own mu/sigma.
#      Pooling all four let reg2 (norm ~1.7e5) crush reg1/3/4 to ~0.013.
#   2. The frozen tail loads through timm checkpoint_filter_fn, so the 48
#      LayerScale gammas hold the real ~0.057 instead of the 1e-5 init.
#   3. Stage-1 meta now records n_prefix/norm_mode and load_stage1_spatial
#      verifies them: a dropped --n_prefix 5 raises instead of silently
#      treating the 4 register tokens as patches.
#   4. Rate: n_patch = T - n_prefix, and side-info counts 5 mu/sigma groups
#      (160 bits) + the patch grid (16 bits) rather than a flat 32.

set -e

WORK=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1
LOG=$WORK/logs/stage2_joint_noclsconv2_dinov3
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

    sp_ckpt="$STAGE1/${blk}_bili65_K4_c4_both_wa_conv_pre_noq_bv_ablation_main_lr0.0003_ep30_n5000_nval0_s42.pt"
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
            --norm_mode split_per_reg_cls_patch \
            --residual_ablation main \
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
