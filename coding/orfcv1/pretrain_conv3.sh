#!/bin/bash
# Stage-1 pretrain: Conv3x3(D,D,3,2,1) spatial codec, no ORFC/PQ
# 4 blocks parallel on GPU 4/5/6/7

set -e

WORK=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1
LOG=$WORK/logs/pretrain_conv3

source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate featcodec2

mkdir -p "$LOG"

BLOCKS=(blk05 blk10 blk15 blk20)
GPUS=(4 5 6 7)

echo "[($(date)] Starting conv3 pretrain: 4 blocks parallel on GPU 4-7"

for i in "${!BLOCKS[@]}"; do
    blk=${BLOCKS[$i]}
    gpu=${GPUS[$i]}
    logfile="$LOG/${blk}_main.log"

    CUDA_VISIBLE_DEVICES=$gpu python -u "$WORK/run_bilinear_residual.py" \
        --stage residual \
        --residual_mode both \
        --layer "$blk" \
        --spatial_down conv3 --spatial_up conv3 \
        --residual_decoder conv \
        --no-residual_orfc --no-residual_quantize \
        --residual_ablation main \
        --residual_epochs 30 \
        --residual_lr 0.0003 \
        --n_val 0 \
        --max_train_images 5000 \
        --seed 42 \
        --batch_size 32 \
        --grad_clip 1.0 \
        > "$logfile" 2>&1 &

    echo "  GPU $gpu: $blk  (pid=$!)"
done

echo "[$(date)] Waiting for all 4 blocks..."
wait
echo "[$(date)] All conv3 pretrain completed."
