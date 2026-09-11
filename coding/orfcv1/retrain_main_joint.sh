#!/bin/bash
# Retrain conv2 main + ORFC joint: 4 blocks × 6 K values = 24 configs
# Each block is one round, 6 K values run in parallel on GPU 0-5
# Reproduces original: Cayley R, λ=0.5, lr=3e-4, spatial_lr_scale=0.1, τ=2.0, 100 epochs

set -e

WORK=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1
CKPT=$WORK/checkpoints/dinov2_vitl14
BACKUP=$CKPT/backup_absorbed
LOG=$WORK/logs/retrain_main_joint

source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate featcodec2

mkdir -p "$BACKUP" "$LOG"

# ── Step 1: backup absorbed checkpoints ──
echo "[$(date)] Backing up absorbed main checkpoints..."
for f in "$CKPT"/*conv2_main_jointopq*.pt; do
    [ -f "$f" ] && cp -n "$f" "$BACKUP/"
done
echo "[$(date)] Backup done: $(ls "$BACKUP" | wc -l) files"

# ── Step 2: train block by block, 6 K in parallel ──
BLOCKS=(blk05 blk10 blk15 blk20)
KS=(4 8 16 64 256 512)

for blk in "${BLOCKS[@]}"; do
    echo ""
    echo "============================================================"
    echo "[$(date)] Round: $blk — launching 6 K values on GPU 0-5"
    echo "============================================================"

    for i in "${!KS[@]}"; do
        k=${KS[$i]}
        logfile="$LOG/${blk}_K${k}.log"
        CUDA_VISIBLE_DEVICES=$i python -u "$WORK/run_bilinear_orfc_joint.py" \
            --layer "$blk" \
            --K "$k" \
            --embedding_dim 32 \
            --bottleneck_dim 1024 \
            --residual_ablation main \
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
