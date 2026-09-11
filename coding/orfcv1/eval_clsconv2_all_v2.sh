#!/bin/bash
set -e

WORK=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate featcodec2
cd "$WORK"

BLOCKS=(blk05 blk10 blk15 blk20)
KS=(4 8 16 64 256 512)

LOGDIR="$WORK/logs/eval_clsconv2"
mkdir -p "$LOGDIR"

COMMON="--embedding_dim 32 --bottleneck_dim 1024 --lmbda 0.5 --lr 3e-4 --spatial_lr_scale 0.1 --epochs 100 --max_train_images 5000 --n_val 200 --seed 42 --residual_ablation main --cls_mode conv2"

echo "============================================================"
echo "[$(date)] Phase 1: R-absorption (eval_absorbed_v2.py) — all 24"
echo "============================================================"

for blk in "${BLOCKS[@]}"; do
    echo "[$(date)] Absorbed: $blk — 6 K values on GPU 0-5"
    for i in "${!KS[@]}"; do
        k=${KS[$i]}
        logfile="$LOGDIR/absorbed_${blk}_K${k}.log"
        CUDA_VISIBLE_DEVICES=$i python -u eval_absorbed_v2.py \
            --layer "$blk" --K "$k" $COMMON \
            > "$logfile" 2>&1 &
        echo "  GPU $i: $blk K=$k (pid=$!)"
    done
    echo "[$(date)] Waiting for $blk absorbed..."
    wait
    echo "[$(date)] $blk absorbed done."
done

echo ""
echo "============================================================"
echo "[$(date)] Phase 2: Task eval — blk10/15/20 only (blk05 done)"
echo "============================================================"

for blk in blk10 blk15 blk20; do
    echo "[$(date)] Tasks: $blk — 6 K values on GPU 0-5"
    for i in "${!KS[@]}"; do
        k=${KS[$i]}
        logfile="$LOGDIR/tasks_${blk}_K${k}.log"
        CUDA_VISIBLE_DEVICES=$i python -u eval_bilinear_orfc_joint_tasks.py \
            --layer "$blk" --K "$k" $COMMON \
            --skip_orfc \
            > "$logfile" 2>&1 &
        echo "  GPU $i: $blk K=$k (pid=$!)"
    done
    echo "[$(date)] Waiting for $blk tasks..."
    wait
    echo "[$(date)] $blk tasks done."
done

echo ""
echo "[$(date)] All evaluation completed."
