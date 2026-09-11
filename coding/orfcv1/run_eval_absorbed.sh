#!/bin/bash
# Evaluate absorbed vs original vs ORFC for all 24 main configs
# 6 GPUs parallel per block round

set -e

WORK=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1
LOG=$WORK/logs/eval_absorbed

source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate featcodec2

mkdir -p "$LOG"

BLOCKS=(blk05 blk10 blk15 blk20)
KS=(4 8 16 64 256 512)

for blk in "${BLOCKS[@]}"; do
    echo ""
    echo "============================================================"
    echo "[$(date)] Eval round: $blk"
    echo "============================================================"

    for i in "${!KS[@]}"; do
        k=${KS[$i]}
        logfile="$LOG/${blk}_K${k}.log"
        CUDA_VISIBLE_DEVICES=$i python -u "$WORK/eval_absorbed_v2.py" \
            --layer "$blk" \
            --K "$k" \
            --residual_ablation main \
            --tasks seg,depth \
            > "$logfile" 2>&1 &
        echo "  GPU $i: $blk K=$k  (pid=$!)"
    done

    echo "[$(date)] Waiting for $blk eval to finish..."
    wait
    echo "[$(date)] $blk eval done."
done

echo ""
echo "[$(date)] All evaluation completed."
