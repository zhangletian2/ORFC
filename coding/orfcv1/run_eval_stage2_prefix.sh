#!/bin/bash
WORK=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1
LOG=$WORK/logs/eval_stage2_prefix
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate featcodec2
mkdir -p "$LOG"
cd "$WORK"
for blk in blk05 blk10 blk15 blk20; do
    i=0
    for k in 4 8 16 64 256 512; do
        CUDA_VISIBLE_DEVICES=$i python -u eval_stage2_prefix.py $blk $k \
            > "$LOG/${blk}_K${k}.log" 2>&1 &
        i=$((i+1))
    done
    wait
    echo "$blk done"
done
echo "ALL DONE"
