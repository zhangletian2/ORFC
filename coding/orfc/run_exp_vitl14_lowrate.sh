#!/bin/bash
# ================================================================
#  DINOv2 ViT-G/14 low-rate: K=4 and K=8 on 3 layers
#
#  6 experiments on 6 GPUs (one each), ~15 min total
#
#  GPU 0: blk09 K=4   GPU 3: blk09 K=8
#  GPU 1: blk19 K=4   GPU 4: blk19 K=8
#  GPU 2: blk29 K=4   GPU 5: blk29 K=8
# ================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export TORCH_HOME="$PROJECT_ROOT/pretrained"


PYTHON=python
SCRIPT="run_soft_pq.py"
LOG_DIR="logs_vitl14_lowrate"
mkdir -p "$LOG_DIR"

BACKBONE="dinov2_vitl14"

COMMON="--backbone $BACKBONE \
        --bottleneck_dim 1024 --warm_start_opq \
        --embedding_dim 32 --lr 0.0003 --lmbda 0.1 \
        --tau_start 0.5 --tau_end 0.005 \
        --epochs 100 --max_train_images 5000 \
        --batch_size 32 --seed 42"

run_one() {
    local gpu=$1 layer=$2 K=$3 tag=$4
    local logfile="${LOG_DIR}/${tag}.log"
    echo "  [GPU $gpu] RUN   ${tag}  ->  $logfile"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON $SCRIPT \
        --layer "$layer" --K "$K" $COMMON \
        > "$logfile" 2>&1
    local ret=$?
    if [ $ret -eq 0 ]; then
        echo "  [GPU $gpu] DONE  ${tag}"
    else
        echo "  [GPU $gpu] FAIL  ${tag}  (exit $ret)"
    fi
    return $ret
}

echo "================================================================"
echo "  ViT-G/14 low-rate: K={4,8} × {blk09,blk19,blk29}"
echo "  6 experiments on 6 GPUs, ~15 min"
echo "  $(date)"
echo "================================================================"

run_one 0 blk20 8 "blk20_K8" &
run_one 1 blk20 16 "blk20_K16" &
run_one 2 blk20 32 "blk20_K32" &

echo "All 6 jobs launched. Waiting..."
wait

echo ""
echo "================================================================"
echo "  All done at $(date)"
echo "================================================================"
echo ""

for f in "$LOG_DIR"/*.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    if grep -q "Summary:" "$f" 2>/dev/null; then
        acc=$(grep "Codec.*Acc=" "$f" | tail -1 | grep -oP 'Acc=[\d.]+' | head -1)
        miou=$(grep "Codec.*mIoU=" "$f" | tail -1 | grep -oP 'mIoU=[\d.]+' | head -1)
        rate=$(grep "rANS=" "$f" | tail -1 | grep -oP 'rANS=[\d.]+' | head -1)
        echo "  $name: $acc $miou rANS=${rate}bpt"
    elif grep -q "Error\|Traceback" "$f" 2>/dev/null; then
        echo "  $name: FAILED"
    else
        echo "  $name: (running)"
    fi
done
