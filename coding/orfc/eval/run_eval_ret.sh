#!/bin/bash
# ============================================================
#  Soft-PQ / OPQ COCO Retrieval evaluation
#  Usage:
#    bash run_eval_ret.sh              # both (default)
#    bash run_eval_ret.sh opq          # OPQ only
#    bash run_eval_ret.sh softpq       # Soft-PQ only
# ============================================================
set -uo pipefail

METHOD="${1:-both}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/eval_ret_soft_pq.py"
FEAT_ROOT="/data4/workspace/zlt/featcodec/ORFC/features/coco_ret/siglip2_so400m"
TRAIN_FEAT_ROOT="/data4/workspace/zlt/featcodec/ORFC/features/train/siglip2_so400m"
META_JSON="/data4/workspace/zlt/featcodec/ORFC/tools/coco_ret_features/subset_meta.json"
CKPT_DIR="$SCRIPT_DIR/checkpoints/siglip2_so400m"
OUT_DIR="$SCRIPT_DIR/results/ret_soft_pq"
LOG_DIR="$SCRIPT_DIR/logs/ret_eval"
mkdir -p "$OUT_DIR" "$LOG_DIR"

CONDA_ENV=siglip_codec
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run() {
    local gpu=$1 layer=$2
    echo "[$(date '+%H:%M:%S')] GPU $gpu  START  $layer  method=$METHOD"

    local extra_args=""
    if [ "$METHOD" != "softpq" ]; then
        extra_args="--train_feat_root $TRAIN_FEAT_ROOT"
    fi

    CUDA_VISIBLE_DEVICES=$gpu \
        conda run -n $CONDA_ENV \
        python "$SCRIPT" \
            --layer "$layer" \
            --feat_root "$FEAT_ROOT" \
            $extra_args \
            --meta_json "$META_JSON" \
            --ckpt_dir "$CKPT_DIR" \
            --output "$OUT_DIR/${layer}.json" \
            --method "$METHOD" \
        > "$LOG_DIR/${layer}_${METHOD}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] GPU $gpu  DONE   $layer"
    else
        echo "[$(date '+%H:%M:%S')] GPU $gpu  FAIL   $layer  (exit $rc)"
    fi
}

echo "============================================================"
echo "  COCO Retrieval Evaluation  [method=$METHOD]"
echo "  Started: $(date)"
echo "============================================================"

run 0 blk07 &
run 1 blk15 &
run 2 blk23 &

wait

echo ""
echo "============================================================"
echo "  All done at $(date)"
echo "============================================================"
echo "Logs:    $LOG_DIR/"
echo "Results: $OUT_DIR/"
