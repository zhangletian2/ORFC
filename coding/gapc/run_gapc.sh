#!/usr/bin/env bash
# GAPC multi-GPU experiment launcher (one GPU per layer in parallel).
#
# Scope: paper Algorithm 1 only (no random / top-k baselines).
# Tasks:
#   * classification (default): ImageNet sweep on features/train, report on
#     features/test.
#   * --eval_seg / EVAL_SEG=1 : ALSO run the VOC2012 mIoU + bpfp pipeline.
#   * --skip_cls  / SKIP_CLS=1: skip the classification pipeline entirely
#     (requires EVAL_SEG=1).  Use `run_gapc_seg.sh` for a pre-wired version.
#
# Train / test sample pools are disjoint and the runner raises on any
# overlap — a guarantee, not just a check.
#
# Layer-specific threshold grids (|L| scale of DINOv2-ViT-L/14 raw latents):
#   blk05: abs_mean ≈ 0.038   -> θ ∈ [0.005, 0.15]
#   blk10: abs_mean ≈ 0.058   -> θ ∈ [0.005, 0.2]
#   blk15: abs_mean ≈ 0.159   -> θ ∈ [0.02, 0.5]
#   blk20: abs_mean ≈ 0.688   -> θ ∈ [0.1, 1.5]
#
# Usage examples:
#   # cls only, default 4 layers x 4 GPUs
#   bash run_gapc.sh
#
#   # cls + VOC2012 mIoU sweep on the same 4 GPUs
#   EVAL_SEG=1 bash run_gapc.sh
#
#   # paper-extension quant_bits sweep (cls)
#   QUANT_BITS=16,8,4 bash run_gapc.sh
#
#   # one layer at a time, explicit GPU, seg only
#   GPUS=3 LAYERS=blk10 EVAL_SEG=1 SKIP_CLS=1 bash run_gapc.sh
#
# Output:
#   results/<backbone>/<layer>_gapc_zip6[_q{QB}][_seg]_ntr<Ntr>_nte<Nte>_s42.json
#   logs/<timestamp>/<layer>_q<QB>_gpu<G>.log

set -euo pipefail

PYTHON=/home/user/anaconda3/envs/featcodec2/bin/python
THISDIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$THISDIR/run_gapc.py"
PROJ_ROOT="$(cd "$THISDIR/../../.." && pwd)"

# ================================================================
#                    user-overridable knobs
# ================================================================

GPUS="${GPUS:-0,1,2,3}"
LAYERS="${LAYERS:-blk05,blk10,blk15,blk20}"
QUANT_BITS="${QUANT_BITS:-32}"         # 32 (paper) | 16 | 8 | 4 (comma-sep)

# ---- task selection ----
SKIP_CLS="${SKIP_CLS:-0}"              # 1 = skip cls (seg only)
EVAL_SEG="${EVAL_SEG:-0}"              # 1 = also run seg

# ---- classification ----
MAX_TRAIN="${MAX_TRAIN:-500}"
MAX_TEST="${MAX_TEST:-0}"              # 0 = all test images
CHUNK="${CHUNK:-64}"
ZIP_LEVEL="${ZIP_LEVEL:-6}"
ZIP_WORKERS="${ZIP_WORKERS:-8}"
BACKBONE="${BACKBONE:-dinov2_vitl14}"
SEED="${SEED:-42}"

# ---- segmentation ----
SEG_TRAIN_FEAT_ROOT="${SEG_TRAIN_FEAT_ROOT:-$PROJ_ROOT/features/voc2012_5000}"
SEG_TRAIN_IMAGE_LIST="${SEG_TRAIN_IMAGE_LIST:-$PROJ_ROOT/utils/voc2012_all_5000.txt}"
# Default 500 matches the classification train sample size.
# Set to 0 to use all ~1192 seg-GT-valid images (slower; larger preload).
MAX_SEG_TRAIN="${MAX_SEG_TRAIN:-500}"

SEG_TEST_FEAT_ROOT="${SEG_TEST_FEAT_ROOT:-$PROJ_ROOT/features/voc2012_100}"
SEG_TEST_IMAGE_LIST="${SEG_TEST_IMAGE_LIST:-$PROJ_ROOT/utils/voc2012_val_100.txt}"
MAX_SEG_TEST="${MAX_SEG_TEST:-0}"

VOC_ROOT="${VOC_ROOT:-$PROJ_ROOT/data/VOCdevkit/VOC2012}"

LOG_DIR="${LOG_DIR:-$THISDIR/logs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

# ================================================================
#                    per-layer threshold grids
# ================================================================

TH_blk05="0.005 0.01 0.015 0.02 0.025 0.03 0.04 0.05 0.07 0.1 0.15"
TH_blk10="0.005 0.015 0.025 0.035 0.05 0.065 0.075 0.085 0.1 0.15 0.2"
TH_blk15="0.02 0.05 0.08 0.12 0.16 0.2 0.25 0.3 0.35 0.4 0.5"
TH_blk20="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0 1.2 1.5"

IFS=',' read -ra LAYER_ARR <<< "$LAYERS"
IFS=',' read -ra GPU_ARR   <<< "$GPUS"
IFS=',' read -ra QUANT_ARR <<< "$QUANT_BITS"

if [ "${#LAYER_ARR[@]}" -gt "${#GPU_ARR[@]}" ]; then
    echo "[ERROR] Not enough GPUs (${#GPU_ARR[@]}) for ${#LAYER_ARR[@]} layers."
    echo "        Either reduce LAYERS= or extend GPUS=."
    exit 1
fi

if [ "$SKIP_CLS" = "1" ] && [ "$EVAL_SEG" != "1" ]; then
    echo "[ERROR] SKIP_CLS=1 requires EVAL_SEG=1 (otherwise nothing runs)."
    exit 1
fi

get_th () {
    local LAYER=$1
    case "$LAYER" in
        blk05) echo "$TH_blk05" ;;
        blk10) echo "$TH_blk10" ;;
        blk15) echo "$TH_blk15" ;;
        blk20) echo "$TH_blk20" ;;
        *)
            echo "[WARN] No default thresholds for $LAYER — using blk10 grid" >&2
            echo "$TH_blk10" ;;
    esac
}

# ================================================================
#                 launch one (layer, QB) job
# ================================================================

run_one () {
    local LAYER=$1
    local GPU=$2
    local QB=$3
    local LOG="$LOG_DIR/${LAYER}_q${QB}_gpu${GPU}.log"
    local TH
    TH="$(get_th "$LAYER")"

    local SKIP_ARGS=""
    if [ "$SKIP_CLS" = "1" ]; then
        SKIP_ARGS="--skip_cls"
    fi

    local SEG_ARGS=""
    if [ "$EVAL_SEG" = "1" ]; then
        SEG_ARGS="--eval_seg \
            --seg_train_feat_root $SEG_TRAIN_FEAT_ROOT \
            --seg_train_image_list $SEG_TRAIN_IMAGE_LIST \
            --max_seg_train_images $MAX_SEG_TRAIN \
            --seg_test_feat_root $SEG_TEST_FEAT_ROOT \
            --seg_test_image_list $SEG_TEST_IMAGE_LIST \
            --max_seg_test_images $MAX_SEG_TEST \
            --voc_root $VOC_ROOT"
    fi

    echo "[launch] $LAYER / q${QB} on GPU $GPU (log: $LOG)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SCRIPT" \
        --layer "$LAYER" \
        --backbone "$BACKBONE" \
        --threshold_list $TH \
        --quant_bits "$QB" \
        --max_train_images "$MAX_TRAIN" \
        --max_test_images "$MAX_TEST" \
        --chunk_images "$CHUNK" \
        --zip_level "$ZIP_LEVEL" \
        --zip_workers "$ZIP_WORKERS" \
        --seed "$SEED" \
        $SKIP_ARGS \
        $SEG_ARGS \
        > "$LOG" 2>&1
}

export -f run_one get_th
export PYTHON SCRIPT BACKBONE MAX_TRAIN MAX_TEST CHUNK ZIP_LEVEL ZIP_WORKERS
export SEED LOG_DIR SKIP_CLS EVAL_SEG VOC_ROOT
export SEG_TRAIN_FEAT_ROOT SEG_TRAIN_IMAGE_LIST MAX_SEG_TRAIN
export SEG_TEST_FEAT_ROOT SEG_TEST_IMAGE_LIST MAX_SEG_TEST
export TH_blk05 TH_blk10 TH_blk15 TH_blk20

# ================================================================
#                           banner
# ================================================================

echo "=========================================================================="
echo "  GAPC multi-GPU launcher"
echo "  backbone=$BACKBONE"
echo "  layers=(${LAYER_ARR[*]})  GPUs=(${GPU_ARR[*]})"
echo "  quant_bits=(${QUANT_ARR[*]})"
TASKS="cls"
if [ "$SKIP_CLS" = "1" ]; then TASKS=""; fi
if [ "$EVAL_SEG" = "1" ]; then
    if [ -z "$TASKS" ]; then TASKS="seg"; else TASKS="$TASKS,seg"; fi
fi
echo "  tasks=$TASKS"
if [ "$SKIP_CLS" != "1" ]; then
    echo "  [cls] max_train=$MAX_TRAIN  max_test=$MAX_TEST  chunk=$CHUNK"
    echo "        zip_level=$ZIP_LEVEL  zip_workers=$ZIP_WORKERS"
fi
if [ "$EVAL_SEG" = "1" ]; then
    echo "  [seg] train : $SEG_TRAIN_FEAT_ROOT  (max=$MAX_SEG_TRAIN)"
    echo "        list  : $SEG_TRAIN_IMAGE_LIST"
    echo "        test  : $SEG_TEST_FEAT_ROOT  (max=$MAX_SEG_TEST)"
    echo "        list  : $SEG_TEST_IMAGE_LIST"
    echo "        voc   : $VOC_ROOT"
fi
echo "  logs: $LOG_DIR"
echo "=========================================================================="

# ================================================================
#  Outer: QB ; Inner: LAYER (parallel across GPUs, one per layer)
# ================================================================

for QB in "${QUANT_ARR[@]}"; do
    echo ""
    echo "---- quant_bits=$QB ----"
    pids=()
    for i in "${!LAYER_ARR[@]}"; do
        LAYER="${LAYER_ARR[$i]}"
        GPU="${GPU_ARR[$i]}"
        run_one "$LAYER" "$GPU" "$QB" &
        pids+=($!)
    done
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=$((failed+1)); fi
    done
    if [ "$failed" -ne 0 ]; then
        echo "[ERROR] $failed job(s) failed in q=$QB — see $LOG_DIR"
        exit 1
    fi
    echo "[done]  quant_bits=$QB"
done

echo ""
echo "=========================================================================="
echo "  All sweeps complete."
echo "  Results : $THISDIR/results/$BACKBONE/*.json"
echo "  Logs    : $LOG_DIR/*.log"
echo "=========================================================================="
