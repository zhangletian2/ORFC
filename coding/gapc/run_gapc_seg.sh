#!/usr/bin/env bash
# GAPC segmentation-only multi-GPU launcher.
#
# Purpose
# -------
# Classification R-D curves are already cached in results/<backbone>/.
# This script runs *only* the VOC2012 segmentation task:
#
#   * Train sweep  : features/voc2012_5000 (filtered to images with
#                    SegmentationClass GT; default 500 samples, matching
#                    the classification train size).
#   * Test report  : features/voc2012_100 (100 held-out images).
#   * Quantisation extension : quant_bits ∈ {16, 8, 4} (post-GAPC scalar
#     quantisation of the kept columns).  Paper-strict fp32 (quant_bits=32)
#     is NOT run here — set QUANT_BITS=32 to add it back.
#
# Parallelism
# -----------
# 4 layers x 4 GPUs: blk05,blk10,blk15,blk20 -> GPU 0,1,2,3.
# Per wave: each GPU runs one layer through the whole seg sweep.
# Waves are sequential over quant_bits so a GPU is never contended.
#
# Train / test overlap is *guaranteed empty* by the runner (raises on any
# shared basename; disjoint pools are further pre-filtered to images that
# have both .npy features AND a SegmentationClass .png).
#
# Usage
# -----
#   # default: blk05,10,15,20 × q={16,8,4} on GPU 0..3
#   bash run_gapc_seg.sh
#
#   # only blk10, only 8-bit, GPU 2
#   GPUS=2 LAYERS=blk10 QUANT_BITS=8 bash run_gapc_seg.sh
#
#   # larger seg train sample (all ~1192 GT-valid VOC2012 trainaug images)
#   MAX_SEG_TRAIN=0 bash run_gapc_seg.sh
#
# Output
#   results/<backbone>/<layer>_seg_zip6_q{16,8,4}_segtr<Ntr>_segte<Nte>_s42.json
#   logs/seg_<timestamp>/<layer>_q<QB>_gpu<G>.log

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
QUANT_BITS="${QUANT_BITS:-16,8,4}"           # paper-extension low-bpfp curves

CHUNK="${CHUNK:-64}"
ZIP_LEVEL="${ZIP_LEVEL:-6}"
ZIP_WORKERS="${ZIP_WORKERS:-8}"
BACKBONE="${BACKBONE:-dinov2_vitl14}"
SEED="${SEED:-42}"

# ---- segmentation data roots ----
SEG_TRAIN_FEAT_ROOT="${SEG_TRAIN_FEAT_ROOT:-$PROJ_ROOT/features/voc2012_5000}"
SEG_TRAIN_IMAGE_LIST="${SEG_TRAIN_IMAGE_LIST:-$PROJ_ROOT/utils/voc2012_all_5000.txt}"
# 500 matches the classification training sample size (bpfp R-D points).
# Set 0 to use ALL GT-valid images in voc2012_5000 (~1192, ≈2.4× slower).
MAX_SEG_TRAIN="${MAX_SEG_TRAIN:-500}"

SEG_TEST_FEAT_ROOT="${SEG_TEST_FEAT_ROOT:-$PROJ_ROOT/features/voc2012_100}"
SEG_TEST_IMAGE_LIST="${SEG_TEST_IMAGE_LIST:-$PROJ_ROOT/utils/voc2012_val_100.txt}"
MAX_SEG_TEST="${MAX_SEG_TEST:-0}"

VOC_ROOT="${VOC_ROOT:-$PROJ_ROOT/data/VOCdevkit/VOC2012}"

LOG_DIR="${LOG_DIR:-$THISDIR/logs/seg_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

# ================================================================
#                    per-layer threshold grids
# ================================================================
# These are the same θ values used for the classification runner, picked
# off the |L| scale of DINOv2-ViT-L/14 raw latents:
#   blk05: abs_mean ≈ 0.038  blk10: ≈ 0.058  blk15: ≈ 0.159  blk20: ≈ 0.688
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
#                 launch one (layer, QB) seg job
# ================================================================

run_one_seg () {
    local LAYER=$1
    local GPU=$2
    local QB=$3
    local LOG="$LOG_DIR/${LAYER}_q${QB}_gpu${GPU}.log"
    local TH
    TH="$(get_th "$LAYER")"

    echo "[launch] seg  $LAYER  q${QB}  GPU $GPU  (log: $LOG)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SCRIPT" \
        --layer "$LAYER" \
        --backbone "$BACKBONE" \
        --threshold_list $TH \
        --quant_bits "$QB" \
        --chunk_images "$CHUNK" \
        --zip_level "$ZIP_LEVEL" \
        --zip_workers "$ZIP_WORKERS" \
        --seed "$SEED" \
        --skip_cls \
        --eval_seg \
        --seg_train_feat_root "$SEG_TRAIN_FEAT_ROOT" \
        --seg_train_image_list "$SEG_TRAIN_IMAGE_LIST" \
        --max_seg_train_images "$MAX_SEG_TRAIN" \
        --seg_test_feat_root "$SEG_TEST_FEAT_ROOT" \
        --seg_test_image_list "$SEG_TEST_IMAGE_LIST" \
        --max_seg_test_images "$MAX_SEG_TEST" \
        --voc_root "$VOC_ROOT" \
        > "$LOG" 2>&1
}

export -f run_one_seg get_th
export PYTHON SCRIPT BACKBONE CHUNK ZIP_LEVEL ZIP_WORKERS SEED LOG_DIR
export VOC_ROOT
export SEG_TRAIN_FEAT_ROOT SEG_TRAIN_IMAGE_LIST MAX_SEG_TRAIN
export SEG_TEST_FEAT_ROOT SEG_TEST_IMAGE_LIST MAX_SEG_TEST
export TH_blk05 TH_blk10 TH_blk15 TH_blk20

# ================================================================
#                           banner
# ================================================================

echo "=========================================================================="
echo "  GAPC SEG-ONLY multi-GPU launcher   (quant-ext: ${QUANT_BITS})"
echo "  backbone=$BACKBONE"
echo "  layers=(${LAYER_ARR[*]})   GPUs=(${GPU_ARR[*]})"
echo "  quant_bits=(${QUANT_ARR[*]})"
echo "  [seg] train : $SEG_TRAIN_FEAT_ROOT   (max=$MAX_SEG_TRAIN)"
echo "        list  : $SEG_TRAIN_IMAGE_LIST"
echo "        test  : $SEG_TEST_FEAT_ROOT   (max=$MAX_SEG_TEST)"
echo "        list  : $SEG_TEST_IMAGE_LIST"
echo "        voc   : $VOC_ROOT"
echo "  zip_level=$ZIP_LEVEL  zip_workers=$ZIP_WORKERS  chunk=$CHUNK"
echo "  logs: $LOG_DIR"
echo "=========================================================================="

# ================================================================
#  Outer : QB ; Inner : LAYER in parallel (one GPU per layer)
# ================================================================

for QB in "${QUANT_ARR[@]}"; do
    echo ""
    echo "---- quant_bits=$QB ----"
    pids=()
    for i in "${!LAYER_ARR[@]}"; do
        LAYER="${LAYER_ARR[$i]}"
        GPU="${GPU_ARR[$i]}"
        run_one_seg "$LAYER" "$GPU" "$QB" &
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
echo "  All segmentation sweeps complete."
echo "  Results : $THISDIR/results/$BACKBONE/*_seg_*.json"
echo "  Logs    : $LOG_DIR/*.log"
echo "=========================================================================="
