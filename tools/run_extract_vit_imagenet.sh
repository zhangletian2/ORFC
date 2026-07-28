#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Qwen3-VL ViT-only ImageNet feature extraction launcher
#
# Usage:
#   bash run_extract_vit_imagenet.sh train 4   # extract train (5000), 4 GPUs
#   bash run_extract_vit_imagenet.sh test  2   # extract test  (500),  2 GPUs
#   bash run_extract_vit_imagenet.sh all   4   # extract both, 4 GPUs
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SUBSET="${1:-all}"
NUM_WORKERS="${2:-1}"

# ──── Paths ────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="/data4/workspace/zlt/cache/hf/hub/Qwen3-VL-4B-Thinking"
IMG_ROOT="/data4/workspace/zlt/featcodec/ORFC/data/imagenet/images/val"
LAYER=5
DTYPE="bf16"

PYTHON="/home/user/anaconda3/envs/qwen3vl_codec/bin/python"
SCRIPT="${PROJECT_ROOT}/tools/extract_vit_imagenet.py"

# train: 5000 samples
TRAIN_LIST="${PROJECT_ROOT}/utils/imagenet_selected_pathname5000.txt"
TRAIN_OUT="${PROJECT_ROOT}/features/train/qwen3vl_4b/blk05"

# test: 500 samples
TEST_LIST="${PROJECT_ROOT}/utils/imagenet_selected_pathname500.txt"
TEST_OUT="${PROJECT_ROOT}/features/test/qwen3vl_4b/blk05"

# ──── Function ────
run_extract() {
    local TAG="$1"
    local LIST_FILE="$2"
    local OUT_DIR="$3"
    local N_WORKERS="$4"
    local N_SAMPLES
    N_SAMPLES=$(wc -l < "$LIST_FILE")

    echo ""
    echo "========================================"
    echo "  ViT Feature Extraction: $TAG"
    echo "========================================"
    echo "  Model:   $(basename $MODEL_PATH)"
    echo "  Layer:   block[$LAYER]"
    echo "  Samples: $N_SAMPLES"
    echo "  Workers: $N_WORKERS"
    echo "  Output:  $OUT_DIR"
    echo "========================================"
    echo ""

    mkdir -p "$OUT_DIR"
    local T0=$SECONDS

    if [ "$N_WORKERS" -eq 1 ]; then
        $PYTHON "$SCRIPT" \
            --model_path "$MODEL_PATH" \
            --root "$IMG_ROOT" \
            --list "$LIST_FILE" \
            --layer "$LAYER" \
            --out_dir "$OUT_DIR" \
            --dtype "$DTYPE"
    else
        local pids=()
        local LOG_DIR="${OUT_DIR}/logs"
        mkdir -p "$LOG_DIR"

        for ((i=0; i<N_WORKERS; i++)); do
            CUDA_VISIBLE_DEVICES=$i \
            $PYTHON "$SCRIPT" \
                --model_path "$MODEL_PATH" \
                --root "$IMG_ROOT" \
                --list "$LIST_FILE" \
                --layer "$LAYER" \
                --out_dir "$OUT_DIR" \
                --dtype "$DTYPE" \
                --device "cuda" \
                --num_workers "$N_WORKERS" \
                --worker_id "$i" \
                > "${LOG_DIR}/worker_${i}.log" 2>&1 &
            pids+=($!)
            echo "  Started worker $i (PID ${pids[-1]})"
        done

        echo "  Waiting for $N_WORKERS workers..."
        local failed=0
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                failed=$((failed + 1))
            fi
        done

        echo ""
        for ((i=0; i<N_WORKERS; i++)); do
            echo "  --- Worker $i ---"
            tail -3 "${LOG_DIR}/worker_${i}.log"
            echo ""
        done

        if [ "$failed" -gt 0 ]; then
            echo "  [WARN] $failed worker(s) failed, check logs in $LOG_DIR"
        fi
    fi

    # Merge grid_thw JSON files from all workers
    echo "  Merging grid_thw metadata..."
    $PYTHON -c "
import json, glob, os
out_dir = '$OUT_DIR'
merged = {}
for f in sorted(glob.glob(os.path.join(out_dir, 'grid_thw_worker*.json'))):
    with open(f) as fp:
        merged.update(json.load(fp))
    os.remove(f)
out_path = os.path.join(out_dir, 'grid_thw.json')
with open(out_path, 'w') as fp:
    json.dump(merged, fp)
print(f'  grid_thw.json: {len(merged)} entries')
"

    local ELAPSED=$((SECONDS - T0))
    local N_FILES
    N_FILES=$(find "$OUT_DIR" -name "*.npy" | wc -l)
    local TOTAL_SIZE
    TOTAL_SIZE=$(du -sh "$OUT_DIR" | cut -f1)

    echo "  ----------------------------------------"
    echo "  $TAG done in ${ELAPSED}s"
    echo "  Files: $N_FILES .npy ($TOTAL_SIZE)"
    echo "  Path:  $OUT_DIR"
    echo "  ----------------------------------------"
}

# ──── Main ────
case "$SUBSET" in
    train)
        run_extract "train (5000)" "$TRAIN_LIST" "$TRAIN_OUT" "$NUM_WORKERS"
        ;;
    test)
        run_extract "test (500)" "$TEST_LIST" "$TEST_OUT" "$NUM_WORKERS"
        ;;
    all)
        run_extract "train (5000)" "$TRAIN_LIST" "$TRAIN_OUT" "$NUM_WORKERS"
        run_extract "test (500)" "$TEST_LIST" "$TEST_OUT" "$NUM_WORKERS"
        ;;
    *)
        echo "Usage: $0 {train|test|all} [num_gpus]"
        exit 1
        ;;
esac

echo ""
echo "All extraction complete."
