#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Qwen3-VL ViT-only 特征提取启动器
# 只跑视觉塔，不走 LLM，~50-100x 加速
#
# 用法：
#   bash run_extract_vit.sh          # 单卡提取
#   NUM_WORKERS=4 bash run_extract_vit.sh  # 4 卡并行
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ──── 配置 ────
MODEL_PATH="/data4/workspace/zlt/cache/hf/hub/Qwen3-VL-4B-Thinking"
DATA_DIR="/data4/workspace/zlt/featcodec/ORFC/data/MMBench"
SAMPLE_LIST="/data4/workspace/zlt/featcodec/ORFC/utils/mmbench_en_val_1000.txt"
OUT_DIR="/data4/workspace/zlt/featcodec/ORFC/features/mmbench_en_val_1000/qwen3vl_4b/blk05"
LAYER=5
SPLIT="validation"
DATA_SUBDIR="en"
DTYPE="bf16"

NUM_WORKERS="${NUM_WORKERS:-1}"
PYTHON="/home/user/anaconda3/envs/qwen3vl_codec/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/extract_vit_features.py"

# ──── 运行 ────
mkdir -p "$OUT_DIR"
echo "╔═══════════════════════════════════════════════╗"
echo "║  ViT-only Feature Extraction                  ║"
echo "╠═══════════════════════════════════════════════╣"
echo "║  Model:   $(basename $MODEL_PATH)"
echo "║  Layer:   block[$LAYER]"
echo "║  Samples: $(wc -l < "$SAMPLE_LIST")"
echo "║  Workers: $NUM_WORKERS"
echo "║  Output:  $OUT_DIR"
echo "╚═══════════════════════════════════════════════╝"
echo ""

T0=$SECONDS

if [ "$NUM_WORKERS" -eq 1 ]; then
    $PYTHON "$SCRIPT" \
        --model_path "$MODEL_PATH" \
        --data_dir "$DATA_DIR" \
        --split "$SPLIT" \
        --data_subdir "$DATA_SUBDIR" \
        --sample_list "$SAMPLE_LIST" \
        --layer "$LAYER" \
        --out_dir "$OUT_DIR" \
        --dtype "$DTYPE"
else
    pids=()
    for ((i=0; i<NUM_WORKERS; i++)); do
        CUDA_VISIBLE_DEVICES=$i \
        $PYTHON "$SCRIPT" \
            --model_path "$MODEL_PATH" \
            --data_dir "$DATA_DIR" \
            --split "$SPLIT" \
            --data_subdir "$DATA_SUBDIR" \
            --sample_list "$SAMPLE_LIST" \
            --layer "$LAYER" \
            --out_dir "$OUT_DIR" \
            --dtype "$DTYPE" \
            --device "cuda" \
            --num_workers "$NUM_WORKERS" \
            --worker_id "$i" \
            > "${OUT_DIR}/worker_${i}.log" 2>&1 &
        pids+=($!)
        echo "  Started worker $i (PID ${pids[-1]})"
    done
    echo "  Waiting for all workers..."
    for pid in "${pids[@]}"; do
        wait $pid
    done
    echo ""
    for ((i=0; i<NUM_WORKERS; i++)); do
        echo "=== Worker $i ==="
        tail -3 "${OUT_DIR}/worker_${i}.log"
        echo ""
    done
fi

ELAPSED=$((SECONDS - T0))
N_FILES=$(ls "$OUT_DIR"/*.npy 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh "$OUT_DIR" | cut -f1)

echo "════════════════════════════════════════════════"
echo "  Done in ${ELAPSED}s"
echo "  Files: $N_FILES .npy ($TOTAL_SIZE)"
echo "  Path:  $OUT_DIR"
echo "════════════════════════════════════════════════"
