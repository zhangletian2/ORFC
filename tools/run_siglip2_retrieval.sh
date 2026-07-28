#!/bin/bash
# SigLIP2 COCO Retrieval: 特征提取 + 检索评估
# 使用 utils/ 下固定的 500 张图片列表
#
# 用法: bash run_siglip2_retrieval.sh [extract|direct|replay|compare|all]

set -e

export HF_HUB_OFFLINE=1

# ─── 路径配置 ───
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UTILS_DIR="${PROJECT_ROOT}/utils"
FEATURES_DIR="${PROJECT_ROOT}/features"

COCO_ROOT="/data4/workspace/zlt/CAPO_ret/dataset/coco2014"
IMAGE_LIST="${UTILS_DIR}/coco_selected_pathname500.txt"
CAPTION_JSON="${UTILS_DIR}/coco_selected_caption500.json"
FEAT_ROOT="${FEATURES_DIR}/coco_ret/siglip2_so400m"
MODEL_ID="google/siglip2-so400m-patch14-224"

LAYERS="7,15,23"
DEVICE="cuda"

CMD=${1:-all}

# ─── Step 1: 提取中间层特征 ───
if [[ "$CMD" == "extract" || "$CMD" == "all" ]]; then
    echo "============================================"
    echo " Step 1: 提取 SigLIP2 中间层特征"
    echo "============================================"
    python "${SCRIPT_DIR}/siglip2_ret_extract.py" \
        --model_id      "${MODEL_ID}" \
        --image_list    "${IMAGE_LIST}" \
        --image_root    "${COCO_ROOT}" \
        --image_subdir  val2014 \
        --out_root      "${FEAT_ROOT}" \
        --layers        "${LAYERS}" \
        --batch_size    32 \
        --num_workers   4 \
        --device        "${DEVICE}"
    echo ""
fi

# ─── Step 2: 直接推理检索评估 ───
if [[ "$CMD" == "direct" || "$CMD" == "all" ]]; then
    echo "============================================"
    echo " Step 2: Direct Inference 检索评估"
    echo "============================================"
    python "${SCRIPT_DIR}/siglip2_retrieval.py" direct \
        --model_id          "${MODEL_ID}" \
        --meta_json         "${CAPTION_JSON}" \
        --image_root        "${COCO_ROOT}" \
        --image_batch_size  32 \
        --text_batch_size   128 \
        --output            "${FEAT_ROOT}/results_direct.json" \
        --device            "${DEVICE}"
    echo ""
fi

# ─── Step 3: 中间层回放检索评估 ───
if [[ "$CMD" == "replay" || "$CMD" == "all" ]]; then
    echo "============================================"
    echo " Step 3: Replay 检索评估"
    echo "============================================"
    python "${SCRIPT_DIR}/siglip2_retrieval.py" replay \
        --model_id          "${MODEL_ID}" \
        --meta_json         "${CAPTION_JSON}" \
        --feature_root      "${FEAT_ROOT}" \
        --layers            blk07 blk15 blk23 \
        --text_batch_size   128 \
        --output            "${FEAT_ROOT}/results_replay.json" \
        --device            "${DEVICE}"
    echo ""
fi

# ─── 或: 一步对比 ───
if [[ "$CMD" == "compare" ]]; then
    echo "============================================"
    echo " Compare: Direct vs Replay"
    echo "============================================"
    python "${SCRIPT_DIR}/siglip2_retrieval.py" compare \
        --model_id          "${MODEL_ID}" \
        --meta_json         "${CAPTION_JSON}" \
        --image_root        "${COCO_ROOT}" \
        --feature_root      "${FEAT_ROOT}" \
        --layers            blk07 blk15 blk23 \
        --image_batch_size  32 \
        --text_batch_size   128 \
        --output            "${FEAT_ROOT}/results_compare.json" \
        --device            "${DEVICE}"
fi

echo "Done!"
