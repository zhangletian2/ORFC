#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Qwen3-VL-8B-Instruct  MMStar 分层抽取 / 分层回放 Pipeline
#
# 三个子命令：
#   baseline  — 标准评测（精度基线）
#   extract   — hook 抽取 ViT block[8] 特征 + 同步评测
#   replay    — 从保存/重建特征回放推理
#
# 对齐 https://github.com/QwenLM/Qwen3-VL 官方 Instruct 推理参数
#
# 用法:
#   bash run_qwen3vl_instruct_feat_pipeline.sh baseline [GPU_ID]
#   bash run_qwen3vl_instruct_feat_pipeline.sh extract  [GPU_ID] [LAYER]
#   bash run_qwen3vl_instruct_feat_pipeline.sh replay   [GPU_ID] [LAYER] [FEAT_DIR]
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ──── 路径配置 ────
MODEL_PATH="/data4/workspace/zlt/cache/hf/hub/Qwen3-VL-8B-Instruct"
DATA_PATH="/data4/workspace/zlt/featcodec/data/MMStar/mmstar.parquet"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/qwen3vl_instruct_feat_pipeline.py"
FEAT_BASE="${SCRIPT_DIR}/../features"
RESULT_BASE="${SCRIPT_DIR}/../results/qwen3vl_8b_instruct_feat"

PYTHON="/home/user/anaconda3/envs/qwen3vl_codec/bin/python"

# ──── 官方 Instruct 推理参数 ────
SEED=3407
TEMPERATURE=0.7
TOP_P=0.8
TOP_K=20
REP_PENALTY=1.0
PRESENCE_PENALTY=1.5
MAX_TOKENS=32768
DTYPE="bf16"

# ──── 通用参数字符串 ────
COMMON_ARGS="--model_path ${MODEL_PATH} \
    --data_path  ${DATA_PATH} \
    --dtype      ${DTYPE} \
    --seed       ${SEED} \
    --max_new_tokens ${MAX_TOKENS} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --top_k ${TOP_K} \
    --repetition_penalty ${REP_PENALTY} \
    --presence_penalty ${PRESENCE_PENALTY}"

# ──── 子命令分发 ────
CMD="${1:-baseline}"
GPU_ID="${2:-3}"

case "${CMD}" in
  baseline)
    RESULT_DIR="${RESULT_BASE}/baseline"
    mkdir -p "${RESULT_DIR}"

    echo "══════════════════════════════════════════════════════════════"
    echo "  [baseline] Qwen3-VL-8B-Instruct MMStar"
    echo "  GPU: ${GPU_ID}  seed=${SEED}  temp=${TEMPERATURE}"
    echo "══════════════════════════════════════════════════════════════"

    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} "${EVAL_SCRIPT}" \
        baseline ${COMMON_ARGS} \
        --output "${RESULT_DIR}/mmstar_baseline.json" \
        2>&1 | tee "${RESULT_DIR}/log.txt"
    ;;

  extract)
    LAYER="${3:-8}"
    LAYER_FMT=$(printf '%02d' "${LAYER}")
    RESULT_DIR="${RESULT_BASE}/extract_blk${LAYER_FMT}"
    FEAT_DIR="${FEAT_BASE}/qwen3vl_8b_instruct_mmstar/blk${LAYER_FMT}"
    mkdir -p "${FEAT_DIR}" "${RESULT_DIR}"

    echo "══════════════════════════════════════════════════════════════"
    echo "  [extract] Qwen3-VL-8B-Instruct MMStar"
    echo "  GPU: ${GPU_ID}  layer=${LAYER}  seed=${SEED}  temp=${TEMPERATURE}"
    echo "  Features → ${FEAT_DIR}"
    echo "══════════════════════════════════════════════════════════════"

    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} "${EVAL_SCRIPT}" \
        extract ${COMMON_ARGS} \
        --layer "${LAYER}" \
        --out_dir "${FEAT_DIR}" \
        --output "${RESULT_DIR}/mmstar_extract.json" \
        2>&1 | tee "${RESULT_DIR}/log.txt"
    ;;

  replay)
    LAYER="${3:-8}"
    LAYER_FMT=$(printf '%02d' "${LAYER}")
    FEAT_DIR="${4:-${FEAT_BASE}/qwen3vl_8b_instruct_mmstar/blk${LAYER_FMT}}"
    RESULT_DIR="${RESULT_BASE}/replay_blk${LAYER_FMT}"
    mkdir -p "${RESULT_DIR}"

    echo "══════════════════════════════════════════════════════════════"
    echo "  [replay] Qwen3-VL-8B-Instruct MMStar"
    echo "  GPU: ${GPU_ID}  layer=${LAYER}  seed=${SEED}  temp=${TEMPERATURE}"
    echo "  Features ← ${FEAT_DIR}"
    echo "══════════════════════════════════════════════════════════════"

    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} "${EVAL_SCRIPT}" \
        replay ${COMMON_ARGS} \
        --layer "${LAYER}" \
        --feat_dir "${FEAT_DIR}" \
        --output "${RESULT_DIR}/mmstar_replay.json" \
        2>&1 | tee "${RESULT_DIR}/log.txt"
    ;;

  compare)
    LAYER="${3:-8}"
    NUM="${4:-20}"
    LAYER_FMT=$(printf '%02d' "${LAYER}")
    FEAT_DIR="${FEAT_BASE}/qwen3vl_8b_instruct_mmstar/blk${LAYER_FMT}"
    RESULT_DIR="${RESULT_BASE}/compare_blk${LAYER_FMT}"
    mkdir -p "${RESULT_DIR}"

    echo "══════════════════════════════════════════════════════════════"
    echo "  [compare] Qwen3-VL-8B-Instruct MMStar"
    echo "  GPU: ${GPU_ID}  layer=${LAYER}  num=${NUM}  seed=${SEED}"
    echo "  Features ← ${FEAT_DIR}"
    echo "══════════════════════════════════════════════════════════════"

    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} "${EVAL_SCRIPT}" \
        compare ${COMMON_ARGS} \
        --layer "${LAYER}" \
        --feat_dir "${FEAT_DIR}" \
        --num_samples "${NUM}" \
        --output "${RESULT_DIR}/mmstar_compare.json" \
        2>&1 | tee "${RESULT_DIR}/log.txt"
    ;;

  *)
    echo "Usage: $0 {baseline|extract|replay|compare} [GPU_ID] [LAYER] [FEAT_DIR|NUM]"
    exit 1
    ;;
esac

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Done! Results in: ${RESULT_DIR}"
echo "══════════════════════════════════════════════════════════════"
