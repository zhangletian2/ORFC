#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Qwen3-VL-8B-Instruct  MMStar 全量评测
#
# 对齐 https://github.com/QwenLM/Qwen3-VL 官方 Instruct 推理参数:
#   seed=3407, temperature=0.7, top_p=0.8, top_k=20,
#   repetition_penalty=1.0, presence_penalty=1.5,
#   out_seq_length=32768
#
# 环境: conda activate qwen3vl_codec
# 用法: bash run_qwen3vl_instruct_mmstar.sh [GPU_ID]
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ──── 路径配置 ────
MODEL_PATH="/data4/workspace/zlt/cache/hf/hub/Qwen3-VL-8B-Instruct"
DATA_PATH="/data4/workspace/zlt/featcodec/data/MMStar/mmstar.parquet"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/qwen3vl_instruct_mmstar_eval.py"
RESULT_DIR="${SCRIPT_DIR}/../results/qwen3vl_8b_instruct_mmstar"

PYTHON="/home/user/anaconda3/envs/qwen3vl_codec/bin/python"

mkdir -p "${RESULT_DIR}"

# ──── 官方 Instruct 推理参数 (https://github.com/QwenLM/Qwen3-VL) ────
SEED=3407
TEMPERATURE=0.7
TOP_P=0.8
TOP_K=20
REP_PENALTY=1.0
PRESENCE_PENALTY=1.5
MAX_TOKENS=32768
DTYPE="bf16"

# ──── GPU 配置 ────
GPU_ID="${1:-3}"

echo "══════════════════════════════════════════════════════════════"
echo "  Qwen3-VL-8B-Instruct MMStar Full Evaluation"
echo "  GPU: ${GPU_ID}  |  model=Instruct (no thinking)"
echo "  seed=${SEED}, temperature=${TEMPERATURE}, top_p=${TOP_P}, top_k=${TOP_K}"
echo "  presence_penalty=${PRESENCE_PENALTY}, max_new_tokens=${MAX_TOKENS}"
echo "══════════════════════════════════════════════════════════════"
echo ""

LOG_FILE="${RESULT_DIR}/log.txt"
OUTPUT_FILE="${RESULT_DIR}/mmstar_instruct.json"

echo "[$(date '+%H:%M:%S')] Starting on GPU${GPU_ID} ..."
CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} "${EVAL_SCRIPT}" \
    --model_path "${MODEL_PATH}" \
    --data_path  "${DATA_PATH}" \
    --dtype      "${DTYPE}" \
    --seed       "${SEED}" \
    --max_new_tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --repetition_penalty "${REP_PENALTY}" \
    --presence_penalty "${PRESENCE_PENALTY}" \
    --output "${OUTPUT_FILE}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Done!"
echo "  Results: ${OUTPUT_FILE}"
echo "  Log:     ${LOG_FILE}"
echo "══════════════════════════════════════════════════════════════"
