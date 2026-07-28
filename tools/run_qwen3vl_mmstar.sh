#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Qwen3-VL-8B-Thinking  MMStar 全量评测
#
# 注意: 该模型的 chat_template 硬编码 <think> 在 generation prompt 中，
#       始终处于 thinking 模式，无法关闭。
#
# 严格对齐官方推理参数:
#   temperature=1.0, top_p=0.95, top_k=20,
#   repetition_penalty=1.0, out_seq_length=40960
#
# 环境: conda activate qwen3vl_codec
# 用法: bash run_qwen3vl_mmstar.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ──── 路径配置 ────
MODEL_PATH="/data4/workspace/zlt/cache/hf/hub/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b"
DATA_PATH="/data4/workspace/zlt/featcodec/data/MMStar/mmstar.parquet"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/qwen3vl_mmstar_eval.py"
RESULT_DIR="${SCRIPT_DIR}/../results/qwen3vl_8b_thinking_mmstar"

PYTHON="/home/user/anaconda3/envs/qwen3vl_codec/bin/python"

mkdir -p "${RESULT_DIR}"

# ──── 官方推理参数 ────
TEMPERATURE=1.0
TOP_P=0.95
TOP_K=20
REP_PENALTY=1.0
MAX_TOKENS=40960
DTYPE="bf16"

# ──── GPU 配置 ────
GPU_ID="${1:-6}"

echo "══════════════════════════════════════════════════════════════"
echo "  Qwen3-VL-8B-Thinking MMStar Full Evaluation"
echo "  GPU: ${GPU_ID}  |  thinking=ON (always)"
echo "  temperature=${TEMPERATURE}, top_p=${TOP_P}, top_k=${TOP_K}"
echo "  max_new_tokens=${MAX_TOKENS}"
echo "══════════════════════════════════════════════════════════════"
echo ""

LOG_FILE="${RESULT_DIR}/log_think_on.txt"
OUTPUT_FILE="${RESULT_DIR}/mmstar_think_on.json"

echo "[$(date '+%H:%M:%S')] Starting on GPU${GPU_ID} ..."
CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} "${EVAL_SCRIPT}" \
    --model_path "${MODEL_PATH}" \
    --data_path  "${DATA_PATH}" \
    --dtype      "${DTYPE}" \
    --max_new_tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --repetition_penalty "${REP_PENALTY}" \
    --output "${OUTPUT_FILE}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Done!"
echo "  Results: ${OUTPUT_FILE}"
echo "  Log:     ${LOG_FILE}"
echo "══════════════════════════════════════════════════════════════"
