#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Qwen3-VL-4B-Thinking  MMBench 分层抽取 / 回放 评测
# 环境: conda activate qwen3vl_codec
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ──── 路径配置 ────
MODEL_PATH="/data4/workspace/zlt/cache/hf/hub/Qwen3-VL-4B-Thinking"
DATA_DIR="/data4/workspace/zlt/featcodec/ORFC/data/MMBench"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE="${SCRIPT_DIR}/qwen3vl_feat_pipeline.py"
RESULT_DIR="${SCRIPT_DIR}/../results/qwen3vl_mmbench"
FEAT_DIR="${SCRIPT_DIR}/../features/mmbench_en_val_1000/qwen3vl_4b/blk05"
SAMPLE_LIST="${SCRIPT_DIR}/../utils/mmbench_en_val_1000.txt"

SPLIT="validation"           # validation 有 ground-truth；test 需要提交
LAYER=5                     # ViT block 切分索引（前半 0-5，后半 6-23）
MAX_SAMPLES=""              # 留空=全量，调试时设为 "--max_samples 20"
DTYPE="bf16"
DEVICE="cuda"
THINK_FLAG=""               # 默认开启 thinking；设 "--no_think" 关闭
MAX_TOKENS=2048             # VLMEvalKit Qwen-VL 默认 2048
CIRCULAR="--circular"       # CircularEval（与官方排行榜对齐）；留空则 single-pass
SAMPLE_ARG="--sample_list ${SAMPLE_LIST}"  # 使用固定采样列表；留空=全量

PYTHON="/home/user/anaconda3/envs/qwen3vl_codec/bin/python"

mkdir -p "${RESULT_DIR}"

# ─────────────────────────────────────────────────────────────
# 1) Baseline：标准推理（无特征操作）
# ─────────────────────────────────────────────────────────────
run_baseline() {
    echo "══════ [1/3] Baseline ══════"
    ${PYTHON} "${PIPELINE}" \
        --model_path "${MODEL_PATH}" \
        --data_dir   "${DATA_DIR}" \
        --split      "${SPLIT}" \
        --device     "${DEVICE}" \
        --dtype      "${DTYPE}" \
        --max_new_tokens "${MAX_TOKENS}" \
        ${MAX_SAMPLES} \
        ${SAMPLE_ARG} \
        ${THINK_FLAG} \
        baseline \
        ${CIRCULAR} \
        --output "${RESULT_DIR}/baseline.json"
}

# ─────────────────────────────────────────────────────────────
# 2) Extract：抽取 ViT block[LAYER] 特征
# ─────────────────────────────────────────────────────────────
run_extract() {
    echo "══════ [2/3] Extract (layer=${LAYER}) ══════"
    echo "  提示: 推荐使用 run_extract_vit.sh 做 ViT-only 提取（更快）"
    echo "  特征已在: ${FEAT_DIR}"
    ${PYTHON} "${PIPELINE}" \
        --model_path "${MODEL_PATH}" \
        --data_dir   "${DATA_DIR}" \
        --split      "${SPLIT}" \
        --device     "${DEVICE}" \
        --dtype      "${DTYPE}" \
        --max_new_tokens "${MAX_TOKENS}" \
        ${MAX_SAMPLES} \
        ${SAMPLE_ARG} \
        ${THINK_FLAG} \
        extract \
        --layer   "${LAYER}" \
        --out_dir "${FEAT_DIR}" \
        --output  "${RESULT_DIR}/extract.json"
}

# ─────────────────────────────────────────────────────────────
# 3) Replay：从保存特征回放推理
# ─────────────────────────────────────────────────────────────
run_replay() {
    echo "══════ [3/3] Replay (layer=${LAYER}) ══════"
    ${PYTHON} "${PIPELINE}" \
        --model_path "${MODEL_PATH}" \
        --data_dir   "${DATA_DIR}" \
        --split      "${SPLIT}" \
        --device     "${DEVICE}" \
        --dtype      "${DTYPE}" \
        --max_new_tokens "${MAX_TOKENS}" \
        ${MAX_SAMPLES} \
        ${SAMPLE_ARG} \
        ${THINK_FLAG} \
        replay \
        --layer    "${LAYER}" \
        --feat_dir "${FEAT_DIR}" \
        ${CIRCULAR} \
        --output   "${RESULT_DIR}/replay.json"
}

# ─────────────────────────────────────────────────────────────
# 执行（可单独调用: bash run_qwen3vl_mmbench.sh baseline）
# ─────────────────────────────────────────────────────────────
CMD="${1:-all}"

case "${CMD}" in
    baseline)  run_baseline ;;
    extract)   run_extract  ;;
    replay)    run_replay   ;;
    all)
        run_baseline
        run_extract
        run_replay
        echo ""
        echo "══════ Done ══════"
        echo "Results: ${RESULT_DIR}/"
        echo "  baseline.json  — 标准推理精度"
        echo "  extract.json   — 抽取模式精度（应与 baseline 一致）"
        echo "  replay.json    — 回放模式精度（验证特征可复现）"
        ;;
    *)
        echo "Usage: $0 {baseline|extract|replay|all}"
        exit 1 ;;
esac
