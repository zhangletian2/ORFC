#!/usr/bin/env bash
set -euo pipefail

# ==================================================================
#  深度估计评测 - 多 GPU 并行 (Soft-PQ & OPQ)
#
#  所有路径、参数、checkpoint 配置在此脚本集中指定，
#  eval_depth_soft_pq.py 只提供纯评测接口。
#
#  Usage:
#    bash run_depth_eval_parallel.sh                         # 默认 4 GPU
#    GPUS=0,1 bash run_depth_eval_parallel.sh               # 指定 GPU
#    GPUS=0 LAYERS="blk05 blk10" bash run_depth_eval_parallel.sh
#    METHODS="opq" bash run_depth_eval_parallel.sh           # 仅 OPQ
#    METHODS="soft_pq opq" bash run_depth_eval_parallel.sh   # 两者都跑
# ==================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- conda ---
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "Cannot find conda" >&2; exit 1
fi
conda activate featcodec2

export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/pretrained}"
cd "$SCRIPT_DIR"

# ==================================================================
#  * 集中配置区 *  修改以下变量即可适配不同实验
# ==================================================================

# --- 数据 & 特征 ---
FEAT_ROOT="${FEAT_ROOT:-$PROJECT_ROOT/features/nyu_depth_80/dinov2_vitl14}"
DATA_ROOT="${DATA_ROOT:-/data4/workspace/zlt/LaMoFC/Data_example/dinov2/dpt/source/NYU_Test80}"
SPLIT_FILE="${SPLIT_FILE:-/data4/workspace/zlt/LaMoFC/Data_example/dinov2/dpt/source/nyu_test_80.txt}"

# --- Checkpoint & 配置 ---
CKPT_DIR="${CKPT_DIR:-$SCRIPT_DIR/checkpoints/dinov2_vitl14}"
CONFIG_JSON="${CONFIG_JSON:-$SCRIPT_DIR/depth_eval_configs.json}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/data4/workspace/zlt/cache/torch/hub/checkpoints}"

# --- 模型参数 ---
MODEL="${MODEL:-vitl14}"
DIM="${DIM:-1024}"
NORM_MODE="${NORM_MODE:-per_image}"

# --- 评测方法 (soft_pq / opq / soft_pq opq) ---
METHODS="${METHODS:-soft_pq opq}"

# --- OPQ 参数 ---
OPQ_TRAIN_FEAT_ROOT="${OPQ_TRAIN_FEAT_ROOT:-$PROJECT_ROOT/features/train/dinov2_vitl14}"
OPQ_CACHE_DIR="${OPQ_CACHE_DIR:-$SCRIPT_DIR/opq_cache/dinov2_vitl14_depth}"
OPQ_ITER="${OPQ_ITER:-20}"
OPQ_KMEANS_ITER="${OPQ_KMEANS_ITER:-100}"
OPQ_MAX_TRAIN="${OPQ_MAX_TRAIN:-5000}"

# --- 并行 ---
GPUS="${GPUS:-0,1,2,3}"
LAYERS=(${LAYERS:-blk05 blk10 blk15 blk20})

LOG_DIR="$SCRIPT_DIR/logs/depth_eval"
mkdir -p "$LOG_DIR"

# ==================================================================

IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

echo "================================================================"
echo "  Depth Eval - Soft-PQ & OPQ (multi-GPU)"
echo "  GPUs:         ${GPUS} (${NUM_GPUS} devices)"
echo "  Layers:       ${LAYERS[*]}"
echo "  Methods:      ${METHODS}"
echo "  FEAT_ROOT:    ${FEAT_ROOT}"
echo "  DATA_ROOT:    ${DATA_ROOT}"
echo "  CKPT_DIR:     ${CKPT_DIR}"
echo "  CONFIG_JSON:  ${CONFIG_JSON}"
echo "  WEIGHTS_ROOT: ${WEIGHTS_ROOT}"
echo "  MODEL:        ${MODEL}, DIM=${DIM}, NORM=${NORM_MODE}"
echo "  OPQ_TRAIN:    ${OPQ_TRAIN_FEAT_ROOT}"
echo "  OPQ_CACHE:    ${OPQ_CACHE_DIR}"
echo "  OPQ params:   iter=${OPQ_ITER}, kmeans=${OPQ_KMEANS_ITER}, max_train=${OPQ_MAX_TRAIN}"
echo "  Logs:         ${LOG_DIR}/"
echo "================================================================"
echo ""

# --- 按 GPU 分组 layers ---
declare -A GPU_LAYERS
for i in "${!LAYERS[@]}"; do
  gpu_idx=$((i % NUM_GPUS))
  gpu_id=${GPU_LIST[$gpu_idx]}
  GPU_LAYERS[$gpu_id]+="${LAYERS[$i]} "
done

# --- 启动 ---
PIDS=()
GPU_IDS=()

for gpu_id in "${!GPU_LAYERS[@]}"; do
  layers_str="${GPU_LAYERS[$gpu_id]}"
  layers_str="${layers_str% }"
  log_name=$(echo "$layers_str" | tr ' ' '_')
  LOG="${LOG_DIR}/${log_name}.log"

  echo "[RUN]  GPU ${gpu_id}: ${layers_str} -> ${LOG}"

  CUDA_VISIBLE_DEVICES=$gpu_id python eval_depth_soft_pq.py \
    --feat_root      "$FEAT_ROOT" \
    --data_root      "$DATA_ROOT" \
    --split_file     "$SPLIT_FILE" \
    --ckpt_dir       "$CKPT_DIR" \
    --weights_root   "$WEIGHTS_ROOT" \
    --config_json    "$CONFIG_JSON" \
    --model          "$MODEL" \
    --dim            "$DIM" \
    --norm_mode      "$NORM_MODE" \
    --methods        $METHODS \
    --opq_train_feat_root "$OPQ_TRAIN_FEAT_ROOT" \
    --opq_cache_dir  "$OPQ_CACHE_DIR" \
    --opq_iter       "$OPQ_ITER" \
    --opq_kmeans_iter "$OPQ_KMEANS_ITER" \
    --opq_max_train  "$OPQ_MAX_TRAIN" \
    --layers         $layers_str \
    > "$LOG" 2>&1 &

  PIDS+=($!)
  GPU_IDS+=($gpu_id)
done

echo ""
echo "Launched ${#PIDS[@]} jobs. Waiting..."
echo ""

# --- 等待 ---
FAILURES=0
for i in "${!PIDS[@]}"; do
  PID=${PIDS[$i]}
  GPU=${GPU_IDS[$i]}
  layers_str="${GPU_LAYERS[$GPU]}"
  layers_str="${layers_str% }"
  if wait $PID; then
    echo "[OK]   GPU ${GPU}: ${layers_str} (pid=${PID})"
  else
    echo "[FAIL] GPU ${GPU}: ${layers_str} (pid=${PID}, exit=$?)"
    FAILURES=$((FAILURES + 1))
  fi
done

# --- 汇总 ---
echo ""
echo "================================================================"
if [ $FAILURES -eq 0 ]; then
  echo "  All ${#PIDS[@]} jobs completed successfully!"
else
  echo "  ${FAILURES}/${#PIDS[@]} jobs FAILED."
fi
echo "================================================================"
echo ""
echo "--- Summary ---"
echo ""
printf "%-8s %-8s %-16s %8s %8s %8s\n" "Layer" "Method" "Config" "BPFP" "RMSE" "dRMSE"
echo "--------------------------------------------------------------"

for layer in "${LAYERS[@]}"; do
  found=false
  for logfile in "$LOG_DIR"/*.log; do
    if grep -q "^${layer}" "$logfile" 2>/dev/null; then
      grep "^${layer}" "$logfile"
      found=true
      break
    fi
  done
  if [ "$found" = true ]; then
    echo ""
  fi
done

echo "================================================================"
echo "  Logs: ${LOG_DIR}/"
echo "================================================================"

exit $FAILURES
