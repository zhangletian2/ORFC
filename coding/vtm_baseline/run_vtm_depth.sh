#!/bin/bash
set -e

# ============================================================
#  VTM Feature Codec + DINOv2 Depth Evaluation Pipeline
#
#  Phase 1: VTM encode/decode ALL layers/QPs (CPU only, no GPU)
#  Phase 2: Load model → Anchor + Replay ALL (GPU)
#  Phase 3: Summary report (BPFP + RMSE + ΔRMSE)
#
#  Usage:  conda activate featcodec2 && bash run_vtm_depth.sh [GPU_ID]
#  Example: bash run_vtm_depth.sh 1    # use cuda:1
# ============================================================

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ─────────────── 路径配置 ───────────────
FEAT_ROOT=$PROJECT_ROOT/features/nyu_depth_80
DATA_ROOT=/data4/workspace/zlt/LaMoFC/Data_example/dinov2/dpt/source/NYU_Test80
SPLIT_FILE=/data4/workspace/zlt/LaMoFC/Data_example/dinov2/dpt/source/nyu_test_80.txt
WEIGHTS_ROOT=/data4/workspace/zlt/cache/torch/hub/checkpoints

# ─────────────── VTM 配置 ───────────────
VTM_ENCODER=$SCRIPT_DIR/EncoderAppStatic
VTM_DECODER=$SCRIPT_DIR/DecoderAppStatic
VTM_CFG=$SCRIPT_DIR/encoder_intra_vtm.cfg
TMP_DIR=$SCRIPT_DIR/_vtm_tmp_depth
BIT_DEPTH=10
VTM_WORKERS=16

# ─────────────── 实验配置 ───────────────
MODEL=dinov2_vitl14
LAYERS_SHALLOW=(blk05 blk10 blk15)
LAYERS_DEEP=(blk20)
QPS_SHALLOW=(22 25 27 30 32 35)
QPS_DEEP=(0 2 5 7 10 12)
DEVICE=cuda

RESULT_DIR=$FEAT_ROOT/${MODEL}/eval_vtm
mkdir -p $RESULT_DIR

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  VTM Feature Codec + DINOv2 NYU Depth Evaluation Pipeline    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model:      $MODEL"
echo "║  Shallow:    ${LAYERS_SHALLOW[*]}  QPs: ${QPS_SHALLOW[*]}"
echo "║  Deep:       ${LAYERS_DEEP[*]}          QPs: ${QPS_DEEP[*]}"
echo "║  GPU:        $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "║  Workers:    $VTM_WORKERS"
echo "║  Features:   $FEAT_ROOT/$MODEL"
echo "║  Data:       $DATA_ROOT"
echo "╚══════════════════════════════════════════════════════════════╝"

# ==================================================================
#  Phase 1: VTM encode/decode ALL layers/QPs (CPU only, no GPU)
# ==================================================================
echo ""
echo "========================================"
echo "  Phase 1: VTM 编解码（纯 CPU，不占显存）"
echo "========================================"

cd $SCRIPT_DIR

for layer in "${LAYERS_SHALLOW[@]}"; do
    echo "  >> $layer (QPs: ${QPS_SHALLOW[*]})"
    python vtm_baseline.py \
        --feat_root $FEAT_ROOT \
        --models $MODEL \
        --layers $layer \
        --qps "${QPS_SHALLOW[@]}" \
        --bit_depth $BIT_DEPTH \
        --vtm_encoder $VTM_ENCODER \
        --vtm_decoder $VTM_DECODER \
        --vtm_cfg $VTM_CFG \
        --tmp_dir $TMP_DIR \
        --workers $VTM_WORKERS
done

for layer in "${LAYERS_DEEP[@]}"; do
    echo "  >> $layer (QPs: ${QPS_DEEP[*]})"
    python vtm_baseline.py \
        --feat_root $FEAT_ROOT \
        --models $MODEL \
        --layers $layer \
        --qps "${QPS_DEEP[@]}" \
        --bit_depth $BIT_DEPTH \
        --vtm_encoder $VTM_ENCODER \
        --vtm_decoder $VTM_DECODER \
        --vtm_cfg $VTM_CFG \
        --tmp_dir $TMP_DIR \
        --workers $VTM_WORKERS
done

echo ""
echo "  ✓ Phase 1 完成: 全部 VTM 编解码结束"

# ==================================================================
#  Phase 2: Load model → Anchor + Replay ALL (GPU)
# ==================================================================
echo ""
echo "========================================"
echo "  Phase 2: 加载模型，Anchor + 回放评测"
echo "========================================"

cd $PROJECT_ROOT

# --- 2a: Anchor（回放原始无压缩特征） ---
ANCHOR_LOG=$RESULT_DIR/anchor.log
if [ -f "$ANCHOR_LOG" ]; then
    echo "  Anchor: 跳过（已有结果）"
else
    echo "  >> Anchor: 回放原始特征（blk05 blk10 blk15 blk20）"
    python tools/dinov2_depth_pipeline.py replay \
        --model vitl14 \
        --data_root $DATA_ROOT \
        --split $SPLIT_FILE \
        --weights_root $WEIGHTS_ROOT \
        --feature_root $FEAT_ROOT/$MODEL \
        --layer blk05 blk10 blk15 blk20 \
        --device $DEVICE \
        2>&1 | tee $ANCHOR_LOG
fi

# --- 2b: Shallow layers (blk05 blk10 blk15) per QP ---
for qp in "${QPS_SHALLOW[@]}"; do
    decoded_dir=$FEAT_ROOT/${MODEL}/decoded/vtm/${qp}
    log_file=$RESULT_DIR/shallow_qp${qp}.log

    if [ -f "$log_file" ]; then
        echo "  shallow/QP${qp}: 跳过（已有结果）"
        continue
    fi

    echo "  >> Replay: blk05 blk10 blk15 / QP${qp}"
    python tools/dinov2_depth_pipeline.py replay \
        --model vitl14 \
        --data_root $DATA_ROOT \
        --split $SPLIT_FILE \
        --weights_root $WEIGHTS_ROOT \
        --feature_root $decoded_dir \
        --layer blk05 blk10 blk15 \
        --org_feature_root $FEAT_ROOT/$MODEL \
        --device $DEVICE \
        2>&1 | tee $log_file
done

# --- 2c: Deep layers (blk20) per QP ---
for qp in "${QPS_DEEP[@]}"; do
    decoded_dir=$FEAT_ROOT/${MODEL}/decoded/vtm/${qp}
    log_file=$RESULT_DIR/deep_qp${qp}.log

    if [ -f "$log_file" ]; then
        echo "  blk20/QP${qp}: 跳过（已有结果）"
        continue
    fi

    echo "  >> Replay: blk20 / QP${qp}"
    python tools/dinov2_depth_pipeline.py replay \
        --model vitl14 \
        --data_root $DATA_ROOT \
        --split $SPLIT_FILE \
        --weights_root $WEIGHTS_ROOT \
        --feature_root $decoded_dir \
        --layer blk20 \
        --org_feature_root $FEAT_ROOT/$MODEL \
        --device $DEVICE \
        2>&1 | tee $log_file
done

echo ""
echo "  ✓ Phase 2 完成: 全部回放评测结束"

# ==================================================================
#  Phase 3: Summary report (RMSE + ΔRMSE only)
# ==================================================================
echo ""
echo "========================================"
echo "  Phase 3: 汇总结果"
echo "========================================"

python3 - <<'PYSCRIPT'
import os, csv, re
import numpy as np

project_root = "/data4/workspace/zlt/featcodec/ORFC"
feat_root = os.path.join(project_root, "features/nyu_depth_80")
model = "dinov2_vitl14"
result_dir = os.path.join(feat_root, model, "eval_vtm")

layers_shallow = ["blk05", "blk10", "blk15"]
layers_deep = ["blk20"]
qps_shallow = [17, 22, 25, 27, 30, 32, 35]
qps_deep = [0, 2, 5, 7, 10, 12]

def parse_metrics(log_path, layer):
    """从 replay log 中提取指定 layer 的 RMSE 和 FeatMSE"""
    if not os.path.exists(log_path):
        return {}
    with open(log_path, 'r') as f:
        content = f.read()
    metrics = {}
    # 匹配: blkXX    0.5443   0.1426   0.0602   0.8035   0.9672   0.9934
    pattern = rf'{layer}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    match = re.search(pattern, content)
    if match:
        metrics['rmse'] = float(match.group(1))
    # 匹配 FeatMSE (第7个数字)
    mse_pattern = rf'{layer}\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)'
    mse_match = re.search(mse_pattern, content)
    if mse_match:
        metrics['feat_mse'] = float(mse_match.group(1))
    return metrics

def get_bpfp(qp, layer):
    """从 VTM stats CSV 获取平均 BPFP"""
    stats_csv = os.path.join(feat_root, model, "decoded", "vtm", str(qp), "_stats.csv")
    if not os.path.exists(stats_csv):
        return None
    bpfps = []
    with open(stats_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("layer") == layer:
                bpfps.append(float(row["bpfp"]))
    return np.mean(bpfps) if bpfps else None

# ─── 解析 Anchor ───
anchor_log = os.path.join(result_dir, "anchor.log")
anchor_rmse = {}
for layer in layers_shallow + layers_deep:
    m = parse_metrics(anchor_log, layer)
    if 'rmse' in m:
        anchor_rmse[layer] = m['rmse']

# ─── 构建结果表 ───
all_configs = []
for layer in layers_shallow:
    for qp in qps_shallow:
        all_configs.append((layer, qp))
for layer in layers_deep:
    for qp in qps_deep:
        all_configs.append((layer, qp))

print()
print("=" * 60)
print("  DINOv2 ViT-L/14 NYU Depth V2: VTM Codec 评测汇总")
print("=" * 60)
print(f"{'Layer':<8} {'QP':>4} {'BPFP':>8} {'FeatMSE':>12} {'RMSE':>8} {'ΔRMSE':>8}")
print("-" * 56)

prev_layer = None
for layer, qp in all_configs:
    if prev_layer is not None and layer != prev_layer:
        print()
    prev_layer = layer

    if layer in layers_shallow:
        log_path = os.path.join(result_dir, f"shallow_qp{qp}.log")
    else:
        log_path = os.path.join(result_dir, f"deep_qp{qp}.log")

    metrics = parse_metrics(log_path, layer)
    bpfp = get_bpfp(qp, layer)

    bpfp_str = f"{bpfp:.4f}" if bpfp is not None else "-"
    rmse_str = f"{metrics['rmse']:.4f}" if 'rmse' in metrics else "-"
    mse_str = f"{metrics['feat_mse']:.6f}" if 'feat_mse' in metrics else "-"

    if 'rmse' in metrics and layer in anchor_rmse:
        d_rmse = metrics['rmse'] - anchor_rmse[layer]
        d_str = f"{d_rmse:+.4f}"
    else:
        d_str = "-"

    print(f"{layer:<8} {qp:>4} {bpfp_str:>8} {mse_str:>12} {rmse_str:>8} {d_str:>8}")

print("-" * 56)
if anchor_rmse:
    anchors = ", ".join(f"{l}={v:.4f}" for l, v in anchor_rmse.items())
    print(f"  Anchor (无压缩): {anchors}")
else:
    print("  Anchor: 未找到 anchor.log")
print("=" * 60)

PYSCRIPT

echo ""
echo "========================================"
echo "  Pipeline 完成!  结果: $RESULT_DIR"
echo "========================================"
