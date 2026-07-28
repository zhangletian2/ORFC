#!/bin/bash
set -e

# ============================================================
#  VTM Feature Codec + SigLIP2 Zero-shot Classification Pipeline
#
#  Phase 1: VTM encode/decode ALL layers/QPs (CPU only, no GPU)
#  Phase 2: Load SigLIP2 → Anchor + Replay ALL (GPU)
#  Phase 3: Summary report (BPFP + Acc@1 + Acc@5 + ΔAcc@1)
#
#  Usage:  conda activate siglip_codec && bash run_vtm_siglip2.sh [GPU_ID]
#  Example: bash run_vtm_siglip2.sh 0
# ============================================================

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ─────────────── 路径配置 ───────────────
FEAT_ROOT=$PROJECT_ROOT/features/test
LABEL_FILE=/data4/workspace/zlt/featcodec/utils/imagenet_selected_label500.txt
CLASSNAMES=$PROJECT_ROOT/utils/classnames.txt

# ─────────────── VTM 配置 ───────────────
VTM_ENCODER=$SCRIPT_DIR/EncoderAppStatic
VTM_DECODER=$SCRIPT_DIR/DecoderAppStatic
VTM_CFG=$SCRIPT_DIR/encoder_intra_vtm.cfg
TMP_DIR=$SCRIPT_DIR/_vtm_tmp_siglip2
BIT_DEPTH=10
VTM_WORKERS=16

# ─────────────── 实验配置 ───────────────
MODEL=siglip2_so400m
LAYERS_SHALLOW=(blk07)
LAYERS_DEEP=(blk15 blk23)
QPS_SHALLOW=(22 25 27 30 32)
QPS_DEEP=(0 2 5 7 10 12)
DEVICE=cuda

RESULT_DIR=$FEAT_ROOT/${MODEL}/eval_vtm
mkdir -p $RESULT_DIR

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  VTM Feature Codec + SigLIP2 Zero-shot Classification       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model:      $MODEL"
echo "║  Shallow:    ${LAYERS_SHALLOW[*]}  QPs: ${QPS_SHALLOW[*]}"
echo "║  Deep:       ${LAYERS_DEEP[*]}  QPs: ${QPS_DEEP[*]}"
echo "║  GPU:        $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "║  Workers:    $VTM_WORKERS"
echo "║  Features:   $FEAT_ROOT/$MODEL"
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
#  Phase 2: Load SigLIP2 → Anchor + Replay ALL (GPU)
# ==================================================================
echo ""
echo "========================================"
echo "  Phase 2: 加载 SigLIP2，Anchor + 回放评测"
echo "========================================"

cd $PROJECT_ROOT

# --- 2a: Anchor（回放原始无压缩特征） ---
ANCHOR_LOG=$RESULT_DIR/anchor.log
if [ -f "$ANCHOR_LOG" ]; then
    echo "  Anchor: 跳过（已有结果）"
else
    echo "  >> Anchor: 回放原始特征（blk07 blk15 blk23）"
    python tools/siglip2_feat_pipeline.py replay \
        --feature_root $FEAT_ROOT/$MODEL \
        --layer blk07 blk15 blk23 \
        --labels $LABEL_FILE \
        --classnames $CLASSNAMES \
        --device $DEVICE \
        2>&1 | tee $ANCHOR_LOG
fi

# --- 2b: Shallow layers (blk07) per QP ---
for qp in "${QPS_SHALLOW[@]}"; do
    decoded_dir=$FEAT_ROOT/${MODEL}/decoded/vtm/${qp}
    log_file=$RESULT_DIR/shallow_qp${qp}.log

    if [ -f "$log_file" ]; then
        echo "  shallow/QP${qp}: 跳过（已有结果）"
        continue
    fi

    echo "  >> Replay: ${LAYERS_SHALLOW[*]} / QP${qp}"
    python tools/siglip2_feat_pipeline.py replay \
        --feature_root $decoded_dir \
        --layer "${LAYERS_SHALLOW[@]}" \
        --labels $LABEL_FILE \
        --classnames $CLASSNAMES \
        --device $DEVICE \
        2>&1 | tee $log_file
done

# --- 2c: Deep layers (blk15 blk23) per QP ---
for qp in "${QPS_DEEP[@]}"; do
    decoded_dir=$FEAT_ROOT/${MODEL}/decoded/vtm/${qp}
    log_file=$RESULT_DIR/deep_qp${qp}.log

    if [ -f "$log_file" ]; then
        echo "  deep/QP${qp}: 跳过（已有结果）"
        continue
    fi

    echo "  >> Replay: ${LAYERS_DEEP[*]} / QP${qp}"
    python tools/siglip2_feat_pipeline.py replay \
        --feature_root $decoded_dir \
        --layer "${LAYERS_DEEP[@]}" \
        --labels $LABEL_FILE \
        --classnames $CLASSNAMES \
        --device $DEVICE \
        2>&1 | tee $log_file
done

echo ""
echo "  ✓ Phase 2 完成: 全部回放评测结束"

# ==================================================================
#  Phase 3: Summary report (BPFP + Acc@1 + Acc@5 + ΔAcc@1)
# ==================================================================
echo ""
echo "========================================"
echo "  Phase 3: 汇总结果"
echo "========================================"

python3 - <<'PYSCRIPT'
import os, csv, re
import numpy as np

project_root = os.environ.get("PROJECT_ROOT", "/data4/workspace/zlt/featcodec/ORFC")
feat_root = os.path.join(project_root, "features", "test")
model = "siglip2_so400m"
result_dir = os.path.join(feat_root, model, "eval_vtm")

layers_shallow = ["blk07"]
layers_deep = ["blk15", "blk23"]
qps_shallow = [22, 25, 27, 30, 32]
qps_deep = [0, 2, 5, 7, 10, 12]

def parse_acc(log_path, layer):
    if not os.path.exists(log_path):
        return {}
    with open(log_path, 'r') as f:
        content = f.read()
    # siglip2_feat_pipeline.py replay 输出格式:
    # blk07		70.00%		90.00%		1.23s
    pattern = rf'{layer}\s+([\d.]+)%\s+([\d.]+)%'
    match = re.search(pattern, content)
    if match:
        return {'acc1': float(match.group(1)), 'acc5': float(match.group(2))}
    return {}

def get_bpfp(qp, layer):
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

# Anchor
anchor_log = os.path.join(result_dir, "anchor.log")
anchor = {}
for layer in layers_shallow + layers_deep:
    m = parse_acc(anchor_log, layer)
    if m:
        anchor[layer] = m

# Results
all_configs = []
for layer in layers_shallow:
    for qp in qps_shallow:
        all_configs.append((layer, qp))
for layer in layers_deep:
    for qp in qps_deep:
        all_configs.append((layer, qp))

print()
print("=" * 72)
print("  SigLIP2 So400m/14 ImageNet: VTM Codec 评测汇总")
print("=" * 72)
print(f"{'Layer':<8} {'QP':>4} {'BPFP':>8} {'Acc@1':>8} {'Acc@5':>8} {'ΔAcc@1':>8}")
print("-" * 48)

prev_layer = None
for layer, qp in all_configs:
    if prev_layer is not None and layer != prev_layer:
        print()
    prev_layer = layer

    if layer in layers_shallow:
        log_path = os.path.join(result_dir, f"shallow_qp{qp}.log")
    else:
        log_path = os.path.join(result_dir, f"deep_qp{qp}.log")

    metrics = parse_acc(log_path, layer)
    bpfp = get_bpfp(qp, layer)

    bpfp_str = f"{bpfp:.4f}" if bpfp is not None else "-"
    acc1_str = f"{metrics['acc1']:.2f}%" if 'acc1' in metrics else "-"
    acc5_str = f"{metrics['acc5']:.2f}%" if 'acc5' in metrics else "-"

    if 'acc1' in metrics and layer in anchor:
        d_acc1 = metrics['acc1'] - anchor[layer]['acc1']
        d_str = f"{d_acc1:+.2f}%"
    else:
        d_str = "-"

    print(f"{layer:<8} {qp:>4} {bpfp_str:>8} {acc1_str:>8} {acc5_str:>8} {d_str:>8}")

print("-" * 48)
if anchor:
    anchors = ", ".join(f"{l}: Acc@1={v['acc1']:.2f}% Acc@5={v['acc5']:.2f}%" for l, v in anchor.items())
    print(f"  Anchor (无压缩): {anchors}")
else:
    print("  Anchor: 未找到 anchor.log")
print("=" * 72)

PYSCRIPT

echo ""
echo "========================================"
echo "  Pipeline 完成!  结果: $RESULT_DIR"
echo "========================================"
