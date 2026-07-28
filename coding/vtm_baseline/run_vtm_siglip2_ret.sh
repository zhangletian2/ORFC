#!/bin/bash
set -e

# ============================================================
#  VTM Feature Codec + SigLIP2 COCO Retrieval Pipeline
#
#  Phase 1: VTM encode/decode ALL layers/QPs (CPU only, no GPU)
#  Phase 2: Load SigLIP2 → Anchor + Replay ALL → 检索评估 (GPU)
#  Phase 3: Summary report (BPFP + I2T/T2I R@K)
#
#  Usage:  conda activate siglip_codec && bash run_vtm_siglip2_ret.sh [GPU_ID]
#  Example: bash run_vtm_siglip2_ret.sh 0
# ============================================================

GPU_ID=${1:-7}
export CUDA_VISIBLE_DEVICES=$GPU_ID
export HF_HUB_OFFLINE=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ─────────────── 路径配置 ───────────────
FEAT_ROOT=$PROJECT_ROOT/features/coco_ret
MODEL=siglip2_so400m
CAPTION_JSON=$PROJECT_ROOT/utils/coco_selected_caption500.json
MODEL_ID="google/siglip2-so400m-patch14-224"

# ─────────────── VTM 配置 ───────────────
VTM_ENCODER=$SCRIPT_DIR/EncoderAppStatic
VTM_DECODER=$SCRIPT_DIR/DecoderAppStatic
VTM_CFG=$SCRIPT_DIR/encoder_intra_vtm.cfg
TMP_DIR=$SCRIPT_DIR/_vtm_tmp_siglip2_ret
BIT_DEPTH=10
VTM_WORKERS=15

# ─────────────── 实验配置 ───────────────
LAYERS_SHALLOW=(blk07)
LAYERS_DEEP=(blk15 blk23)
QPS_SHALLOW=(32 30 27 25 22)
QPS_DEEP=(10 7 5 2 0)
DEVICE=cuda

RESULT_DIR=$FEAT_ROOT/${MODEL}/eval_vtm_ret
mkdir -p $RESULT_DIR

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  VTM Feature Codec + SigLIP2 COCO Retrieval                 ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model:      $MODEL"
echo "║  Shallow:    ${LAYERS_SHALLOW[*]}  QPs: ${QPS_SHALLOW[*]}"
echo "║  Deep:       ${LAYERS_DEEP[*]}  QPs: ${QPS_DEEP[*]}"
echo "║  GPU:        $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "║  Workers:    $VTM_WORKERS"
echo "║  Features:   $FEAT_ROOT/$MODEL"
echo "║  Caption:    $CAPTION_JSON"
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
echo "  Phase 1 完成: 全部 VTM 编解码结束"

# ==================================================================
#  Phase 2: Load SigLIP2 → Anchor + Replay ALL → 检索评估 (GPU)
# ==================================================================
echo ""
echo "========================================"
echo "  Phase 2: 加载 SigLIP2，Anchor + 回放检索评测"
echo "========================================"

cd $PROJECT_ROOT

# --- 2a: Anchor（回放原始无压缩特征） ---
ANCHOR_LOG=$RESULT_DIR/anchor.log
if [ -f "$ANCHOR_LOG" ]; then
    echo "  Anchor: 跳过（已有结果）"
else
    echo "  >> Anchor: 回放原始特征 (blk07 blk15 blk23)"
    python tools/siglip2_retrieval.py replay \
        --model_id   $MODEL_ID \
        --meta_json  $CAPTION_JSON \
        --feature_root $FEAT_ROOT/$MODEL \
        --layers     blk07 blk15 blk23 \
        --text_batch_size 128 \
        --output     $RESULT_DIR/results_anchor.json \
        --device     $DEVICE \
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
    python tools/siglip2_retrieval.py replay \
        --model_id   $MODEL_ID \
        --meta_json  $CAPTION_JSON \
        --feature_root $decoded_dir \
        --layers     "${LAYERS_SHALLOW[@]}" \
        --text_batch_size 128 \
        --output     $RESULT_DIR/results_shallow_qp${qp}.json \
        --device     $DEVICE \
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
    python tools/siglip2_retrieval.py replay \
        --model_id   $MODEL_ID \
        --meta_json  $CAPTION_JSON \
        --feature_root $decoded_dir \
        --layers     "${LAYERS_DEEP[@]}" \
        --text_batch_size 128 \
        --output     $RESULT_DIR/results_deep_qp${qp}.json \
        --device     $DEVICE \
        2>&1 | tee $log_file
done

echo ""
echo "  Phase 2 完成: 全部回放检索评测结束"

# ==================================================================
#  Phase 3: Summary report
# ==================================================================
echo ""
echo "========================================"
echo "  Phase 3: 汇总结果"
echo "========================================"

export PROJECT_ROOT FEAT_ROOT MODEL RESULT_DIR

python3 - <<'PYSCRIPT'
import os, csv, json
import numpy as np

feat_root = os.environ["FEAT_ROOT"]
model = os.environ["MODEL"]
result_dir = os.environ["RESULT_DIR"]

layers_shallow = ["blk07"]
layers_deep = ["blk15", "blk23"]
qps_shallow = [32, 30, 27, 25, 22]
qps_deep = [10, 7, 5, 2, 0]

def load_ret_json(json_path, layer):
    if not os.path.exists(json_path):
        return {}
    with open(json_path) as f:
        data = json.load(f)
    if layer in data:
        return data[layer]
    return data

def get_bpfp(qp, layer):
    stats_csv = os.path.join(feat_root, model, "decoded", "vtm", str(qp), "_stats.csv")
    if not os.path.exists(stats_csv):
        return None
    bpfps = []
    with open(stats_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("layer") == layer:
                bpfps.append(float(row["bpfp"]))
    return np.mean(bpfps) if bpfps else None

# Anchor
anchor = {}
anchor_json = os.path.join(result_dir, "results_anchor.json")
if os.path.exists(anchor_json):
    with open(anchor_json) as f:
        anchor = json.load(f)

# Collect all
all_configs = []
for l in layers_shallow:
    for q in qps_shallow:
        all_configs.append((l, q, "shallow"))
for l in layers_deep:
    for q in qps_deep:
        all_configs.append((l, q, "deep"))

print()
print("=" * 110)
print("  SigLIP2 So400m/14 COCO Retrieval: VTM Codec 全量评测汇总")
print("=" * 110)

print(f"{'Layer':<8} {'QP':>4} {'BPFP':>8}  "
      f"{'i2t_R@1':>8} {'i2t_R@5':>8} {'i2t_R@10':>9}  "
      f"{'t2i_R@1':>8} {'t2i_R@5':>8} {'t2i_R@10':>9}  "
      f"{'Di2t_R1':>8}")
print("-" * 95)

# Print anchor rows
for layer in layers_shallow + layers_deep:
    a = anchor.get(layer, {})
    if a:
        print(f"{layer:<8} {'anc':>4} {'---':>8}  "
              f"{a.get('i2t_R@1',0):>7.2f}% {a.get('i2t_R@5',0):>7.2f}% "
              f"{a.get('i2t_R@10',0):>8.2f}%  "
              f"{a.get('t2i_R@1',0):>7.2f}% {a.get('t2i_R@5',0):>7.2f}% "
              f"{a.get('t2i_R@10',0):>8.2f}%  {'---':>8}")
print()

prev_layer = None
for layer, qp, kind in all_configs:
    if prev_layer is not None and layer != prev_layer:
        print()
    prev_layer = layer

    json_path = os.path.join(result_dir, f"results_{kind}_qp{qp}.json")
    m = load_ret_json(json_path, layer)
    bpfp = get_bpfp(qp, layer)

    bpfp_s = f"{bpfp:.4f}" if bpfp is not None else "---"

    if m:
        a = anchor.get(layer, {})
        di2t = m.get("i2t_R@1", 0) - a.get("i2t_R@1", 0) if a else 0
        print(f"{layer:<8} {qp:>4} {bpfp_s:>8}  "
              f"{m.get('i2t_R@1',0):>7.2f}% {m.get('i2t_R@5',0):>7.2f}% "
              f"{m.get('i2t_R@10',0):>8.2f}%  "
              f"{m.get('t2i_R@1',0):>7.2f}% {m.get('t2i_R@5',0):>7.2f}% "
              f"{m.get('t2i_R@10',0):>8.2f}%  "
              f"{di2t:>+7.2f}%")
    else:
        print(f"{layer:<8} {qp:>4} {bpfp_s:>8}  {'---':>8} {'---':>8} {'---':>9}  "
              f"{'---':>8} {'---':>8} {'---':>9}  {'---':>8}")

print("-" * 95)
if anchor:
    for layer in layers_shallow + layers_deep:
        a = anchor.get(layer, {})
        if a:
            print(f"  Anchor {layer}: i2t_R@1={a.get('i2t_R@1',0):.2f}% "
                  f"i2t_R@5={a.get('i2t_R@5',0):.2f}% "
                  f"i2t_R@10={a.get('i2t_R@10',0):.2f}%")
print("=" * 110)

PYSCRIPT

echo ""
echo "========================================"
echo "  Pipeline 完成!  结果: $RESULT_DIR"
echo "========================================"
