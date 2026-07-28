#!/usr/bin/env bash
set -euo pipefail
# ============================================================
#  Qwen3-VL VTM Codec quick verification
#  5 samples x N QPs
#
#  Usage:
#    conda activate qwen3vl_codec && bash verify_vtm_qwen3vl.sh [GPU_ID]
# ============================================================

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- paths ---
ORIG_FEAT_DIR=$PROJECT_ROOT/features/mmbench_en_val_1000/qwen3vl_4b/blk05
WORK_DIR=$SCRIPT_DIR/_qwen3vl_verify
FEAT_ROOT=$WORK_DIR/feat

MODEL_PATH="/data4/workspace/zlt/cache/hf/hub/Qwen3-VL-4B-Thinking"
DATA_DIR=$PROJECT_ROOT/data/MMBench
SAMPLE_LIST=$PROJECT_ROOT/utils/mmbench_en_val_1000.txt

# --- VTM ---
VTM_ENCODER=$SCRIPT_DIR/EncoderAppStatic
VTM_DECODER=$SCRIPT_DIR/DecoderAppStatic
VTM_CFG=$SCRIPT_DIR/encoder_intra_vtm.cfg
TMP_DIR=$WORK_DIR/_vtm_tmp
BIT_DEPTH=10
VTM_WORKERS=6

# --- experiment ---
MODEL_NAME=qwen3vl_4b
LAYER_NAME=blk05
LAYER_IDX=5
QPS=(12 17 22 25 27 30 32)
N_VERIFY=5

PYTHON="/home/user/anaconda3/envs/qwen3vl_codec/bin/python"

echo ""
echo "========================================================"
echo "  Qwen3-VL VTM Codec Verification"
echo "  Samples:  $N_VERIFY"
echo "  QPs:      ${QPS[*]}"
echo "  Layer:    block[$LAYER_IDX] ($LAYER_NAME)"
echo "  GPU:      $GPU_ID"
echo "  Features: $ORIG_FEAT_DIR"
echo "========================================================"

T_START=$SECONDS

# ==================================================================
#  Phase 0: copy N samples to verify workspace
# ==================================================================
echo ""
echo "---- Phase 0: select $N_VERIFY samples ----"

mkdir -p "$FEAT_ROOT/$MODEL_NAME/$LAYER_NAME"

$PYTHON - "$ORIG_FEAT_DIR" "$FEAT_ROOT/$MODEL_NAME/$LAYER_NAME" "$N_VERIFY" <<'PYSCRIPT'
import numpy as np, os, sys, shutil
src_dir, dst_dir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
files = sorted(f for f in os.listdir(src_dir) if f.endswith(".npy"))[:n]
for f in files:
    shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
    feat = np.load(os.path.join(dst_dir, f))
    stem = f.replace(".npy", "")
    print("  %s: shape=%s  range=[%.3f, %.3f]" % (stem, feat.shape, feat.min(), feat.max()))
PYSCRIPT

echo "  done: $N_VERIFY samples ready"

# ==================================================================
#  Phase 1: VTM encode/decode (CPU only)
# ==================================================================
echo ""
echo "---- Phase 1: VTM encode/decode (QPs: ${QPS[*]}) ----"

mkdir -p "$TMP_DIR"

$PYTHON "$SCRIPT_DIR/vtm_baseline.py" \
    --feat_root "$FEAT_ROOT" \
    --models "$MODEL_NAME" \
    --layers "$LAYER_NAME" \
    --qps "${QPS[@]}" \
    --bit_depth "$BIT_DEPTH" \
    --vtm_encoder "$VTM_ENCODER" \
    --vtm_decoder "$VTM_DECODER" \
    --vtm_cfg "$VTM_CFG" \
    --tmp_dir "$TMP_DIR" \
    --workers "$VTM_WORKERS"

echo ""
echo "  done: VTM encode/decode finished"

# ==================================================================
#  Phase 2: replay evaluation (baseline + anchor + per-QP)
# ==================================================================
echo ""
echo "---- Phase 2: model load + replay evaluation ----"

$PYTHON "$SCRIPT_DIR/verify_qwen3vl_eval.py" \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --orig_feat_dir "$FEAT_ROOT/$MODEL_NAME/$LAYER_NAME" \
    --decoded_root "$FEAT_ROOT/$MODEL_NAME/decoded/vtm" \
    --stats_root "$FEAT_ROOT/$MODEL_NAME/decoded/vtm" \
    --qps "${QPS[@]}" \
    --layer "$LAYER_IDX" \
    --output "$WORK_DIR/verify_results.json" \
    --no_think \
    --max_new_tokens 2048

T_TOTAL=$((SECONDS - T_START))
echo ""
echo "========================================================"
echo "  Verification done!  Total: ${T_TOTAL}s"
echo "  Results: $WORK_DIR/verify_results.json"
echo "========================================================"
