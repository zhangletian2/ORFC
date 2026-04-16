#!/bin/bash
# ================================================================
#  Unified intro-figure analysis for DOPQ paper (v2)
#
#  Combines gradient-based sensitivity diagnostics (diagnose_subspace)
#  with OPQ correlation sweep (analysis_intro) into one coherent pipeline.
#
#  Figure 2 (sensitivity balance):  ~30-60 min
#    Identity / OPQ / Trained gradient sensitivity + ablation
#  Figure 3 (signal divergence):    same run as Figure 2
#    Cross-metric correlations (sens_lref vs sens_mse, mse vs ablation)
#  Figure 1 (correlation):          ~30-60 min
#    OPQ sweep across 10 (K, emb) configs + load DOPQ results + mIoU
#
#  GPU: single GPU, ~6-8 GB VRAM
# ================================================================
set -e

PYTHON=python
SCRIPT="analysis_intro_v2.py"
GPU="${1:-5}"

# ── Configuration ──
LAYER="blk20"
K=16
EMB=32
CKPT="checkpoints/dinov2_vitl14/blk20_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"

LOG_DIR="logs_analysis_v2/${LAYER}_K${K}_emb${EMB}"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "  DOPQ Intro Figure Analysis (v2 — unified)"
echo "  Layer: $LAYER   K: $K   emb: $EMB   GPU: $GPU"
echo "  Codec: $CKPT"
echo "  $(date)"
echo "================================================================"

# ──────────────────────────────────────────────────────────────────
#  Step 1: Sensitivity analysis (produces data for Figure 2 + 3)
#
#  Builds 3 codecs (Identity, OPQ, Trained from checkpoint):
#    - variance, quantization MSE per group
#    - gradient sensitivity (L_ref and MSE) per group
#    - ablation delta-L_ref per group
#
#  Key outputs:
#    - sens_lref CV: Identity >> OPQ > Trained  (Figure 2)
#    - r(sens_lref, sens_mse) ~ 0              (Figure 3)
#    - r(mse, ablation) < 0 for Trained        (Figure 3)
# ──────────────────────────────────────────────────────────────────
echo ""
echo "  [Step 1] Sensitivity analysis (Figure 2 + 3)..."
echo "  Loading trained codec from checkpoint (no retraining)"
echo ""

CUDA_VISIBLE_DEVICES=$GPU $PYTHON $SCRIPT \
    --mode sensitivity \
    --layer "$LAYER" --K $K --embedding_dim $EMB \
    --codec_path "$CKPT" \
    --n_diag 200 \
    --max_train_images 5000 \
    --batch_size 8 --seed 42 \
    2>&1 | tee "$LOG_DIR/fig23_sensitivity.log"

# ──────────────────────────────────────────────────────────────────
#  Step 2: Correlation analysis (produces data for Figure 1)
#
#  Sweeps OPQ across 10 (K, emb) configs:
#    (4,32) (8,32) (16,32) (32,32) (64,32)
#    (256,32) (64,16) (256,16) (64,8) (256,8)
#
#  For each: measures MSE, L_ref, Acc, mIoU on test set.
#  Also loads existing DOPQ results (with mIoU) from results/soft_pq/.
# ──────────────────────────────────────────────────────────────────
echo ""
echo "  [Step 2] Correlation analysis (Figure 1) with segmentation..."
echo ""

CUDA_VISIBLE_DEVICES=$GPU $PYTHON $SCRIPT \
    --mode correlation \
    --layer "$LAYER" \
    --eval_seg \
    --batch_size 16 \
    --max_train_images 5000 --seed 42 \
    2>&1 | tee "$LOG_DIR/fig1_correlation.log"

echo ""
echo "================================================================"
echo "  All done at $(date)"
echo "  Results in: results/analysis_intro_v2/"
echo "  Logs in:    $LOG_DIR/"
echo ""
echo "  Output files:"
echo "    results/analysis_intro_v2/sensitivity_${LAYER}_K${K}_emb${EMB}_*.json"
echo "    results/analysis_intro_v2/correlation_${LAYER}.json"
echo "================================================================"
