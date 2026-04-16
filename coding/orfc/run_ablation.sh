#!/bin/bash
# ================================================================
# DOPQ Component Ablation — blk20, DINOv2 ViT-L/14
#
# 消融表 (DETR-style):
#   Row (a) OPQ baseline       — 已有 (附带在每次实验输出中)
#   Row (b) L_ref → MSE        — 需补 3 个码率点 (K16/d32 已有)
#   Row (c) − Train R (fzR)    — 需补 3 个码率点 (K16/d32 已有)
#   Row (d) − Train C (fzC)    — 需补 4 个码率点
#   Row (e) − Soft PQ (τ=0)    — 需补 4 个码率点
#   Row (f) Full DOPQ          — 已有全部 4 个码率点
#
# 码率点: K=8/d=32, K=16/d=32, K=64/d=16, K=256/d=16
# 对应 max BPFP ≈ 0.094, 0.125, 0.375, 0.500
#
# 共 14 个新实验, 4 轮 × 4 GPU 并行
# 预计总耗时 ~3-4 小时
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PY=python
SC="$SCRIPT_DIR/run_soft_pq.py"

LOGDIR="$SCRIPT_DIR/logs/logs_ablation"
mkdir -p "$LOGDIR"

BASE="--layer blk20 --bottleneck_dim 1024 --warm_start_opq \
  --lr 0.0003 --epochs 100 --max_train_images 5000 \
  --batch_size 32 --seed 42 --eval_seg --lmbda 0.5"

SOFT="--tau_start 0.5 --tau_end 0.005"

echo "=============================================="
echo " DOPQ Ablation: 14 experiments on GPUs 0-3"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

# ---- Round 1/4: freeze-R ×3 + freeze-C ×1 ----
echo "[$(date '+%H:%M:%S')] Round 1/4: freeze-R ×3 + freeze-C ×1"

CUDA_VISIBLE_DEVICES=0 $PY $SC $BASE $SOFT \
  --K 8   --embedding_dim 32 --freeze_transform \
  >"$LOGDIR/fzR_K8d32.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 $PY $SC $BASE $SOFT \
  --K 64  --embedding_dim 16 --freeze_transform \
  >"$LOGDIR/fzR_K64d16.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 $PY $SC $BASE $SOFT \
  --K 256 --embedding_dim 16 --freeze_transform \
  >"$LOGDIR/fzR_K256d16.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 $PY $SC $BASE $SOFT \
  --K 8   --embedding_dim 32 --freeze_codebooks \
  >"$LOGDIR/fzC_K8d32.log" 2>&1 &

wait
echo "[$(date '+%H:%M:%S')] Round 1 done."

# ---- Round 2/4: freeze-C ×3 + hard-PQ ×1 ----
echo "[$(date '+%H:%M:%S')] Round 2/4: freeze-C ×3 + hard-PQ ×1"

CUDA_VISIBLE_DEVICES=0 $PY $SC $BASE $SOFT \
  --K 16  --embedding_dim 32 --freeze_codebooks \
  >"$LOGDIR/fzC_K16d32.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 $PY $SC $BASE $SOFT \
  --K 64  --embedding_dim 16 --freeze_codebooks \
  >"$LOGDIR/fzC_K64d16.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 $PY $SC $BASE $SOFT \
  --K 256 --embedding_dim 16 --freeze_codebooks \
  >"$LOGDIR/fzC_K256d16.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 $PY $SC $BASE \
  --K 8   --embedding_dim 32 --tau_start 0 \
  >"$LOGDIR/hard_K8d32.log" 2>&1 &

wait
echo "[$(date '+%H:%M:%S')] Round 2 done."

# ---- Round 3/4: hard-PQ ×3 + MSE ×1 ----
echo "[$(date '+%H:%M:%S')] Round 3/4: hard-PQ ×3 + MSE ×1"

CUDA_VISIBLE_DEVICES=0 $PY $SC $BASE \
  --K 16  --embedding_dim 32 --tau_start 0 \
  >"$LOGDIR/hard_K16d32.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 $PY $SC $BASE \
  --K 64  --embedding_dim 16 --tau_start 0 \
  >"$LOGDIR/hard_K64d16.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 $PY $SC $BASE \
  --K 256 --embedding_dim 16 --tau_start 0 \
  >"$LOGDIR/hard_K256d16.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 $PY $SC $BASE $SOFT \
  --K 8   --embedding_dim 32 --mse_loss \
  >"$LOGDIR/mse_K8d32.log" 2>&1 &

wait
echo "[$(date '+%H:%M:%S')] Round 3 done."

# ---- Round 4/4: MSE ×2 ----
echo "[$(date '+%H:%M:%S')] Round 4/4: MSE ×2"

CUDA_VISIBLE_DEVICES=0 $PY $SC $BASE $SOFT \
  --K 64  --embedding_dim 16 --mse_loss \
  >"$LOGDIR/mse_K64d16.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 $PY $SC $BASE $SOFT \
  --K 256 --embedding_dim 16 --mse_loss \
  >"$LOGDIR/mse_K256d16.log" 2>&1 &

wait
echo "[$(date '+%H:%M:%S')] Round 4 done."

echo "=============================================="
echo " All 14 ablation experiments complete!"
echo " Results: coding/vq/v3.4/results/soft_pq/dinov2_vitl14/"
echo " Logs:    $LOGDIR/"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
