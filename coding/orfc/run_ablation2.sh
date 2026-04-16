#!/bin/bash
# ================================================================
# DOPQ 补充消融 — blk20, DINOv2 ViT-L/14
#
#   (e') − Soft PQ (τ=0, λ=0)  — 4 个码率点 (新实验, 纯 Hard PQ 无 ECVQ)
#   (f)  Full DOPQ 重跑         — 4 个码率点 (修复 ckpt 命名 + VOC rate)
#   (c)  − Train R (fzR) 重跑   — 4 个码率点 (修复 ckpt 命名 + VOC rate)
#
# 码率点: K=8/d=32, K=16/d=32, K=64/d=16, K=256/d=16
# 共 12 个实验, 3 轮 × 4 GPU 并行
# 预计总耗时 ~2-3 小时
#
# 之后运行 eval_voc_rate.py 补全 fzC/MSE/HardPQ 的 VOC rate
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PY=python
SC="$SCRIPT_DIR/run_soft_pq.py"
EVAL_VOC="$SCRIPT_DIR/eval_voc_rate.py"

LOGDIR="$SCRIPT_DIR/logs/logs_ablation2"
mkdir -p "$LOGDIR"

BASE="--layer blk20 --bottleneck_dim 1024 --warm_start_opq \
  --lr 0.0003 --epochs 100 --max_train_images 5000 \
  --batch_size 32 --seed 42 --eval_seg"

SOFT="--tau_start 0.5 --tau_end 0.005"

echo "=============================================="
echo " DOPQ Ablation-2: 12 experiments on GPUs 0-3"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

# ---- Round 1/3: Hard PQ λ=0 (4 rate points) ----
echo "[$(date '+%H:%M:%S')] Round 1/3: Hard PQ λ=0 ×4"

CUDA_VISIBLE_DEVICES=0 $PY $SC $BASE \
  --K 8   --embedding_dim 32 --tau_start 0 --lmbda 0 \
  >"$LOGDIR/hard_lm0_K8d32.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 $PY $SC $BASE \
  --K 16  --embedding_dim 32 --tau_start 0 --lmbda 0 \
  >"$LOGDIR/hard_lm0_K16d32.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 $PY $SC $BASE \
  --K 64  --embedding_dim 16 --tau_start 0 --lmbda 0 \
  >"$LOGDIR/hard_lm0_K64d16.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 $PY $SC $BASE \
  --K 256 --embedding_dim 16 --tau_start 0 --lmbda 0 \
  >"$LOGDIR/hard_lm0_K256d16.log" 2>&1 &

wait
echo "[$(date '+%H:%M:%S')] Round 1 done."

# ---- Round 2/3: Full DOPQ 重跑 (4 rate points, ckpt 修复 + VOC rate) ----
echo "[$(date '+%H:%M:%S')] Round 2/3: Full DOPQ ×4 (re-run)"

CUDA_VISIBLE_DEVICES=0 $PY $SC $BASE $SOFT --lmbda 0.5 \
  --K 8   --embedding_dim 32 \
  >"$LOGDIR/full_K8d32.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 $PY $SC $BASE $SOFT --lmbda 0.5 \
  --K 16  --embedding_dim 32 \
  >"$LOGDIR/full_K16d32.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 $PY $SC $BASE $SOFT --lmbda 0.5 \
  --K 64  --embedding_dim 16 \
  >"$LOGDIR/full_K64d16.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 $PY $SC $BASE $SOFT --lmbda 0.5 \
  --K 256 --embedding_dim 16 \
  >"$LOGDIR/full_K256d16.log" 2>&1 &

wait
echo "[$(date '+%H:%M:%S')] Round 2 done."

# ---- Round 3/3: fzR 重跑 (4 rate points, ckpt 修复 + VOC rate) ----
echo "[$(date '+%H:%M:%S')] Round 3/3: fzR ×4 (re-run)"

CUDA_VISIBLE_DEVICES=0 $PY $SC $BASE $SOFT --lmbda 0.5 \
  --K 8   --embedding_dim 32 --freeze_transform \
  >"$LOGDIR/fzR_K8d32.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 $PY $SC $BASE $SOFT --lmbda 0.5 \
  --K 16  --embedding_dim 32 --freeze_transform \
  >"$LOGDIR/fzR_K16d32.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 $PY $SC $BASE $SOFT --lmbda 0.5 \
  --K 64  --embedding_dim 16 --freeze_transform \
  >"$LOGDIR/fzR_K64d16.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 $PY $SC $BASE $SOFT --lmbda 0.5 \
  --K 256 --embedding_dim 16 --freeze_transform \
  >"$LOGDIR/fzR_K256d16.log" 2>&1 &

wait
echo "[$(date '+%H:%M:%S')] Round 3 done."

echo ""
echo "=============================================="
echo " 12 experiments done. Now retroactively adding"
echo " VOC rate to fzC / MSE / HardPQ(λ=0.5) ..."
echo "=============================================="

# Retroactively add VOC rate to existing results that have checkpoints
# fzC: checkpoint at old naming path (without _fzC) — works via fallback
# MSE: checkpoint has _mse tag — works directly
# HardPQ(λ=0.5): checkpoint has no tau tag — works directly
CUDA_VISIBLE_DEVICES=0 $PY $EVAL_VOC --gpu 0 --filter blk20 \
  >"$LOGDIR/eval_voc_rate.log" 2>&1

echo ""
echo "=============================================="
echo " All done!"
echo " Results: coding/vq/v3.4/results/soft_pq/dinov2_vitl14/"
echo " Logs:    $LOGDIR/"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
