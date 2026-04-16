#!/bin/bash
# E3: fzR robustness — fixed rotation (OPQ / random_orth / PCA / identity)
#     + ΔLref codebook training
#
# 验证 ΔLref 对码本的优化是否依赖特定旋转
#   - init_rotation ∈ {opq, random_orth, pca, identity}
#   - K ∈ {16, 64}
#   - 全部 freeze_transform (只训码本)
#   - 不加码率约束 (λ=0)
#
# 其中 opq+fzR 已有结果，此处跑 random_orth / pca / identity 共 6 runs
# GPU: 1, 2 号卡
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$SCRIPT_DIR"


SEG_ARGS="--eval_seg \
  --seg_feat_root $PROJECT_ROOT/features/voc2012_100 \
  --voc_root $PROJECT_ROOT/data/VOCdevkit/VOC2012 \
  --seg_image_list $PROJECT_ROOT/utils/voc2012_val_100.txt"

COMMON="--layer blk20 --norm_mode per_image --max_train_images 5000 \
  --epochs 100 --lr 1e-4 \
  --bottleneck_dim 1024 --freeze_transform \
  --batch_size 32 --n_val 200 \
  --seed 42 --embedding_dim 32 $SEG_ARGS"

GPUS=(7)
NGPU=${#GPUS[@]}
COUNT=0
BATCH=1

run() {
  local gpu=${GPUS[$((COUNT % NGPU))]}
  echo "  [GPU $gpu] $@"
  CUDA_VISIBLE_DEVICES=$gpu python run_soft_pq.py "$@" $COMMON &
  COUNT=$((COUNT + 1))
  if [ $((COUNT % NGPU)) -eq 0 ]; then
    echo "[$(date)] Batch $BATCH ($NGPU jobs) running, waiting..."
    wait
    echo "[$(date)] Batch $BATCH done."
    BATCH=$((BATCH + 1))
  fi
}

echo "[$(date)] E3: rotation robustness (6 runs on GPU 1,2)"
echo "======================================================="

for ROT in random_orth pca identity; do
  for K in 16 64; do
    run --K $K --init_rotation $ROT
  done
done

wait
echo "[$(date)] All 6 runs complete. Results in results/soft_pq/"
echo ""
echo "Compare with existing fzR (opq) results:"
echo "  blk20_K16_emb32_bt1024_ws_fzR_*.json"
echo "  blk20_K64_emb32_bt1024_ws_fzR_*.json"
