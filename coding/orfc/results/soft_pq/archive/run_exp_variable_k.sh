#!/bin/bash
# Phase 4: Variable K_g per group via sensitivity allocation
#
# Core idea: after warmup, use per-group sensitivity to allocate
# non-uniform codebook sizes K_g ∈ {4,8,16,32,64} while maintaining
# the total bit budget sum(log2(K_g)) = G * log2(16) = 128.
# Codebook transition via split/merge avoids distortion spikes.
#
# Experiments:
#   1. baseline uniform K=16 (should already exist → skip)
#   2. variable_K (automatic allocation from sensitivity)
#
# Fixed: blk20, K=16, emb=32, bt=1024, warm_start_opq, epochs=100
# λ ∈ {0.5, 1.0}
#
# Skips runs whose result JSON already exists.  8-GPU parallel.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$SCRIPT_DIR"

RESULT_DIR="results/soft_pq"
mkdir -p "$RESULT_DIR"

SEG_ARGS="--eval_seg \
  --seg_feat_root $PROJECT_ROOT/features/voc2012_100 \
  --voc_root $PROJECT_ROOT/data/VOCdevkit/VOC2012 \
  --seg_image_list $PROJECT_ROOT/utils/voc2012_val_100.txt"

COMMON="--norm_mode per_image --max_train_images 5000 \
  --epochs 100 --lr 1e-4 \
  --bottleneck_dim 1024 --warm_start_opq \
  --batch_size 32 --n_val 200 \
  --seed 42 --embedding_dim 16 $SEG_ARGS"

GPUS=(0 1 2 3 4 5 6 7)
NGPU=${#GPUS[@]}
COUNT=0
BATCH=1
SKIPPED=0
LAUNCHED=0

build_fname() {
  local layer="$1" k="$2" lmbda="$3" use_ste="$4" alpha_mode="$5"
  local tied="$6" varK="$7" embedding_dim="$8"

  local rate_tag=""
  [ "$lmbda" != "0.0" ] && [ "$lmbda" != "0" ] && rate_tag="_lmbda${lmbda}"
  local tied_tag=""
  [ "$tied" = "1" ] && tied_tag="_tied"
  local ste_tag=""
  [ "$use_ste" = "1" ] && ste_tag="_ste"
  local alpha_tag=""
  [ "$alpha_mode" = "sensitivity" ] && alpha_tag="_alphasensitivity"
  local varK_tag=""
  [ "$varK" = "1" ] && varK_tag="_varK"

  # emb${embedding_dim} for embedding dim
  echo "${layer}_K${k}_emb${embedding_dim}_bt1024_ws${rate_tag}${tied_tag}${ste_tag}${alpha_tag}${varK_tag}_lr0.0001_ep100_n5000_s42.json"
}

run() {
  local layer="" k="16" lmbda="0.0" use_ste="0" alpha_mode="uniform"
  local tied="0" varK="0"
  local embedding_dim="16"
  local args=("$@")
  for ((i=0; i<${#args[@]}; i++)); do
    case "${args[$i]}" in
      --layer)            layer="${args[$((i+1))]}" ;;
      --K)                k="${args[$((i+1))]}" ;;
      --lmbda)            lmbda="${args[$((i+1))]}" ;;
      --use_ste)          use_ste="1" ;;
      --tied_transform)   tied="1" ;;
      --alpha_mode)       alpha_mode="${args[$((i+1))]}" ;;
      --variable_K)       varK="1" ;;
      --embedding_dim)    embedding_dim="${args[$((i+1))]}" ;;
    esac
  done

  local fname
  fname=$(build_fname "$layer" "$k" "$lmbda" "$use_ste" "$alpha_mode" \
                      "$tied" "$varK" "$embedding_dim")

  if [ -f "$RESULT_DIR/$fname" ]; then
    echo "  [SKIP] $fname"
    SKIPPED=$((SKIPPED + 1))
    return
  fi

  local gpu=${GPUS[$((COUNT % NGPU))]}
  echo "  [GPU $gpu] $fname"
  CUDA_VISIBLE_DEVICES=$gpu python run_soft_pq.py "$@" $COMMON &
  COUNT=$((COUNT + 1))
  LAUNCHED=$((LAUNCHED + 1))
  if [ $((COUNT % NGPU)) -eq 0 ]; then
    echo "[$(date)] Batch $BATCH ($NGPU slots) launched, waiting..."
    wait
    echo "[$(date)] Batch $BATCH done."
    BATCH=$((BATCH + 1))
  fi
}

VARK="--variable_K --sensitivity_warmup 10"

echo "================================================================"
echo "[$(date)] Phase 4: Variable K_g per group"
echo "================================================================"

# ================================================================
# Group 1: Baselines (uniform K=16)
# ================================================================
for LMBDA in 0.5 1.0; do
  echo ""
  echo "--- blk20 K=16 λ=$LMBDA : baselines ---"

  # free decoder, uniform K (should already exist)
  run --layer blk20 --K 16 --lmbda $LMBDA

  # tied decoder, uniform K (should already exist)
  run --layer blk20 --K 16 --lmbda $LMBDA --tied_transform
done

# ================================================================
# Group 2: Variable K_g (automatic allocation)
# ================================================================
for LMBDA in 0.5 1.0; do
  echo ""
  echo "--- blk20 K=16 λ=$LMBDA : variable K ---"

  # free decoder + variable K (alpha/permute automatically disabled)
  run --layer blk20 --K 16 --lmbda $LMBDA \
      $VARK

  # tied decoder + variable K
  run --layer blk20 --K 16 --lmbda $LMBDA --tied_transform \
      $VARK
done

# Wait for any remaining jobs
if [ $((COUNT % NGPU)) -ne 0 ]; then
  echo "[$(date)] Waiting for final batch..."
  wait
  echo "[$(date)] Final batch done."
fi

echo ""
echo "================================================================"
echo "[$(date)] All done.  Launched=$LAUNCHED  Skipped=$SKIPPED"
echo "================================================================"
