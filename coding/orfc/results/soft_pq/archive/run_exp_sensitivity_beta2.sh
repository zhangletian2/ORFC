#!/bin/bash
# Sensitivity α_g with correct β=2.0, NO permutation
#
# Key experiment: test per-group ΔL_ref sensitivity rate allocation
# on top of OPQ warm-start, with amplification strong enough to
# actually change assignments (β=2 → α spans [0.25, 4.0]).
#
# Previous non-perm sensitivity experiments appear to have run with
# effective β ≈ 0.05 (α only spanned [0.88, 1.14] — near no-op).
# This script explicitly sets --alpha_beta 2.0 to ensure correct behavior.
#
# prior_floor=0.01 prevents -log2(p) explosion when prior peaks.
#
# Matrix: {free, tied} × {baseline, sensitivity} × λ ∈ {0.5, 1.0}
#   = 8 runs total, 1 batch on 8 GPUs
#
# Baselines (uniform α=1) will be skipped if they already exist.
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
  --seed 42 --embedding_dim 32 $SEG_ARGS"

GPUS=(0 1 2 3 4 5 6 7)
NGPU=${#GPUS[@]}
COUNT=0
BATCH=1
SKIPPED=0
LAUNCHED=0

build_fname() {
  local layer="$1" k="$2" lmbda="$3" tied="$4" alpha_mode="$5" pfloor="$6"

  local rate_tag="_lmbda${lmbda}"
  local tied_tag=""
  [ "$tied" = "1" ] && tied_tag="_tied"
  local alpha_tag=""
  [ "$alpha_mode" = "sensitivity" ] && alpha_tag="_alphasensitivity"
  local pfloor_tag=""
  [ "$pfloor" != "0" ] && [ "$pfloor" != "0.0" ] && pfloor_tag="_pf${pfloor}"

  echo "${layer}_K${k}_emb32_bt1024_ws${rate_tag}${tied_tag}${alpha_tag}${pfloor_tag}_lr0.0001_ep100_n5000_s42.json"
}

run() {
  local layer="" k="16" lmbda="0.0" tied="0" alpha_mode="uniform" pfloor="0"
  local args=("$@")
  for ((i=0; i<${#args[@]}; i++)); do
    case "${args[$i]}" in
      --layer)            layer="${args[$((i+1))]}" ;;
      --K)                k="${args[$((i+1))]}" ;;
      --lmbda)            lmbda="${args[$((i+1))]}" ;;
      --tied_transform)   tied="1" ;;
      --alpha_mode)       alpha_mode="${args[$((i+1))]}" ;;
      --prior_floor)      pfloor="${args[$((i+1))]}" ;;
    esac
  done

  local fname
  fname=$(build_fname "$layer" "$k" "$lmbda" "$tied" "$alpha_mode" "$pfloor")

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

SENS="--alpha_mode sensitivity --alpha_beta 2.0 \
  --sensitivity_warmup 10 --sensitivity_interval 10 \
  --prior_floor 0.01"

echo "================================================================"
echo "[$(date)] Sensitivity β=2.0 (no permutation)"
echo "================================================================"

for LMBDA in 0.5 1.0; do
  echo ""
  echo "--- blk20 K=16 λ=$LMBDA ---"

  # A) Baselines (uniform α, prior_floor=0.01 for fair comparison)
  run --layer blk20 --K 16 --lmbda $LMBDA --prior_floor 0.01
  run --layer blk20 --K 16 --lmbda $LMBDA --prior_floor 0.01 --tied_transform

  # B) Sensitivity β=2 (the key experiment)
  run --layer blk20 --K 16 --lmbda $LMBDA $SENS
  run --layer blk20 --K 16 --lmbda $LMBDA $SENS --tied_transform
done

if [ $((COUNT % NGPU)) -ne 0 ]; then
  echo "[$(date)] Waiting for final batch..."
  wait
  echo "[$(date)] Final batch done."
fi

echo ""
echo "================================================================"
echo "[$(date)] All done.  Launched=$LAUNCHED  Skipped=$SKIPPED"
echo "================================================================"
echo ""
echo "Compare results:"
echo "  python3 -c \""
echo "import json, glob"
echo "for f in sorted(glob.glob('$RESULT_DIR/blk20_K16_emb32_bt1024_ws_lmbda*_pf0.01_*.json') +"
echo "                glob.glob('$RESULT_DIR/blk20_K16_emb32_bt1024_ws_lmbda*_alphasensitivity_pf0.01_*.json')):"
echo "    r = json.load(open(f))"
echo "    h = r['history'][-1]"
echo "    tag = f.split('/')[-1].replace('_lr0.0001_ep100_n5000_s42.json', '')"
echo "    print(f'{tag:65s}  acc={r[\"soft_pq_acc\"]:.3f}  mIoU={r.get(\"soft_pq_miou\",-1):.4f}  DL={h[\"loss_distortion\"]:.0f}')"
echo "\""
