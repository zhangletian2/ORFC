#!/bin/bash
# Experiment: per-group α_g sensitivity + STE validation
#
# Phase 1 (this script): NO rate constraint (λ=0)
#   → α_g has no effect (no rate bias), so we only test STE
#   → Does encoder gradient (STE) improve over frozen R_opq under ΔL_ref?
#
#   A) baseline  — warm-start OPQ, no STE  (reuse existing results)
#   B) ste       — warm-start OPQ + STE pass-through
#
# Phase 2 (future): WITH rate constraint (λ>0)
#   → Test α_g sensitivity allocation + STE combos
#
# Fixed: emb=32, bt=1024, warm_start_opq, epochs=100
# Sweep: K ∈ {16, 64},  layers ∈ {blk20, blk05}
#
# Skips runs whose result JSON already exists.
set -e
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

GPUS=(4 5 6 7)
NGPU=${#GPUS[@]}
COUNT=0
BATCH=1
SKIPPED=0
LAUNCHED=0

# Build expected result filename — matches run_soft_pq.py naming exactly
build_fname() {
  local layer="$1" k="$2" lmbda="$3" use_ste="$4" alpha_mode="$5"

  local rate_tag=""
  if [ "$lmbda" != "0.0" ] && [ "$lmbda" != "0" ]; then
    rate_tag="_lmbda${lmbda}"
  fi
  local ste_tag=""
  [ "$use_ste" = "1" ] && ste_tag="_ste"
  local alpha_tag=""
  [ "$alpha_mode" = "sensitivity" ] && alpha_tag="_alphasensitivity"

  echo "${layer}_K${k}_emb32_bt1024_ws${rate_tag}${ste_tag}${alpha_tag}_lr0.0001_ep100_n5000_s42.json"
}

run() {
  local layer="" k="64" lmbda="0.0" use_ste="0" alpha_mode="uniform"
  local args=("$@")
  for ((i=0; i<${#args[@]}; i++)); do
    case "${args[$i]}" in
      --layer)      layer="${args[$((i+1))]}" ;;
      --K)          k="${args[$((i+1))]}" ;;
      --lmbda)      lmbda="${args[$((i+1))]}" ;;
      --use_ste)    use_ste="1" ;;
      --alpha_mode) alpha_mode="${args[$((i+1))]}" ;;
    esac
  done

  local fname
  fname=$(build_fname "$layer" "$k" "$lmbda" "$use_ste" "$alpha_mode")

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

echo "================================================================"
echo "[$(date)] Phase 1: STE validation (λ=0, no rate constraint)"
echo "================================================================"

# ================================================================
# λ=0: baseline vs STE — does encoder gradient help under ΔL_ref?
# ================================================================
echo ""
echo "=== baseline vs STE (λ=0, K ∈ {16, 64}) ==="
for LAYER in blk20 blk05; do
  for K in 16 64; do
    # A) baseline (existing results from run_rd_sweep.sh should be reused)
    run --layer $LAYER --K $K --lmbda 0.0

    # B) STE pass-through
    run --layer $LAYER --K $K --lmbda 0.0 --use_ste
  done
done
wait

echo ""
echo "================================================================"
echo "[$(date)] Done. Launched=$LAUNCHED  Skipped=$SKIPPED"
echo "================================================================"
echo ""
echo "Compare results:"
echo "  python3 -c \""
echo "import json, glob"
echo "for f in sorted(glob.glob('$RESULT_DIR/*K*_emb32_bt1024_ws_*.json') +"
echo "                glob.glob('$RESULT_DIR/*K*_emb32_bt1024_ws_ste_*.json')):"
echo "    r = json.load(open(f))"
echo "    print(f'{f.split(\"/\")[-1]:70s}  acc={r[\"soft_pq_acc\"]:.4f}  '"
echo "          f'DL={r[\"soft_pq_delta_l\"]:.1f}')"
echo "\""
