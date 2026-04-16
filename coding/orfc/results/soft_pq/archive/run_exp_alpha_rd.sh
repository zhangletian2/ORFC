#!/bin/bash
# Phase 2: Per-group α_g sensitivity + STE + tied decoder R-D allocation
#
# Part 1 — Free decoder (4 configs per λ):
#   A) uniform           — α_g = 1 for all groups (baseline)
#   B) sensitivity       — α_g ∝ 1/s_g from ΔL_ref gradient energy
#   C) uniform + STE     — encoder also gets ΔL_ref gradient via STE
#   D) sensitivity + STE — both α_g allocation and encoder gradient
#
# Part 2 — Tied decoder (4 configs per λ):
#   E) tied + uniform           — tied decoder baseline
#   F) tied + sensitivity       — does tied decoder expose true group sensitivity?
#   G) tied + STE               — STE with constrained decoder
#   H) tied + sensitivity + STE — full combination
#
# STE formulation: z_hat = z + z_q - z.detach()
#   → codebook gradient preserved (via gather), encoder gets STE bypass
#
# Tied decoder: W_dec = W_enc^T, prevents decoder from compensating
# inter-group quality differences → exposes true sensitivity for α_g.
#
# The α_g-weighted loss is:  J = Σ_g α_g·R_g·T + λ·D
# High-sensitivity groups get α_g < 1  →  cheaper rate  →  more bits allocated.
#
# Fixed: emb=32, bt=1024, warm_start_opq, epochs=100
# Part 1: K=16, λ ∈ {0.1, 0.5, 1.0}, layers ∈ {blk20, blk05}
# Part 2: K=16, λ ∈ {0.5, 1.0},       layer = blk20
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

GPUS=(0 1 2 3 4 5 6 7)
NGPU=${#GPUS[@]}
COUNT=0
BATCH=1
SKIPPED=0
LAUNCHED=0

build_fname() {
  local layer="$1" k="$2" lmbda="$3" use_ste="$4" alpha_mode="$5" tied="$6"

  local rate_tag=""
  if [ "$lmbda" != "0.0" ] && [ "$lmbda" != "0" ]; then
    rate_tag="_lmbda${lmbda}"
  fi
  local tied_tag=""
  [ "$tied" = "1" ] && tied_tag="_tied"
  local ste_tag=""
  [ "$use_ste" = "1" ] && ste_tag="_ste"
  local alpha_tag=""
  [ "$alpha_mode" = "sensitivity" ] && alpha_tag="_alphasensitivity"

  # tag order must match run_soft_pq.py: ...{rate_tag}{tied_tag}...{ste_tag}{alpha_tag}...
  echo "${layer}_K${k}_emb32_bt1024_ws${rate_tag}${tied_tag}${ste_tag}${alpha_tag}_lr0.0001_ep100_n5000_s42.json"
}

run() {
  local layer="" k="64" lmbda="0.0" use_ste="0" alpha_mode="uniform" tied="0"
  local args=("$@")
  for ((i=0; i<${#args[@]}; i++)); do
    case "${args[$i]}" in
      --layer)           layer="${args[$((i+1))]}" ;;
      --K)               k="${args[$((i+1))]}" ;;
      --lmbda)           lmbda="${args[$((i+1))]}" ;;
      --use_ste)         use_ste="1" ;;
      --tied_transform)  tied="1" ;;
      --alpha_mode)      alpha_mode="${args[$((i+1))]}" ;;
    esac
  done

  local fname
  fname=$(build_fname "$layer" "$k" "$lmbda" "$use_ste" "$alpha_mode" "$tied")

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

# echo "================================================================"
# echo "[$(date)] Phase 2: α_g sensitivity + STE R-D allocation (λ>0)"
# echo "================================================================"

# # ================================================================
# # For each (layer, K, λ): 4 configs (2×2 matrix)
# #   {uniform, sensitivity} × {no-STE, STE}
# # ================================================================
# for LAYER in blk20 blk05; do
#   for K in 16; do
#     for LMBDA in 0.1 0.5 1.0; do
#       echo ""
#       echo "--- $LAYER  K=$K  λ=$LMBDA ---"

#       # A) uniform baseline (no STE)
#       run --layer $LAYER --K $K --lmbda $LMBDA

#       # B) sensitivity-based α_g (no STE)
#       run --layer $LAYER --K $K --lmbda $LMBDA \
#           --alpha_mode sensitivity \
#           --sensitivity_warmup 10 --sensitivity_interval 10

#       # C) uniform + STE
#       run --layer $LAYER --K $K --lmbda $LMBDA --use_ste

#       # D) sensitivity + STE
#       run --layer $LAYER --K $K --lmbda $LMBDA --use_ste \
#           --alpha_mode sensitivity \
#           --sensitivity_warmup 10 --sensitivity_interval 10
#     done
#   done
# done
# wait

echo ""
echo "================================================================"
echo "[$(date)] Part 2: Tied decoder — expose true group sensitivities"
echo "================================================================"

# ================================================================
# Tied decoder: W_dec = W_enc^T  →  decoder cannot independently
# compensate per-group quality differences  →  α_g should show
# larger spread if groups truly differ in ΔL_ref importance.
#
# 4 configs (2×2): {uniform, sensitivity} × {no-STE, STE}
# ================================================================
for LAYER in blk20 blk05; do
  for K in 16; do
    for LMBDA in 0.5 1.0; do
      echo ""
      echo "--- $LAYER  K=$K  λ=$LMBDA  (tied decoder) ---"

      # E) tied + uniform (no STE)
      run --layer $LAYER --K $K --lmbda $LMBDA --tied_transform

      # F) tied + sensitivity (no STE)
      run --layer $LAYER --K $K --lmbda $LMBDA --tied_transform \
          --alpha_mode sensitivity \
          --sensitivity_warmup 10 --sensitivity_interval 10

      # G) tied + STE (uniform)
      run --layer $LAYER --K $K --lmbda $LMBDA --tied_transform --use_ste

      # H) tied + sensitivity + STE
      run --layer $LAYER --K $K --lmbda $LMBDA --tied_transform --use_ste \
          --alpha_mode sensitivity \
          --sensitivity_warmup 10 --sensitivity_interval 10
    done
  done
done
wait

echo ""
echo "================================================================"
echo "[$(date)] Done. Launched=$LAUNCHED  Skipped=$SKIPPED"
echo "================================================================"
echo ""
echo "Compare results ({free,tied} × {uniform,sensitivity} × {no-STE,STE}):"
echo "  python3 -c \""
echo "import json, glob"
echo "files = sorted(glob.glob('$RESULT_DIR/*_lmbda*_lr0.0001_ep100_n5000_s42.json'))"
echo "for f in files:"
echo "    r = json.load(open(f))"
echo "    ri = r.get('rate_info', {})"
echo "    rans = ri.get('rans_bpt', ri.get('xent_rate_bpt', -1))"
echo "    miou = r.get('soft_pq_miou', -1)"
echo "    alpha = 'sens' if 'alphasensitivity' in f else 'unif'"
echo "    ste = '+STE' if '_ste' in f else '    '"
echo "    dec = 'tied' if '_tied' in f else 'free'"
echo "    name = f.split('/')[-1].replace('_lr0.0001_ep100_n5000_s42.json', '')"
echo "    print(f'{name:70s}  {dec:4s} {alpha:4s} {ste:4s}  acc={r[\"soft_pq_acc\"]:.4f}  '"
echo "          f'DL={r[\"soft_pq_delta_l\"]:.0f}  R={rans:.2f}b/t  mIoU={miou:.3f}')"
echo "\""
