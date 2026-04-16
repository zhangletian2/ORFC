#!/bin/bash
# Phase 3: Permute-by-sensitivity + exponential α_g
#
# Core idea: reorder OPQ-rotated dimensions by per-dim ΔL_ref sensitivity
# so that high-sensitivity dims concentrate in early PQ groups, then apply
# exponential α_g to allocate more bits to important groups.
#
# 2×2 ablation:
#   Factor A — decoder: free / tied (W_dec = W_enc^T)
#   Factor B — grouping: original uniform / permute-by-sensitivity
#
# Additional sweeps:  β ∈ {1,2,3},  prior_floor ∈ {0, 0.01},  freeze_transform
#
# Fixed: blk20, K=16, emb=32, bt=1024, warm_start_opq, epochs=100
# λ ∈ {0.5, 1.0}
#
# Skips runs whose result JSON already exists.  8-GPU parallel.
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
  local layer="$1" k="$2" lmbda="$3" use_ste="$4" alpha_mode="$5"
  local tied="$6" perm="$7" beta="$8" pf="$9" freeze="${10}"

  local rate_tag=""
  [ "$lmbda" != "0.0" ] && [ "$lmbda" != "0" ] && rate_tag="_lmbda${lmbda}"
  local tied_tag=""
  [ "$tied" = "1" ] && tied_tag="_tied"
  local fz_tag=""
  [ "$freeze" = "1" ] && fz_tag="_fzR"
  local ste_tag=""
  [ "$use_ste" = "1" ] && ste_tag="_ste"
  local alpha_tag=""
  [ "$alpha_mode" = "sensitivity" ] && alpha_tag="_alphasensitivity"
  local perm_tag=""
  [ "$perm" = "1" ] && perm_tag="_perm"
  local beta_tag=""
  [ "$perm" = "1" ] && [ "$beta" != "2.0" ] && beta_tag="_b${beta}"
  local pf_tag=""
  [ -n "$pf" ] && [ "$pf" != "0" ] && [ "$pf" != "0.0" ] && pf_tag="_pf${pf}"

  echo "${layer}_K${k}_emb32_bt1024_ws${rate_tag}${tied_tag}${fz_tag}${ste_tag}${alpha_tag}${perm_tag}${beta_tag}${pf_tag}_lr0.0001_ep100_n5000_s42.json"
}

run() {
  local layer="" k="16" lmbda="0.0" use_ste="0" alpha_mode="uniform"
  local tied="0" perm="0" beta="2.0" pf="0" freeze="0"
  local args=("$@")
  for ((i=0; i<${#args[@]}; i++)); do
    case "${args[$i]}" in
      --layer)            layer="${args[$((i+1))]}" ;;
      --K)                k="${args[$((i+1))]}" ;;
      --lmbda)            lmbda="${args[$((i+1))]}" ;;
      --use_ste)          use_ste="1" ;;
      --tied_transform)   tied="1" ;;
      --alpha_mode)       alpha_mode="${args[$((i+1))]}" ;;
      --permute_dims)     perm="1" ;;
      --alpha_beta)       beta="${args[$((i+1))]}" ;;
      --prior_floor)      pf="${args[$((i+1))]}" ;;
      --freeze_transform) freeze="1" ;;
    esac
  done

  local fname
  fname=$(build_fname "$layer" "$k" "$lmbda" "$use_ste" "$alpha_mode" \
                      "$tied" "$perm" "$beta" "$pf" "$freeze")

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

SENS="--alpha_mode sensitivity --sensitivity_warmup 10 --sensitivity_interval 5"

echo "================================================================"
echo "[$(date)] Phase 3: Permute-by-sensitivity + exponential α_g"
echo "================================================================"

# ================================================================
# Group 1: Core 2×2 ablation — {free, tied} × {no-perm, perm}
#           + STE variants
# ================================================================
for LMBDA in 0.5 1.0; do
  echo ""
  echo "--- blk20 K=16 λ=$LMBDA : core 2×2 ---"

  # A0B0: free + uniform (baseline, likely exists → skip)
  run --layer blk20 --K 16 --lmbda $LMBDA

  # A1B0: tied + uniform (baseline, likely exists → skip)
  run --layer blk20 --K 16 --lmbda $LMBDA --tied_transform

  # A0B1: free + perm + sensitivity (β=2, pf=0.01)
  run --layer blk20 --K 16 --lmbda $LMBDA \
      $SENS --permute_dims --alpha_beta 2.0 --prior_floor 0.01

  # A1B1: tied + perm + sensitivity (β=2, pf=0.01) ★ main config
  run --layer blk20 --K 16 --lmbda $LMBDA --tied_transform \
      $SENS --permute_dims --alpha_beta 2.0 --prior_floor 0.01

  # A0B1 + STE: free + perm + sensitivity + STE
  run --layer blk20 --K 16 --lmbda $LMBDA --use_ste \
      $SENS --permute_dims --alpha_beta 2.0 --prior_floor 0.01

  # A1B1 + STE: tied + perm + sensitivity + STE
  run --layer blk20 --K 16 --lmbda $LMBDA --tied_transform --use_ste \
      $SENS --permute_dims --alpha_beta 2.0 --prior_floor 0.01
done

# ================================================================
# Group 2: β sweep — tied + perm, β ∈ {1, 3}
# ================================================================
for LMBDA in 0.5 1.0; do
  echo ""
  echo "--- blk20 K=16 λ=$LMBDA : β sweep ---"

  # β=1 (mild amplification)
  run --layer blk20 --K 16 --lmbda $LMBDA --tied_transform \
      $SENS --permute_dims --alpha_beta 1.0 --prior_floor 0.01

  # β=3 (strong amplification)
  run --layer blk20 --K 16 --lmbda $LMBDA --tied_transform \
      $SENS --permute_dims --alpha_beta 3.0 --prior_floor 0.01
done

# ================================================================
# Group 3: Ablations — prior_floor=0, freeze_transform
# ================================================================
for LMBDA in 0.5 1.0; do
  echo ""
  echo "--- blk20 K=16 λ=$LMBDA : ablations ---"

  # No prior floor (pf=0) — test if prior collapse occurs
  run --layer blk20 --K 16 --lmbda $LMBDA --tied_transform \
      $SENS --permute_dims --alpha_beta 2.0

  # Freeze encoder — pure codebook+prior training in permuted space
  run --layer blk20 --K 16 --lmbda $LMBDA --tied_transform --freeze_transform \
      $SENS --permute_dims --alpha_beta 2.0 --prior_floor 0.01
done

# Wait for final batch
wait

echo ""
echo "================================================================"
echo "[$(date)] Done. Launched=$LAUNCHED  Skipped=$SKIPPED"
echo "================================================================"

# ================================================================
# Results comparison table
# ================================================================
echo ""
echo "Results comparison:"
python3 -c "
import json, glob, os

files = sorted(glob.glob('$RESULT_DIR/blk20_K16*_lmbda*_lr0.0001_ep100_n5000_s42.json'))
if not files:
    print('  No result files found.')
else:
    header = f\"{'config':80s}  {'dec':4s} {'α':4s} {'STE':4s} {'perm':4s}  {'Acc':>6s}  {'ΔL_ref':>8s}  {'R(b/t)':>7s}  {'mIoU':>5s}\"
    print(header)
    print('-' * len(header))
    for f in files:
        r = json.load(open(f))
        ri = r.get('rate_info', {})
        rans = ri.get('rans_bpt', ri.get('xent_rate_bpt', -1))
        miou = r.get('soft_pq_miou', -1)
        acc = r.get('soft_pq_acc', -1)
        dl = r.get('soft_pq_delta_l', -1)
        alpha = 'sens' if 'alphasensitivity' in f else 'unif'
        ste = '+STE' if '_ste_' in f or f.endswith('_ste_lr') else '    '
        dec = 'tied' if '_tied' in f else 'free'
        perm = 'perm' if '_perm' in f else '    '
        name = os.path.basename(f).replace('_lr0.0001_ep100_n5000_s42.json', '')
        print(f'{name:80s}  {dec:4s} {alpha:4s} {ste:4s} {perm:4s}  {acc:6.4f}  {dl:8.0f}  {rans:7.2f}  {miou:5.3f}')
"
