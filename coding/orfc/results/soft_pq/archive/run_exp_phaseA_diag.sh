#!/bin/bash
# Phase A: Diagnostic — does tying the decoder expose per-group sensitivity?
#
# Hypothesis: free decoder implicitly compensates group sensitivity differences,
#   masking the α_g signal (observed ±12% spread). Tying/freezing the decoder
#   should expose the true sensitivity structure (expected ±30%+ spread).
#
# Experiment matrix (layer=blk20, emb=32, bt=1024):
#   Decoder constraint × α_mode × K × λ
#
#   Constraints tested:
#     free         — baseline (existing results reused)
#     tied         — W_dec = W_enc^T, no independent decoder
#     tied+orth0.1 — tied + soft orthogonality regularisation
#     fzR          — freeze transform at R_opq (most extreme)
#
#   For each constraint: uniform + sensitivity
#   K ∈ {16, 64},  λ ∈ {0.5, 1.0}
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

# Build expected filename — must match run_soft_pq.py tag construction exactly:
#   {layer}_K{K}_emb32_bt1024_ws{rate_tag}{tied_tag}{orth_tag}{fz_tag}{alpha_tag}_lr..._ep..._n..._s...
build_fname() {
  local layer="$1" k="$2" lmbda="$3" tied="$4" orth="$5" fz="$6" alpha_mode="$7"

  local rate_tag=""
  [ "$lmbda" != "0.0" ] && [ "$lmbda" != "0" ] && rate_tag="_lmbda${lmbda}"

  local tied_tag=""
  [ "$tied" = "1" ] && tied_tag="_tied"

  local orth_tag=""
  [ "$orth" != "0" ] && [ "$orth" != "0.0" ] && orth_tag="_orth${orth}"

  local fz_tag=""
  [ "$fz" = "1" ] && fz_tag="_fzR"

  local alpha_tag=""
  [ "$alpha_mode" = "sensitivity" ] && alpha_tag="_alphasensitivity"

  echo "${layer}_K${k}_emb32_bt1024_ws${rate_tag}${tied_tag}${orth_tag}${fz_tag}${alpha_tag}_lr0.0001_ep100_n5000_s42.json"
}

run() {
  # Parse known flags to build expected filename
  local layer="" k="64" lmbda="0.0" tied="0" orth="0" fz="0" alpha_mode="uniform"
  local args=("$@")
  for ((i=0; i<${#args[@]}; i++)); do
    case "${args[$i]}" in
      --layer)            layer="${args[$((i+1))]}" ;;
      --K)                k="${args[$((i+1))]}" ;;
      --lmbda)            lmbda="${args[$((i+1))]}" ;;
      --tied_transform)   tied="1" ;;
      --orth_lambda)      orth="${args[$((i+1))]}" ;;
      --freeze_transform) fz="1" ;;
      --alpha_mode)       alpha_mode="${args[$((i+1))]}" ;;
    esac
  done

  local fname
  fname=$(build_fname "$layer" "$k" "$lmbda" "$tied" "$orth" "$fz" "$alpha_mode")

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

SENS="--alpha_mode sensitivity --sensitivity_warmup 10 --sensitivity_interval 10"

echo "================================================================"
echo "[$(date)] Phase A: Decoder-constraint diagnostic (blk20)"
echo "================================================================"

for K in 16 64; do
  for LMBDA in 0.5 1.0; do
    echo ""
    echo "====== K=$K  λ=$LMBDA ======"

    # 1) free decoder (baseline — should reuse existing results)
    run --layer blk20 --K $K --lmbda $LMBDA
    run --layer blk20 --K $K --lmbda $LMBDA $SENS

    # 2) tied decoder
    run --layer blk20 --K $K --lmbda $LMBDA --tied_transform
    run --layer blk20 --K $K --lmbda $LMBDA --tied_transform $SENS

    # 3) tied + orth regularisation
    run --layer blk20 --K $K --lmbda $LMBDA --tied_transform --orth_lambda 0.1
    run --layer blk20 --K $K --lmbda $LMBDA --tied_transform --orth_lambda 0.1 $SENS

    # 4) frozen transform (most extreme)
    run --layer blk20 --K $K --lmbda $LMBDA --freeze_transform
    run --layer blk20 --K $K --lmbda $LMBDA --freeze_transform $SENS
  done
done
wait

echo ""
echo "================================================================"
echo "[$(date)] Done. Launched=$LAUNCHED  Skipped=$SKIPPED"
echo "================================================================"
echo ""
echo "Analyse α_g spread across decoder constraints:"
cat <<'PYEOF'
python3 -c "
import json, glob, numpy as np

files = sorted(glob.glob('results/soft_pq/blk20_K*_emb32_bt1024_ws_lmbda*_lr0.0001_ep100_n5000_s42.json')
             + glob.glob('results/soft_pq/blk20_K*_emb32_bt1024_ws_lmbda*_tied*_lr0.0001_ep100_n5000_s42.json')
             + glob.glob('results/soft_pq/blk20_K*_emb32_bt1024_ws_lmbda*_fzR*_lr0.0001_ep100_n5000_s42.json'))

print(f'{'config':62s}  {'acc':>6s}  {'DLref':>8s}  {'mIoU':>6s}  {'ppl':>5s}  {'R_bpt':>6s}  {'a_std':>6s}  {'a_range':>12s}')
print('-' * 120)
for f in files:
    r = json.load(open(f))
    cfg = r['config']
    ri = r.get('rate_info', {})
    h = r['history']

    # Find last alpha_g
    a_std, a_range = '', ''
    for ep in reversed(h):
        if 'alpha_g' in ep:
            ag = np.array(ep['alpha_g'])
            a_std = f'{ag.std():.4f}'
            a_range = f'{ag.min():.2f}~{ag.max():.2f}'
            break

    rans = ri.get('rans_bpt', ri.get('xent_rate_bpt', -1))
    name = f.split('/')[-1].replace('_lr0.0001_ep100_n5000_s42.json', '')
    print(f'{name:62s}  {r[\"soft_pq_acc\"]:6.4f}  {r[\"soft_pq_delta_l\"]:8.0f}  '
          f'{r.get(\"soft_pq_miou\",0):6.4f}  {h[-1][\"perplexity\"]:5.1f}  {rans:6.2f}  '
          f'{a_std:>6s}  {a_range:>12s}')
"
PYEOF
