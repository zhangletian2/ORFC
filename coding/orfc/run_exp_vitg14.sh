#!/bin/bash
# ================================================================
#  DINOv2 ViT-G/14 multi-layer sweep (6 GPUs, chain scheduling)
#
#  3 layers × 6 configs = 18 experiments
#  Each GPU chains: blk09 → blk19 → blk29
#
#  Configs per layer:
#    #1  K=16   emb=32  lr=3e-4  λ=0.5   (low-rate)
#    #2  K=64   emb=32  lr=3e-4  λ=0.5   (mid-rate)
#    #3  K=64   emb=32  lr=3e-4  λ=0     (no rate constraint)
#    #4  K=256  emb=32  lr=3e-4  λ=0.5   (high-rate)
#    #5  K=256  emb=32  lr=5e-4  λ=0     (high-rate, max perf)
#    #6  K=64   emb=16  lr=3e-4  λ=0.5   (more groups)
#
#  ViT-G/14: D=1536, 40 blocks, bottleneck_dim=1536
#  Fixed: τ=0.5→0.005, epochs=100, n=5000, bs=32, seed=42, eval_seg
# ================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export TORCH_HOME="$PROJECT_ROOT/pretrained"


PYTHON=python
SCRIPT="run_soft_pq.py"
LOG_DIR="logs_vitg14"
mkdir -p "$LOG_DIR"

BACKBONE="dinov2_vitg14"
LAYERS=(blk09 blk19 blk29)

CONFIGS=(
  "16  32 0.0003 0.5  K16e32_lr3e4_l05"
  "64  32 0.0003 0.5  K64e32_lr3e4_l05"
  "64  32 0.0003 0.0  K64e32_lr3e4_l00"
  "256 32 0.0003 0.5  K256e32_lr3e4_l05"
  "256 32 0.0005 0.0  K256e32_lr5e4_l00"
  "64  16 0.0003 0.5  K64e16_lr3e4_l05"
)

COMMON_FIXED="--backbone $BACKBONE \
              --bottleneck_dim 1536 --warm_start_opq \
              --tau_start 0.5 --tau_end 0.005 \
              --epochs 100 --max_train_images 5000 \
              --batch_size 16 --seed 42 --eval_seg"

run_one() {
    local gpu=$1 layer=$2 K=$3 emb=$4 lr=$5 lmbda=$6 tag=$7
    local lmbda_arg=""
    if (( $(echo "$lmbda > 0" | bc -l) )); then
        lmbda_arg="--lmbda $lmbda"
    fi

    local logfile="${LOG_DIR}/${layer}_${tag}.log"
    echo "  [GPU $gpu] RUN   ${layer}_${tag}  ->  $logfile"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON $SCRIPT \
        --layer "$layer" --K "$K" --embedding_dim "$emb" \
        --lr "$lr" $lmbda_arg $COMMON_FIXED \
        > "$logfile" 2>&1
    local ret=$?
    if [ $ret -eq 0 ]; then
        echo "  [GPU $gpu] DONE  ${layer}_${tag}"
    else
        echo "  [GPU $gpu] FAIL  ${layer}_${tag}  (exit $ret)"
    fi
    return $ret
}

run_chain() {
    local gpu=$1
    shift
    local K=$1 emb=$2 lr=$3 lmbda=$4 tag=$5
    echo "[GPU $gpu] Chain: $tag  (blk09→blk19→blk29)"
    for layer in "${LAYERS[@]}"; do
        run_one "$gpu" "$layer" "$K" "$emb" "$lr" "$lmbda" "$tag"
    done
    echo "[GPU $gpu] Chain DONE: $tag"
}

echo "================================================================"
echo "  ViT-G/14 multi-layer sweep: 3 layers × 6 configs = 18 experiments"
echo "  6 GPUs, chain scheduling"
echo "  $(date)"
echo "================================================================"
echo ""
echo "  Configs:"
for i in "${!CONFIGS[@]}"; do
    echo "    #$((i+1)): ${CONFIGS[$i]}"
done
echo ""

for i in "${!CONFIGS[@]}"; do
    read -r K emb lr lmbda tag <<< "${CONFIGS[$i]}"
    run_chain "$i" "$K" "$emb" "$lr" "$lmbda" "$tag" &
done

echo ""
echo "All 6 chains launched. Waiting for completion..."
wait
echo ""
echo "================================================================"
echo "  All experiments finished at $(date)"
echo "================================================================"

# ================================================================
#  Summary table
# ================================================================
echo ""
echo "======== ViT-G/14 Results Summary ========"
$PYTHON << 'PYEOF'
import json, glob, os

results_dir = os.path.join('results', 'soft_pq', 'dinov2_vitg14')

rows = []
for f in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
    try:
        with open(f) as fh:
            r = json.load(fh)
    except:
        continue
    c = r.get('config', {})
    layer = c.get('layer', '?')
    K = c.get('K', 0)
    emb = c.get('embedding_dim', 0)
    lr = c.get('lr', 0)
    lmbda = c.get('lmbda', 0)
    mse = c.get('mse_loss', False)
    fzR = c.get('freeze_transform', False)
    if mse or fzR:
        continue
    acc = r.get('soft_pq_acc', 0)
    miou = r.get('soft_pq_miou', 0)
    dl = r.get('soft_pq_delta_l', 0)
    rate = r.get('rate_info', {}).get('rans_bpt', 0)
    opq_acc = r.get('std_opq_acc', 0)
    opq_miou = r.get('std_opq_miou', 0)
    D = 1536
    bpfp = (rate / D) if rate > 0 else 0
    rows.append((layer, K, emb, lr, lmbda, acc, miou, dl, rate, bpfp,
                 opq_acc, opq_miou))

rows.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

print(f"{'Layer':<6} {'K':>4} {'emb':>4} {'lr':>7} {'λ':>4} "
      f"{'Acc':>6} {'mIoU':>6} {'ΔL_ref':>8} {'Rate':>6} {'BPFP':>6} "
      f"{'OPQ_A':>6} {'OPQ_M':>6}")
print('-' * 95)
cur_layer = ''
for row in rows:
    layer, K, emb, lr, lmbda, acc, miou, dl, rate, bpfp, oa, om = row
    if layer != cur_layer:
        if cur_layer:
            print('-' * 95)
        cur_layer = layer
    print(f"{layer:<6} {K:>4} {emb:>4} {lr:>7.5f} {lmbda:>4.1f} "
          f"{acc:>6.3f} {miou:>6.3f} {dl:>8.0f} {rate:>6.1f} {bpfp:>6.4f} "
          f"{oa:>6.3f} {om:>6.3f}")

if rows:
    print()
    print("Best per layer:")
    print(f"{'Layer':<6} {'Best Acc (config)':>40} {'Best mIoU (config)':>40}")
    print('-' * 90)
    for layer in ['blk09', 'blk19', 'blk29']:
        layer_rows = [r for r in rows if r[0] == layer]
        if not layer_rows:
            continue
        best_acc = max(layer_rows, key=lambda x: x[5])
        best_miou = max(layer_rows, key=lambda x: x[6])
        acc_cfg = f"K={best_acc[1]} e={best_acc[2]} lr={best_acc[3]} λ={best_acc[4]} → {best_acc[5]:.3f}"
        miou_cfg = f"K={best_miou[1]} e={best_miou[2]} lr={best_miou[3]} λ={best_miou[4]} → {best_miou[6]:.3f}"
        print(f"{layer:<6} {acc_cfg:>40} {miou_cfg:>40}")
PYEOF

echo ""
echo "======== Per-job status ========"
for f in "$LOG_DIR"/*.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    if grep -q "Codec Acc" "$f" 2>/dev/null; then
        acc=$(grep "Codec Acc" "$f" | tail -1 | grep -oP '[\d.]+' | head -1)
        miou=$(grep "Codec.*mIoU" "$f" | tail -1 | grep -oP '[\d.]+' | head -1)
        echo "  $name: Acc=$acc mIoU=${miou:-N/A}"
    elif grep -q "Error\|Traceback\|FAIL" "$f" 2>/dev/null; then
        echo "  $name: FAILED (check log)"
    else
        echo "  $name: (running or no result)"
    fi
done
