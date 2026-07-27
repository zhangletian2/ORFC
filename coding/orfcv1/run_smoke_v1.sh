#!/bin/bash
# Small end-to-end smoke suite for beta=0, U-only, Joint, and Alternating.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export TORCH_HOME="$PROJECT_ROOT/pretrained"
CONDA="${CONDA:-/home/user/anaconda3/bin/conda}"
PYTHON="$CONDA run --no-capture-output -n featcodec2 python"
SCRIPT="$SCRIPT_DIR/run_v1.py"
RUN_ID="${RUN_ID:-smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="$SCRIPT_DIR/logs/$RUN_ID"
RESULTS_DIR="$SCRIPT_DIR/results/dinov2_vitl14/$RUN_ID"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

COMMON="--layer blk20 --K 8 --embedding_dim 32 --bottleneck_dim 1024 \
        --alpha 0.1 --elastic_tau 0.1 --n_groups_per_batch 2 \
        --probe_group_chunk 2 --heldout_probe_group_chunk 2 \
        --epochs 1 --batch_size 2 --lr 0.0003 \
        --tau_start 0.5 --tau_end 0.005 \
        --n_train 8 --n_val 4 --n_test 8 \
        --val_elasticity_interval 1 --force_val_heldout \
        --opq_iter 1 --kmeans_iter 2 --kmeans_max_samples 4096 \
        --seed 777 --run_id $RUN_ID"

echo "[smoke prep] $RUN_ID"
CUDA_VISIBLE_DEVICES=0 $PYTHON "$SCRIPT" $COMMON \
    --epochs 0 --beta 0 --freeze_codebooks --skip_test_eval \
    --heldout_max_images 4 --result_suffix smoke_prep \
    > "$LOG_DIR/smoke_prep.log" 2>&1 || exit $?

run_one() {
    local gpu=$1 tag=$2 beta=$3 extra=$4
    echo "  [GPU $gpu] $tag"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON "$SCRIPT" $COMMON \
        --beta "$beta" $extra --result_suffix "$tag" \
        > "$LOG_DIR/$tag.log" 2>&1
}

PIDS=()
run_one 0 smoke_b0_joint 0.0 "--step_mode joint" & PIDS+=($!)
run_one 1 smoke_b003_uonly 0.03 "--freeze_codebooks" & PIDS+=($!)
run_one 2 smoke_b003_joint 0.03 "--step_mode joint" & PIDS+=($!)
run_one 3 smoke_b003_alt 0.03 \
    "--step_mode alternating --alt_u_steps 1 --alt_c_steps 1" & PIDS+=($!)

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done
if [ "$FAIL" -ne 0 ]; then
    echo "smoke failed: $FAIL job(s); inspect $LOG_DIR"
    exit "$FAIL"
fi

$PYTHON - "$RESULTS_DIR" <<'PY'
import glob
import json
import math
import os
import sys

root = sys.argv[1]
paths = [
    p for p in glob.glob(os.path.join(root, '*.json'))
    if any(tag in os.path.basename(p) for tag in (
        'smoke_b0_joint', 'smoke_b003_uonly',
        'smoke_b003_joint', 'smoke_b003_alt'))
]
if len(paths) != 4:
    raise SystemExit(f'expected 4 smoke results, found {len(paths)}')
for path in paths:
    with open(path) as f:
        r = json.load(f)
    if not r.get('reconstruction_audit', {}).get('passed'):
        raise SystemExit(f'audit failed: {path}')
    if len(r.get('history', [])) != 1:
        raise SystemExit(f'incomplete history: {path}')
    val = r.get('heldout_metrics', {}).get('val')
    if not val:
        raise SystemExit(f'missing validation heldout: {path}')
    n = val['eps_g']['n_images']
    for row in val['q_g_normalized']['per_group'].values():
        if row['n'] != n:
            raise SystemExit(f'q_g count mismatch: {path}')
    if not val.get('interaction', {}).get('per_pair'):
        raise SystemExit(f'missing interaction: {path}')
    hist = r['history'][0]
    for key in ('peak_memory_mb', 'images_per_sec', 'tokens_per_sec',
                'tail_calls_per_regular_training_batch'):
        if key not in hist or not math.isfinite(float(hist[key])):
            raise SystemExit(f'missing/nonfinite {key}: {path}')
    if float(r['config']['beta']) > 0:
        for key in ('grad_elastic_norm', 'grad_elastic_raw_norm'):
            value = hist.get(key)
            if value is None or not math.isfinite(float(value)) or float(value) <= 0:
                raise SystemExit(f'non-positive real elastic gradient {key}: {path}')
print(f'PASS: {len(paths)} end-to-end smoke configurations')
PY

echo "smoke results: $RESULTS_DIR"
