#!/bin/bash
# Measure real DINOv2 training memory/throughput for probe chunks 1, 2, and 4.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export TORCH_HOME="$PROJECT_ROOT/pretrained"
CONDA="${CONDA:-/home/user/anaconda3/bin/conda}"
PYTHON="$CONDA run --no-capture-output -n featcodec2 python"
SCRIPT="$SCRIPT_DIR/run_v1.py"
RUN_ID="${RUN_ID:-chunkbench_$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="$SCRIPT_DIR/logs/$RUN_ID"
RESULTS_DIR="$SCRIPT_DIR/results/dinov2_vitl14/$RUN_ID"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

COMMON="--layer blk20 --K 8 --embedding_dim 32 --bottleneck_dim 1024 \
        --beta 0.03 --alpha 0.1 --elastic_tau 0.1 \
        --n_groups_per_batch 4 --epochs 1 --batch_size 32 --lr 0.0003 \
        --tau_start 0.5 --tau_end 0.005 \
        --n_train 128 --n_val 32 --n_test 0 \
        --val_elasticity_interval 1 --force_val_heldout \
        --opq_iter 1 --kmeans_iter 2 --kmeans_max_samples 32768 \
        --heldout_probe_group_chunk 4 \
        --skip_test_eval --seed 778 --run_id $RUN_ID"

echo "[chunk benchmark prep] $RUN_ID"
CUDA_VISIBLE_DEVICES=4 $PYTHON "$SCRIPT" $COMMON \
    --epochs 0 --beta 0 --probe_group_chunk 4 --freeze_codebooks \
    --result_suffix chunkbench_prep > "$LOG_DIR/prep.log" 2>&1 || exit $?

PIDS=()
for spec in "4 1" "5 2" "6 4"; do
    read -r gpu chunk <<< "$spec"
    echo "  [GPU $gpu] chunk=$chunk"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON "$SCRIPT" $COMMON \
        --probe_group_chunk "$chunk" \
        --result_suffix "chunkbench_c${chunk}" \
        > "$LOG_DIR/chunk${chunk}.log" 2>&1 &
    PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done
if [ "$FAIL" -ne 0 ]; then
    echo "chunk benchmark failed: $FAIL job(s)"
    exit "$FAIL"
fi

$PYTHON - "$RESULTS_DIR" <<'PY'
import glob
import json
import os
import sys

root = sys.argv[1]
rows = []
for path in glob.glob(os.path.join(root, '*chunkbench_c*.json')):
    with open(path) as f:
        r = json.load(f)
    h = r['history'][0]
    rows.append({
        'probe_group_chunk': int(r['config']['probe_group_chunk']),
        'peak_memory_mb': float(h['peak_memory_mb']),
        'images_per_sec': float(h['images_per_sec']),
        'tokens_per_sec': float(h['tokens_per_sec']),
        'tail_calls_per_regular_training_batch': int(
            h['tail_calls_per_regular_training_batch']),
        'reconstruction_audit_passed': bool(
            r['reconstruction_audit']['passed']),
    })
rows.sort(key=lambda x: x['probe_group_chunk'])
if len(rows) != 3 or not all(x['reconstruction_audit_passed'] for x in rows):
    raise SystemExit(f'incomplete benchmark rows: {rows}')
eligible = [x for x in rows if x['peak_memory_mb'] < 22000]
if not eligible:
    raise SystemExit('no chunk setting leaves the required memory headroom')
chosen = max(eligible, key=lambda x: x['images_per_sec'])
report = {
    'selection_rule': 'highest images/s subject to peak_memory_mb < 22000',
    'rows': rows,
    'selected_probe_group_chunk': chosen['probe_group_chunk'],
}
out = os.path.join(root, 'chunk_benchmark_report.json')
tmp = out + '.tmp'
with open(tmp, 'w') as f:
    json.dump(report, f, indent=2)
os.replace(tmp, out)
print(json.dumps(report, indent=2))
PY
