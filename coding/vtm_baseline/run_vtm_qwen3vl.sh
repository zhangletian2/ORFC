#!/usr/bin/env bash
set -euo pipefail
# ============================================================
#  VTM Feature Codec + Qwen3-VL MMBench replay pipeline
#
#  Phase 1: VTM encode/decode ALL QPs (CPU only)
#  Phase 2: Replay with thinking (multi-GPU parallel)
#  Phase 3: Summary (BPFP + Acc + dAcc)
#
#  Usage:
#    bash run_vtm_qwen3vl.sh NUM_GPUS PHASE
#    bash run_vtm_qwen3vl.sh 8 all       # 8-GPU full pipeline
#    bash run_vtm_qwen3vl.sh 1 vtm       # Phase 1 only (CPU)
#    bash run_vtm_qwen3vl.sh 8 replay    # Phase 2 (8-GPU parallel)
#    bash run_vtm_qwen3vl.sh 1 summary   # Phase 3
# ============================================================

NUM_GPUS=${1:-1}
PHASE=${2:-all}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- paths ---
FEAT_BASE=$PROJECT_ROOT/features/mmbench_en_val_1000
ORIG_FEAT_DIR=$FEAT_BASE/qwen3vl_4b/blk05
SAMPLE_LIST=$PROJECT_ROOT/utils/mmbench_en_val_1000.txt

MODEL_PATH="/data4/workspace/zlt/cache/hf/hub/Qwen3-VL-4B-Thinking"
DATA_DIR=$PROJECT_ROOT/data/MMBench

# --- VTM ---
VTM_ENCODER=$SCRIPT_DIR/EncoderAppStatic
VTM_DECODER=$SCRIPT_DIR/DecoderAppStatic
VTM_CFG=$SCRIPT_DIR/encoder_intra_vtm.cfg
TMP_DIR=$SCRIPT_DIR/_vtm_tmp_qwen3vl
BIT_DEPTH=10
VTM_WORKERS=20

# --- experiment ---
MODEL_NAME=qwen3vl_4b
LAYER_NAME=blk05
LAYER_IDX=5
QPS=(22 25 27 30 32 35 37 42)
MAX_TOKENS=2048

PYTHON="/home/user/anaconda3/envs/qwen3vl_codec/bin/python"
RESULT_DIR=$FEAT_BASE/qwen3vl_4b/eval_vtm
LOG_DIR=$RESULT_DIR/logs

echo ""
echo "================================================================"
echo "  VTM Feature Codec + Qwen3-VL MMBench"
echo "----------------------------------------------------------------"
echo "  Layer:      block[$LAYER_IDX] ($LAYER_NAME)"
echo "  QPs:        ${QPS[*]}"
echo "  GPUs:       $NUM_GPUS"
echo "  VTM workers: $VTM_WORKERS"
echo "  Thinking:   on"
echo "  Phase:      $PHASE"
echo "================================================================"

# ==================================================================
#  Phase 1: VTM encode/decode ALL QPs (CPU)
# ==================================================================
run_vtm() {
    echo ""
    echo "========================================"
    echo "  Phase 1: VTM encode/decode (QPs: ${QPS[*]})"
    echo "========================================"

    mkdir -p "$TMP_DIR"

    $PYTHON "$SCRIPT_DIR/vtm_baseline.py" \
        --feat_root "$FEAT_BASE" \
        --models "$MODEL_NAME" \
        --layers "$LAYER_NAME" \
        --qps "${QPS[@]}" \
        --bit_depth "$BIT_DEPTH" \
        --vtm_encoder "$VTM_ENCODER" \
        --vtm_decoder "$VTM_DECODER" \
        --vtm_cfg "$VTM_CFG" \
        --tmp_dir "$TMP_DIR" \
        --workers "$VTM_WORKERS"

    echo ""
    echo "  Phase 1 done"
}

# ==================================================================
#  Phase 2: Replay (multi-GPU parallel, thinking enabled)
# ==================================================================
run_replay() {
    echo ""
    echo "========================================"
    echo "  Phase 2: Replay ($NUM_GPUS GPUs, thinking=on)"
    echo "========================================"

    mkdir -p "$RESULT_DIR" "$LOG_DIR"

    # --- collect pending tasks ---
    declare -a TASK_LABELS=()
    declare -a TASK_FEAT_DIRS=()
    declare -a TASK_OUTPUTS=()

    if [ ! -f "$RESULT_DIR/anchor.json" ]; then
        TASK_LABELS+=("anchor")
        TASK_FEAT_DIRS+=("$ORIG_FEAT_DIR")
        TASK_OUTPUTS+=("$RESULT_DIR/anchor.json")
    else
        echo "  anchor: skip (exists)"
    fi

    for qp in "${QPS[@]}"; do
        if [ ! -f "$RESULT_DIR/qp${qp}.json" ]; then
            TASK_LABELS+=("QP${qp}")
            TASK_FEAT_DIRS+=("$FEAT_BASE/$MODEL_NAME/decoded/vtm/${qp}/$LAYER_NAME")
            TASK_OUTPUTS+=("$RESULT_DIR/qp${qp}.json")
        else
            echo "  QP${qp}: skip (exists)"
        fi
    done

    N_TASKS=${#TASK_LABELS[@]}
    if [ "$N_TASKS" -eq 0 ]; then
        echo "  All replay tasks done"
        return
    fi

    echo "  Pending: $N_TASKS tasks, $NUM_GPUS GPUs"
    echo ""

    # --- launch in batches of NUM_GPUS ---
    for ((batch_start=0; batch_start<N_TASKS; batch_start+=NUM_GPUS)); do
        batch_end=$((batch_start + NUM_GPUS))
        [ "$batch_end" -gt "$N_TASKS" ] && batch_end=$N_TASKS
        batch_size=$((batch_end - batch_start))

        echo "  -- batch $((batch_start/NUM_GPUS + 1)): ${TASK_LABELS[*]:$batch_start:$batch_size}"

        pids=()
        for ((i=batch_start; i<batch_end; i++)); do
            gpu=$((i - batch_start))
            label=${TASK_LABELS[$i]}
            feat_dir=${TASK_FEAT_DIRS[$i]}
            output=${TASK_OUTPUTS[$i]}
            log_file=$LOG_DIR/${label}.log

            echo "     $label -> GPU $gpu  (log: $log_file)"

            CUDA_VISIBLE_DEVICES=$gpu \
            $PYTHON "$PROJECT_ROOT/tools/qwen3vl_feat_pipeline.py" \
                --model_path "$MODEL_PATH" \
                --data_dir "$DATA_DIR" \
                --split validation --data_subdir en \
                --sample_list "$SAMPLE_LIST" \
                --device cuda --dtype bf16 \
                --max_new_tokens "$MAX_TOKENS" \
                replay \
                --layer "$LAYER_IDX" \
                --feat_dir "$feat_dir" \
                --output "$output" \
                > "$log_file" 2>&1 &
            pids+=($!)
        done

        echo "     waiting for ${#pids[@]} processes..."
        failed=0
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                failed=$((failed + 1))
            fi
        done

        if [ "$failed" -gt 0 ]; then
            echo "  [WARN] $failed process(es) failed in this batch, check logs"
        fi

        # show tail of each log
        for ((i=batch_start; i<batch_end; i++)); do
            label=${TASK_LABELS[$i]}
            log_file=$LOG_DIR/${label}.log
            echo "     [$label] $(tail -1 "$log_file" 2>/dev/null || echo 'no output')"
        done
        echo ""
    done

    echo "  Phase 2 done"
}

# ==================================================================
#  Phase 3: Summary
# ==================================================================
run_summary() {
    echo ""
    echo "========================================"
    echo "  Phase 3: Summary"
    echo "========================================"

    $PYTHON - <<PYSCRIPT
import os, json, csv, numpy as np

result_dir = "$RESULT_DIR"
stats_base = "$FEAT_BASE/$MODEL_NAME/decoded/vtm"
layer = "$LAYER_NAME"
qps = [${QPS[*]// /, }]

def load_acc(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d.get("accuracy"), d.get("correct"), d.get("total")

def get_avg_bpfp(qp):
    csv_path = os.path.join(stats_base, str(qp), "_stats.csv")
    if not os.path.exists(csv_path):
        return None
    vals = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("layer") == layer:
                vals.append(float(row["bpfp"]))
    return np.mean(vals) if vals else None

anchor = load_acc(os.path.join(result_dir, "anchor.json"))

print()
print("=" * 56)
print("  Qwen3-VL VTM Codec MMBench Summary")
print("=" * 56)
print(f"{'Mode':>8} {'QP':>4} {'BPFP':>8} {'Acc%':>8} {'dAcc':>8}")
print("-" * 56)
if anchor:
    print(f"{'anchor':>8} {'--':>4} {'--':>8} {anchor[0]:>7.2f}% {'--':>8}")

for qp in qps:
    r = load_acc(os.path.join(result_dir, f"qp{qp}.json"))
    bpfp = get_avg_bpfp(qp)
    bpfp_s = f"{bpfp:.4f}" if bpfp else "--"
    if r:
        d = f"{r[0] - anchor[0]:+.2f}%" if anchor else "--"
        print(f"{'QP' + str(qp):>8} {qp:>4} {bpfp_s:>8} {r[0]:>7.2f}% {d:>8}")
    else:
        print(f"{'QP' + str(qp):>8} {qp:>4} {bpfp_s:>8} {'--':>8} {'--':>8}")

print("=" * 56)
PYSCRIPT
}

# --- dispatch ---
case "$PHASE" in
    vtm)     run_vtm ;;
    replay)  run_replay ;;
    summary) run_summary ;;
    all)
        run_vtm
        run_replay
        run_summary
        echo ""
        echo "========================================"
        echo "  Done!  Results: $RESULT_DIR"
        echo "========================================"
        ;;
    *)
        echo "Usage: $0 [NUM_GPUS] {all|vtm|replay|summary}"
        exit 1
        ;;
esac
