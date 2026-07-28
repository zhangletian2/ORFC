#!/usr/bin/env bash
set -euo pipefail
# ============================================================
#  VTM Feature Codec + Qwen3-VL-8B-Instruct MMStar Pipeline
#
#  Phase 1 (vtm)    : VTM encode/decode ALL QPs (CPU parallel)
#  Phase 2 (replay) : Replay anchor + QPs (multi-GPU, greedy)
#  Phase 3 (summary): BPFP + Acc + ΔAcc table
#
#  Usage:
#    bash run_vtm_instruct_mmstar.sh PHASE [NUM_GPUS] [MAX_SAMPLES] [GPU_START]
#    bash run_vtm_instruct_mmstar.sh all 4 0 4       # full, 4 GPUs from GPU4
#    bash run_vtm_instruct_mmstar.sh all 1 5 6       # validate 5 samples, GPU6
#    bash run_vtm_instruct_mmstar.sh vtm             # Phase 1 only (CPU)
#    bash run_vtm_instruct_mmstar.sh replay 4 0 4    # Phase 2, GPU4-7
#    bash run_vtm_instruct_mmstar.sh summary         # Phase 3
# ============================================================

PHASE=${1:-all}
NUM_GPUS=${2:-1}
MAX_SAMPLES=${3:-0}   # 0 = all samples
GPU_START=${4:-4}     # first GPU id (skip occupied GPUs)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- paths ---
FEAT_BASE=$PROJECT_ROOT/features
MODEL_NAME=qwen3vl_8b_instruct_mmstar
LAYER_NAME=blk08
LAYER_IDX=8
ORIG_FEAT_DIR=$FEAT_BASE/$MODEL_NAME/$LAYER_NAME

MODEL_PATH="/data4/workspace/zlt/cache/hf/hub/Qwen3-VL-8B-Instruct"
DATA_PATH="$PROJECT_ROOT/data/MMStar/mmstar.parquet"

# --- VTM ---
VTM_ENCODER=$SCRIPT_DIR/EncoderAppStatic
VTM_DECODER=$SCRIPT_DIR/DecoderAppStatic
VTM_CFG=$SCRIPT_DIR/encoder_intra_vtm.cfg
TMP_DIR=$SCRIPT_DIR/_vtm_tmp_instruct_mmstar
BIT_DEPTH=10
VTM_WORKERS=16

# --- experiment ---
QPS=(22 25 27 30 32 35 37 42)

# --- 官方 Instruct 推理参数（与 qwen3vl_instruct_feat_pipeline 保持一致） ---
SEED=3407
TEMPERATURE=0.7
TOP_P=0.8
TOP_K=20
REP_PENALTY=1.0
PRESENCE_PENALTY=1.5
MAX_TOKENS=32768
DTYPE=bf16

PYTHON="/home/user/anaconda3/envs/qwen3vl_codec/bin/python"
EVAL_SCRIPT="$PROJECT_ROOT/tools/qwen3vl_instruct_feat_pipeline.py"

RESULT_DIR=$FEAT_BASE/$MODEL_NAME/eval_vtm
[ "$MAX_SAMPLES" -gt 0 ] && RESULT_DIR="${RESULT_DIR}_val${MAX_SAMPLES}"
LOG_DIR=$RESULT_DIR/logs

echo ""
echo "================================================================"
echo "  VTM Feature Codec + Qwen3-VL-8B-Instruct MMStar"
echo "----------------------------------------------------------------"
echo "  Layer:       block[$LAYER_IDX] ($LAYER_NAME)"
echo "  QPs:         ${QPS[*]}"
echo "  GPUs:        $NUM_GPUS"
echo "  VTM workers: $VTM_WORKERS"
echo "  Sampling:    temp=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K pp=$PRESENCE_PENALTY"
echo "  Phase:       $PHASE"
echo "  GPU start:   $GPU_START"
[ "$MAX_SAMPLES" -gt 0 ] && echo "  MaxSamples:  $MAX_SAMPLES (validation)"
echo "  Results:     $RESULT_DIR"
echo "================================================================"

# ==================================================================
#  Phase 1: VTM encode/decode (CPU)
# ==================================================================
run_vtm() {
    echo ""
    echo "========================================"
    echo "  Phase 1: VTM encode/decode (QPs: ${QPS[*]})"
    echo "========================================"

    mkdir -p "$TMP_DIR"
    local vtm_feat_root="$FEAT_BASE"

    if [ "$MAX_SAMPLES" -gt 0 ]; then
        vtm_feat_root="$TMP_DIR/_staging"
        local staging_in="$vtm_feat_root/$MODEL_NAME/$LAYER_NAME"
        rm -rf "$vtm_feat_root"
        mkdir -p "$staging_in"

        local count=0
        for f in $(ls "$ORIG_FEAT_DIR"/*.npy | sort -V); do
            [ "$count" -ge "$MAX_SAMPLES" ] && break
            ln -sf "$f" "$staging_in/"
            count=$((count + 1))
        done
        echo "  [validation] staged $count files"
    fi

    $PYTHON "$SCRIPT_DIR/vtm_baseline.py" \
        --feat_root "$vtm_feat_root" \
        --models "$MODEL_NAME" \
        --layers "$LAYER_NAME" \
        --qps "${QPS[@]}" \
        --bit_depth "$BIT_DEPTH" \
        --vtm_encoder "$VTM_ENCODER" \
        --vtm_decoder "$VTM_DECODER" \
        --vtm_cfg "$VTM_CFG" \
        --tmp_dir "$TMP_DIR" \
        --workers "$VTM_WORKERS"

    if [ "$MAX_SAMPLES" -gt 0 ]; then
        for qp in "${QPS[@]}"; do
            local src="$vtm_feat_root/$MODEL_NAME/decoded/vtm/$qp/$LAYER_NAME"
            local dst="$FEAT_BASE/$MODEL_NAME/decoded/vtm/$qp/$LAYER_NAME"
            if [ -d "$src" ]; then
                mkdir -p "$dst"
                cp "$src"/*.npy "$dst/"
            fi
            local stats_src="$vtm_feat_root/$MODEL_NAME/decoded/vtm/$qp/_stats.csv"
            local stats_dst="$FEAT_BASE/$MODEL_NAME/decoded/vtm/$qp/_stats.csv"
            [ -f "$stats_src" ] && cp "$stats_src" "$stats_dst"
        done
        rm -rf "$vtm_feat_root"
    fi

    echo ""
    echo "  Phase 1 done"
}

# ==================================================================
#  Phase 2: Replay (multi-GPU, official sampling)
# ==================================================================
run_replay() {
    echo ""
    echo "========================================"
    echo "  Phase 2: Replay ($NUM_GPUS GPUs, seed=$SEED temp=$TEMPERATURE)"
    echo "========================================"

    mkdir -p "$RESULT_DIR" "$LOG_DIR"

    local replay_extra=""
    [ "$MAX_SAMPLES" -gt 0 ] && replay_extra="--max_samples $MAX_SAMPLES"

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
            gpu=$((GPU_START + i - batch_start))
            label=${TASK_LABELS[$i]}
            feat_dir=${TASK_FEAT_DIRS[$i]}
            output=${TASK_OUTPUTS[$i]}
            log_file=$LOG_DIR/${label}.log

            echo "     $label -> GPU $gpu  (log: $log_file)"

            CUDA_VISIBLE_DEVICES=$gpu \
            $PYTHON "$EVAL_SCRIPT" \
                replay \
                --model_path "$MODEL_PATH" \
                --data_path "$DATA_PATH" \
                --dtype "$DTYPE" \
                --seed "$SEED" \
                --max_new_tokens "$MAX_TOKENS" \
                --temperature "$TEMPERATURE" \
                --top_p "$TOP_P" \
                --top_k "$TOP_K" \
                --repetition_penalty "$REP_PENALTY" \
                --presence_penalty "$PRESENCE_PENALTY" \
                $replay_extra \
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
            echo "  [WARN] $failed process(es) failed, check logs"
        fi

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

    local qps_py
    qps_py=$(IFS=,; echo "${QPS[*]}")

    $PYTHON - <<PYSCRIPT
import os, json, csv, numpy as np

result_dir = "$RESULT_DIR"
stats_base = "$FEAT_BASE/$MODEL_NAME/decoded/vtm"
layer = "$LAYER_NAME"
qps = [$qps_py]

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
print("=" * 60)
print("  Qwen3-VL-8B-Instruct  VTM Codec  MMStar")
print("=" * 60)
print(f"{'Mode':>8} {'QP':>4} {'BPFP':>8} {'Acc%':>8} {'dAcc':>8}")
print("-" * 60)
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

print("=" * 60)
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
        echo "Usage: $0 {all|vtm|replay|summary} [NUM_GPUS] [MAX_SAMPLES] [GPU_START]"
        exit 1
        ;;
esac
