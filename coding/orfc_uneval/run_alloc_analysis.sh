#!/usr/bin/env bash
# ============================================================================
# Allocation extremity analysis: min_bits sweep + multi-seed OPQ
#
# Two approaches to fix extreme DP allocation:
#   A) Increase min_bits (1->2->3->4) to constrain allocation range
#   B) Multiple OPQ seeds to find more balanced importance distribution
#
# Stage 1+2 only (OPQ + importance + allocation). No training.
# With --eval_init_mse: also runs k-means init + MSE for seed comparison.
#
# Schedule: 2 layers x 5 seeds = 10 runs, each GPU runs 1 process at a time.
# ============================================================================
set -u
cd "$(dirname "$0")"
mkdir -p logs/alloc_analysis results/alloc_analysis

PYTHON=${PYTHON:-conda run --no-capture-output -n featcodec2 python -u}
NUM_GPUS=${NUM_GPUS:-5}
K=${K:-16}
EDIM=${EDIM:-32}
MAX_BITS=${MAX_BITS:-10}
MIN_BITS_LIST=${MIN_BITS_LIST:-"1,2,3,4"}
BACKBONE=${BACKBONE:-dinov2_vitl14}

SEEDS=(42 43 44 45 46)
LAYERS=("blk05" "blk20")

# 1 = also compute init MSE (slower but needed for seed selection)
EVAL_MSE=${EVAL_MSE:-1}

MSE_FLAG=""
[ "$EVAL_MSE" = "1" ] && MSE_FLAG="--eval_init_mse"

# ---------------------------------------------------------------------------
run_one() {
    local gpu=$1 layer=$2 seed=$3
    local logf="logs/alloc_analysis/${layer}_K${K}_e${EDIM}_s${seed}.log"
    echo "[$(date +%H:%M:%S)] gpu=${gpu}  ${layer}  seed=${seed}  -> ${logf}"
    CUDA_VISIBLE_DEVICES=${gpu} $PYTHON analyze_allocation.py \
        --layer "${layer}" --K "${K}" --embedding_dim "${EDIM}" \
        --seed "${seed}" \
        --min_bits_list "${MIN_BITS_LIST}" --max_bits "${MAX_BITS}" \
        --backbone "${BACKBONE}" \
        ${MSE_FLAG} \
        > "${logf}" 2>&1
    local rc=$?
    [ $rc -ne 0 ] && echo "  [!!] ${layer} seed=${seed} FAILED (rc=${rc})" >&2 \
                   || echo "  [ok] ${layer} seed=${seed}"
}

# ---------------------------------------------------------------------------
# Build job list
# ---------------------------------------------------------------------------
ALL_JOBS=()
for layer in "${LAYERS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        ALL_JOBS+=("${layer}:${seed}")
    done
done
TOTAL=${#ALL_JOBS[@]}

echo "============================================================"
echo "  Allocation Analysis: ${TOTAL} runs"
echo "  ${#SEEDS[@]} seeds x ${#LAYERS[@]} layers, ${NUM_GPUS} GPUs"
echo "  K=${K}  e=${EDIM}  min_bits_list=${MIN_BITS_LIST}"
echo "  eval_init_mse=${EVAL_MSE}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Launch in waves: each wave = NUM_GPUS parallel jobs (1 per GPU)
# ---------------------------------------------------------------------------
START_TS=$(date +%s)
idx=0
while [ $idx -lt $TOTAL ]; do
    PIDS=()
    gpu=0
    while [ $gpu -lt $NUM_GPUS ] && [ $idx -lt $TOTAL ]; do
        IFS=':' read -r layer seed <<< "${ALL_JOBS[$idx]}"
        run_one "$gpu" "$layer" "$seed" &
        PIDS+=($!)
        gpu=$((gpu + 1))
        idx=$((idx + 1))
    done
    echo "  >> wave: ${#PIDS[@]} jobs on GPUs 0..$((gpu-1)), waiting..."
    for pid in "${PIDS[@]}"; do wait "$pid"; done
done
END_TS=$(date +%s)
echo ""
echo "All ${TOTAL} runs done.  elapsed=$(( (END_TS-START_TS)/60 ))m$(( (END_TS-START_TS)%60 ))s"

# ============================================================================
# Summary table
# ============================================================================
echo ""
echo "=================================================================="
echo "  Summary: Allocation Patterns"
echo "=================================================================="

for layer in "${LAYERS[@]}"; do
    D=1024; G=$((D / EDIM))
    LOG2K=$(python3 -c "import math; print(int(math.log2(${K})))")
    BUDGET=$((G * LOG2K))

    echo ""
    echo "--- ${layer}  K=${K}  budget=${BUDGET}  G=${G} ---"
    echo ""
    printf "  %-4s | %-4s | %5s %5s %4s | %5s | %s\n" \
           "seed" "minb" "n_min" "n_max" "uniq" "std" "histogram"
    echo "  ---------------------------------------------------------------"

    for seed in "${SEEDS[@]}"; do
        logf="logs/alloc_analysis/${layer}_K${K}_e${EDIM}_s${seed}.log"
        if [ ! -f "$logf" ]; then
            printf "  %-4s | %-40s\n" "$seed" "LOG MISSING"
            continue
        fi
        imp_cv=$(grep "importance cv=" "$logf" | head -1 | \
                 grep -oP 'cv=[0-9.]+' | sed 's/cv=//')

        while IFS= read -r line; do
            minb=$(echo "$line" | awk '{print $1}')
            n_min=$(echo "$line" | awk '{print $3}')
            n_max=$(echo "$line" | awk '{print $4}')
            uniq=$(echo "$line" | awk '{print $5}')
            bstd=$(echo "$line" | awk '{print $7}')
            hist=$(echo "$line" | grep -oP '\d+bit:\d+' | tr '\n' ' ')
            printf "  %-4s | %-4s | %5s %5s %4s | %5s | %s\n" \
                   "$seed" "$minb" "$n_min" "$n_max" "$uniq" "$bstd" "$hist"
        done < <(grep "bit:" "$logf" | grep -v "^  ---" | grep -v "INFEASIBLE")

        printf "  %-4s |      imp_cv=%-8s\n" "" "${imp_cv:--}"
        echo "  ---------------------------------------------------------------"
    done
done

# Init MSE summary
if [ "$EVAL_MSE" = "1" ]; then
    echo ""
    echo "=================================================================="
    echo "  Summary: Init MSE (lower is better)"
    echo "=================================================================="
    for layer in "${LAYERS[@]}"; do
        echo ""
        echo "--- ${layer} ---"
        printf "  %-4s | %-8s |" "seed" "imp_cv"
        for mb in $(echo "$MIN_BITS_LIST" | tr ',' ' '); do
            printf " minb=%-4s |" "$mb"
        done
        echo ""
        echo "  ---------------------------------------------------------------"
        for seed in "${SEEDS[@]}"; do
            logf="logs/alloc_analysis/${layer}_K${K}_e${EDIM}_s${seed}.log"
            imp_cv=$(grep "importance cv=" "$logf" 2>/dev/null | head -1 | \
                     grep -oP 'cv=[0-9.]+' | sed 's/cv=//')
            printf "  %-4s | %-8s |" "$seed" "${imp_cv:--}"
            for mb in $(echo "$MIN_BITS_LIST" | tr ',' ' '); do
                mse=$(grep "min_bits=${mb}:" "$logf" 2>/dev/null | \
                      grep -oP 'init_MSE=[0-9.]+' | sed 's/init_MSE=//')
                printf " %-9s |" "${mse:--}"
            done
            echo ""
        done
    done
fi

echo ""
echo "JSON: results/alloc_analysis/"
echo "Logs: logs/alloc_analysis/"
