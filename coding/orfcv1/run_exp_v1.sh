#!/bin/bash
# ORFC-v1 controlled delivery experiment.
# Phase A/B run concurrently on 8 GPUs; Phase C starts after validation-only
# beta selection.  Only after the configuration is frozen does Phase D train
# seeds 42/43/44 and evaluate the untouched test split.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export TORCH_HOME="$PROJECT_ROOT/pretrained"

CONDA_ENV="featcodec2"
CONDA="${CONDA:-/home/user/anaconda3/bin/conda}"
PYTHON="$CONDA run --no-capture-output -n $CONDA_ENV python"
SCRIPT="$SCRIPT_DIR/run_v1.py"
RUN_ID="${RUN_ID:-orfcv1_$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="$SCRIPT_DIR/logs/$RUN_ID"
RESULTS_DIR="$SCRIPT_DIR/results/dinov2_vitl14/$RUN_ID"
RHO_D="${RHO_D:-0.05}"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

COMMON="--layer blk20 --K 8 --embedding_dim 32 \
        --bottleneck_dim 1024 \
        --lr 0.0003 \
        --tau_start 0.5 --tau_end 0.005 \
        --epochs 100 --batch_size 32 \
        --n_train 4500 --n_val 500 --n_test 0 \
        --val_elasticity_interval 20 \
        --probe_group_chunk 1 --heldout_probe_group_chunk 4 \
        --opq_iter 20 --kmeans_iter 100 \
        --seed 42 --run_id $RUN_ID"

FAIL_COUNT=0

run_one() {
    local gpu=$1 beta=$2 alpha=$3 tau_el=$4 tag=$5 extra=$6
    local logfile="${LOG_DIR}/${tag}.log"
    echo "  [GPU $gpu] RUN   ${tag} -> $logfile"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON "$SCRIPT" \
        --beta "$beta" --alpha "$alpha" --elastic_tau "$tau_el" \
        $COMMON $extra \
        --result_suffix "$tag" \
        > "$logfile" 2>&1
    local ret=$?
    if [ $ret -eq 0 ]; then
        echo "  [GPU $gpu] DONE  ${tag}"
    else
        echo "  [GPU $gpu] FAIL  ${tag} (exit $ret)"
    fi
    return $ret
}

wait_and_collect() {
    local phase=$1
    shift
    local phase_fails=0
    local pid
    for pid in "$@"; do
        wait "$pid" 2>/dev/null || phase_fails=$((phase_fails + 1))
    done
    if [ $phase_fails -gt 0 ]; then
        echo "  ERROR: $phase_fails job(s) failed in $phase"
        FAIL_COUNT=$((FAIL_COUNT + phase_fails))
    fi
}

require_no_failures() {
    local phase=$1
    if [ "$FAIL_COUNT" -gt 0 ]; then
        echo "FATAL: stopping after $phase because $FAIL_COUNT job(s) failed."
        exit "$FAIL_COUNT"
    fi
}

echo "================================================================"
echo " ORFC-v1 delivery run: $RUN_ID"
echo " Results: $RESULTS_DIR"
echo " Started: $(date)"
echo "================================================================"

# -----------------------------------------------------------------
# Pre-step: create shared OPQ/caches and calibrate elastic_tau.
# This calibration run never loads test features or labels.
# -----------------------------------------------------------------
echo "[Pre-step] OPQ, shared caches, validation elasticity pilot"
CUDA_VISIBLE_DEVICES=0 $PYTHON "$SCRIPT" \
    --layer blk20 --K 8 --embedding_dim 32 --bottleneck_dim 1024 \
    --beta 0 --alpha 0.1 --elastic_tau 1.0 \
    --n_groups_per_batch 4 \
    --probe_group_chunk 1 --heldout_probe_group_chunk 4 \
    --epochs 0 --batch_size 32 --lr 0.0003 \
    --tau_start 0.5 --tau_end 0.005 \
    --n_train 4500 --n_val 500 --n_test 0 \
    --opq_iter 20 --kmeans_iter 100 \
    --freeze_codebooks --force_val_heldout --heldout_max_images 200 \
    --skip_test_eval \
    --seed 42 --run_id "$RUN_ID" --result_suffix prep \
    > "$LOG_DIR/prep.log" 2>&1
PREP_RET=$?
if [ "$PREP_RET" -ne 0 ]; then
    echo "FATAL: preparation failed; see $LOG_DIR/prep.log"
    exit "$PREP_RET"
fi

ELASTIC_TAU=$($PYTHON - "$RESULTS_DIR" <<'PY'
import glob, json, os, sys
paths = glob.glob(os.path.join(sys.argv[1], '*_prep.json'))
if len(paths) != 1:
    raise SystemExit(f'expected one prep result, found {len(paths)}')
r = json.load(open(paths[0]))
span = r['heldout_metrics']['val']['eps_g']['global_span']
print(f'{max(float(span) / 4.0, 1e-4):.8g}')
PY
)
echo "  Calibrated elastic_tau=$ELASTIC_TAU (val epsilon span / 4)"
printf '%s\n' "$ELASTIC_TAU" > "$RESULTS_DIR/elastic_tau.txt"

# -----------------------------------------------------------------
# Phase A + B: eight independent candidates on all eight GPUs.
# -----------------------------------------------------------------
echo "[Phase A+B] launching 8 candidates on GPUs 0-7"
PIDS=()
run_one 0 0.0  0.1 "$ELASTIC_TAU" phaseA_b0_a01   "--freeze_codebooks --skip_test_eval" & PIDS+=($!)
run_one 1 0.01 0.1 "$ELASTIC_TAU" phaseA_b001_a01 "--freeze_codebooks --skip_test_eval" & PIDS+=($!)
run_one 2 0.03 0.1 "$ELASTIC_TAU" phaseA_b003_a01 "--freeze_codebooks --skip_test_eval" & PIDS+=($!)
run_one 3 0.1  0.1 "$ELASTIC_TAU" phaseA_b01_a01  "--freeze_codebooks --skip_test_eval" & PIDS+=($!)
run_one 4 0.0  0.1 "$ELASTIC_TAU" phaseB_b0_a01   "--step_mode joint --force_val_heldout --skip_test_eval" & PIDS+=($!)
run_one 5 0.01 0.1 "$ELASTIC_TAU" phaseB_b001_a01 "--step_mode joint --skip_test_eval" & PIDS+=($!)
run_one 6 0.03 0.1 "$ELASTIC_TAU" phaseB_b003_a01 "--step_mode joint --skip_test_eval" & PIDS+=($!)
run_one 7 0.1  0.1 "$ELASTIC_TAU" phaseB_b01_a01  "--step_mode joint --skip_test_eval" & PIDS+=($!)
wait_and_collect "Phase A+B" "${PIDS[@]}"
require_no_failures "Phase A+B"

# -----------------------------------------------------------------
# Phase C: validation-only selection from Phase B, then ratios.
# -----------------------------------------------------------------
echo "[Phase C] selecting beta from current-run Phase B"
BEST_BETA=$($PYTHON "$SCRIPT_DIR/select_phasec.py" \
    --results-dir "$RESULTS_DIR" --rho-d "$RHO_D" \
    --output "$RESULTS_DIR/phasec_selection.json")
echo "  Selected beta=$BEST_BETA"

PIDS=()
run_one 0 "$BEST_BETA" 0.1 "$ELASTIC_TAU" phaseC_u1c1 \
    "--step_mode alternating --alt_u_steps 1 --alt_c_steps 1 --skip_test_eval" & PIDS+=($!)
run_one 1 "$BEST_BETA" 0.1 "$ELASTIC_TAU" phaseC_u2c1 \
    "--step_mode alternating --alt_u_steps 2 --alt_c_steps 1 --skip_test_eval" & PIDS+=($!)
run_one 2 "$BEST_BETA" 0.1 "$ELASTIC_TAU" phaseC_u1c2 \
    "--step_mode alternating --alt_u_steps 1 --alt_c_steps 2 --skip_test_eval" & PIDS+=($!)
wait_and_collect "Phase C" "${PIDS[@]}"
require_no_failures "Phase C"

# -----------------------------------------------------------------
# Phase D: freeze the selected B/C config, then run three independent seeds.
# This is the first and only phase that loads labels/test features.
# -----------------------------------------------------------------
echo "[Phase D] selecting final configuration and running seeds 42/43/44"
FINAL_SPEC=$($PYTHON "$SCRIPT_DIR/select_final.py" \
    --results-dir "$RESULTS_DIR" --rho-d "$RHO_D" \
    --output "$RESULTS_DIR/final_selection.json")
read -r FINAL_MODE FINAL_BETA FINAL_U FINAL_C <<< "$FINAL_SPEC"
echo "  Final: mode=$FINAL_MODE beta=$FINAL_BETA u=$FINAL_U c=$FINAL_C"

if [ "$FINAL_MODE" = "alternating" ]; then
    FINAL_EXTRA="--step_mode alternating --alt_u_steps $FINAL_U --alt_c_steps $FINAL_C"
else
    FINAL_EXTRA="--step_mode joint"
fi

PIDS=()
run_one 0 "$FINAL_BETA" 0.1 "$ELASTIC_TAU" final_seed42 \
    "$FINAL_EXTRA --seed 42" & PIDS+=($!)
run_one 1 "$FINAL_BETA" 0.1 "$ELASTIC_TAU" final_seed43 \
    "$FINAL_EXTRA --seed 43" & PIDS+=($!)
run_one 2 "$FINAL_BETA" 0.1 "$ELASTIC_TAU" final_seed44 \
    "$FINAL_EXTRA --seed 44" & PIDS+=($!)
wait_and_collect "Phase D" "${PIDS[@]}"
require_no_failures "Phase D"

echo "================================================================"
echo " ORFC-v1 delivery run completed: $RUN_ID"
echo " Finished: $(date)"
echo " Results: $RESULTS_DIR"
echo "================================================================"
