#!/bin/bash
# ORFC-v1.1 parallel validation driver.
#
# Runs 9 training experiments across GPUs 3-7 in two waves:
#   Wave 1: 5 jobs (1 per GPU)
#   Wave 2: 4 jobs (GPUs 3-6)
# Then selection + U-only sequentially.
#
# Assumes k8_00_opq_val and k8_01_original_val already complete.
#
# Usage:
#   bash run_exp_v1_1_parallel.sh [RUN_ID]

set -euo pipefail

RUN_ID=${1:-"v1_1_20260726T011657Z"}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

CONDA_BIN=${CONDA_BIN:-/home/user/anaconda3/bin/conda}
PYTHON_RUN=("${CONDA_BIN}" run --no-capture-output -n featcodec2 python)

RESULT_DIR="${SCRIPT_DIR}/results/dinov2_vitl14/${RUN_ID}"
LOG_DIR="${SCRIPT_DIR}/logs/${RUN_ID}"
SELECTION="${RESULT_DIR}/selection_v1_1.json"
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

FROZEN_K8_OPQ="${SCRIPT_DIR}/artifacts/dinov2_vitl14/opq_blk20_K8_emb32_per_image_n4500_ss1608637542_is787846414_oi20_ki100_km2000000_orfcv1-delivery-20260726.npz"

COMMON=(
    --seed 42
    --layer blk20
    --embedding_dim 32
    --bottleneck_dim 1024
    --norm_mode per_image
    --epochs 100
    --lr 3e-4
    --batch_size 32
    --grad_clip 1.0
    --tau_start 0.5
    --tau_end 0.005
    --tau_schedule exponential
    --n_groups_per_batch 4
    --n_interaction_pairs 4
    --val_elasticity_interval 20
    --probe_group_chunk 4
    --heldout_probe_group_chunk 4
    --n_train 4500
    --n_val 500
    --d0_normalize per_element
    --run_id "${RUN_ID}"
    --K 8
    --opq_artifact "${FROZEN_K8_OPQ}"
    --skip_test_eval
)

launch() {
    local gpu=$1 log_name=$2; shift 2
    echo "[$(date +%H:%M:%S)] Launching ${log_name} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES=${gpu} PYTHONUNBUFFERED=1 \
        "${PYTHON_RUN[@]}" run_v1_1.py "$@" \
        > "${LOG_DIR}/${log_name}.log" 2>&1 &
}

json_value() {
    "${PYTHON_RUN[@]}" -c \
        "import json; d=json.load(open('$1')); print($2)"
}

PIDS=()

echo "============================================================"
echo "  ORFC-v1.1 Parallel Validation"
echo "  RUN_ID=${RUN_ID}, GPUs=3-7"
echo "============================================================"

# ---- Wave 1: 5 jobs on GPUs 3-7 ----
echo ""
echo "=== Wave 1: 5 jobs (GPUs 3-7) ==="

launch 3 k8_legacy_gr0p05 \
    "${COMMON[@]}" --response_objective legacy \
    --alpha_list 0.1,0.5,1.0 \
    --alpha_weights 0.333333333333,0.333333333333,0.333333333334 \
    --beta 1 --beta_target_ratio 0.05 --result_suffix legacy_gr0p05
PIDS+=($!)

launch 4 k8_legacy_gr0p10 \
    "${COMMON[@]}" --response_objective legacy \
    --alpha_list 0.1,0.5,1.0 \
    --alpha_weights 0.333333333333,0.333333333333,0.333333333334 \
    --beta 1 --beta_target_ratio 0.10 --result_suffix legacy_gr0p10
PIDS+=($!)

launch 5 k8_legacy_gr0p20 \
    "${COMMON[@]}" --response_objective legacy \
    --alpha_list 0.1,0.5,1.0 \
    --alpha_weights 0.333333333333,0.333333333333,0.333333333334 \
    --beta 1 --beta_target_ratio 0.20 --result_suffix legacy_gr0p20
PIDS+=($!)

launch 6 k8_fixed_energy_gr0p05 \
    "${COMMON[@]}" --response_objective fixed_energy \
    --alpha_list 0.1,0.5,1.0 \
    --alpha_weights 0.333333333333,0.333333333333,0.333333333334 \
    --beta 1 --beta_target_ratio 0.05 --result_suffix fixed_energy_gr0p05
PIDS+=($!)

launch 7 k8_fixed_energy_gr0p10 \
    "${COMMON[@]}" --response_objective fixed_energy \
    --alpha_list 0.1,0.5,1.0 \
    --alpha_weights 0.333333333333,0.333333333333,0.333333333334 \
    --beta 1 --beta_target_ratio 0.10 --result_suffix fixed_energy_gr0p10
PIDS+=($!)

echo "  Waiting for Wave 1 (${#PIDS[@]} jobs)..."
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "  FAILED: PID ${pid}"
        FAIL=$((FAIL + 1))
    fi
done
echo "  Wave 1 done. Failures: ${FAIL}"
if [[ ${FAIL} -gt 0 ]]; then
    echo "  Check logs in ${LOG_DIR}/ for failures"
fi

# ---- Wave 2: 4 jobs on GPUs 3-6 ----
PIDS=()
echo ""
echo "=== Wave 2: 4 jobs (GPUs 3-6) ==="

launch 3 k8_fixed_energy_gr0p20 \
    "${COMMON[@]}" --response_objective fixed_energy \
    --alpha_list 0.1,0.5,1.0 \
    --alpha_weights 0.333333333333,0.333333333333,0.333333333334 \
    --beta 1 --beta_target_ratio 0.20 --result_suffix fixed_energy_gr0p20
PIDS+=($!)

launch 4 k8_operational_gr0p05 \
    "${COMMON[@]}" --response_objective operational \
    --alpha_list 0.1,0.5,1.0 --alpha_weights 0,0,1 \
    --beta 1 --beta_target_ratio 0.05 --result_suffix operational_gr0p05
PIDS+=($!)

launch 5 k8_operational_gr0p10 \
    "${COMMON[@]}" --response_objective operational \
    --alpha_list 0.1,0.5,1.0 --alpha_weights 0,0,1 \
    --beta 1 --beta_target_ratio 0.10 --result_suffix operational_gr0p10
PIDS+=($!)

launch 6 k8_operational_gr0p20 \
    "${COMMON[@]}" --response_objective operational \
    --alpha_list 0.1,0.5,1.0 --alpha_weights 0,0,1 \
    --beta 1 --beta_target_ratio 0.20 --result_suffix operational_gr0p20
PIDS+=($!)

echo "  Waiting for Wave 2 (${#PIDS[@]} jobs)..."
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "  FAILED: PID ${pid}"
        FAIL=$((FAIL + 1))
    fi
done
echo "  Wave 2 done. Failures: ${FAIL}"
if [[ ${FAIL} -gt 0 ]]; then
    echo "  Check logs in ${LOG_DIR}/ for failures"
fi

# ---- Selection ----
echo ""
echo "=== Running selection ==="
"${PYTHON_RUN[@]}" select_v1_1.py \
    --result_dir "${RESULT_DIR}" --output "${SELECTION}" \
    2>&1 | tee "${LOG_DIR}/k8_selection.log"

if [[ ! -f "${SELECTION}" ]]; then
    echo "Selection failed — no output file." >&2
    exit 4
fi

objective=$(json_value "${SELECTION}" "d['chosen']['objective']")
beta=$(json_value "${SELECTION}" "d['chosen']['effective_beta']")
weights=$(json_value "${SELECTION}" "','.join(str(x) for x in d['chosen']['alpha_weights'])")

echo "  Selected: objective=${objective}, beta=${beta}"

# ---- U-only causal control ----
echo ""
echo "=== K8 selected U-only causal control (GPU 3) ==="
CUDA_VISIBLE_DEVICES=3 "${PYTHON_RUN[@]}" run_v1_1.py \
    "${COMMON[@]}" \
    --response_objective "${objective}" --alpha_list 0.1,0.5,1.0 \
    --alpha_weights "${weights}" --beta "${beta}" \
    --freeze_codebooks --result_suffix selected_u_only \
    2>&1 | tee "${LOG_DIR}/k8_selected_u_only.log"

echo ""
echo "============================================================"
echo "  Validation complete."
echo "  Results: ${RESULT_DIR}/"
echo "  Selection: ${SELECTION}"
echo "  Next: review selection, then run postselect"
echo "============================================================"
