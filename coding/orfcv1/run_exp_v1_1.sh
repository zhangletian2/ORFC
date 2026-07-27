#!/bin/bash
# ORFC-v1.1 protocol driver (§15).
#
# Usage:
#   bash run_exp_v1_1.sh [GPU] [RUN_ID] [validation|postselect|all]
#
# validation: K8 frozen baselines, 3 objectives × 3 calibrated gradient
#             ratios, validation-only selection, and selected U-only control.
# postselect: frozen K8 final test, K256 validation/training/transfer/final test.
# all:        validation followed by postselect.

set -euo pipefail

GPU=${1:-0}
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ID=${2:-"v1_1_${TIMESTAMP}"}
MODE=${3:-validation}

if [[ "${MODE}" != "validation" && "${MODE}" != "postselect" && "${MODE}" != "all" ]]; then
    echo "MODE must be validation, postselect, or all" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"
export CUDA_VISIBLE_DEVICES=${GPU}

CONDA_BIN=${CONDA_BIN:-/home/user/anaconda3/bin/conda}
PYTHON_RUN=("${CONDA_BIN}" run --no-capture-output -n featcodec2 python)

RESULT_DIR="${SCRIPT_DIR}/results/dinov2_vitl14/${RUN_ID}"
LOG_DIR="${SCRIPT_DIR}/logs/${RUN_ID}"
SELECTION="${RESULT_DIR}/selection_v1_1.json"
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

FROZEN_K8_OPQ=${FROZEN_K8_OPQ:-"${SCRIPT_DIR}/artifacts/dinov2_vitl14/opq_blk20_K8_emb32_per_image_n4500_ss1608637542_is787846414_oi20_ki100_km2000000_orfcv1-delivery-20260726.npz"}
FROZEN_K8_OPQ_REF=${FROZEN_K8_OPQ_REF:-"${RESULT_DIR%/${RUN_ID}}/formal_k8_20260726T175547Z/posthoc_OPQ_seed42.json"}
FROZEN_K8_ORFC_CKPT=${FROZEN_K8_ORFC_CKPT:-"${SCRIPT_DIR}/checkpoints/dinov2_vitl14/formal_k8_20260726T175547Z/blk20_K8_emb32_b0.0_a0.1_lr0.0003_tau0.0063977931_gpb4_joint_ep100_s42_phaseB_b0_a01.pt"}
FROZEN_K8_ORFC_REF=${FROZEN_K8_ORFC_REF:-"${RESULT_DIR%/${RUN_ID}}/formal_k8_20260726T175547Z/posthoc_Original_ORFC_seed42.json"}

for required in "${CONDA_BIN}" "${FROZEN_K8_OPQ}" "${FROZEN_K8_OPQ_REF}" \
                "${FROZEN_K8_ORFC_CKPT}" "${FROZEN_K8_ORFC_REF}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Required frozen input not found: ${required}" >&2
        exit 3
    fi
done

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
)

run_logged() {
    local log_name=$1
    shift
    "${PYTHON_RUN[@]}" run_v1_1.py "$@" 2>&1 | tee "${LOG_DIR}/${log_name}.log"
}

json_value() {
    local path=$1
    local expression=$2
    "${PYTHON_RUN[@]}" -c \
        "import json; d=json.load(open('${path}')); print(${expression})"
}

run_validation() {
    echo "=== K8 OPQ validation (frozen historical artifact) ==="
    run_logged k8_00_opq_val \
        "${COMMON[@]}" --K 8 --opq_artifact "${FROZEN_K8_OPQ}" \
        --response_objective none --beta 0 --skip_test_eval \
        --result_suffix opq_val

    echo "=== K8 Original ORFC validation (frozen historical checkpoint) ==="
    run_logged k8_01_original_val \
        "${COMMON[@]}" --K 8 --opq_artifact "${FROZEN_K8_OPQ}" \
        --response_objective legacy --beta 0 --skip_test_eval \
        --eval_checkpoint "${FROZEN_K8_ORFC_CKPT}" \
        --result_suffix original_val

    for objective in legacy fixed_energy operational; do
        if [[ "${objective}" == "operational" ]]; then
            weights="0,0,1"
        else
            weights="0.333333333333,0.333333333333,0.333333333334"
        fi
        for ratio in 0.05 0.10 0.20; do
            suffix="${objective}_gr${ratio/./p}"
            echo "=== K8 ${objective} Joint, target ratio ${ratio} ==="
            run_logged "k8_${suffix}" \
                "${COMMON[@]}" --K 8 --opq_artifact "${FROZEN_K8_OPQ}" \
                --response_objective "${objective}" \
                --alpha_list 0.1,0.5,1.0 --alpha_weights "${weights}" \
                --beta 1 --beta_target_ratio "${ratio}" --skip_test_eval \
                --result_suffix "${suffix}"
        done
    done

    "${PYTHON_RUN[@]}" select_v1_1.py \
        --result_dir "${RESULT_DIR}" --output "${SELECTION}" \
        2>&1 | tee "${LOG_DIR}/k8_selection.log"

    local objective beta weights checkpoint
    objective=$(json_value "${SELECTION}" "d['chosen']['objective']")
    beta=$(json_value "${SELECTION}" "d['chosen']['effective_beta']")
    weights=$(json_value "${SELECTION}" "','.join(str(x) for x in d['chosen']['alpha_weights'])")
    checkpoint=$(json_value "${SELECTION}" "d['chosen']['checkpoint']")

    echo "=== K8 selected U-only causal control ==="
    run_logged k8_selected_u_only \
        "${COMMON[@]}" --K 8 --opq_artifact "${FROZEN_K8_OPQ}" \
        --response_objective "${objective}" --alpha_list 0.1,0.5,1.0 \
        --alpha_weights "${weights}" --beta "${beta}" \
        --freeze_codebooks --skip_test_eval \
        --result_suffix selected_u_only

    echo "Selection frozen: ${checkpoint}"
}

run_postselect() {
    if [[ ! -f "${SELECTION}" ]]; then
        echo "Missing validation selection: ${SELECTION}" >&2
        exit 4
    fi
    local objective beta weights checkpoint
    objective=$(json_value "${SELECTION}" "d['chosen']['objective']")
    beta=$(json_value "${SELECTION}" "d['chosen']['effective_beta']")
    weights=$(json_value "${SELECTION}" "','.join(str(x) for x in d['chosen']['alpha_weights'])")
    checkpoint=$(json_value "${SELECTION}" "d['chosen']['checkpoint']")

    echo "=== K8 final: frozen Original ORFC (reuse previous Acc) ==="
    run_logged k8_final_original \
        "${COMMON[@]}" --K 8 --opq_artifact "${FROZEN_K8_OPQ}" \
        --opq_reference_json "${FROZEN_K8_OPQ_REF}" \
        --response_objective legacy --beta 0 \
        --eval_checkpoint "${FROZEN_K8_ORFC_CKPT}" \
        --codec_reference_json "${FROZEN_K8_ORFC_REF}" \
        --result_suffix final_original

    echo "=== K8 final: frozen selected Joint (new Acc, exactly once) ==="
    run_logged k8_final_selected \
        "${COMMON[@]}" --K 8 --opq_artifact "${FROZEN_K8_OPQ}" \
        --opq_reference_json "${FROZEN_K8_OPQ_REF}" \
        --response_objective "${objective}" --alpha_list 0.1,0.5,1.0 \
        --alpha_weights "${weights}" --beta "${beta}" \
        --eval_checkpoint "${checkpoint}" \
        --result_suffix final_selected

    echo "=== K256 OPQ validation ==="
    run_logged k256_00_opq_val \
        "${COMMON[@]}" --K 256 --response_objective none --beta 0 \
        --skip_test_eval --result_suffix k256_opq_val

    echo "=== K256 beta=0 Joint validation ==="
    run_logged k256_01_original_train \
        "${COMMON[@]}" --K 256 --response_objective legacy --beta 0 \
        --skip_test_eval --result_suffix k256_original

    echo "=== K256 frozen K8-objective transfer validation ==="
    run_logged k256_02_transfer_train \
        "${COMMON[@]}" --K 256 --response_objective "${objective}" \
        --alpha_list 0.1,0.5,1.0 --alpha_weights "${weights}" \
        --beta "${beta}" --skip_test_eval \
        --result_suffix k256_transfer

    local k256_base_json k256_transfer_json k256_base_ckpt k256_transfer_ckpt
    k256_base_json=$(find "${RESULT_DIR}" -maxdepth 1 -name '*K256*_k256_original.json' -print -quit)
    k256_transfer_json=$(find "${RESULT_DIR}" -maxdepth 1 -name '*K256*_k256_transfer.json' -print -quit)
    if [[ -z "${k256_base_json}" || -z "${k256_transfer_json}" ]]; then
        echo "Could not resolve K256 validation result files" >&2
        exit 5
    fi
    k256_base_ckpt=$(json_value "${k256_base_json}" "d['checkpoint']")
    k256_transfer_ckpt=$(json_value "${k256_transfer_json}" "d['checkpoint']")

    echo "=== K256 OPQ final test (new Acc, exactly once) ==="
    run_logged k256_03_opq_final \
        "${COMMON[@]}" --K 256 --response_objective none --beta 0 \
        --result_suffix k256_opq_final
    local k256_opq_ref
    k256_opq_ref="${RESULT_DIR}/blk20_K256_opq_s42_k256_opq_final.json"

    echo "=== K256 beta=0 final test ==="
    run_logged k256_04_original_final \
        "${COMMON[@]}" --K 256 --response_objective legacy --beta 0 \
        --opq_reference_json "${k256_opq_ref}" \
        --eval_checkpoint "${k256_base_ckpt}" \
        --result_suffix k256_original_final

    echo "=== K256 transfer final test ==="
    run_logged k256_05_transfer_final \
        "${COMMON[@]}" --K 256 --response_objective "${objective}" \
        --alpha_list 0.1,0.5,1.0 --alpha_weights "${weights}" \
        --beta "${beta}" --opq_reference_json "${k256_opq_ref}" \
        --eval_checkpoint "${k256_transfer_ckpt}" \
        --result_suffix k256_transfer_final
}

echo "ORFC-v1.1 RUN_ID=${RUN_ID}, GPU=${GPU}, MODE=${MODE}"
if [[ "${MODE}" == "validation" || "${MODE}" == "all" ]]; then
    run_validation
fi
if [[ "${MODE}" == "postselect" || "${MODE}" == "all" ]]; then
    run_postselect
fi

echo "Complete. Results: ${RESULT_DIR}"
echo "Logs: ${LOG_DIR}"
