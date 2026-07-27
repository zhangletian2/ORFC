#!/bin/bash
# Evaluate all available v1.1 checkpoints + v1 baselines on GPU 7
# Classification (ImageNet-500) + Segmentation (VOC2012-100)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

GPU=${1:-7}
CONDA_BIN=${CONDA_BIN:-/home/user/anaconda3/bin/conda}

V1_1_CKPT_DIR="checkpoints/dinov2_vitl14/v1_1_20260726T011657Z"
V1_CKPT_DIR="checkpoints/dinov2_vitl14/formal_k8_20260726T175547Z"
OPQ_ART="artifacts/dinov2_vitl14/opq_blk20_K8_emb32_per_image_n4500_ss1608637542_is787846414_oi20_ki100_km2000000_orfcv1-delivery-20260726.npz"

TIMESTAMP=$(date +%Y%m%dT%H%M%S)
OUT_DIR="results/dinov2_vitl14/eval_compare_${TIMESTAMP}"
LOG_DIR="logs/eval_compare_${TIMESTAMP}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# v1 baselines to include
V1_BASELINES=(
    "${V1_CKPT_DIR}/blk20_K8_emb32_b0.0_a0.1_lr0.0003_tau0.0063977931_gpb4_joint_ep100_s42_phaseB_b0_a01.pt"
    "${V1_CKPT_DIR}/blk20_K8_emb32_b0.1_a0.1_lr0.0003_tau0.0063977931_gpb4_joint_ep100_s42_final_seed42.pt"
)

# Verify v1 baselines exist
for f in "${V1_BASELINES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "WARNING: v1 baseline not found: $f"
    fi
done

echo "============================================================"
echo "  ORFC v1/v1.1 Comprehensive Evaluation"
echo "  GPU: ${GPU}"
echo "  v1.1 ckpts: ${V1_1_CKPT_DIR}"
echo "  Output: ${OUT_DIR}"
echo "  Time: ${TIMESTAMP}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU} ${CONDA_BIN} run --no-capture-output -n featcodec2 \
    python eval_v1_1_all.py \
        --gpu 0 \
        --opq_artifact "${OPQ_ART}" \
        --ckpt_dirs "${V1_1_CKPT_DIR}" \
        --v1_baselines "${V1_BASELINES[@]}" \
        --out "${OUT_DIR}/eval_comparison.json" \
    2>&1 | tee "${LOG_DIR}/eval.log"

echo ""
echo "Done. Results: ${OUT_DIR}/eval_comparison.json"
echo "Log: ${LOG_DIR}/eval.log"
