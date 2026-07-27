#!/bin/bash
# ORFCv1 segmentation evaluation on VOC2012 (100 images)
# Uses GPU 5-7 in parallel
#
# Evaluates:
#   - 3 final seeds (seed42/43/44, best config: joint β=0.1)
#   - Phase B β=0 baseline (no elastic loss)
#   - Phase A β=0.1 (U-only, freeze codebooks)
#   - Phase C u2c1 (alternating, best variant)
#
# Usage:  bash run_eval_seg_v1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CKPT_BASE="${SCRIPT_DIR}/checkpoints/dinov2_vitl14/formal_k8_20260726T175547Z"
RESULT_DIR="${SCRIPT_DIR}/results/dinov2_vitl14/formal_k8_20260726T175547Z"
LOG_DIR="${SCRIPT_DIR}/logs/seg_eval_20260726"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

CONDA_ENV="featcodec2"
LAYER="blk20"
NORM="per_image"

# Checkpoints to evaluate (6 total → 2 per GPU)
declare -a CKPTS=(
    # GPU 5: final seed42 + seed43
    "${CKPT_BASE}/blk20_K8_emb32_b0.1_a0.1_lr0.0003_tau0.0063977931_gpb4_joint_ep100_s42_final_seed42.pt"
    "${CKPT_BASE}/blk20_K8_emb32_b0.1_a0.1_lr0.0003_tau0.0063977931_gpb4_joint_ep100_s43_final_seed43.pt"
    # GPU 6: final seed44 + Phase B β=0
    "${CKPT_BASE}/blk20_K8_emb32_b0.1_a0.1_lr0.0003_tau0.0063977931_gpb4_joint_ep100_s44_final_seed44.pt"
    "${CKPT_BASE}/blk20_K8_emb32_b0.0_a0.1_lr0.0003_tau0.0063977931_gpb4_joint_ep100_s42_phaseB_b0_a01.pt"
    # GPU 7: Phase A β=0.1 + Phase C u2c1
    "${CKPT_BASE}/blk20_K8_emb32_b0.1_a0.1_lr0.0003_tau0.0063977931_gpb4_fzC_joint_ep100_s42_phaseA_b01_a01.pt"
    "${CKPT_BASE}/blk20_K8_emb32_b0.1_a0.1_lr0.0003_tau0.0063977931_gpb4_alternating_u2c1_ep100_s42_phaseC_u2c1.pt"
)

GPU_IDS=(5 6 7)

# Verify all checkpoints exist
for ckpt in "${CKPTS[@]}"; do
    if [ ! -f "$ckpt" ]; then
        echo "ERROR: checkpoint not found: $ckpt"
        exit 1
    fi
done
echo "All ${#CKPTS[@]} checkpoints verified."

# Launch evaluation jobs
PIDS=()
for gpu_idx in 0 1 2; do
    gpu=${GPU_IDS[$gpu_idx]}
    c1=${CKPTS[$((gpu_idx * 2))]}
    c2=${CKPTS[$((gpu_idx * 2 + 1))]}

    tag1=$(basename "$c1" .pt)
    tag2=$(basename "$c2" .pt)
    log="${LOG_DIR}/gpu${gpu}.log"

    echo "[GPU ${gpu}] Evaluating:"
    echo "  1) ${tag1}"
    echo "  2) ${tag2}"
    echo "  Log: ${log}"

    (
        conda run --no-capture-output -n "${CONDA_ENV}" \
            python "${SCRIPT_DIR}/eval_seg_v1.py" \
                --ckpt "$c1" \
                --layer "${LAYER}" --norm_mode "${NORM}" \
                --gpu "${gpu}" \
                --out_dir "${RESULT_DIR}" \
                2>&1

        conda run --no-capture-output -n "${CONDA_ENV}" \
            python "${SCRIPT_DIR}/eval_seg_v1.py" \
                --ckpt "$c2" \
                --layer "${LAYER}" --norm_mode "${NORM}" \
                --gpu "${gpu}" \
                --out_dir "${RESULT_DIR}" \
                2>&1
    ) > "${log}" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} parallel jobs (PIDs: ${PIDS[*]})"
echo "Logs: ${LOG_DIR}/"

# Wait and collect exit codes
FAILURES=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    gpu=${GPU_IDS[$i]}
    if wait "$pid"; then
        echo "[GPU ${gpu}] DONE (PID ${pid})"
    else
        echo "[GPU ${gpu}] FAILED (PID ${pid}, exit=$?)"
        FAILURES=$((FAILURES + 1))
    fi
done

# Merge all per-ckpt result JSONs into one summary
echo ""
if [ $FAILURES -eq 0 ]; then
    echo "All jobs completed successfully."
    echo "Merging results..."
    conda run --no-capture-output -n "${CONDA_ENV}" python -c "
import json, glob, os
result_dir = '${RESULT_DIR}'
files = sorted(glob.glob(os.path.join(result_dir, 'seg_eval_*.json')))
all_r = []
for f in files:
    with open(f) as fh:
        data = json.load(fh)
        if isinstance(data, list):
            all_r.extend(data)
        else:
            all_r.append(data)
# Deduplicate by ckpt path
seen = set()
unique = []
for r in all_r:
    k = r.get('ckpt', '')
    if k not in seen:
        seen.add(k)
        unique.append(r)
out = os.path.join(result_dir, 'seg_eval_all.json')
with open(out, 'w') as f:
    json.dump(unique, f, indent=2, default=str)
print(f'Merged {len(unique)} results -> {out}')
print()
for r in unique:
    name = os.path.basename(r['ckpt']).replace('.pt','')
    print(f\"  {name}\")
    print(f\"    mIoU={r['miou']:.4f}  aAcc={r['acc']:.4f}\")
if len(unique) > 1:
    import statistics
    mious = [r['miou'] for r in unique]
    print(f\"\n  Overall mean mIoU = {statistics.mean(mious):.4f}\")
    # Final seeds only
    finals = [r for r in unique if 'final_seed' in r['ckpt']]
    if len(finals) > 1:
        fm = [r['miou'] for r in finals]
        print(f\"  Final seeds mean  = {statistics.mean(fm):.4f} ± {statistics.stdev(fm):.4f}\")
"
else
    echo "WARNING: ${FAILURES} job(s) failed. Check logs in ${LOG_DIR}/"
fi
