#!/bin/bash
# dtufc seg R-D sweep: 4 rounds (one per blk), 5 lambdas x 5 GPUs
# NO truncation. Uses packed mmap data + kmeans_8bit_notrun mapping.
#
# Schedule: sequential rounds, parallel lambdas within each round.
#   Round 1: preload blk05 → launch 5 lambda jobs → wait
#   Round 2: preload blk10 → launch 5 lambda jobs → wait
#   ...
#
# Lambda design (targeting BPFP 0~1.5):
#   0.0005 → ~0.1   0.001 → ~0.3   0.003 → ~0.7
#   0.005  → ~1.0   0.01  → ~1.5
#
# Resume: loads checkpoint.pth.tar if present.
# Skip:   skips if final_eval_seg_results.json exists.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPRESSAI_DIR="$SCRIPT_DIR/coding/CompressAI"
export PYTHONPATH="$PROJECT_ROOT/coding/orfc:$COMPRESSAI_DIR:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

WORK_DIR="$PROJECT_ROOT"
BACKBONE=dinov2_vitl14
PREPROCESS_DIR=$WORK_DIR/features/preprocess
SEG_TEST_ROOT=$WORK_DIR/features/voc2012_100
GT_ROOT=$WORK_DIR/data/VOCdevkit/VOC2012/SegmentationClass
SEG_HEAD_PATH=$PROJECT_ROOT/pretrained/dinov2_vitl14_voc2012_linear_head.pth

TRANSFORM_TYPE=kmeans
BIT_DEPTH=8

LAYERS=(blk05 blk10 blk15 blk20)
LAMBDA_ALL=(0.0005 0.001 0.003 0.005 0.01)
DEVICES=(0 1 2 3 4)

EPOCHS=16
EVAL_INTERVAL=1

for LAYER in "${LAYERS[@]}"; do
  echo ""
  echo "============================================================"
  echo "  Round: ${LAYER}  ($(date '+%H:%M:%S'))"
  echo "============================================================"

  # --- Preload mmap file into page cache (sequential read) ---
  PACKED=$PREPROCESS_DIR/data_seg_notrun/${BACKBONE}/${LAYER}/train_all.npy
  if [ -f "$PACKED" ]; then
    echo "  Preloading $LAYER ($(du -h "$PACKED" | cut -f1))..."
    dd if="$PACKED" of=/dev/null bs=4M 2>/dev/null
    echo "  Preload done."
  fi

  # --- Launch 5 lambda jobs in parallel ---
  MAPPING=$PREPROCESS_DIR/transform_mapping/${TRANSFORM_TYPE}_${BIT_DEPTH}bit_notrun/${BACKBONE}/${LAYER}.json
  TRAIN_DATA=$PREPROCESS_DIR/data_seg_notrun/${BACKBONE}/${LAYER}
  LAUNCHED=0

  for idx in "${!LAMBDA_ALL[@]}"; do
    LAMBDA=${LAMBDA_ALL[$idx]}
    DEVICE=${DEVICES[$idx]}
    OUT_DIR=$WORK_DIR/coding/dtufc/checkpoints_notrun/seg_${BACKBONE}_${LAYER}_lmb${LAMBDA}
    mkdir -p "${OUT_DIR}"

    # Skip if final results already exist
    if [ -f "${OUT_DIR}/final_eval_seg_results.json" ]; then
      echo "  [SKIP] ${LAYER} lmb=${LAMBDA} — results exist"
      continue
    fi

    # Resume from checkpoint if available
    CKPT_ARG=""
    if [ -f "${OUT_DIR}/checkpoint.pth.tar" ]; then
      CKPT_ARG="--checkpoint ${OUT_DIR}/checkpoint.pth.tar"
      echo "  [RESUME] ${LAYER} lmb=${LAMBDA} on GPU ${DEVICE}"
    else
      echo "  [START]  ${LAYER} lmb=${LAMBDA} on GPU ${DEVICE}"
    fi

    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python "$SCRIPT_DIR/train.py" \
      --model bmshj2018-hyperprior \
      --model_type ${BACKBONE} \
      --layer ${LAYER} \
      --task seg \
      --savepath "${OUT_DIR}/checkpoint.pth.tar" \
      --log_dir "${OUT_DIR}/tb" \
      --train_data "${TRAIN_DATA}" \
      --mapping "${MAPPING}" \
      --seg_test_root "${SEG_TEST_ROOT}" \
      --gt_root "${GT_ROOT}" \
      --seg_head_path "${SEG_HEAD_PATH}" \
      --bit_depth ${BIT_DEPTH} \
      --epochs ${EPOCHS} \
      --eval_interval ${EVAL_INTERVAL} \
      --num-workers 0 \
      --lambda ${LAMBDA} \
      --batch-size 128 \
      --test-batch-size 64 \
      --patch-size 256 256 \
      ${CKPT_ARG} \
      --cuda \
      > "${OUT_DIR}/log.txt" 2>&1 &

    LAUNCHED=$((LAUNCHED + 1))
  done

  if [ ${LAUNCHED} -gt 0 ]; then
    echo "  Waiting for ${LAYER} (${LAUNCHED} jobs)..."
    wait
    echo "  ${LAYER} done. ($(date '+%H:%M:%S'))"
  else
    echo "  ${LAYER}: all configs already complete, skipping."
  fi
done

echo ""
echo "============================================================"
echo "  All rounds complete. ($(date '+%H:%M:%S'))"
echo "============================================================"
