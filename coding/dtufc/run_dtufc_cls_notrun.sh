#!/bin/bash
# dtufc no-truncation R-D sweep: 4 rounds (one per blk), 5 lambdas x 5 GPUs
# Uses data_notrun + kmeans_8bit_notrun mapping (dtufc original: trun_flag=False)
# Resumes from existing checkpoints_notrun if available.
#
# Lambda design rationale (based on pilot runs):
#   blk05: lmb0.002->BPFP~0.35, lmb0.005->BPFP~1.22
#   blk10: lmb0.002->BPFP~0.48, lmb0.005->BPFP~1.47
#   blk15: lmb0.001->BPFP~0.34, lmb0.005->BPFP~1.72(too high)
#   blk20: lmb0.001->BPFP~0.26, lmb0.003->BPFP~1.02
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPRESSAI_DIR="$SCRIPT_DIR/coding/CompressAI"
export PYTHONPATH="$PROJECT_ROOT/coding/orfc:$COMPRESSAI_DIR:$PYTHONPATH"
WORK_DIR="$PROJECT_ROOT"
BACKBONE=dinov2_vitl14

DEVICES=(4 5 6 7)
GT_PATH=$WORK_DIR/utils/imagenet_selected_label500.txt
PREPROCESS_DIR=$WORK_DIR/features/preprocess

TRANSFORM_TYPE=kmeans
BIT_DEPTH=8

# Per-layer 5 lambdas targeting BPFP 0~1.5
declare -A LAMBDAS
LAMBDAS[blk05]="0.001 0.002 0.003 0.005"
LAMBDAS[blk10]="0.001 0.002 0.003 0.005"
LAMBDAS[blk15]="0.001 0.002 0.003 0.005"
LAMBDAS[blk20]="0.001 0.002 0.003 0.005"

LAYERS=(blk05 blk10 blk15 blk20)
EPOCHS=300
EVAL_INTERVAL=20

for LAYER in "${LAYERS[@]}"; do
  echo "========== Round: ${LAYER} =========="
  MAPPING=$PREPROCESS_DIR/transform_mapping/${TRANSFORM_TYPE}_${BIT_DEPTH}bit_notrun/${BACKBONE}/${LAYER}.json
  TRAIN_DATA=$PREPROCESS_DIR/data_notrun/${BACKBONE}/${LAYER}

  read -ra LMB_ARR <<< "${LAMBDAS[$LAYER]}"
  for idx in "${!LMB_ARR[@]}"; do
    LAMBDA=${LMB_ARR[$idx]}
    DEVICE=${DEVICES[$idx]}
    OUT_DIR=$WORK_DIR/coding/dtufc/checkpoints_notrun/${BACKBONE}_${LAYER}_lmb${LAMBDA}
    mkdir -p ${OUT_DIR}

    CKPT_ARG=""
    if [ -f "${OUT_DIR}/checkpoint.pth.tar" ]; then
      CKPT_ARG="--checkpoint ${OUT_DIR}/checkpoint.pth.tar"
      echo "  Resuming ${LAYER} lambda=${LAMBDA} on GPU ${DEVICE} (from ckpt)"
    else
      echo "  Starting ${LAYER} lambda=${LAMBDA} on GPU ${DEVICE} (fresh)"
    fi

    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python $SCRIPT_DIR/train.py \
      --model bmshj2018-hyperprior \
      --model_type ${BACKBONE} \
      --layer ${LAYER} \
      --trun_low 0 \
      --trun_high 0 \
      --savepath ${OUT_DIR}/checkpoint.pth.tar \
      --log_dir ${OUT_DIR}/tb \
      --train_data ${TRAIN_DATA} \
      --test_data ${WORK_DIR}/features \
      --mapping ${MAPPING} \
      --gt_path ${GT_PATH} \
      --bit_depth ${BIT_DEPTH} \
      --epochs ${EPOCHS} \
      --num-workers 8 \
      --lambda ${LAMBDA} \
      --batch-size 128 \
      --test-batch-size 64 \
      --patch-size 256 256 \
      --eval_interval ${EVAL_INTERVAL} \
      ${CKPT_ARG} \
      --cuda \
      >> ${OUT_DIR}/log.txt 2>&1 &
  done
  echo "  Waiting for ${LAYER} (5 jobs)..."
  wait
  echo "  ${LAYER} done."
done
echo "All rounds complete."
