#!/bin/bash
# dtufc R-D curve sweep: 4 rounds (one per blk), each round 5 lambdas on 5 GPUs
# Resumes from existing checkpoints if available.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPRESSAI_DIR="$SCRIPT_DIR/coding/CompressAI"
export PYTHONPATH="$PROJECT_ROOT/coding/orfc:$COMPRESSAI_DIR:$PYTHONPATH"
WORK_DIR="$PROJECT_ROOT"
BACKBONE=dinov2_vitl14

DEVICES=(3 4 5 6 7)
GT_PATH=$WORK_DIR/utils/imagenet_selected_label500.txt
PREPROCESS_DIR=$WORK_DIR/features/preprocess

declare -A TRUN_LOW TRUN_HIGH
TRUN_LOW[blk05]=-1;  TRUN_HIGH[blk05]=1
TRUN_LOW[blk10]=-1;  TRUN_HIGH[blk10]=1
TRUN_LOW[blk15]=-2;  TRUN_HIGH[blk15]=2
TRUN_LOW[blk20]=-5;  TRUN_HIGH[blk20]=5

TRANSFORM_TYPE=kmeans
BIT_DEPTH=8

declare -A LAMBDAS
LAMBDAS[blk05]="0.0005 0.001 0.002 0.003 0.005"
LAMBDAS[blk10]="0.0005 0.001 0.002 0.003 0.005"
LAMBDAS[blk15]="0.0003 0.0005 0.001 0.002 0.003"
LAMBDAS[blk20]="0.0002 0.0005 0.001 0.002 0.003"

LAYERS=(blk05 blk10 blk15 blk20)
EPOCHS=500
CLS_INTERVAL=20

for LAYER in "${LAYERS[@]}"; do
  echo "========== Round: ${LAYER} =========="
  MAPPING=$PREPROCESS_DIR/transform_mapping/${TRANSFORM_TYPE}_${BIT_DEPTH}bit/${BACKBONE}/${LAYER}.json
  TRAIN_DATA=$PREPROCESS_DIR/data/${BACKBONE}/${LAYER}

  read -ra LMB_ARR <<< "${LAMBDAS[$LAYER]}"
  for idx in "${!LMB_ARR[@]}"; do
    LAMBDA=${LMB_ARR[$idx]}
    DEVICE=${DEVICES[$idx]}
    OUT_DIR=$WORK_DIR/coding/dtufc/checkpoints/${BACKBONE}_${LAYER}_lmb${LAMBDA}
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
      --trun_low ${TRUN_LOW[$LAYER]} \
      --trun_high ${TRUN_HIGH[$LAYER]} \
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
      --cls_eval_interval ${CLS_INTERVAL} \
      ${CKPT_ARG} \
      --cuda \
      >> ${OUT_DIR}/log.txt 2>&1 &
  done
  echo "  Waiting for ${LAYER} (5 jobs)..."
  wait
  echo "  ${LAYER} done."
done
echo "All rounds complete."
