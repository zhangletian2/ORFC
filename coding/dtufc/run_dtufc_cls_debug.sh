#!/bin/bash
# dtufc reproduction: kmeans preprocessing + hyperprior compression + cls evaluation
# Follows ORFC/coding/CompressAI/run_hyperprior_cls.sh framework
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPRESSAI_DIR="$SCRIPT_DIR/coding/CompressAI"
export PYTHONPATH="$PROJECT_ROOT/coding/orfc:$COMPRESSAI_DIR:$PYTHONPATH"
WORK_DIR="$PROJECT_ROOT"
BACKBONE=dinov2_vitl14

LAMBDA_ALL=(0.005)
DEVICES=(4 5 6 7)
GT_PATH=$WORK_DIR/utils/imagenet_selected_label500.txt
PREPROCESS_DIR=$WORK_DIR/features/preprocess

declare -A TRUN_LOW TRUN_HIGH
TRUN_LOW[blk05]=-1;  TRUN_HIGH[blk05]=1
TRUN_LOW[blk10]=-1;  TRUN_HIGH[blk10]=1
TRUN_LOW[blk15]=-2;  TRUN_HIGH[blk15]=2
TRUN_LOW[blk20]=-5;  TRUN_HIGH[blk20]=5

TRANSFORM_TYPE=kmeans
BIT_DEPTH=8
LAYERS=(blk05 blk10 blk15 blk20)

for li in "${!LAYERS[@]}"; do
  LAYER=${LAYERS[$li]}
  DEVICE=${DEVICES[$li]}
  MAPPING=$PREPROCESS_DIR/transform_mapping/${TRANSFORM_TYPE}_${BIT_DEPTH}bit/${BACKBONE}/${LAYER}.json
  TRAIN_DATA=$PREPROCESS_DIR/data/${BACKBONE}/${LAYER}
  for LAMBDA in "${LAMBDA_ALL[@]}"; do
    OUT_DIR=$WORK_DIR/coding/dtufc/checkpoints/${BACKBONE}_${LAYER}_lmb${LAMBDA}
    mkdir -p ${OUT_DIR}

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
      --epochs 800 \
      --num-workers 8 \
      --lambda ${LAMBDA} \
      --batch-size 128 \
      --test-batch-size 64 \
      --patch-size 256 256 \
      --cuda \
      > ${OUT_DIR}/log.txt 2>&1 &
  done
done
wait
