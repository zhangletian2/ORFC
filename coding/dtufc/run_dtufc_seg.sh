#!/bin/bash
# dtufc reproduction: kmeans preprocessing + hyperprior compression + seg evaluation
# Follows ORFC/coding/CompressAI/run_hyperprior_seg.sh framework
#
# Prerequisites:
#   1. Pre-crop seg features (if not done):
#      python $PROJECT_ROOT/tools/precrop_seg_features.py \
#        --base_src $PROJECT_ROOT/features/voc2012_5000/dinov2_vitl14 \
#        --base_dst $PROJECT_ROOT/features/voc2012_5000_crops/train/dinov2_vitl14 \
#        --layers blk05 blk10 blk15 blk20 --num_crops 50 --workers 8
#
#   2. Preprocess seg data (offline kmeans quantization):
#      python $PREPROCESS_DIR/preprocess.py fit_seg_mapping \
#        --layers blk20 --seg_train_root $PROJECT_ROOT/features/voc2012_5000
#      python $PREPROCESS_DIR/preprocess.py generate_seg_data \
#        --layers blk20 \
#        --seg_train_root $PROJECT_ROOT/features/voc2012_5000_crops \
#        --seg_test_root $PROJECT_ROOT/features/voc2012_100

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPRESSAI_DIR="$SCRIPT_DIR/coding/CompressAI"
export PYTHONPATH="$PROJECT_ROOT/coding/orfc:$COMPRESSAI_DIR:$PYTHONPATH"
WORK_DIR="$PROJECT_ROOT"
BACKBONE=dinov2_vitl14

# blk20 lambdas (same as CompressAI seg pipeline)
LAMBDA_ALL=(0.002 0.003 0.005 0.007)
DEVICES=(0 1 2 3)

PREPROCESS_DIR=$WORK_DIR/features/preprocess
SEG_TEST_ROOT=$WORK_DIR/features/voc2012_100
GT_ROOT=$WORK_DIR/data/VOCdevkit/VOC2012/SegmentationClass
SEG_HEAD_PATH=$PROJECT_ROOT/pretrained/dinov2_vitl14_voc2012_linear_head.pth

# 250k patches, batch=128 → ~1953 batches/epoch
# 16 epochs ≈ 800 epochs (original 5k images) in gradient updates
EPOCHS=16

declare -A TRUN_LOW TRUN_HIGH
TRUN_LOW[blk05]=-1;  TRUN_HIGH[blk05]=1
TRUN_LOW[blk10]=-1;  TRUN_HIGH[blk10]=1
TRUN_LOW[blk15]=-2;  TRUN_HIGH[blk15]=2
TRUN_LOW[blk20]=-5;  TRUN_HIGH[blk20]=5
TRUN_LOW[blk23]=-10; TRUN_HIGH[blk23]=10

TRANSFORM_TYPE=kmeans
BIT_DEPTH=8
LAYERS=(blk20)

for LAYER in "${LAYERS[@]}"; do
  MAPPING=$PREPROCESS_DIR/transform_mapping/${TRANSFORM_TYPE}_${BIT_DEPTH}bit/${BACKBONE}/${LAYER}.json
  TRAIN_DATA=$PREPROCESS_DIR/data_seg/${BACKBONE}/${LAYER}
  for idx in "${!LAMBDA_ALL[@]}"; do
    LAMBDA=${LAMBDA_ALL[$idx]}
    DEVICE=${DEVICES[$idx]}
    OUT_DIR=$WORK_DIR/coding/dtufc/checkpoints/seg_${BACKBONE}_${LAYER}_lmb${LAMBDA}
    mkdir -p ${OUT_DIR}

    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python $SCRIPT_DIR/train.py \
      --model bmshj2018-hyperprior \
      --model_type ${BACKBONE} \
      --layer ${LAYER} \
      --task seg \
      --trun_low ${TRUN_LOW[$LAYER]} \
      --trun_high ${TRUN_HIGH[$LAYER]} \
      --savepath ${OUT_DIR}/checkpoint.pth.tar \
      --log_dir ${OUT_DIR}/tb \
      --train_data ${TRAIN_DATA} \
      --mapping ${MAPPING} \
      --seg_test_root ${SEG_TEST_ROOT} \
      --gt_root ${GT_ROOT} \
      --seg_head_path ${SEG_HEAD_PATH} \
      --bit_depth ${BIT_DEPTH} \
      --epochs ${EPOCHS} \
      --eval_interval 4 \
      --num-workers 16 \
      --lambda ${LAMBDA} \
      --batch-size 128 \
      --test-batch-size 64 \
      --patch-size 256 256 \
      --cuda \
      > ${OUT_DIR}/log.txt 2>&1 &
  done
  wait
done
