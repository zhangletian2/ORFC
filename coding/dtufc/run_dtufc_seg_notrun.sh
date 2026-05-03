#!/bin/bash
# dtufc seg experiment: 4 layers x 1 lambda, NO truncation
# Uses data_seg_notrun + kmeans_8bit_notrun mapping
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPRESSAI_DIR="$SCRIPT_DIR/coding/CompressAI"
export PYTHONPATH="$PROJECT_ROOT/coding/orfc:$COMPRESSAI_DIR:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
WORK_DIR="$PROJECT_ROOT"
BACKBONE=dinov2_vitl14

LAMBDA_ALL=(0.005)
DEVICES=(0 1 2 3)

PREPROCESS_DIR=$WORK_DIR/features/preprocess
SEG_TEST_ROOT=$WORK_DIR/features/voc2012_100
GT_ROOT=$WORK_DIR/data/VOCdevkit/VOC2012/SegmentationClass
SEG_HEAD_PATH=$PROJECT_ROOT/pretrained/dinov2_vitl14_voc2012_linear_head.pth

TRANSFORM_TYPE=kmeans
BIT_DEPTH=8
LAYERS=(blk05 blk10 blk15 blk20)

# 250k patches, batch=128 -> ~1953 iters/epoch
# 16 epochs ~ 800 epochs with 5k images in total gradient updates
EPOCHS=16
EVAL_INTERVAL=1

# Sequential preload: read packed npy into page cache one-by-one
echo "=== Preloading mmap files into page cache ==="
for LAYER in "${LAYERS[@]}"; do
  PACKED=$PREPROCESS_DIR/data_seg_notrun/${BACKBONE}/${LAYER}/train_all.npy
  if [ -f "$PACKED" ]; then
    echo "  Preloading $LAYER ($(du -h "$PACKED" | cut -f1))..."
    dd if="$PACKED" of=/dev/null bs=4M 2>/dev/null
  fi
done
echo "=== Preload done, starting training ==="

for li in "${!LAYERS[@]}"; do
  LAYER=${LAYERS[$li]}
  DEVICE=${DEVICES[$li]}
  MAPPING=$PREPROCESS_DIR/transform_mapping/${TRANSFORM_TYPE}_${BIT_DEPTH}bit_notrun/${BACKBONE}/${LAYER}.json
  TRAIN_DATA=$PREPROCESS_DIR/data_seg_notrun/${BACKBONE}/${LAYER}
  for LAMBDA in "${LAMBDA_ALL[@]}"; do
    OUT_DIR=$WORK_DIR/coding/dtufc/checkpoints_notrun/seg_${BACKBONE}_${LAYER}_lmb${LAMBDA}
    mkdir -p ${OUT_DIR}

    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python -u $SCRIPT_DIR/train.py \
      --model bmshj2018-hyperprior \
      --model_type ${BACKBONE} \
      --layer ${LAYER} \
      --task seg \
      --savepath ${OUT_DIR}/checkpoint.pth.tar \
      --log_dir ${OUT_DIR}/tb \
      --train_data ${TRAIN_DATA} \
      --mapping ${MAPPING} \
      --seg_test_root ${SEG_TEST_ROOT} \
      --gt_root ${GT_ROOT} \
      --seg_head_path ${SEG_HEAD_PATH} \
      --bit_depth ${BIT_DEPTH} \
      --epochs ${EPOCHS} \
      --eval_interval ${EVAL_INTERVAL} \
      --num-workers 0 \
      --lambda ${LAMBDA} \
      --batch-size 128 \
      --test-batch-size 64 \
      --patch-size 256 256 \
      --cuda \
      > ${OUT_DIR}/log.txt 2>&1 &
  done
done
wait
