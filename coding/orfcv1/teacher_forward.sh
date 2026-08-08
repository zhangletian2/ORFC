#!/bin/bash
# Teacher (= tail(features)) forward for the six new caches, one GPU each.
# teacher_blkNN = blocks[NN+1:] + backbone.norm applied to the raw features,
# matching allocation_train._distortions.
set -uo pipefail
cd /data4/workspace/zlt/featcodec/ORFC/coding/orfcv1
PY=/home/user/anaconda3/envs/featcodec2/bin/python
C=artifacts/dinov2_vitl14/cache
TAG=ss20260730

i=2
for spec in "blk05 5 train 4500" "blk05 5 val 500" "blk05 5 test 3000" \
            "blk20 20 train 4500" "blk20 20 val 500" "blk20 20 test 3000"; do
  set -- $spec; B=$1; L=$2; S=$3; N=$4
  CUDA_VISIBLE_DEVICES=$i PYTHONPATH=$PWD nohup $PY build_cache.py teacher \
      --layer $L --features $C/features_${S}_${B}_n${N}_${TAG}.npy \
      --out $C/teacher_${S}_${B}_n${N}_${TAG}.npy --batch 16 \
      > /tmp/teacher_${B}_${S}.log 2>&1 &
  echo "gpu$i <- ${B} ${S} n${N} (pid $!)"
  i=$((i+1))
done
wait
echo "ALL TEACHER FORWARDS DONE"
