#!/bin/bash
# Stack the six new feature caches. IO-bound, so sequential.
#   train n4500 = codebook fitting   (train pool, frozen split index)
#   val   n500  = dev                (train pool, frozen split index)
#   test  n3000 = measure            (val pool, whole thing)
# Tag ss20260730 marks the uniform-provenance re-extraction of 2026-07-30.
set -euo pipefail
cd /data4/workspace/zlt/featcodec/ORFC/coding/orfcv1
PY=/home/user/anaconda3/envs/featcodec2/bin/python
S=artifacts/dinov2_vitl14/split
C=artifacts/dinov2_vitl14/cache
TAG=ss20260730

for B in blk05 blk20; do
  echo "=== $B train n4500"
  $PY build_cache.py features --pool train --block $B \
      --index $S/train_idx_n4500.npy --out $C/features_train_${B}_n4500_${TAG}.npy
  echo "=== $B val n500 (dev)"
  $PY build_cache.py features --pool train --block $B \
      --index $S/dev_idx_n500.npy --out $C/features_val_${B}_n500_${TAG}.npy
  echo "=== $B test n3000 (measure)"
  $PY build_cache.py features --pool val --block $B \
      --out $C/features_test_${B}_n3000_${TAG}.npy
done
echo "ALL FEATURE STACKS DONE"
