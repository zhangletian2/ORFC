#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PY=${PY:-/home/user/anaconda3/envs/featcodec2/bin/python}
MENU=${MENU:?set MENU to menu_warmup.pt from run_allocation_short.sh}
CALIBRATION=${CALIBRATION:?set CALIBRATION to its calibration_R192.npz}
RUN_ID=${RUN_ID:-allocation_full_$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-results/dinov2_vitl14/rate_allocation/$RUN_ID}
CACHE=artifacts/dinov2_vitl14/cache
EPOCHS=${EPOCHS:-100}
TRAIN_IMAGES=${TRAIN_IMAGES:-4500}
IMAGES=${IMAGES:-32}
STEPS_PER_EPOCH=$(( (TRAIN_IMAGES + IMAGES - 1) / IMAGES ))

# The original 5k pool is split into 4.5k optimisation and 500 held-out
# validation images. This preserves test isolation while matching its scale.
mkdir -p "$OUT"/logs
train_one() {
  local gpu=$1 name=$2 ratio=$3
  CUDA_VISIBLE_DEVICES=$gpu "$PY" allocation_train.py short \
    --codec "$MENU" \
    --calibration "$CALIBRATION" \
    --features "$CACHE/features_train_blk20_n4500_ss1608637542.npy" \
    --teachers "$CACHE/teacher_train_blk20_n4500_ss1608637542.npy" \
    --hard-features "$CACHE/features_val_blk20_n500_ss1608637542.npy" \
    --hard-teachers "$CACHE/teacher_val_blk20_n500_ss1608637542.npy" \
    --output "$OUT/$name.json" --checkpoint "$OUT/$name.pt" \
    --epochs "$EPOCHS" --train-images "$TRAIN_IMAGES" --images "$IMAGES" \
    --refresh-images 64 --refresh-steps "$STEPS_PER_EPOCH" \
    --select-steps "$STEPS_PER_EPOCH" --log-steps 25 \
    --hard-images 128 --hard-image-offset 0 --hard-batch-size 16 \
    --allocations 8 --allocation-chunk 8 \
    --tau-start 0.5 --tau-end 0.005 --schedule-unit epoch --lr 0.0003 \
    --rotation-lr "${ROTATION_LR:-0.0003}" \
    --monotonic-tolerance 0.001 --remainder-grad-ratio "$ratio" \
    --dynamic-allocations --dynamic-single 32 --dynamic-random 32 \
    >"$OUT/logs/$name.log" 2>&1
}

train_one "${GPU_BASE:-4}" target_only 0 &
p0=$!
train_one "${GPU_PROXY:-5}" target_recovery 0.25 &
p1=$!
wait "$p0" "$p1"
echo "$OUT"
