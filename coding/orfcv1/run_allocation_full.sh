#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PY=${PY:-/home/user/anaconda3/envs/featcodec2/bin/python}
MENU=${MENU:?set MENU to a validated multi-mode warmup checkpoint}
CALIBRATION=${CALIBRATION:?set CALIBRATION to a 300-image discrete calibration}
RUN_ID=${RUN_ID:-allocation_full_$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-results/dinov2_vitl14/rate_allocation/$RUN_ID}
CACHE=artifacts/dinov2_vitl14/cache
EPOCHS=${EPOCHS:-100}
TRAIN_IMAGES=${TRAIN_IMAGES:-3900}
IMAGES=${IMAGES:-32}
STEPS_PER_EPOCH=$(( (TRAIN_IMAGES + IMAGES - 1) / IMAGES ))
REFRESH_EPOCHS=${REFRESH_EPOCHS:-10}
REFRESH_STEPS=$(( STEPS_PER_EPOCH * REFRESH_EPOCHS ))
STATE="$OUT/shared_initial_outer.npz"

mkdir -p "$OUT"/logs
COMMON=(
  --codec "$MENU" --calibration "$CALIBRATION"
  --features "$CACHE/features_train_blk20_n4500_ss1608637542.npy"
  --teachers "$CACHE/teacher_train_blk20_n4500_ss1608637542.npy"
  --hard-features "$CACHE/features_val_blk20_n500_ss1608637542.npy"
  --hard-teachers "$CACHE/teacher_val_blk20_n500_ss1608637542.npy"
  --epochs "$EPOCHS" --train-images "$TRAIN_IMAGES" --train-image-offset 600
  --images "$IMAGES" --refresh-steps "$REFRESH_STEPS"
  --select-steps "$REFRESH_STEPS" --log-steps 25
  --hard-images 300 --hard-image-offset 0 --hard-batch-size 16
  --allocations 64 --allocation-chunk 8
  --tau-start 0.5 --tau-end 0.005 --schedule-unit epoch --lr 0.0003
  --rotation-lr "${ROTATION_LR:-0.0003}" --monotonic-tolerance 0.001
  --dynamic-allocations --dynamic-single 32 --dynamic-random 32
  --outer-refresh --outer-calibration-images 300
  --outer-calibration-offset 0 --outer-mining-images 300
  --outer-mining-offset 300 --outer-batch-size 8 --outer-group-chunk 8
  --minimum-saved-calibration-images 300 --outer-eps 0.01
  --ideal-set-size 256 --ideal-batch-size 16
  --primary-target outer_operational_best --candidate-mean-weight 0.05
  --recovery-aggregate max --aux-calibration-images 64
  --audit-bootstraps 2000 --seed 42
)

CUDA_VISIBLE_DEVICES="${GPU_STATE:-4}" "$PY" allocation_train.py short \
  "${COMMON[@]}" --remainder-grad-ratio 0 \
  --output "$OUT/shared_unused.json" --checkpoint "$OUT/shared_unused.pt" \
  --outer-state-output "$STATE" --prepare-outer-only \
  >"$OUT/logs/shared_outer.log" 2>&1

train_one() {
  local gpu=$1 name=$2 ratio=$3 objective=$4
  shift 4
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" allocation_train.py short \
    "${COMMON[@]}" --initial-outer-state "$STATE" \
    --remainder-grad-ratio "$ratio" --auxiliary-objective "$objective" "$@" \
    --output "$OUT/$name.json" --checkpoint "$OUT/$name.pt" \
    >"$OUT/logs/$name.log" 2>&1
}

train_one "${GPU_BASE:-4}" primary_only 0 recovery &
p0=$!
train_one "${GPU_RECOVERY:-5}" primary_recovery 0.25 recovery \
  --outer-gated-recovery --recovery-pairs "${RECOVERY_PAIRS:-4}" &
p1=$!
train_one "${GPU_REMAINDER:-6}" primary_remainder 0.25 remainder_range &
p2=$!
failed=0
wait "$p0" || failed=1
wait "$p1" || failed=1
wait "$p2" || failed=1
if (( failed )); then
  echo "one or more full-training arms failed; inspect $OUT/logs" >&2
  exit 1
fi
echo "$OUT"
