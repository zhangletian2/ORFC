#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PY=${PY:-/home/user/anaconda3/envs/featcodec2/bin/python}
MENU=${MENU:?set MENU to a multi-mode warmup checkpoint}
CALIBRATION=${CALIBRATION:?set CALIBRATION to discrete calibration_R192.npz}
RUN_ID=${RUN_ID:-outer_inner_short_$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-results/dinov2_vitl14/rate_allocation/$RUN_ID}
CACHE=artifacts/dinov2_vitl14/cache
STEPS=${STEPS:-10}
OUTER_CALIBRATION_IMAGES=${OUTER_CALIBRATION_IMAGES:-300}
OUTER_MINING_OFFSET=${OUTER_MINING_OFFSET:-300}
TRAIN_IMAGE_OFFSET=${TRAIN_IMAGE_OFFSET:-364}

mkdir -p "$OUT/logs"
run_one() {
  local gpu=$1 name=$2 ratio=$3
  CUDA_VISIBLE_DEVICES=$gpu "$PY" allocation_train.py short \
    --codec "$MENU" --calibration "$CALIBRATION" \
    --features "$CACHE/features_train_blk20_n4500_ss1608637542.npy" \
    --teachers "$CACHE/teacher_train_blk20_n4500_ss1608637542.npy" \
    --hard-features "$CACHE/features_val_blk20_n500_ss1608637542.npy" \
    --hard-teachers "$CACHE/teacher_val_blk20_n500_ss1608637542.npy" \
    --output "$OUT/$name.json" --checkpoint "$OUT/$name.pt" \
    --steps "$STEPS" --images 4 --train-images 512 \
    --train-image-offset "$TRAIN_IMAGE_OFFSET" \
    --hard-images 64 --hard-image-offset 0 \
    --hard-batch-size 8 --allocation-chunk 8 --allocations 16 \
    --refresh-steps 5 --select-steps 5 --log-steps 1 \
    --outer-refresh --outer-calibration-images "$OUTER_CALIBRATION_IMAGES" \
    --minimum-saved-calibration-images 300 \
    --outer-calibration-offset 0 --outer-mining-images 64 \
    --outer-mining-offset "$OUTER_MINING_OFFSET" --outer-batch-size 8 \
    --outer-group-chunk 8 --outer-eps 0.01 \
    --dynamic-allocations --dynamic-single 32 --dynamic-random 32 \
    --ideal-set-size "${IDEAL_SET_SIZE:-256}" \
    --ideal-batch-size "${IDEAL_BATCH_SIZE:-16}" \
    --primary-target "${PRIMARY_TARGET:-outer_operational_best}" \
    --candidate-mean-weight "${CANDIDATE_MEAN_WEIGHT:-0}" \
    --recovery-margin 0 --remainder-grad-ratio "$ratio" \
    --rotation-lr 0.00001 --lr 0.00001 \
    --tau-start 0.01 --tau-end 0.01 --seed 42 \
    >"$OUT/logs/$name.log" 2>&1
}

run_one "${GPU_BASE:-4}" primary_only 0 &
p0=$!
run_one "${GPU_RECOVERY:-5}" primary_recovery 0.25 &
p1=$!
wait "$p0" "$p1"
echo "$OUT"
