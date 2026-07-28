#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PY=${PY:-/home/user/anaconda3/envs/featcodec2/bin/python}
RUN_ID=${RUN_ID:-allocation_short_$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-results/dinov2_vitl14/rate_allocation/$RUN_ID}
CACHE=artifacts/dinov2_vitl14/cache
INIT=${INIT:-opq}
WARMUP_STEPS=${WARMUP_STEPS:-100}
SHORT_STEPS=${SHORT_STEPS:-10}

mkdir -p "$OUT"/logs
CUDA_VISIBLE_DEVICES=${GPU_INIT:-4} "$PY" p1_fixed_rate.py prepare \
  --arm "$INIT" --features "$CACHE/features_train_blk20_n4500_ss1608637542.npy" \
  --output "$OUT/menu_init.pt" --source-kind "$INIT" \
  --mode-bits 3,4,5,6,7,8 --groups 32 --images 1000 \
  --max-vectors 62500 --kmeans-iters 50 --opq-bits 6 --opq-iters 20 \
  --seed 42 \
  >"$OUT/logs/menu_prepare.log" 2>&1

CUDA_VISIBLE_DEVICES=${GPU_INIT:-4} "$PY" allocation_train.py warmup \
  --codec "$OUT/menu_init.pt" \
  --features "$CACHE/features_train_blk20_n4500_ss1608637542.npy" \
  --teachers "$CACHE/teacher_train_blk20_n4500_ss1608637542.npy" \
  --hard-features "$CACHE/features_val_blk20_n500_ss1608637542.npy" \
  --hard-teachers "$CACHE/teacher_val_blk20_n500_ss1608637542.npy" \
  --output "$OUT/menu_warmup.json" --checkpoint "$OUT/menu_warmup.pt" \
  --steps "$WARMUP_STEPS" --log-steps 10 --images 4 \
  --hard-images 64 --hard-image-offset 0 --allocation-chunk 2 --seed 42 \
  >"$OUT/logs/menu_warmup.log" 2>&1

mkdir -p "$OUT/calibration"
CUDA_VISIBLE_DEVICES=${GPU_INIT:-4} "$PY" p1_fixed_rate.py calibrate \
  --arm "$INIT" --codec "$OUT/menu_warmup.pt" \
  --features "$CACHE/features_train_blk20_n4500_ss1608637542.npy" \
  --output-dir "$OUT/calibration" --mode-bits 3,4,5,6,7,8 \
  --budgets 192 --reference-bits 6 --images 64 --group-chunk 8 \
  --single 32 --random 32 --seed 42 \
  >"$OUT/logs/calibration.log" 2>&1

short_run() {
  local gpu=$1 name=$2 ratio=$3
  CUDA_VISIBLE_DEVICES=$gpu "$PY" allocation_train.py short \
    --codec "$OUT/menu_warmup.pt" \
    --calibration "$OUT/calibration/calibration_R192.npz" \
    --features "$CACHE/features_train_blk20_n4500_ss1608637542.npy" \
    --teachers "$CACHE/teacher_train_blk20_n4500_ss1608637542.npy" \
    --hard-features "$CACHE/features_val_blk20_n500_ss1608637542.npy" \
    --hard-teachers "$CACHE/teacher_val_blk20_n500_ss1608637542.npy" \
    --output "$OUT/$name.json" --checkpoint "$OUT/$name.pt" \
    --steps "$SHORT_STEPS" --refresh-steps 5 --images 4 \
    --refresh-images 32 --hard-images 64 --hard-image-offset 64 \
    --allocations 64 --allocation-chunk 4 --dynamic-allocations \
    --monotonic-tolerance 0.001 \
    --remainder-grad-ratio "$ratio" \
    >"$OUT/logs/$name.log" 2>&1
}

short_run "${GPU_BASE:-4}" candidate_only 0 &
p0=$!
short_run "${GPU_PROXY:-5}" candidate_remainder 0.25 &
p1=$!
wait "$p0" "$p1"
echo "$OUT"
