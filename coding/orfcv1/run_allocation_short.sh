#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PY=${PY:-/home/user/anaconda3/envs/featcodec2/bin/python}
RUN_ID=${RUN_ID:-allocation_short_$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-results/dinov2_vitl14/rate_allocation/$RUN_ID}
CACHE=artifacts/dinov2_vitl14/cache
P1=results/dinov2_vitl14/p1_fixed_rate/formal_p1_corrected_20260727T090628Z
ANCHOR=artifacts/dinov2_vitl14/p1_menus/orfc_k64_anchor.pt
REPAIR_STEPS=${REPAIR_STEPS:-100}
SHORT_STEPS=${SHORT_STEPS:-10}

mkdir -p "$OUT"/logs
CUDA_VISIBLE_DEVICES=${GPU_REPAIR:-4} "$PY" p1_fixed_rate.py prepare \
  --arm candidate --features "$CACHE/features_train_blk20_n4500_ss1608637542.npy" \
  --output "$OUT/menu_anchored.pt" \
  --source-kind checkpoint --source "$ANCHOR" \
  --anchor-codec "$ANCHOR" --anchor-bits 6 \
  --mode-bits 3,4,5,6,7,8 --groups 32 --images 1000 \
  --max-vectors 62500 --kmeans-iters 100 --seed 42 \
  >"$OUT/logs/menu_prepare.log" 2>&1

CUDA_VISIBLE_DEVICES=${GPU_REPAIR:-4} "$PY" allocation_train.py repair \
  --codec "$OUT/menu_anchored.pt" \
  --features "$CACHE/features_train_blk20_n4500_ss1608637542.npy" \
  --teachers "$CACHE/teacher_train_blk20_n4500_ss1608637542.npy" \
  --hard-features "$CACHE/features_val_blk20_n500_ss1608637542.npy" \
  --hard-teachers "$CACHE/teacher_val_blk20_n500_ss1608637542.npy" \
  --output "$OUT/menu_repair.json" --checkpoint "$OUT/menu_repaired.pt" \
  --steps "$REPAIR_STEPS" --eval-steps 10 --images 4 \
  --hard-images 64 --hard-image-offset 0 --allocation-chunk 2 \
  --minimum-high-rate-gain 0 \
  >"$OUT/logs/menu_repair.log" 2>&1

short_run() {
  local gpu=$1 name=$2 ratio=$3
  CUDA_VISIBLE_DEVICES=$gpu "$PY" allocation_train.py short \
    --codec "$OUT/menu_repaired.pt" \
    --allocation-file "$P1/response/measurement_R192.npz" \
    --calibration "$P1/response/calibration_R192.npz" \
    --features "$CACHE/features_train_blk20_n4500_ss1608637542.npy" \
    --teachers "$CACHE/teacher_train_blk20_n4500_ss1608637542.npy" \
    --hard-features "$CACHE/features_val_blk20_n500_ss1608637542.npy" \
    --hard-teachers "$CACHE/teacher_val_blk20_n500_ss1608637542.npy" \
    --output "$OUT/$name.json" --checkpoint "$OUT/$name.pt" \
    --steps "$SHORT_STEPS" --refresh-steps 5 --images 4 \
    --refresh-images 8 --hard-images 64 --hard-image-offset 64 \
    --allocations 64 --allocation-chunk 4 \
    --remainder-grad-ratio "$ratio" \
    >"$OUT/logs/$name.log" 2>&1
}

short_run "${GPU_BASE:-4}" candidate_only 0 &
p0=$!
short_run "${GPU_PROXY:-5}" candidate_remainder 0.25 &
p1=$!
wait "$p0" "$p1"
echo "$OUT"
