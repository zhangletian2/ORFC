#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PY=${PY:-/home/user/anaconda3/envs/featcodec2/bin/python}
RUN_ID=${RUN_ID:?set RUN_ID}
P1=${P1:-results/dinov2_vitl14/p1_fixed_rate/formal_p1_corrected_20260727T090628Z}
BASE=${BASE:-results/dinov2_vitl14/p1_fixed_rate/formal_p1_20260727T084459Z/response/menu.pt}
OUT=${OUT:-$P1/pretrain_checks/$RUN_ID}
CACHE=artifacts/dinov2_vitl14/cache
MEAS=$P1/response/measurement_R192.npz
CAL=$P1/response/calibration_R192.npz
DEV=$OUT/alloc_holdout_dev_R192.npz
ALLOCATIONS=${ALLOCATIONS:-64}
PREV=$P1/pretrain_checks/p2_compare_formal_20260727/p2_independent_20260727T143713Z
exclude=("$MEAS")
for path in "$PREV"/alloc_holdout_{dev,final}_R192.npz; do
  [[ ! -f $path ]] || exclude+=("$path")
done

mkdir -p "$OUT"/{dev,logs}
"$PY" p2_pretrain.py manifest --source "$MEAS" --calibration "$CAL" \
  --exclude "${exclude[@]}" --output "$DEV" --samples 261 --seed 3042

common=(
  --codec "$BASE" --measurement "$MEAS" --calibration "$CAL"
  --features "$CACHE/features_train_blk20_n4500_ss1608637542.npy"
  --teachers "$CACHE/teacher_train_blk20_n4500_ss1608637542.npy"
  --hard-features "$CACHE/features_val_blk20_n500_ss1608637542.npy"
  --hard-teachers "$CACHE/teacher_val_blk20_n500_ss1608637542.npy"
  --parameters u --assignment soft --anchor-weight 1
  --images 8 --allocations "$ALLOCATIONS" --allocation-chunk 4
  --hard-images 32 --hard-batch-size 4 --hard-allocation-chunk 8
  --steps 10 --refresh-steps 0 --lr 1e-5 --checkpoint-every 1
  --grad-stats-every 0 --seed 42
)

run_arm() {
  local gpu=$1 name=$2 ratio=$3 extra
  if [[ $name == mean ]]; then
    extra=(--remainder-weight 0)
  else
    extra=(--target-remainder-grad-ratio "$ratio")
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$PY" p2_pretrain.py step "${common[@]}" \
    "${extra[@]}" --output "$OUT/$name.json" --checkpoint "$OUT/$name.pt" \
    >"$OUT/logs/train_$name.log" 2>&1
}

run_arm "${GPU_MEAN:-4}" mean 0 & p0=$!
run_arm "${GPU_R025:-5}" r025 0.25 & p1=$!
run_arm "${GPU_R050:-6}" r050 0.50 & p2=$!
run_arm "${GPU_R100:-7}" r100 1.00 & p3=$!
wait "$p0" "$p1" "$p2" "$p3"

eval_arm() {
  local gpu=$1 name=$2 ckpt stem
  for ckpt in "$OUT/$name".step*.pt; do
    stem=${ckpt%.pt}
    stem=${stem##*/}
    CUDA_VISIBLE_DEVICES=$gpu "$PY" p2_pretrain.py arrays \
      --codec "$ckpt" --measurement "$DEV" --calibration "$CAL" \
      --coefficients "$CAL" --reference-codec "$BASE" \
      --features "$CACHE/features_val_blk20_n500_ss1608637542.npy" \
      --teachers "$CACHE/teacher_val_blk20_n500_ss1608637542.npy" \
      --image-offset 128 --images 128 --batch-size 4 --allocation-chunk 8 \
      --output "$OUT/dev/$stem.npz"
  done >"$OUT/logs/eval_$name.log" 2>&1
}

eval_arm "${GPU_MEAN:-4}" mean & q0=$!
eval_arm "${GPU_R025:-5}" r025 & q1=$!
eval_arm "${GPU_R050:-6}" r050 & q2=$!
eval_arm "${GPU_R100:-7}" r100 & q3=$!
wait "$q0" "$q1" "$q2" "$q3"

"$PY" p2_pretrain.py match \
  --mean-inputs "$OUT"/dev/mean.step*.json \
  --joint-inputs "$OUT"/dev/r{025,050,100}.step*.json \
  --output "$OUT/matched_validation.json"
echo "$OUT/matched_validation.json"
