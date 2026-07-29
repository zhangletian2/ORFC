#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PY=${PY:-/home/user/anaconda3/envs/featcodec2/bin/python}
MENU=${MENU:?set MENU to a multi-mode warmup checkpoint}
CALIBRATION=${CALIBRATION:?set CALIBRATION to discrete calibration_R192.npz}
RUN_ID=${RUN_ID:-multi_outer_medium_$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-results/dinov2_vitl14/rate_allocation/$RUN_ID}
CACHE=artifacts/dinov2_vitl14/cache
STEPS=${STEPS:-50}
REFRESH_STEPS=${REFRESH_STEPS:-10}
STATE="$OUT/shared_initial_outer.npz"
if (( REFRESH_STEPS < 1 || STEPS < REFRESH_STEPS )); then
  echo "require 1 <= REFRESH_STEPS <= STEPS" >&2
  exit 2
fi

GPU_STATE=${GPU_STATE:-4}
GPU_BASE=${GPU_BASE:-4}
GPU_RECOVERY=${GPU_RECOVERY:-5}
GPU_REMAINDER=${GPU_REMAINDER:-6}
seen=" "
for gpu in "$GPU_STATE" "$GPU_BASE" "$GPU_RECOVERY" "$GPU_REMAINDER"; do
  [[ "$seen" == *" $gpu "* ]] && continue
  seen+="$gpu "
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid \
    --format=csv,noheader,nounits)
  if [[ -n "$pids" ]]; then
    echo "physical GPU $gpu is occupied by PID(s): $pids" >&2
    exit 3
  fi
done

COMMON=(
  --codec "$MENU" --calibration "$CALIBRATION"
  --features "$CACHE/features_train_blk20_n4500_ss1608637542.npy"
  --teachers "$CACHE/teacher_train_blk20_n4500_ss1608637542.npy"
  --hard-features "$CACHE/features_val_blk20_n500_ss1608637542.npy"
  --hard-teachers "$CACHE/teacher_val_blk20_n500_ss1608637542.npy"
  --steps "$STEPS" --images 4 --train-images 512 --train-image-offset 600
  --hard-images 300 --hard-image-offset 0 --hard-batch-size 8
  --allocation-chunk 8 --allocations 16 --select-steps 10 --log-steps 5
  --outer-refresh --outer-calibration-images 300
  --minimum-saved-calibration-images 300 --outer-calibration-offset 0
  --outer-mining-images 300 --outer-mining-offset 300
  --outer-batch-size 8 --outer-group-chunk 8 --outer-eps 0.01
  --dynamic-allocations --dynamic-single 32 --dynamic-random 32
  --ideal-set-size 256 --ideal-batch-size 16
  --primary-target outer_operational_best --candidate-mean-weight 0.05
  --recovery-margin 0 --rotation-lr 0.00001 --lr 0.00001
  --recovery-aggregate max --aux-calibration-images 32
  --audit-bootstraps 1000 --restart-scheduler-on-refresh
  --tau-start 0.01 --tau-end 0.01 --seed 42
)

mkdir -p "$OUT/logs"
trap 'printf "driver_exit=%s\n" "$?" >"$OUT/logs/driver_exit.txt"' EXIT
CUDA_VISIBLE_DEVICES="$GPU_STATE" "$PY" allocation_train.py short \
  "${COMMON[@]}" --refresh-steps 0 --remainder-grad-ratio 0 \
  --output "$OUT/shared_unused.json" --checkpoint "$OUT/shared_unused.pt" \
  --outer-state-output "$STATE" --prepare-outer-only \
  >"$OUT/logs/shared_outer.log" 2>&1

run_one() {
  local gpu=$1 name=$2 ratio=$3 objective=$4
  shift 4
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" allocation_train.py short \
    "${COMMON[@]}" --refresh-steps "$REFRESH_STEPS" \
    --initial-outer-state "$STATE" --remainder-grad-ratio "$ratio" \
    --auxiliary-objective "$objective" "$@" \
    --output "$OUT/$name.json" --checkpoint "$OUT/$name.pt" \
    >"$OUT/logs/$name.log" 2>&1
}

run_one "$GPU_BASE" primary_only 0 recovery &
p0=$!
run_one "$GPU_RECOVERY" primary_recovery 0.25 recovery \
  --outer-gated-recovery --recovery-pairs "${RECOVERY_PAIRS:-4}" &
p1=$!
run_one "$GPU_REMAINDER" primary_remainder 0.25 remainder_range &
p2=$!
set +e
wait "$p0"; s0=$?
wait "$p1"; s1=$?
wait "$p2"; s2=$?
set -e
printf "primary_only\t%s\nprimary_recovery\t%s\nprimary_remainder\t%s\n" \
  "$s0" "$s1" "$s2" | tee "$OUT/logs/arm_status.tsv"
if (( s0 || s1 || s2 )); then
  echo "one or more medium-training arms failed; inspect $OUT/logs" >&2
  exit 1
fi
echo "$OUT"
