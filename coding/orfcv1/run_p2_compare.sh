#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PY=${PY:-/home/user/anaconda3/envs/featcodec2/bin/python}
RUN_ID=${RUN_ID:-p2_compare_$(date -u +%Y%m%dT%H%M%SZ)}
ROOT=results/dinov2_vitl14/p1_fixed_rate
P1=${P1:-$ROOT/formal_p1_corrected_20260727T090628Z}
MENU=${MENU:-$ROOT/formal_p1_20260727T084459Z/response/menu.pt}
CACHE=artifacts/dinov2_vitl14/cache
OUT=${OUT:-$P1/pretrain_checks/$RUN_ID}
STEPS=${STEPS:-10}
REFRESH_STEPS=${REFRESH_STEPS:-2}
IMAGES=${IMAGES:-8}
HARD_IMAGES=${HARD_IMAGES:-128}
ALLOCATIONS=${ALLOCATIONS:-64}
LR=${LR:-1e-5}

mkdir -p "$OUT"
common=(
  --codec "$MENU"
  --measurement "$P1/response/measurement_R192.npz"
  --calibration "$P1/response/calibration_R192.npz"
  --features "$CACHE/features_train_blk20_n4500_ss1608637542.npy"
  --teachers "$CACHE/teacher_train_blk20_n4500_ss1608637542.npy"
  --hard-features "$CACHE/features_val_blk20_n500_ss1608637542.npy"
  --hard-teachers "$CACHE/teacher_val_blk20_n500_ss1608637542.npy"
  --images "$IMAGES" --allocations "$ALLOCATIONS"
  --allocation-chunk 4 --hard-images "$HARD_IMAGES"
  --hard-batch-size 4 --hard-allocation-chunk 8
  --steps "$STEPS" --refresh-steps "$REFRESH_STEPS"
  --lr "$LR" --seed 42
)

run_arm() {
  local gpu=$1 name=$2 parameters=$3 assignment=$4
  CUDA_VISIBLE_DEVICES=$gpu "$PY" p2_pretrain.py step \
    "${common[@]}" --parameters "$parameters" --assignment "$assignment" \
    --output "$OUT/$name.json" --checkpoint "$OUT/$name.pt" \
    >"$OUT/$name.log" 2>&1
}

run_arm "${GPU_U_SOFT:-4}" u_soft u soft &
pid_u_soft=$!
run_arm "${GPU_U_STE:-5}" u_ste u ste &
pid_u_ste=$!
run_arm "${GPU_CODEBOOK_STE:-6}" codebook_ste codebook ste &
pid_codebook=$!
wait "$pid_u_soft"
wait "$pid_u_ste"
wait "$pid_codebook"

"$PY" p2_pretrain.py compare \
  --inputs "$OUT/u_soft.json" "$OUT/u_ste.json" "$OUT/codebook_ste.json" \
  --names u_soft u_ste codebook_ste --output "$OUT/comparison.json"
echo "$OUT/comparison.json"
