#!/usr/bin/env bash
# ============================================================================
# Non-uniform bit-allocation PQ — best config grid.
#
# Best hyperparams from sweep:  λ=0.5  τ=const(0.5)  lr=1e-3
#
# 6 rate points × 4 layers = 24 configs.
#   #1  K=4    e=32  G=32   B=64    bpfp=0.0625
#   #2  K=8    e=32  G=32   B=96    bpfp=0.09375
#   #3  K=16   e=32  G=32   B=128   bpfp=0.125
#   #4  K=64   e=32  G=32   B=192   bpfp=0.1875
#   #5  K=64   e=16  G=64   B=384   bpfp=0.375
#   #6  K=256  e=16  G=64   B=512   bpfp=0.5
#
# Schedule: 4 rounds, each round = 1 blk × 6 configs on 6 GPUs.
#   Round order: blk05 → blk10 → blk15 → blk20  (longest first)
#
# Includes --eval_seg for VOC2012 mIoU.
# ============================================================================
set -u
cd "$(dirname "$0")"
mkdir -p logs results checkpoints

PYTHON=${PYTHON:-python -u}
NUM_GPUS=${NUM_GPUS:-6}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-100}
N_TRAIN=${N_TRAIN:-5000}
BACKBONE=${BACKBONE:-dinov2_vitl14}

# Best hyperparams from sweep
LMBDA=0.5
PFLOOR=0.0
LR=1e-3
TAU_START=0.5
TAU_END=0.5
TAU_SCHED=exponential
MIN_BITS=1
MAX_BITS=10
ALLOC_OBJ=linear
MONOTONIC=1
GRAD_CLIP=1.0

LAYERS=("blk05" "blk10" "blk15" "blk20")

# 6 rate points: "K embedding_dim"
RATE_POINTS=(
  "4   32"
  "8   32"
  "16  32"
  "64  32"
  "64  16"
  "256 16"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log2_int() {
  case $1 in
    2) echo 1;; 4) echo 2;; 8) echo 3;; 16) echo 4;; 32) echo 5;;
    64) echo 6;; 128) echo 7;; 256) echo 8;; 512) echo 9;; 1024) echo 10;;
    *) echo "[!!] unsupported K=$1" >&2; exit 1;;
  esac
}

lr_to_float() {
  case "$1" in
    1e-3|0.001) echo "0.001";;
    3e-4|0.0003) echo "0.0003";;
    *) echo "$1";;
  esac
}

pick_batch_size() {
  local edim=$1
  if [ "$edim" -le 8 ]; then
    echo 16
  else
    echo 32
  fi
}

predict_result_json() {
  local layer=$1 K=$2 edim=$3
  local lr_f=$(lr_to_float "$LR")
  local suffix="bestgrid_${layer}_K${K}_e${edim}"
  local rate_tag="_lmbda${LMBDA}"
  local tau_tag="_tau${TAU_START}"
  local mono_tag=""
  [ "$MONOTONIC" = "1" ] && mono_tag="_mono"
  local t="${layer}_Kref${K}_emb${edim}"
  t+="_minb${MIN_BITS}_maxb${MAX_BITS}_${ALLOC_OBJ}${mono_tag}${rate_tag}${tau_tag}"
  t+="_lr${lr_f}_ep${EPOCHS}_n${N_TRAIN}_s${SEED}_${suffix}"
  echo "results/uneval_pq/${BACKBONE}/${t}.json"
}

is_run_done() { [ -f "$(predict_result_json "$@")" ]; }

run_one_cfg() {
  local gpu=$1 layer=$2 K=$3 edim=$4
  local m=$((1024 / edim))
  local logK=$(log2_int "$K")
  local B=$((m * logK))
  local bs=$(pick_batch_size "$edim")
  local mono_sfx=""
  [ "$MONOTONIC" = "1" ] && mono_sfx="_mono"
  local suffix="bestgrid_${layer}_K${K}_e${edim}"
  local logf="logs/bestgrid_${layer}_K${K}_e${edim}_${ALLOC_OBJ}${mono_sfx}_minb${MIN_BITS}.log"

  echo "[$(date +%H:%M:%S)] gpu=${gpu}  ${layer}  K=${K}  e=${edim}  m=${m}  B=${B}  bs=${bs}  -> ${logf}"
  local mono_flag=""
  [ "$MONOTONIC" = "1" ] && mono_flag="--monotonic"

  CUDA_VISIBLE_DEVICES=${gpu} $PYTHON run_uneval_pq.py \
      --layer "${layer}" --backbone "${BACKBONE}" \
      --K "${K}" --embedding_dim "${edim}" \
      --min_bits "${MIN_BITS}" --max_bits "${MAX_BITS}" \
      --alloc_objective "${ALLOC_OBJ}" ${mono_flag} \
      --epochs "${EPOCHS}" --lr "${LR}" \
      --batch_size "${bs}" --grad_clip "${GRAD_CLIP}" \
      --max_train_images "${N_TRAIN}" \
      --tau_start "${TAU_START}" --tau_end "${TAU_END}" \
      --tau_schedule "${TAU_SCHED}" \
      --lmbda "${LMBDA}" --prior_floor "${PFLOOR}" \
      --eval_seg \
      --result_suffix "${suffix}" --seed "${SEED}" \
      > "${logf}" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "  [!!] gpu=${gpu}  ${layer}_K${K}_e${edim}  FAILED (exit=${rc})  see ${logf}" >&2
  else
    echo "  [ok] gpu=${gpu}  ${layer}_K${K}_e${edim}"
  fi
}

# ---------------------------------------------------------------------------
# Run: 4 rounds (1 layer per round), 6 configs per round on 6 GPUs
# ---------------------------------------------------------------------------
TOTAL=$(( ${#LAYERS[@]} * ${#RATE_POINTS[@]} ))
FORCE_RERUN=${FORCE_RERUN:-0}

echo "============================================================"
echo "  Non-uniform PQ best-config grid: ${TOTAL} configs"
echo "  6 rate points × 4 layers"
echo "  Fixed: λ=${LMBDA}  τ=${TAU_START}(const)  lr=${LR}"
echo "         pfloor=${PFLOOR}  alloc: min=${MIN_BITS} max=${MAX_BITS} obj=${ALLOC_OBJ} monotonic=${MONOTONIC}"
echo "  EPOCHS=${EPOCHS}  SEED=${SEED}  BACKBONE=${BACKBONE}"
echo "  Schedule: 4 rounds (1 blk/round × 6 GPUs)"
echo "  eval_seg=true"
echo "  FORCE_RERUN=${FORCE_RERUN}"
echo "============================================================"

START_TS=$(date +%s); SKIPPED=0; LAUNCHED=0

for layer in "${LAYERS[@]}"; do
  echo ""
  echo "=== Round: ${layer} (6 configs on ${NUM_GPUS} GPUs) ==="

  gpu_idx=0
  ROUND_PIDS=()

  for rp in "${RATE_POINTS[@]}"; do
    read -r K edim <<< "$rp"

    if [ "$FORCE_RERUN" != "1" ] && is_run_done "$layer" "$K" "$edim"; then
      rj=$(predict_result_json "$layer" "$K" "$edim")
      echo "[skip] ${layer}  K=${K}  e=${edim}  (have ${rj##*/})"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    run_one_cfg "$gpu_idx" "$layer" "$K" "$edim" &
    ROUND_PIDS+=($!)
    LAUNCHED=$((LAUNCHED + 1))
    gpu_idx=$((gpu_idx + 1))
  done

  if [ ${#ROUND_PIDS[@]} -gt 0 ]; then
    echo "  Waiting for ${#ROUND_PIDS[@]} jobs in ${layer} round..."
    for pid in "${ROUND_PIDS[@]}"; do
      wait "$pid"
    done
    echo "  ${layer} round done."
  else
    echo "  ${layer} round: all skipped."
  fi
done

END_TS=$(date +%s); ELAPSED=$((END_TS - START_TS))
echo ""
echo "Done. launched=${LAUNCHED} skipped=${SKIPPED} total=${TOTAL} elapsed=$((ELAPSED/60))m$((ELAPSED%60))s"

# ================================================================
# Summary table
# ================================================================
SUMMARY="logs/bestgrid_summary.txt"
{
  echo "=== Non-uniform PQ best-config grid ==="
  echo "    λ=${LMBDA}  τ=${TAU_START}(const)  lr=${LR}  ep=${EPOCHS}"
  echo "    alloc: min=${MIN_BITS} max=${MAX_BITS} obj=${ALLOC_OBJ} monotonic=${MONOTONIC}"
  echo ""
  printf "  %-5s | %4s %3s %4s | %7s %7s %7s | %7s %7s | %7s %7s %7s | %7s\n" \
    "layer" "K" "e" "B" "OPQ-Acc" "NU-Acc" "Δ(Acc)" "ΔL-OPQ" "ΔL-NU" "OPQ-IoU" "NU-IoU" "Δ(IoU)" "rANS"
  echo "  ---------------------------------------------------------------------------------------------------------------"

  for layer in "${LAYERS[@]}"; do
    for rp in "${RATE_POINTS[@]}"; do
      read -r K edim <<< "$rp"
      m=$((1024 / edim))
      logK=$(log2_int "$K")
      B=$((m * logK))
      local mono_sfx=""
      [ "$MONOTONIC" = "1" ] && mono_sfx="_mono"
      f="logs/bestgrid_${layer}_K${K}_e${edim}_${ALLOC_OBJ}${mono_sfx}_minb${MIN_BITS}.log"
      if [ ! -f "$f" ]; then
        printf "  %-5s | %4s %3s %4s | %7s %7s %7s | %7s %7s | %7s %7s %7s | %7s\n" \
          "$layer" "$K" "$edim" "$B" "MISS" "MISS" "MISS" "MISS" "MISS" "MISS" "MISS" "MISS" "MISS"
        continue
      fi
      opq_acc=$(grep "OPQ Acc =" "$f" | head -1 | grep -oP '[0-9.]+$')
      nu_acc=$(grep "Non-uniform Acc =" "$f" | head -1 | grep -oP '= [0-9.]+' | head -1 | sed 's/= //')
      delta_acc=$(grep "Δ(Acc)=" "$f" | head -1 | grep -oP 'Δ\(Acc\)=[+\-][0-9.]+' | head -1 | sed 's/Δ(Acc)=//')
      opq_dl=$(grep "OPQ baseline.*Acc=.*ΔL_ref=" "$f" | head -1 | grep -oP 'ΔL_ref=[0-9.]+' | head -1 | sed 's/ΔL_ref=//')
      nu_dl=$(grep "Non-uniform.*Acc=.*ΔL_ref=" "$f" | head -1 | grep -oP 'ΔL_ref=[0-9.]+' | head -1 | sed 's/ΔL_ref=//')
      opq_iou=$(grep "OPQ.*mIoU" "$f" | head -1 | grep -oP '= [0-9.]+' | head -1 | sed 's/= //')
      nu_iou=$(grep "Non-uniform mIoU" "$f" | head -1 | grep -oP '= [0-9.]+' | head -1 | sed 's/= //')
      delta_iou=$(grep "Δ(mIoU)" "$f" | head -1 | grep -oP '= [+\-][0-9.]+' | head -1 | sed 's/= //')
      rans=$(grep "rANS=" "$f" | grep -v "VOC\|seg" | tail -1 | grep -oP 'rANS=[0-9.]+' | head -1 | sed 's/rANS=//')
      printf "  %-5s | %4s %3s %4s | %7s %7s %7s | %7s %7s | %7s %7s %7s | %7s\n" \
        "$layer" "$K" "$edim" "$B" \
        "${opq_acc:--}" "${nu_acc:--}" "${delta_acc:--}" \
        "${opq_dl:--}" "${nu_dl:--}" \
        "${opq_iou:--}" "${nu_iou:--}" "${delta_iou:--}" \
        "${rans:--}"
    done
    echo "  ---------------------------------------------------------------------------------------------------------------"
  done
} | tee "$SUMMARY"
echo ""
echo "Summary: ${SUMMARY}"
