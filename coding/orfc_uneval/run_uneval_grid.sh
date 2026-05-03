#!/usr/bin/env bash
# ============================================================================
# Non-uniform bit-allocation PQ (frozen OPQ rotation) — full grid experiment.
#
# 5 rate points × 4 layers = 20 configs.
# Rate points (same as VAQ-mid grid v2):
#   #1  K=4   e=32  m=32   B=64    bpfp=0.0625
#   #2  K=16  e=32  m=32   B=128   bpfp=0.125
#   #3  K=64  e=32  m=32   B=192   bpfp=0.1875
#   #4  K=64  e=16  m=64   B=384   bpfp=0.375
#   #5  K=64  e=8   m=128  B=768   bpfp=0.75
#
# Per-layer LR (from VAQ lr sweep):
#   blk05: lr=2e-3
#   blk10/15/20: lr=1e-3
#
# epochs=100, tau=0.5→0.005 (exponential), ΔL_ref, lmbda=0, grad_clip=1
# Allocation: importance-weighted DP, min_bits=1, max_bits=10, objective=rd
#
# Memory: e=8 → G=128, auto-selects BS=16 (else BS=32).
# ============================================================================
set -u
cd "$(dirname "$0")"
mkdir -p logs results checkpoints

PYTHON=${PYTHON:-python -u}
NUM_GPUS=${NUM_GPUS:-5}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-100}
N_TRAIN=${N_TRAIN:-5000}
BACKBONE=${BACKBONE:-dinov2_vitl14}
TRAIN_BS_DEFAULT=${TRAIN_BS:-32}

MIN_BITS=${MIN_BITS:-1}
MAX_BITS=${MAX_BITS:-10}
ALLOC_OBJ=${ALLOC_OBJ:-rd}

TAU_START=${TAU_START:-0.5}
TAU_END=${TAU_END:-0.005}
TAU_SCHED=${TAU_SCHED:-exponential}

# ---------------------------------------------------------------------------
# Configurations: "layer K embedding_dim"
# ---------------------------------------------------------------------------
CONFIGS=(
  # ===== blk05 (5) – lr=2e-3 =====
  # "blk05   4  32"
  "blk05  16  32"
  # "blk05  64  32"
  # "blk05  64  16"
  # "blk05  64   8"
  # ===== blk10 (5) – lr=1e-3 =====
  # "blk10   4  32"
  "blk10  16  32"
  # "blk10  64  32"
  # "blk10  64  16"
  # "blk10  64   8"
  # ===== blk15 (5) – lr=1e-3 =====
  # "blk15   4  32"
  "blk15  16  32"
  # "blk15  64  32"
  # "blk15  64  16"
  # "blk15  64   8"
  # ===== blk20 (5) – lr=1e-3 =====
  # "blk20   4  32"
  "blk20  16  32"
  # "blk20  64  32"
  # "blk20  64  16"
  # "blk20  64   8"
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

pick_lr() {
  case "$1" in
    blk05) echo "2e-3" ;;
    *)     echo "1e-3" ;;
  esac
}

lr_to_float() {
  case "$1" in
    1e-3|0.001)   echo "0.001";;
    2e-3|0.002)   echo "0.002";;
    3e-3|0.003)   echo "0.003";;
    5e-4|0.0005)  echo "0.0005";;
    3e-4|0.0003)  echo "0.0003";;
    1e-4|0.0001)  echo "0.0001";;
    *)            echo "$1";;
  esac
}

pick_batch_size() {
  local edim=$1
  if [ "$edim" -le 8 ]; then
    echo 16
  else
    echo "${TRAIN_BS_DEFAULT}"
  fi
}

predict_result_json() {
  local layer=$1 K=$2 edim=$3
  local lr=$(pick_lr "$layer")
  local lr_f=$(lr_to_float "$lr")
  local suffix="uneval_grid_${layer}_K${K}_e${edim}"
  local tau_tag="_tau${TAU_START}"
  local t="${layer}_Kref${K}_emb${edim}"
  t+="_minb${MIN_BITS}_maxb${MAX_BITS}_${ALLOC_OBJ}${tau_tag}"
  t+="_lr${lr_f}_ep${EPOCHS}_n${N_TRAIN}_s${SEED}_${suffix}"
  echo "results/uneval_pq/${BACKBONE}/${t}.json"
}

is_run_done() { [ -f "$(predict_result_json "$@")" ]; }

run_one_cfg() {
  local gpu=$1 layer=$2 K=$3 edim=$4
  local m=$((1024 / edim))
  local logK=$(log2_int "$K")
  local B=$((m * logK))
  local lr=$(pick_lr "$layer")
  local bs=$(pick_batch_size "$edim")
  local tag="uneval_grid_${layer}_K${K}_e${edim}"
  local logf="logs/uneval_${tag}.log"

  echo "[$(date +%H:%M:%S)] gpu=${gpu}  ${layer}  K=${K}  e=${edim}  m=${m}  B=${B}  lr=${lr}  bs=${bs}  -> ${logf}"
  CUDA_VISIBLE_DEVICES=${gpu} $PYTHON run_uneval_pq.py \
      --layer "${layer}" --backbone "${BACKBONE}" \
      --K "${K}" --embedding_dim "${edim}" \
      --min_bits "${MIN_BITS}" --max_bits "${MAX_BITS}" \
      --alloc_objective "${ALLOC_OBJ}" \
      --epochs "${EPOCHS}" --lr "${lr}" \
      --batch_size "${bs}" --grad_clip 1.0 \
      --max_train_images "${N_TRAIN}" \
      --tau_start "${TAU_START}" --tau_end "${TAU_END}" \
      --tau_schedule "${TAU_SCHED}" \
      --lmbda 0.0 --prior_floor 0.0 \
      --result_suffix "${tag}" --seed "${SEED}" \
      > "${logf}" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "  [!!] gpu=${gpu}  ${tag}  FAILED (exit=${rc})  see ${logf}" >&2
  else
    echo "  [ok] gpu=${gpu}  ${tag}"
  fi
}

# ---------------------------------------------------------------------------
# Worker queue
# ---------------------------------------------------------------------------
declare -A SLOT_PID
for s in $(seq 0 $((NUM_GPUS - 1))); do SLOT_PID[$s]=""; done
wait_for_free_slot() {
  while true; do
    for s in $(seq 0 $((NUM_GPUS - 1))); do
      local pid=${SLOT_PID[$s]}
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then echo "$s"; return; fi
    done; sleep 5
  done
}

TOTAL=${#CONFIGS[@]}
FORCE_RERUN=${FORCE_RERUN:-0}

echo "============================================================"
echo "  Non-uniform bit-alloc PQ (frozen OPQ): ${TOTAL} configs"
echo "  5 rate points × 4 layers"
echo "  EPOCHS=${EPOCHS}  SEED=${SEED}  BACKBONE=${BACKBONE}"
echo "  LR: blk05=2e-3  blk10/15/20=1e-3"
echo "  BS: e<=8 -> 16, else -> ${TRAIN_BS_DEFAULT}"
echo "  tau=${TAU_START} -> ${TAU_END} (${TAU_SCHED})"
echo "  alloc: min_bits=${MIN_BITS} max_bits=${MAX_BITS} obj=${ALLOC_OBJ}"
echo "  lmbda=0  ΔL_ref  grad_clip=1.0"
echo "  FORCE_RERUN=${FORCE_RERUN}"
echo "============================================================"

START_TS=$(date +%s); SKIPPED=0; LAUNCHED=0
for cfg in "${CONFIGS[@]}"; do
  read -r layer K edim <<< "$cfg"
  if [ "$FORCE_RERUN" != "1" ] && is_run_done "$layer" "$K" "$edim"; then
    rj=$(predict_result_json "$layer" "$K" "$edim")
    echo "[skip] ${layer}  K=${K}  e=${edim}  (have ${rj##*/})"
    SKIPPED=$((SKIPPED + 1)); continue
  fi
  slot=$(wait_for_free_slot)
  run_one_cfg "$slot" "$layer" "$K" "$edim" &
  SLOT_PID[$slot]=$!; LAUNCHED=$((LAUNCHED + 1))
done
wait
END_TS=$(date +%s); ELAPSED=$((END_TS - START_TS))
echo ""
echo "Done. launched=${LAUNCHED} skipped=${SKIPPED} total=${TOTAL} elapsed=$((ELAPSED/60))m$((ELAPSED%60))s"

# ================================================================
# Summary table
# ================================================================
SUMMARY="logs/uneval_grid_summary.txt"
{
  echo "=== Non-uniform bit-alloc PQ (frozen OPQ): 5 rate × 4 layers ==="
  echo "    blk05: lr=2e-3  |  blk10/15/20: lr=1e-3  |  epochs=${EPOCHS}"
  echo "    alloc: min=${MIN_BITS} max=${MAX_BITS} obj=${ALLOC_OBJ}"
  echo "    tau: ${TAU_START} -> ${TAU_END} (${TAU_SCHED})"
  echo ""
  printf "  %-5s | %4s %3s %4s | %5s %3s | %7s %7s %7s | %7s %7s | %7s\n" \
    "layer" "K" "e" "B" "lr" "bs" "OPQ-Acc" "NU-Acc" "Δ(Acc)" "ΔL-OPQ" "ΔL-NU" "rANS"
  echo "  -------------------------------------------------------------------------------------------------"

  for cfg in "${CONFIGS[@]}"; do
    read -r layer K edim <<< "$cfg"
    m=$((1024 / edim))
    logK=$(log2_int "$K")
    B=$((m * logK))
    lr=$(pick_lr "$layer")
    bs=$(pick_batch_size "$edim")
    tag="uneval_grid_${layer}_K${K}_e${edim}"
    f="logs/uneval_${tag}.log"
    if [ ! -f "$f" ]; then
      printf "  %-5s | %4s %3s %4s | %5s %3s | %7s %7s %7s | %7s %7s | %7s\n" \
        "$layer" "$K" "$edim" "$B" "$lr" "$bs" "MISS" "MISS" "MISS" "MISS" "MISS" "MISS"
      continue
    fi
    opq_acc=$(grep "OPQ Acc =" "$f" | head -1 | grep -oP '[0-9.]+$')
    nu_acc=$(grep "Non-uniform Acc =" "$f" | head -1 | grep -oP '= [0-9.]+' | head -1 | sed 's/= //')
    delta_acc=$(grep "Δ(Acc)=" "$f" | head -1 | grep -oP 'Δ\(Acc\)=[+\-][0-9.]+' | head -1 | sed 's/Δ(Acc)=//')
    opq_dl=$(grep "OPQ baseline.*Acc=.*ΔL_ref=" "$f" | head -1 | grep -oP 'ΔL_ref=[0-9.]+' | head -1 | sed 's/ΔL_ref=//')
    nu_dl=$(grep "Non-uniform.*Acc=.*ΔL_ref=" "$f" | head -1 | grep -oP 'ΔL_ref=[0-9.]+' | head -1 | sed 's/ΔL_ref=//')
    rans=$(grep "rANS=" "$f" | tail -1 | grep -oP 'rANS=[0-9.]+' | head -1 | sed 's/rANS=//')
    printf "  %-5s | %4s %3s %4s | %5s %3s | %7s %7s %7s | %7s %7s | %7s\n" \
      "$layer" "$K" "$edim" "$B" "$lr" "$bs" \
      "${opq_acc:--}" "${nu_acc:--}" "${delta_acc:--}" "${opq_dl:--}" "${nu_dl:--}" "${rans:--}"
  done
} | tee "$SUMMARY"
echo ""
echo "Summary: ${SUMMARY}"
