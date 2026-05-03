#!/usr/bin/env bash
# ============================================================================
# Hyperparameter sweep for Non-uniform bit-allocation PQ.
#
# Fixed:  K=16  e=32  epochs=100  seed=42  n_train=5000  alloc=rd
# Sweep:  λ ∈ {0, 0.5, 1.0} × τ ∈ {const, anneal} × lr ∈ {3e-4, 1e-3}
# Layers: blk05, blk20
#
# Total:  3 × 2 × 2 × 2 = 24 configs
#
# τ modes:
#   const  – tau_start=0.5, tau_end=0.5   (constant temperature)
#   anneal – tau_start=0.5, tau_end=0.005  (exponential annealing)
#
# λ > 0 uses prior_floor=0.01 to prevent -log2(p) explosion (ORFC convention).
#
# Scheduling: blk05 configs first (≈4× runtime of blk20) so longer jobs
#   start early and shorter blk20 jobs fill remaining GPU slots.
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

K=16
EMB=32
MIN_BITS=1
MAX_BITS=10
ALLOC_OBJ=rd
BS=32
GRAD_CLIP=1.0

# ---------------------------------------------------------------------------
# Configurations: "layer lmbda tau_mode lr"
#   blk05 first (longest jobs) → blk20 fills gaps
# ---------------------------------------------------------------------------
CONFIGS=(
  # ===== blk05 (12 configs, ~4× runtime) =====
  "blk05 0   const  3e-4"
  "blk05 0   const  1e-3"
  "blk05 0   anneal 3e-4"
  "blk05 0   anneal 1e-3"
  "blk05 0.5 const  3e-4"
  "blk05 0.5 const  1e-3"
  "blk05 0.5 anneal 3e-4"
  "blk05 0.5 anneal 1e-3"
  "blk05 1.0 const  3e-4"
  "blk05 1.0 const  1e-3"
  "blk05 1.0 anneal 3e-4"
  "blk05 1.0 anneal 1e-3"
  # ===== blk20 (12 configs, ~1× runtime) =====
  "blk20 0   const  3e-4"
  "blk20 0   const  1e-3"
  "blk20 0   anneal 3e-4"
  "blk20 0   anneal 1e-3"
  "blk20 0.5 const  3e-4"
  "blk20 0.5 const  1e-3"
  "blk20 0.5 anneal 3e-4"
  "blk20 0.5 anneal 1e-3"
  "blk20 1.0 const  3e-4"
  "blk20 1.0 const  1e-3"
  "blk20 1.0 anneal 3e-4"
  "blk20 1.0 anneal 1e-3"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
lr_to_float() {
  case "$1" in
    3e-4|0.0003) echo "0.0003";;
    1e-3|0.001)  echo "0.001";;
    2e-3|0.002)  echo "0.002";;
    *)           echo "$1";;
  esac
}

make_suffix() {
  local lmbda=$1 tau_mode=$2 lr=$3
  local lt
  case "$lmbda" in
    0|0.0) lt="L0";;
    0.5)   lt="L05";;
    1.0)   lt="L10";;
    *)     lt="L${lmbda}";;
  esac
  local lr_tag
  case "$lr" in
    3e-4|0.0003) lr_tag="3e4";;
    1e-3|0.001)  lr_tag="1e3";;
    *)           lr_tag="$lr";;
  esac
  echo "sweep_${lt}_T${tau_mode}_lr${lr_tag}"
}

predict_result_json() {
  local layer=$1 lmbda=$2 tau_mode=$3 lr=$4
  local lr_f=$(lr_to_float "$lr")
  local suffix=$(make_suffix "$lmbda" "$tau_mode" "$lr")
  local rate_tag=""
  if [ "$lmbda" != "0" ] && [ "$lmbda" != "0.0" ]; then
    rate_tag="_lmbda${lmbda}"
  fi
  local t="${layer}_Kref${K}_emb${EMB}"
  t+="_minb${MIN_BITS}_maxb${MAX_BITS}_${ALLOC_OBJ}${rate_tag}_tau0.5"
  t+="_lr${lr_f}_ep${EPOCHS}_n${N_TRAIN}_s${SEED}_${suffix}"
  echo "results/uneval_pq/${BACKBONE}/${t}.json"
}

is_run_done() { [ -f "$(predict_result_json "$@")" ]; }

run_one_cfg() {
  local gpu=$1 layer=$2 lmbda=$3 tau_mode=$4 lr=$5

  local tau_start=0.5
  local tau_end tau_sched
  if [ "$tau_mode" = "const" ]; then
    tau_end=0.5
    tau_sched="exponential"
  else
    tau_end=0.005
    tau_sched="exponential"
  fi


  local suffix=$(make_suffix "$lmbda" "$tau_mode" "$lr")
  local logf="logs/uneval_sweep_${layer}_${suffix}.log"

  echo "[$(date +%H:%M:%S)] gpu=${gpu}  ${layer}  λ=${lmbda}  τ=${tau_mode}  lr=${lr} -> ${logf}"
  CUDA_VISIBLE_DEVICES=${gpu} $PYTHON run_uneval_pq.py \
      --layer "${layer}" --backbone "${BACKBONE}" \
      --K "${K}" --embedding_dim "${EMB}" \
      --min_bits "${MIN_BITS}" --max_bits "${MAX_BITS}" \
      --alloc_objective "${ALLOC_OBJ}" \
      --epochs "${EPOCHS}" --lr "${lr}" \
      --batch_size "${BS}" --grad_clip "${GRAD_CLIP}" \
      --max_train_images "${N_TRAIN}" \
      --tau_start "${tau_start}" --tau_end "${tau_end}" \
      --tau_schedule "${tau_sched}" \
      --lmbda "${lmbda}" \
      --result_suffix "${suffix}" --seed "${SEED}" \
      > "${logf}" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "  [!!] gpu=${gpu}  ${layer}_${suffix}  FAILED (exit=${rc})  see ${logf}" >&2
  else
    echo "  [ok] gpu=${gpu}  ${layer}_${suffix}"
  fi
}

# ---------------------------------------------------------------------------
# Worker queue  (6 GPUs, 1 process per GPU)
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
echo "  Non-uniform PQ hyperparam sweep: ${TOTAL} configs"
echo "  Fixed: K=${K}  e=${EMB}  epochs=${EPOCHS}  seed=${SEED}"
echo "  Sweep: λ ∈ {0, 0.5, 1.0} × τ ∈ {const, anneal}"
echo "         × lr ∈ {3e-4, 1e-3} × layer ∈ {blk05, blk20}"
echo "  BACKBONE=${BACKBONE}  NUM_GPUS=${NUM_GPUS}"
echo "  Schedule: blk05 first (≈4× runtime)"
echo "  FORCE_RERUN=${FORCE_RERUN}"
echo "============================================================"

START_TS=$(date +%s); SKIPPED=0; LAUNCHED=0
for cfg in "${CONFIGS[@]}"; do
  read -r layer lmbda tau_mode lr <<< "$cfg"
  if [ "$FORCE_RERUN" != "1" ] && is_run_done "$layer" "$lmbda" "$tau_mode" "$lr"; then
    rj=$(predict_result_json "$layer" "$lmbda" "$tau_mode" "$lr")
    echo "[skip] ${layer}  λ=${lmbda}  τ=${tau_mode}  lr=${lr}  (have ${rj##*/})"
    SKIPPED=$((SKIPPED + 1)); continue
  fi
  slot=$(wait_for_free_slot)
  run_one_cfg "$slot" "$layer" "$lmbda" "$tau_mode" "$lr" &
  SLOT_PID[$slot]=$!; LAUNCHED=$((LAUNCHED + 1))
done
wait
END_TS=$(date +%s); ELAPSED=$((END_TS - START_TS))
echo ""
echo "Done. launched=${LAUNCHED} skipped=${SKIPPED} total=${TOTAL} elapsed=$((ELAPSED/60))m$((ELAPSED%60))s"

# ================================================================
# Summary table
# ================================================================
SUMMARY="logs/uneval_sweep_summary.txt"
{
  echo "=== Non-uniform PQ sweep: K=${K} e=${EMB} ep=${EPOCHS} ==="
  echo "    alloc: min=${MIN_BITS} max=${MAX_BITS} obj=${ALLOC_OBJ}"
  echo ""
  printf "  %-5s | %4s %6s %5s | %7s %7s %7s | %7s %7s | %7s\n" \
    "layer" "λ" "τ" "lr" "OPQ-Acc" "NU-Acc" "Δ(Acc)" "ΔL-OPQ" "ΔL-NU" "rANS"
  echo "  ----------------------------------------------------------------------------------"

  for cfg in "${CONFIGS[@]}"; do
    read -r layer lmbda tau_mode lr <<< "$cfg"
    suffix=$(make_suffix "$lmbda" "$tau_mode" "$lr")
    f="logs/uneval_sweep_${layer}_${suffix}.log"
    if [ ! -f "$f" ]; then
      printf "  %-5s | %4s %6s %5s | %7s %7s %7s | %7s %7s | %7s\n" \
        "$layer" "$lmbda" "$tau_mode" "$lr" "MISS" "MISS" "MISS" "MISS" "MISS" "MISS"
      continue
    fi
    opq_acc=$(grep "OPQ Acc =" "$f" | head -1 | grep -oP '[0-9.]+$')
    nu_acc=$(grep "Non-uniform Acc =" "$f" | head -1 | grep -oP '= [0-9.]+' | head -1 | sed 's/= //')
    delta_acc=$(grep "Δ(Acc)=" "$f" | head -1 | grep -oP 'Δ\(Acc\)=[+\-][0-9.]+' | head -1 | sed 's/Δ(Acc)=//')
    opq_dl=$(grep "OPQ baseline.*Acc=.*ΔL_ref=" "$f" | head -1 | grep -oP 'ΔL_ref=[0-9.]+' | head -1 | sed 's/ΔL_ref=//')
    nu_dl=$(grep "Non-uniform.*Acc=.*ΔL_ref=" "$f" | head -1 | grep -oP 'ΔL_ref=[0-9.]+' | head -1 | sed 's/ΔL_ref=//')
    rans=$(grep "rANS=" "$f" | tail -1 | grep -oP 'rANS=[0-9.]+' | head -1 | sed 's/rANS=//')
    printf "  %-5s | %4s %6s %5s | %7s %7s %7s | %7s %7s | %7s\n" \
      "$layer" "$lmbda" "$tau_mode" "$lr" \
      "${opq_acc:--}" "${nu_acc:--}" "${delta_acc:--}" "${opq_dl:--}" "${nu_dl:--}" "${rans:--}"
  done
} | tee "$SUMMARY"
echo ""
echo "Summary: ${SUMMARY}"
