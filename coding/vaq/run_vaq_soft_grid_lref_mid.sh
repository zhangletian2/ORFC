#!/usr/bin/env bash
# ============================================================================
# VAQ-mid grid (v2) trained with ΔL_ref distillation.
#
# 5 rate points × 4 layers = 20 configs.
#
# Rate points (K=64 spine for high rates, avoids K=256 dead-entry problem):
#   #1  K=4   e=32  m=32   B=64    bpfp=0.0625
#   #2  K=16  e=32  m=32   B=128   bpfp=0.125
#   #3  K=64  e=32  m=32   B=192   bpfp=0.1875
#   #4  K=64  e=16  m=64   B=384   bpfp=0.375
#   #5  K=64  e=8   m=128  B=768   bpfp=0.75
#
# Removed vs v1:  K8e32(B=96), K256e32(B=256), K256e16(B=512), K32e32(blk20)
#   - K8e32 too close to K16e32
#   - K256 configs suffer dead entries & non-monotonic accuracy
#   - Replaced by e=8 high-rate point which shows clear accuracy gains
#
# Per-layer LR (from lr sweep + highrate tune):
#   blk05: lr=2e-3  (under-optimised at 1e-3 due to 19-block frozen tail)
#   blk10: lr=1e-3
#   blk15: lr=1e-3
#   blk20: lr=1e-3  (2e-3 slightly hurts; frozen PCA is the ceiling)
#
# epochs=100 for all (snapshot data confirms ep100 ≈ ep200 for e=8 configs).
#
# Memory: e=8 -> G=128, [G,N,K_max] tensors ~4 GiB at BS=32 -> OOM on 24 GB.
#   Script auto-selects BS=16 for e<=8, BS=32 otherwise.
#
# Fixed: tau=1.0, fzR=1, lmbda=0, use_lref, eval_seg
# ============================================================================
set -u
cd "$(dirname "$0")"
mkdir -p logs results checkpoints_soft

PYTHON=${PYTHON:-python -u}
NUM_GPUS=${NUM_GPUS:-5}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-100}
N_TRAIN=${N_TRAIN:-5000}
BACKBONE=${BACKBONE:-dinov2_vitl14}
TRAIN_BS_DEFAULT=${TRAIN_BS:-32}

# ---------------------------------------------------------------------------
# Configurations: "layer K embedding_dim"
# ---------------------------------------------------------------------------
CONFIGS=(
  # ===== blk05 (5) – lr=2e-3 =====
  "blk05   4  32"
  "blk05  16  32"
  "blk05  64  32"
  "blk05  64  16"
  "blk05  64   8"
  # ===== blk10 (5) – lr=1e-3 =====
  "blk10   4  32"
  "blk10  16  32"
  "blk10  64  32"
  "blk10  64  16"
  "blk10  64   8"
  # ===== blk15 (5) – lr=1e-3 =====
  "blk15   4  32"
  "blk15  16  32"
  "blk15  64  32"
  "blk15  64  16"
  "blk15  64   8"
  # ===== blk20 (5) – lr=1e-3 =====
  "blk20   4  32"
  "blk20  16  32"
  "blk20  64  32"
  "blk20  64  16"
  "blk20  64   8"
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

lr_to_tag() {
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
  local m=$((1024 / edim))
  local logK=$(log2_int "$K")
  local B=$((m * logK))
  local lr=$(pick_lr "$layer")
  local lr_tag=$(lr_to_tag "$lr")
  local suffix="grid_lref_mid2_${layer}_K${K}_e${edim}"
  local t="${layer}_vaqsoft_B${B}_m${m}_min1_max10_objlinear_lref"
  t+="_ep${EPOCHS}_lr${lr_tag}_tau1.0-1.0_fzR1_fzC0_n${N_TRAIN}_s${SEED}_${suffix}"
  echo "results/vaq_soft/${BACKBONE}/${t}.json"
}

is_run_done() { [ -f "$(predict_result_json "$@")" ]; }

run_one_cfg() {
  local gpu=$1 layer=$2 K=$3 edim=$4
  local m=$((1024 / edim))
  local logK=$(log2_int "$K")
  local B=$((m * logK))
  local lr=$(pick_lr "$layer")
  local bs=$(pick_batch_size "$edim")
  local tag="grid_lref_mid2_${layer}_K${K}_e${edim}"
  local logf="logs/vaqsoft_${tag}.log"

  echo "[$(date +%H:%M:%S)] gpu=${gpu}  ${layer}  K=${K}  e=${edim}  B=${B}  lr=${lr}  bs=${bs}  -> ${logf}"
  CUDA_VISIBLE_DEVICES=${gpu} $PYTHON run_vaq_soft.py \
      --layer "${layer}" --backbone "${BACKBONE}" \
      --bit_budget "${B}" --num_subspaces "${m}" \
      --min_bits 1 --max_bits 10 --bit_alloc_objective linear \
      --epochs "${EPOCHS}" --lr "${lr}" \
      --batch_size "${bs}" --grad_clip 1.0 \
      --max_train_images "${N_TRAIN}" \
      --tau_start 1.0 --tau_end 1.0 \
      --freeze_transform \
      --lmbda 0.0 --prior_floor 0.0 --no_prior_init \
      --use_lref --eval_seg \
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
echo "  VAQ-mid grid v2: 5 rate points × 4 layers = ${TOTAL} configs"
echo "  Rate: B = {64, 128, 192, 384, 768} bits/token"
echo "  EPOCHS=${EPOCHS}  SEED=${SEED}  BACKBONE=${BACKBONE}"
echo "  LR: blk05=2e-3  blk10/15/20=1e-3"
echo "  BS: e<=8 -> 16, else -> ${TRAIN_BS_DEFAULT}"
echo "  tau=1.0  fzR=1  lmbda=0  use_lref  eval_seg"
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
SUMMARY="logs/vaqsoft_grid_lref_mid2_summary.txt"
{
  echo "=== VAQ-mid grid v2: 5 rate points × 4 layers ==="
  echo "    blk05: lr=2e-3  |  blk10/15/20: lr=1e-3  |  epochs=${EPOCHS}"
  echo ""
  printf "  %-5s | %4s %3s %4s | %5s %3s | %7s %7s %7s | %7s %7s\n" \
    "layer" "K" "e" "B" "lr" "bs" "S1-Acc" "S2-Acc" "S2-mIoU" "rANS_c" "rANS_s"
  echo "  --------------------------------------------------------------------------------------"

  for cfg in "${CONFIGS[@]}"; do
    read -r layer K edim <<< "$cfg"
    m=$((1024 / edim))
    logK=$(log2_int "$K")
    B=$((m * logK))
    lr=$(pick_lr "$layer")
    bs=$(pick_batch_size "$edim")
    tag="grid_lref_mid2_${layer}_K${K}_e${edim}"
    f="logs/vaqsoft_${tag}.log"
    if [ ! -f "$f" ]; then
      printf "  %-5s | %4s %3s %4s | %5s %3s | %7s %7s %7s | %7s %7s\n" \
        "$layer" "$K" "$edim" "$B" "$lr" "$bs" "MISS" "MISS" "MISS" "MISS" "MISS"
      continue
    fi
    s1=$(grep "VAQ Acc =" "$f" | head -1 | grep -oP '[0-9.]+$')
    s2=$(grep "VAQ-Soft Acc =" "$f" | head -1 | grep -oP '[0-9.]+$')
    miou=$(grep "VAQ-Soft   mIoU =" "$f" | head -1 | grep -oP '[0-9.]+$')
    rans_c=$(grep "^\s*\* rANS=" "$f" | tail -1 | grep -oP 'rANS=[0-9.]+' | head -1 | sed 's/rANS=//')
    rans_s=$(grep "VAQ-Soft VOC rANS=" "$f" | head -1 | grep -oP 'rANS=[0-9.]+' | head -1 | sed 's/rANS=//')
    printf "  %-5s | %4s %3s %4s | %5s %3s | %7s %7s %7s | %7s %7s\n" \
      "$layer" "$K" "$edim" "$B" "$lr" "$bs" \
      "${s1:--}" "${s2:--}" "${miou:--}" "${rans_c:--}" "${rans_s:--}"
  done
} | tee "$SUMMARY"
echo ""
echo "Summary: ${SUMMARY}"
