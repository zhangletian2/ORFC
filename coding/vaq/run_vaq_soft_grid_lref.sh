#!/usr/bin/env bash
# ============================================================================
# VAQ-Soft full grid trained with ΔL_ref distillation (λ=0 only).
#
# Same 28 (layer, K, embedding_dim) configs as run_vaq_soft_grid_full.sh but
# with --use_lref enabled and --lmbda 0 (no rate term, since the previous
# sweep showed λ>0 in the R+λD formulation gives no R-D benefit and
# significantly hurts both Acc and mIoU).
#
# Distortion = ||tail(X) - tail(inv_norm(codec(Y)))||²
#   - tail = frozen ViT blocks [layer_idx+1 ..] + final norm
#   - teacher cache pre-computed once at train start (CPU memory ~5 GB / 5000
#     images for vitl14)
#
# Defaults (winners of earlier sweeps):
#   --epochs 100 --lr 1e-3 --batch_size 32 --grad_clip 1.0
#   --max_bits 10 --min_bits 1 --bit_alloc_objective linear
#   --tau 1.0 -> 1.0  (matches λ-grid; tau=1 is "Gumbel-soft constant")
#   --max_train_images 5000
#
# Memory note: forwarding the *student* through the frozen tail in autograd
# mode is the single largest GPU cost. At batch_size=32 + worst codebook
# (B=512, K_max=1024), peak is ~19 GB / 24 GB on a single L40S/A6000. Drop
# TRAIN_BS if your GPUs are smaller.
#
# Total: 28 runs (4 layers x 7 codebook shapes; blk20 swaps K=64 e=16 -> K=32 e=32).
# Dispatched on a worker queue (NUM_GPUS slots, default 4 -- override with env).
# ============================================================================

set -u
cd "$(dirname "$0")"
mkdir -p logs results checkpoints_soft

PYTHON=${PYTHON:-python -u}
NUM_GPUS=${NUM_GPUS:-4}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-100}
N_TRAIN=${N_TRAIN:-5000}
BACKBONE=${BACKBONE:-dinov2_vitl14}
TRAIN_BS=${TRAIN_BS:-32}   # batch for everything (kmeans init, train, eval)

# ---------------------------------------------------------------------------
# Configurations: "layer K embedding_dim"   (lambda fixed to 0.0)
# ---------------------------------------------------------------------------
CONFIGS=(
  # ===== blk05 (7) =====
  "blk05   4  32"
  "blk05   8  32"
  "blk05  16  32"
  "blk05  64  32"
  "blk05 256  32"
  "blk05  64  16"
  "blk05 256  16"
  # ===== blk10 (7) =====
  "blk10   4  32"
  "blk10   8  32"
  "blk10  16  32"
  "blk10  64  32"
  "blk10 256  32"
  "blk10  64  16"
  "blk10 256  16"
  # ===== blk15 (7) =====
  "blk15   4  32"
  "blk15   8  32"
  "blk15  16  32"
  "blk15  64  32"
  "blk15 256  32"
  "blk15  64  16"
  "blk15 256  16"
  # ===== blk20 (7) =====
  "blk20   4  32"
  "blk20   8  32"
  "blk20  16  32"
  "blk20  32  32"
  "blk20  64  32"
  "blk20 256  32"
  "blk20 256  16"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log2_int() {
  case $1 in
    2)   echo 1 ;;
    4)   echo 2 ;;
    8)   echo 3 ;;
    16)  echo 4 ;;
    32)  echo 5 ;;
    64)  echo 6 ;;
    128) echo 7 ;;
    256) echo 8 ;;
    *)   echo "[!!] unsupported K=$1 (must be power of 2 in [2..256])" >&2; exit 1 ;;
  esac
}

# Predict the result-JSON path that run_vaq_soft.py writes at the end of
# run_experiment(). Mirrors the f-string in run_vaq_soft.py::out_tag with
# the L_ref tag and λ=0 (no _lmbda part):
#   {layer}_vaqsoft_B{B}_m{m}_min1_max10_objlinear_lref
#   _ep{EPOCHS}_lr0.001_tau1.0-1.0_fzR0_fzC0_n{N_TRAIN}_s{SEED}_{suffix}.json
predict_result_json() {
  local layer=$1 K=$2 edim=$3
  local m=$((1024 / edim))
  local logK
  logK=$(log2_int "$K")
  local B=$((m * logK))
  local suffix="grid_lref_${layer}_K${K}_e${edim}"
  local out_tag="${layer}_vaqsoft_B${B}_m${m}_min1_max10_objlinear_lref"
  out_tag+="_ep${EPOCHS}_lr0.001_tau1.0-1.0_fzR0_fzC0_n${N_TRAIN}_s${SEED}_${suffix}"
  echo "results/vaq_soft/${BACKBONE}/${out_tag}.json"
}

is_run_done() {
  local rj
  rj=$(predict_result_json "$@")
  [ -f "$rj" ]
}

run_one_cfg() {
  local gpu=$1 layer=$2 K=$3 edim=$4
  local m=$((1024 / edim))
  local logK
  logK=$(log2_int "$K")
  local B=$((m * logK))
  local tag="grid_lref_${layer}_K${K}_e${edim}"
  local logf="logs/vaqsoft_${tag}.log"
  echo "[$(date +%H:%M:%S)] gpu=${gpu}  ${layer}  K=${K}  e=${edim}  (m=${m}, B=${B}, λ=0, ΔL_ref)  -> ${logf}"
  CUDA_VISIBLE_DEVICES=${gpu} $PYTHON run_vaq_soft.py \
      --layer "${layer}" --backbone "${BACKBONE}" \
      --bit_budget "${B}" --num_subspaces "${m}" \
      --min_bits 1 --max_bits 10 \
      --bit_alloc_objective linear \
      --epochs "${EPOCHS}" --lr 1e-3 \
      --batch_size "${TRAIN_BS}" --grad_clip 1.0 \
      --max_train_images "${N_TRAIN}" \
      --tau_start 1.0 --tau_end 1.0 \
      --use_lref \
      --eval_seg \
      --result_suffix "${tag}" \
      --seed "${SEED}" \
      > "${logf}" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "  [!!] gpu=${gpu}  ${tag}  FAILED (exit=${rc})  see ${logf}" >&2
  else
    echo "  [ok] gpu=${gpu}  ${tag}"
  fi
}

# ---------------------------------------------------------------------------
# Worker queue: NUM_GPUS slots; pull next config as each finishes.
# ---------------------------------------------------------------------------
declare -A SLOT_PID
declare -A SLOT_CFG
for s in $(seq 0 $((NUM_GPUS - 1))); do
  SLOT_PID[$s]=""
  SLOT_CFG[$s]=""
done

wait_for_free_slot() {
  while true; do
    for s in $(seq 0 $((NUM_GPUS - 1))); do
      local pid=${SLOT_PID[$s]}
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        echo "$s"
        return
      fi
    done
    sleep 5
  done
}

TOTAL=${#CONFIGS[@]}
FORCE_RERUN=${FORCE_RERUN:-0}
echo "============================================================"
echo "  Dispatching ${TOTAL} L_ref configs across ${NUM_GPUS} GPUs"
echo "  EPOCHS=${EPOCHS}  N_TRAIN=${N_TRAIN}  SEED=${SEED}"
echo "  BACKBONE=${BACKBONE}  TRAIN_BS=${TRAIN_BS}"
echo "  loss = MSE(tail(X), tail(inv_norm(codec(Y))))   (λ=0)"
echo "  FORCE_RERUN=${FORCE_RERUN}  (set FORCE_RERUN=1 to redo finished runs)"
echo "============================================================"

START_TS=$(date +%s)
SKIPPED=0
LAUNCHED=0

for cfg in "${CONFIGS[@]}"; do
  # shellcheck disable=SC2086
  read -r layer K edim <<< "$cfg"
  if [ "$FORCE_RERUN" != "1" ] && is_run_done "$layer" "$K" "$edim"; then
    rj=$(predict_result_json "$layer" "$K" "$edim")
    echo "[skip] ${layer}  K=${K}  e=${edim}   (have ${rj##*/})"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi
  slot=$(wait_for_free_slot)
  run_one_cfg "$slot" "$layer" "$K" "$edim" &
  SLOT_PID[$slot]=$!
  SLOT_CFG[$slot]="$cfg"
  LAUNCHED=$((LAUNCHED + 1))
done

wait
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo
echo "============================================================"
echo "  Done.  launched=${LAUNCHED}  skipped=${SKIPPED}  total=${TOTAL}"
echo "  Elapsed: $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "============================================================"

# ---------------------------------------------------------------------------
# Summary table.
# ---------------------------------------------------------------------------
SUMMARY="logs/vaqsoft_grid_lref_summary.txt"
{
  printf "%-8s | %-4s | %-3s | %-4s | %-7s | %-7s | %-7s | %-7s | %-7s\n" \
    "layer" "K" "e" "B" "S1-Acc" "S2-Acc" "Δacc" "S2-mIoU" "rANS"
  echo "------------------------------------------------------------------------------------"
  for cfg in "${CONFIGS[@]}"; do
    read -r layer K edim <<< "$cfg"
    m=$((1024 / edim))
    logK=$(log2_int "$K")
    B=$((m * logK))
    tag="grid_lref_${layer}_K${K}_e${edim}"
    f="logs/vaqsoft_${tag}.log"
    if [ ! -f "$f" ]; then
      printf "%-8s | %-4s | %-3s | %-4s | %-7s | %-7s | %-7s | %-7s | %-7s\n" \
        "$layer" "$K" "$edim" "$B" "MISS" "MISS" "-" "-" "-"
      continue
    fi
    s1=$(grep "^  Stage1 VAQ      Acc=" "$f" | head -1 | sed 's/.*Acc=//' | awk '{print $1}')
    s2=$(grep "^  Stage2 VAQ-Soft Acc=" "$f" | head -1 | sed 's/.*Acc=//' | awk '{print $1}')
    dacc=$(grep "^  Stage2 VAQ-Soft Acc=" "$f" | head -1 | grep -oE "delta=[+-][0-9.]+" | sed 's/delta=//')
    miou=$(grep "^  Stage2 VAQ-Soft mIoU=" "$f" | head -1 | sed 's/.*mIoU=//' | awk '{print $1}' | sed 's/(.*$//')
    rans=$(grep -E "^  \* rANS=" "$f" | tail -1 | grep -oE "rANS=[0-9.]+" | head -1 | sed 's/rANS=//')
    printf "%-8s | %-4s | %-3s | %-4s | %-7s | %-7s | %-7s | %-7s | %-7s\n" \
      "$layer" "$K" "$edim" "$B" \
      "${s1:--}" "${s2:--}" "${dacc:--}" "${miou:--}" "${rans:--}"
  done
} | tee "$SUMMARY"

echo
echo "Summary saved: ${SUMMARY}"
