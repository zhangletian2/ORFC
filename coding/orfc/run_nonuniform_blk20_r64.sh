#!/usr/bin/env bash
# ================================================================
#  blk20 R64 non-uniform-allocation ground truth (17 histograms)
#
#  t = 0..16:  t groups@K8, (32-2t)@K4, t groups@K2  (all rate=64 bpt)
#  t=0 is the same-口径 uniform ORFC control (through this exact runner).
#  Non-uniform arms are t=1..16 (the 16 experiments requested).
#
#  ORFC-consistent: OPQ warm-start + joint R/codebook training,
#  lmbda=0, ep100, lr3e-4, tau0.5->0.005, batch32, 5000 imgs, seed42.
#
#  6 GPUs, ~3 waves.  Run detached with:
#    nohup bash run_nonuniform_blk20_r64.sh > .../launch.log 2>&1 &
# ================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PY=/home/user/anaconda3/envs/featcodec2/bin/python
LOG_DIR="$SCRIPT_DIR/results/soft_pq_nonuniform/dinov2_vitl14/logs"
mkdir -p "$LOG_DIR"

COMMON="--layer blk20 --base_K 4 --embedding_dim 32 \
        --epochs 100 --lr 0.0003 --lmbda 0.0 \
        --tau_start 0.5 --tau_end 0.005 \
        --batch_size 32 --max_train_images 5000 --seed 42 --eval_seg"

# 2 lanes per GPU (12 lanes).  Each lane "gpu:comma-separated-t" runs its t's
# sequentially; lanes run concurrently -> 2 jobs/GPU at a time, ~2 waves.
LANES=(
  "0:0,12"  "0:6"
  "1:1,13"  "1:7"
  "2:2,14"  "2:8"
  "3:3,15"  "3:9"
  "4:4,16"  "4:10"
  "5:5"     "5:11"
)

run_lane() {
  local gpu=$1 csv=$2
  IFS=',' read -ra ts <<< "$csv"
  for t in "${ts[@]}"; do
    local log="$LOG_DIR/blk20_R64_t$(printf '%02d' "$t").log"
    echo "[GPU $gpu] START t=$t -> $log ($(date '+%H:%M:%S'))"
    CUDA_VISIBLE_DEVICES=$gpu $PY -u run_soft_pq_nonuniform.py --t "$t" $COMMON \
      > "$log" 2>&1
    local rc=$?
    echo "[GPU $gpu] DONE  t=$t rc=$rc ($(date '+%H:%M:%S'))"
  done
}

echo "================================================================"
echo "  blk20 R64 non-uniform ground truth (2/GPU)  $(date)"
for spec in "${LANES[@]}"; do echo "  lane GPU${spec%%:*}: t=${spec#*:}"; done
echo "  logs: $LOG_DIR"
echo "================================================================"

PIDS=()
for spec in "${LANES[@]}"; do
  run_lane "${spec%%:*}" "${spec#*:}" &
  PIDS+=($!)
done
wait "${PIDS[@]}"

echo ""
echo "================================================================"
echo "  ALL DONE  $(date)"
printf "%-6s %-12s %-8s %-8s\n" "t" "TailMSE" "Acc" "mIoU"
for t in $(seq 0 16); do
  log="$LOG_DIR/blk20_R64_t$(printf '%02d' "$t").log"
  [ -f "$log" ] || continue
  line=$(grep "^SUMMARY" "$log" | tail -1)
  echo "  $line"
done
echo "================================================================"
