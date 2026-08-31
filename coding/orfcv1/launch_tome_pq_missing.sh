#!/bin/bash
# Missing ToMe R+PQ rate points, hyperparams aligned to shareR ORFC ckpts.
# Round 1: blk05 GPU4/5, blk20 GPU0/1/3
# Round 2: blk10 GPU4/5, blk15 GPU6/7
# blk20 K4 (λ=0 lr=5e-4 ep=300) is already running on GPU 2 — not launched here.
set -euo pipefail
cd /data4/workspace/zlt/featcodec/ORFC/coding/orfcv1
PY=/home/user/anaconda3/envs/featcodec2/bin/python
LOG=results/similarity_merge_pq/dinov2_vitl14/logs
mkdir -p "$LOG"

run_k() {
  local gpu=$1 layer=$2 k=$3 epochs=$4 lr=$5 lmbda=$6
  local log="$LOG/${layer}_K${k}_lmbda${lmbda}_lr${lr}_ep${epochs}.log"
  echo "[$(date '+%F %T')] GPU${gpu} ${layer} K${k}  ep=${epochs} lr=${lr} lmbda=${lmbda}"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u run_similarity_merge_pq.py \
    --layer "$layer" --K "$k" --match_mode tome --merge_frac 0.25 \
    --epochs "$epochs" --lr "$lr" --lmbda "$lmbda" \
    --tau_start 0.5 --tau_end 0.005 \
    --save_codec --skip_ceiling \
    2>&1 | tee -a "$log"
}

echo "======== round 1  $(date '+%F %T') ========"
run_k 4 blk05 64  100 0.0005 0.5 &
run_k 5 blk05 256 100 0.0003 0.5 &
run_k 0 blk20 32  100 0.0003 0.5 &
run_k 1 blk20 64  100 0.0003 0.0 &
run_k 3 blk20 256 100 0.0005 0.0 &
wait

echo "======== round 2  $(date '+%F %T') ========"
# started separately: blk10 GPU6/7, blk15 GPU0/1 (see zlt_tome_r2)
wait

echo "======== done  $(date '+%F %T') ========"
