#!/usr/bin/env bash
# DINOv3 stage-2 (noclsconv2 joint ORFC-PQ) downstream eval: ADE20K semseg +
# NYU depth, full eval sets so the numbers line up with the stage-1 run and the
# full-set bypass anchors (mIoU 0.5307 / RMSE 0.3474).
# One block per round, the 6 K values in parallel on GPU 0-5.
set -u

EVAL=/data4/workspace/zlt/featcodec/ORFC/coding/orfc/eval
CK=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1/checkpoints/dinov3_vitl16
PY=/data4/workspace/zlt/featcodec/CoFAI/.venv/bin/python
LOG=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1/logs/stage2_tasks_dinov3_noclsconv2
SUF=_emb32_bt1024_ws_lmbda0.5_tau2.0_te2.0_tscon_hlr0.1_bv_lr0.0003_ep100_n5000_nval200_s42

mkdir -p "$LOG"
cd "$EVAL" || exit 1

for blk in blk05 blk10 blk15 blk20; do
  gpu=0
  for K in 4 8 16 64 256 512; do
    STEM=${blk}_conv2_main_jointopq_K${K}${SUF}
    if [[ ! -f $CK/${STEM}.pt || ! -f $CK/${STEM}_spatial.pt ]]; then
      echo "MISSING $STEM" >&2
      gpu=$((gpu + 1))
      continue
    fi
    # CUDA_VISIBLE_DEVICES rather than --device: something inside the cofai
    # backbone/head construction still opens a ~386 MiB primary context on
    # cuda:0 no matter what torch.cuda.set_device() says.  Masking the other
    # cards makes that physically impossible.
    CUDA_VISIBLE_DEVICES=$gpu "$PY" eval_dinov3_tasks.py \
      --stage1_ckpt "$CK/${STEM}_spatial.pt" \
      --orfc_ckpt "$CK/${STEM}.pt" \
      --layer "$blk" \
      --stage1_ablation main \
      --tasks semseg,depth \
      --device cuda:0 \
      --force \
      > "$LOG/${blk}_K${K}.log" 2>&1 &
    gpu=$((gpu + 1))
  done
  wait
  echo "=== $blk done ($(date '+%F %T'))"
done

echo "=== ALL DONE ==="
grep -H "mIoU\|rmse\|RMSE" "$LOG"/*.log
