#!/usr/bin/env bash
# Stage-1 seg/depth eval for the DINOv3 noclsconv2 bilinear codecs.
# Retrained after two fixes: per-reg mu/sigma in opq.batch_normalize_gpu and
# the timm checkpoint_filter_fn for the frozen tail's LayerScale gammas.
# Eval additionally threads the true patch grid (meta['token_hw']) into the
# codec instead of guessing it from the token count.
set -u

ORFC=/data4/workspace/zlt/featcodec/ORFC/coding/orfc
RES=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1/results/bilinear_residual/dinov3_vitl16
PY=/data4/workspace/zlt/featcodec/CoFAI/.venv/bin/python
LOGDIR=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1/logs/eval_stage1_dinov3_gridfix
SUF=bili65_K4_c4_both_wa_conv_pre_noq_bv_ablation_main_lr0.0003_ep30_n5000_nval0_s42

mkdir -p "$LOGDIR"
cd "$ORFC/eval" || exit 1

gpu=0
for blk in blk05 blk10 blk15 blk20; do
  ckpt="$RES/${blk}_${SUF}.pt"
  if [[ ! -f $ckpt ]]; then
    echo "MISSING $ckpt" >&2
    continue
  fi
  echo "launch $blk on cuda:$gpu"
  "$PY" eval_dinov3_tasks.py \
    --stage1_ckpt "$ckpt" \
    --layer "$blk" \
    --tasks semseg,depth \
    --device "cuda:$gpu" \
    --force \
    > "$LOGDIR/$blk.log" 2>&1 &
  gpu=$((gpu + 1))
done

wait
echo "=== all done ==="
grep -H "mIoU\|rmse\|RMSE\|\[eval\] done" "$LOGDIR"/*.log
