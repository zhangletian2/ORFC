#!/bin/bash
# Eval-only pass over EXISTING DINOv3 stage-1 weights (no training).
WORK=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1
LOG=$WORK/logs/eval_stage1_dinov3
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate featcodec2
mkdir -p "$LOG"
BLOCKS=(blk05 blk10 blk15 blk20); GPUS=(0 1 2 3)
for i in "${!BLOCKS[@]}"; do
    blk=${BLOCKS[$i]}
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python -u "$WORK/run_bilinear_residual.py" \
        --stage eval --residual_mode both --layer "$blk" \
        --backbone dinov3_vitl16 --n_prefix 5 --norm_mode split_per_reg_cls_patch \
        --spatial_down conv2 --spatial_up conv2 --cls_mode conv2 \
        --residual_decoder conv --no-residual_orfc --no-residual_quantize \
        --residual_ablation main --residual_epochs 30 --residual_lr 0.0003 \
        --n_val 0 --max_train_images 5000 --seed 42 --batch_size 32 \
        --grad_clip 1.0 --skip_test_acc > "$LOG/${blk}_eval.log" 2>&1 &
done
wait
echo "EVAL DONE"
