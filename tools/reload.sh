SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKDIR=$PROJECT_ROOT
IMROOT=$PROJECT_ROOT/data/imagenet/images/val
CKPT=$PROJECT_ROOT/pretrained
SAVE=$PROJECT_ROOT/features
LABELS=$PROJECT_ROOT/utils/imagenet_selected_label500.txt
NAMES=$PROJECT_ROOT/data/imagenet/classnames.txt
TYPE=test
DEVICE=7
BLK=("blk05" "blk10" "blk15" "blk20")
export TORCH_HOME=$PROJECT_ROOT/pretrained
export CUDA_VISIBLE_DEVICES=${DEVICE}

# ================= Dino =================
# echo "Dino"
# for LMD in 0.1 1 10
# do
# echo "lambda=${LMD}"
#   python dinov2_feat_pipeline_simple.py replay \
#     --feature_root $SAVE/$TYPE/dinov2_vitl14/split_decoded/vq_emb2048_dim32_lmb1_residual_trunc_mse_lmbda${LMD} \
#     --layer ${BLK[*]} \
#     --labels $LABELS \
#     --weights_root $CKPT \
#     --head_layers 1
# done

# ================= CLIP =================
# echo "CLIP"
#   for LMD in 0.01
# do
# echo "lambda=${LMD}"
#   python clip_feat_pipeline_simple.py replay \
#     --feature_root $SAVE/$TYPE/clip_vitl14/split_decoded/trunc_mse_lmbda${LMD} \
#     --layer blk05 \
#     --labels $LABELS \
#     --classnames $NAMES
# done

# # ================= Dino =================
# echo "Dino"
# echo "recon_kmeans"
#   python dinov2_feat_pipeline_simple.py replay \
#     --feature_root $SAVE/recon_${TYPE}_kmeans_10bit/dinov2_vitl14 \
#     --layer ${BLK[*]} \
#     --labels $LABELS \
#     --weights_root $CKPT \
#     --head_layers 1

# # ================= CLIP =================
# echo "CLIP"
# echo "recon_kmeans"
#   python clip_feat_pipeline_simple.py replay \
#     --feature_root $SAVE/recon_${TYPE}_kmeans_10bit/clip_vitl14 \
#     --layer ${BLK[*]} \
#     --labels $LABELS \
#     --classnames $NAMES

# ================= Dino VTM 实验 =================
# echo "=============================================="
# echo "Dino VTM Experiments"
# echo "=============================================="

# echo ""
# echo "=== blk05 ==="
# for QP in 12 22 32 42
# do
# echo "blk05 Qp=${QP}"
# python dinov2_feat_pipeline_simple.py replay \
#   --feature_root $SAVE/$TYPE/dinov2_vitl14/decoded/vtm/${QP} \
#   --layer blk05 \
#   --labels $LABELS \
#   --weights_root $CKPT \
#   --head_layers 1
# done

# echo ""
# echo "=== blk09 ==="
# for QP in 17 25 27 30
# do
# echo "blk09 Qp=${QP}"
# python dinov2_feat_pipeline_simple.py replay \
#   --feature_root $SAVE/$TYPE/dinov2_vitg14/decoded/vtm/${QP} \
#   --layer blk09 \
#   --model vitg14 \
#   --labels $LABELS \
#   --weights_root $CKPT \
#   --head_layers 1
# done

# echo ""
# echo "=== blk19 ==="
# for QP in 15 17 20
# do
# echo "blk19 Qp=${QP}"
# python dinov2_feat_pipeline_simple.py replay \
#   --feature_root $SAVE/$TYPE/dinov2_vitg14/decoded/vtm/${QP} \
#   --layer blk19 \
#   --model vitg14 \
#   --labels $LABELS \
#   --weights_root $CKPT \
#   --head_layers 1
# done

# echo ""
# echo "=== blk29 ==="
# for QP in 0 2 5 7 10
# do
# echo "blk29 Qp=${QP}"
# python dinov2_feat_pipeline_simple.py replay \
#   --feature_root $SAVE/$TYPE/dinov2_vitg14/decoded/vtm/${QP} \
#   --layer blk29 \
#   --model vitg14 \
#   --labels $LABELS \
#   --weights_root $CKPT \
#   --head_layers 1
# done

# echo ""
# echo "=============================================="
# echo "All Dino VTM experiments completed!"
# echo "=============================================="

# # ================= CLIP VTM 实验 =================
echo "CLIP"

echo "=== blk10 ==="
for QP in 12 17
do
echo "Qp=${QP}"
 python clip_feat_pipeline_simple.py replay \
   --feature_root $SAVE/$TYPE/clip_vitl14/decoded/vtm/${QP} \
   --layer blk10 \
   --labels $LABELS \
   --classnames $NAMES
done

echo "=== blk15 ==="
for QP in 0 2 5 7 10 12
do
echo "Qp=${QP}"
 python clip_feat_pipeline_simple.py replay \
   --feature_root $SAVE/$TYPE/clip_vitl14/decoded/vtm/${QP} \
   --layer blk15 \
   --labels $LABELS \
   --classnames $NAMES
done

echo "=== blk20 ==="
for QP in 17 22
do
echo "Qp=${QP}"
 python clip_feat_pipeline_simple.py replay \
   --feature_root $SAVE/$TYPE/clip_vitl14/decoded/vtm/${QP} \
   --layer blk20 \
   --labels $LABELS \
   --classnames $NAMES
done

