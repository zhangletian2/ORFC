SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES=7
IMROOT=$PROJECT_ROOT/data/imagenet/images/val
LIST=$PROJECT_ROOT/utils/imagenet_selected_pathname2000.txt
CKPT=$PROJECT_ROOT/pretrained
SAVE=$PROJECT_ROOT/features
LABELS=$PROJECT_ROOT/utils/imagenet_selected_label2000.txt
NAMES=$PROJECT_ROOT/data/imagenet/classnames.txt
TYPE=test2
BLK=(blk05 blk10 blk15 blk20)
export TORCH_HOME=$PROJECT_ROOT/pretrained

## ================= CLIP =================
# python clip_feat_pipeline_simple.py extract \
#  --root   $IMROOT \
#  --list   $LIST \
#  --out_root $SAVE/$TYPE/clip_vitl14 \
#  --blocks 10,15,20

# for BLK in blk05 blk11 blk17 blk23
# do
#  python clip_feat_pipeline_simple.py replay \
#    --feature_root $SAVE/$TYPE/clip_vitl14 \
#    --layer $BLK \
#    --labels $LABELS \
#    --classnames $NAMES
# done
# ================= Swin =================
#python swin_feat_pipeline_simple.py extract \
#  --root   $IMROOT \
#  --list   $LIST \
#  --out_root $SAVE/$TYPE/swin_large \
#  --stages 1,2,3,4

#for STAGE in 1 2 3 4
#do
#  python swin_feat_pipeline_simple.py replay \
#    --feature_root $SAVE/$TYPE/swin_large \
#    --layer stage$STAGE \
#    --labels $LABELS
#done
# ================= Dino =================
python dinov2_feat_pipeline_simple.py extract \
  --root   $IMROOT \
  --list   $LIST \
  --weights_root  $CKPT \
  --model  vitl14 \
  --head_layers 1 \
  --out_root $SAVE/$TYPE/dinov2_vitl14 \
  --blocks 5,10,15,20  \
  --write_manifest

python dinov2_feat_pipeline_simple.py replay \
  --feature_root $SAVE/$TYPE/dinov2_vitl14 \
  --layer ${BLK[*]} \
  --labels $LABELS \
  --weights_root $CKPT \
  --model vitl14 \
  --head_layers 1

## ======== EVA (Still in Progress) ========
#python eva02_feat_pipeline_simple.py extract \
#   --root   $IMROOT \
#   --list   $LIST \
#   --out_root $SAVE/eva02_vitl14 \
#   --blocks 5,11,17,23
#
#for BLK in blk05 blk11 blk17 blk23
#do
#  python eva02_feat_pipeline_simple.py replay \
#    --feature_root $SAVE/eva02_vitl14 \
#    --layer $BLK \
#    --labels $LABELS
#done
