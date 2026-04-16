SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

python chen2019.py \
  --feat_root "$PROJECT_ROOT/features/test" \
  --models dinov2_vitl14 clip_vitl14 \
  --layers blk05 blk11 blk17 blk23 \
  --qps 27 \
  --hm_root "$SCRIPT_DIR/HM-16.21" \
  --hm_cfg encoder_intra_main_rext.cfg \
  --tmpdir ./hm_tmp