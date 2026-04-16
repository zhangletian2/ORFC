SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

python stat.py \
  --feat_root "$PROJECT_ROOT/features/test" \
  --models dinov2_vitl14 clip_vitl14 \
  --layers blk05 blk11 blk17 blk23 \
  --qps 27
