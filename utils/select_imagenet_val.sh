SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python "$SCRIPT_DIR/select_imagenet_val.py" \
    --num 2000 \
    --exclude_pathname \
    "$SCRIPT_DIR/imagenet_selected_pathname5000.txt" \
    --include_pathname \
    "$SCRIPT_DIR/imagenet_selected_pathname500.txt" \
    --out_pathname imagenet_selected_pathname2000.txt \
    --out_label imagenet_selected_label2000.txt