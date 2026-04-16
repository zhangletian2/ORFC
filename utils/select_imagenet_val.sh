SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python "$SCRIPT_DIR/select_imagenet_val.py" \
    --num 20000 \
    --exclude_pathname \
    "$SCRIPT_DIR/imagenet_selected_pathname500.txt" \
    "$SCRIPT_DIR/imagenet_selected_pathname1000.txt" \
    --out_pathname imagenet_selected_pathname20000.txt \
    --out_label imagenet_selected_label20000.txt