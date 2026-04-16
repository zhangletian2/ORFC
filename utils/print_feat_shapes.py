import os
import numpy as np

def print_feature_shapes(root_dir=""):
    """
    递归遍历特征目录，读取每个 .npy 文件并输出其 shape。
    支持结构：
      features/
        ├── clip_vit14/blk05/*.npy
        ├── dinov2_vitl14/blk05/*.npy
        └── swin_large/stage1/*.npy
    """
    for model_name in sorted(os.listdir(root_dir)):
        model_path = os.path.join(root_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        print(f"\n=== Model: {model_name} ===")
        for layer_name in sorted(os.listdir(model_path)):
            layer_path = os.path.join(model_path, layer_name)
            if not os.path.isdir(layer_path):
                continue

            npy_files = [f for f in os.listdir(layer_path) if f.endswith(".npy")]
            if not npy_files:
                continue

            # 只读取第一个文件来查看shape（所有文件shape应一致）
            npy_path = os.path.join(layer_path, npy_files[0])
            try:
                feat = np.load(npy_path, allow_pickle=False)
                print(f"{layer_name:>10}: {feat.shape}")
            except Exception as e:
                print(f"{layer_name:>10}: [Error reading {npy_path}] {e}")

if __name__ == "__main__":
    _project_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    print_feature_shapes(os.path.join(_project_root, "features", "train"))
