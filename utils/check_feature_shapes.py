#!/usr/bin/env python3
import os
import sys
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

def list_npy(dir_path: Path):
    return sorted([p for p in dir_path.iterdir() if p.suffix == ".npy"])

def safe_shape(npy_path: Path):
    # 仅映射读取，避免把整数组载入内存
    arr = np.load(npy_path, allow_pickle=False, mmap_mode="r")
    return tuple(arr.shape)

def print_header(text):
    print(f"\n=== {text} ===")

def main(root="features", show_examples=3):
    root = Path(root)
    if not root.exists():
        print(f"[Error] Root dir not found: {root}")
        sys.exit(2)

    any_mismatch = False
    total_layers = 0
    empty_layers = []

    for model_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        print_header(f"Model: {model_dir.name}")

        layer_dirs = [p for p in model_dir.iterdir() if p.is_dir()]
        if not layer_dirs:
            print("  (no layer subfolders)")
            continue

        for layer_dir in sorted(layer_dirs):
            total_layers += 1
            files = list_npy(layer_dir)
            if not files:
                empty_layers.append(str(layer_dir))
                print(f"{layer_dir.name:>10}: [EMPTY] no .npy files")
                continue

            shapes = []
            for f in files:
                try:
                    shapes.append(safe_shape(f))
                except Exception as e:
                    shapes.append(("READ_ERROR",))
                    print(f"        [ReadError] {f}: {e}")

            cnt = Counter(shapes)
            if len(cnt) == 1:
                shp = next(iter(cnt.keys()))
                n = cnt[shp]
                print(f"{layer_dir.name:>10}: {shp}  (files: {n})")
            else:
                any_mismatch = True
                print(f"{layer_dir.name:>10}: **MISMATCH**")
                for shp, n in cnt.most_common():
                    print(f"        shape={shp}, files={n}")
                # 给出每种shape的若干示例文件
                by_shape = defaultdict(list)
                for f, shp in zip(files, shapes):
                    by_shape[shp].append(f)
                for shp, flist in by_shape.items():
                    examples = ", ".join([str(p.name) for p in flist[:show_examples]])
                    more = "" if len(flist) <= show_examples else f" (+{len(flist)-show_examples} more)"
                    print(f"        e.g. shape={shp}: {examples}{more}")

    # 末尾总结
    print_header("Summary")
    print(f"Scanned layers: {total_layers}")
    if empty_layers:
        print(f"Empty layers ({len(empty_layers)}):")
        for p in empty_layers:
            print(f"  - {p}")
    if any_mismatch:
        print("Result: shape inconsistencies found.")
        sys.exit(1)
    else:
        print("Result: all layers are consistent.")
        sys.exit(0)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Check feature shapes per model/layer.")
    _project_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--root", default=os.path.join(_project_root, "features", "test"), help="Root directory containing model folders")
    ap.add_argument("--examples", type=int, default=3, help="Number of example files to show per shape")
    args = ap.parse_args()
    main(root=args.root, show_examples=args.examples)
