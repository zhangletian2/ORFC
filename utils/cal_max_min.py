#!/usr/bin/env python3
# stat_clip_npy.py
import os, sys, argparse, numpy as np
from glob import glob

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # 没装 tqdm 也能跑

def stats_for_folder(folder: str):
    npy_files = sorted(glob(os.path.join(folder, "*.npy")))
    if not npy_files:
        return None, 0

    gmin, gmax = None, None
    for fp in tqdm(npy_files, desc=os.path.basename(folder), unit="file"):
        try:
            arr = np.load(fp, mmap_mode='r')  # 内存友好
            # 用 nan 安全的统计，避免异常值中断
            mn = np.nanmin(arr)
            mx = np.nanmax(arr)
            gmin = mn if gmin is None else min(gmin, mn)
            gmax = mx if gmax is None else max(gmax, mx)
        except Exception as e:
            print(f"[WARN] 跳过损坏文件: {fp} ({e})", file=sys.stderr)
            continue
    return (gmin, gmax), len(npy_files)

def main():
    ap = argparse.ArgumentParser(description="统计每层特征.npy文件的全局最小/最大值")
    _project_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--root", default=os.path.join(_project_root, "features", "train", "dinov2_vitl14"))
    ap.add_argument("--out", help="可选：将结果保存为 CSV 文件路径")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"[ERR] 根目录不存在: {root}", file=sys.stderr); sys.exit(1)

    # 只统计第一层子目录（blk05/blk11/...）
    subdirs = [d for d in sorted(os.listdir(root))
               if os.path.isdir(os.path.join(root, d))]

    results = []
    print("\n子目录统计：\n")
    print(f"{'subdir':<10} {'count':>7} {'min':>15} {'max':>15}")
    print("-"*52)
    for sd in subdirs:
        folder = os.path.join(root, sd)
        (gmin_gmax, cnt) = stats_for_folder(folder)
        if gmin_gmax is None:
            print(f"{sd:<10} {cnt:>7} {'(empty)':>15} {'(empty)':>15}")
            results.append((sd, cnt, "", ""))
        else:
            gmin, gmax = gmin_gmax
            print(f"{sd:<10} {cnt:>7} {gmin:>15.6g} {gmax:>15.6g}")
            results.append((sd, cnt, gmin, gmax))

    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subdir", "count", "min", "max"])
            for row in results:
                w.writerow(row)
        print(f"\n已保存到: {args.out}")

if __name__ == "__main__":
    main()
