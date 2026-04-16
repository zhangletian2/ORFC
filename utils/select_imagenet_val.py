#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path

IMG_EXTS = {".jpeg", ".jpg", ".png", ".bmp", ".JPEG", ".JPG", ".PNG", ".BMP"}

def read_classnames(classnames_path):
    wnids = []
    wnid_to_idx = {}
    with open(classnames_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            wnid = line.split()[0]
            wnids.append(wnid)
            wnid_to_idx[wnid] = i  # 行号即label索引（从0开始）
    return wnids, wnid_to_idx

def collect_images_per_class(val_root, wnids):
    val_root = Path(val_root)
    images = {}
    for wnid in wnids:
        cls_dir = val_root / wnid
        files = []
        if cls_dir.is_dir():
            for p in cls_dir.iterdir():
                if p.is_file() and p.suffix in IMG_EXTS:
                    files.append(p)
        files.sort()  # 确定性
        images[wnid] = files
    return images

def allocate_counts(total, num_classes):
    base = total // num_classes
    rem = total % num_classes
    return [base + (1 if i < rem else 0) for i in range(num_classes)]

def read_exclude_pairs(exclude_paths):
    """
    exclude 文件格式：<wnid> <stem>（例如：n01440764 ILSVRC2012_val_00000293）
    支持多个排除文件路径（列表）
    返回 set[(wnid, stem)]
    """
    exclude = set()
    if not exclude_paths:
        return exclude
    
    for exclude_path in exclude_paths:
        p = Path(exclude_path)
        if not p.exists():
            print(f"[Warn] 排除文件不存在：{p}，将跳过该文件。")
            continue
        count_before = len(exclude)
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                wnid, stem = parts[0], parts[1]
                exclude.add((wnid, stem))
        count_added = len(exclude) - count_before
        print(f"[Info] 从 {p.name} 加载排除样本：{count_added} 条")
    
    print(f"[Info] 总共需排除样本数：{len(exclude)}")
    return exclude

def main():
    ap = argparse.ArgumentParser(description="Select ImageNet val images and write pathname/label txts (with exclusion).")
    _project_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--val_root", type=str, default=os.path.join(_project_root, "data", "imagenet", "images", "val"),
                    help="ImageNet val 根目录（下面是1000个wnid子目录）。")
    ap.add_argument("--classnames", type=str, default=os.path.join(_project_root, "data", "imagenet", "classnames.txt"),
                    help="classnames.txt 路径，格式：'wnid class_name'")
    ap.add_argument("--num", type=int, default=10000,
                    help="选取的总图片数（默认5000）")
    ap.add_argument("--out_dir", type=str, default=os.path.join(_project_root, "utils"),
                    help="输出txt所在目录（默认当前目录）")
    ap.add_argument("--out_pathname", type=str, default="imagenet_selected_pathname10000.txt",
                    help="路径清单输出文件名")
    ap.add_argument("--out_label", type=str, default="imagenet_selected_label10000.txt",
                    help="标签清单输出文件名")
    ap.add_argument("--exclude_pathname", type=str, nargs='*', default=[],
                    help="已选样本列表（支持多个txt文件路径），将跳过其中项。例如：--exclude_pathname a.txt b.txt")
    args = ap.parse_args()

    wnids, wnid_to_idx = read_classnames(args.classnames)
    images_per_class = collect_images_per_class(args.val_root, wnids)
    exclude_pairs = read_exclude_pairs(args.exclude_pathname)

    per_class_counts = allocate_counts(args.num, len(wnids))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f_path = out_dir / args.out_pathname
    f_label = out_dir / args.out_label

    selected = 0
    skipped_by_exclude = 0

    with open(f_path, "w", encoding="utf-8") as fp, open(f_label, "w", encoding="utf-8") as fl:
        for i, wnid in enumerate(wnids):
            need = per_class_counts[i]
            pool = images_per_class.get(wnid, [])
            if not pool or need <= 0:
                continue

            taken = 0
            for p in pool:
                stem = p.stem
                if (wnid, stem) in exclude_pairs:
                    skipped_by_exclude += 1
                    continue
                # 选中该样本
                fp.write(f"{wnid} {stem}\n")
                fl.write(f"{stem} {wnid_to_idx[wnid]}\n")
                selected += 1
                taken += 1
                if taken >= need:
                    break

    print(f"[Done] 实际写入 {selected} 张（目标 {args.num}）。")
    print(f"[Info] 因去重跳过：{skipped_by_exclude} 张。")
    print(f" - Pathname list: {f_path}")
    print(f" - Label list   : {f_label}")
    if selected != args.num:
        print("[Warn] 未达到目标数量，可能由于某些类别可选样本不足或排除过多；"
              "可降低 --num、或不均匀分配、或允许跨类补齐。")

if __name__ == "__main__":
    main()
