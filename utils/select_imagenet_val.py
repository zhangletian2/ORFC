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

def read_pair_list(file_paths, label):
    """
    读取 pathname 文件列表，格式：<wnid> <stem>
    返回 set[(wnid, stem)]
    """
    pairs = set()
    if not file_paths:
        return pairs

    for fpath in file_paths:
        p = Path(fpath)
        if not p.exists():
            print(f"[Warn] {label}文件不存在：{p}，将跳过该文件。")
            continue
        count_before = len(pairs)
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                wnid, stem = parts[0], parts[1]
                pairs.add((wnid, stem))
        count_added = len(pairs) - count_before
        print(f"[Info] 从 {p.name} 加载{label}样本：{count_added} 条")

    print(f"[Info] 总共{label}样本数：{len(pairs)}")
    return pairs

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
    ap.add_argument("--include_pathname", type=str, nargs='*', default=[],
                    help="必须包含的样本列表（支持多个txt文件路径），这些样本会优先写入输出，剩余配额再从池中补充。")
    args = ap.parse_args()

    wnids, wnid_to_idx = read_classnames(args.classnames)
    images_per_class = collect_images_per_class(args.val_root, wnids)
    exclude_pairs = read_pair_list(args.exclude_pathname, "排除")
    include_pairs = read_pair_list(args.include_pathname, "强制包含")

    # 按类别统计 include 中各类已有多少
    from collections import defaultdict
    include_per_class = defaultdict(list)
    for wnid, stem in include_pairs:
        include_per_class[wnid].append(stem)

    per_class_counts = allocate_counts(args.num, len(wnids))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f_path = out_dir / args.out_pathname
    f_label = out_dir / args.out_label

    selected = 0
    skipped_by_exclude = 0
    included_count = 0

    with open(f_path, "w", encoding="utf-8") as fp, open(f_label, "w", encoding="utf-8") as fl:
        for i, wnid in enumerate(wnids):
            need = per_class_counts[i]
            pool = images_per_class.get(wnid, [])

            # 1) 先写入该类中的 include 样本
            inc_stems = include_per_class.get(wnid, [])
            taken = 0
            for stem in inc_stems:
                fp.write(f"{wnid} {stem}\n")
                fl.write(f"{stem} {wnid_to_idx[wnid]}\n")
                selected += 1
                included_count += 1
                taken += 1

            # 2) 剩余配额从池中补充
            remaining = need - taken
            if remaining <= 0 or not pool:
                continue
            already_written = set(inc_stems)
            for p in pool:
                stem = p.stem
                if stem in already_written:
                    continue
                if (wnid, stem) in exclude_pairs:
                    skipped_by_exclude += 1
                    continue
                fp.write(f"{wnid} {stem}\n")
                fl.write(f"{stem} {wnid_to_idx[wnid]}\n")
                selected += 1
                remaining -= 1
                if remaining <= 0:
                    break

    print(f"[Done] 实际写入 {selected} 张（目标 {args.num}）。")
    print(f"[Info] 其中强制包含：{included_count} 张，因排除跳过：{skipped_by_exclude} 张。")
    print(f" - Pathname list: {f_path}")
    print(f" - Label list   : {f_label}")
    if selected != args.num:
        print("[Warn] 未达到目标数量，可能由于某些类别可选样本不足或排除过多；"
              "可降低 --num、或不均匀分配、或允许跨类补齐。")

if __name__ == "__main__":
    main()
