#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从VOC2012分割验证集中选择指定数量的图片，支持排除已选样本。
python utils/select_voc2012_val.py \
    --num 5000 \
    --exclude utils/voc2012_val_100.txt \
    --out_name voc2012_trainval_5000.txt
"""

import os
import argparse
import random
from pathlib import Path

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_VOC_ROOT = os.path.join(_PROJECT_ROOT, "data", "VOCdevkit", "VOC2012")


def load_image_list(voc_root, split="val"):
    """
    读取图片列表
    
    Args:
        voc_root: VOC2012根目录
        split: 
            - "val": 分割验证集（1449张）
            - "train": 分割训练集（1464张）
            - "trainval": 分割训练+验证（2913张）
            - "seg_all": 扫描SegmentationClass（2913张，有分割标注）
            - "all": 扫描JPEGImages（17125张，全部图片）
    """
    voc_root = Path(voc_root)
    
    if split == "all":
        # 扫描所有原始图片（不需要分割标注）
        img_dir = voc_root / "JPEGImages"
        names = [p.stem for p in img_dir.glob("*.jpg")]
        names.sort()
        return names
    elif split == "seg_all":
        # 扫描有分割标注的图片
        seg_dir = voc_root / "SegmentationClass"
        names = [p.stem for p in seg_dir.glob("*.png")]
        names.sort()
        return names
    else:
        # 从txt文件读取
        txt_path = voc_root / f"ImageSets/Segmentation/{split}.txt"
        with open(txt_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]


def read_name_list(file_paths, label):
    """
    读取名称列表，支持多个文件
    格式：每行一个图片名（不含扩展名）
    返回 set 和保序 list
    """
    names_set = set()
    names_ordered = []
    if not file_paths:
        return names_set, names_ordered

    for fpath in file_paths:
        p = Path(fpath)
        if not p.exists():
            print(f"[Warn] {label}文件不存在：{p}，将跳过。")
            continue
        count_before = len(names_set)
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name and name not in names_set:
                    names_set.add(name)
                    names_ordered.append(name)
        count_added = len(names_set) - count_before
        print(f"[Info] 从 {p.name} 加载{label}样本：{count_added} 条")

    print(f"[Info] 总共{label}样本数：{len(names_set)}")
    return names_set, names_ordered


def main():
    ap = argparse.ArgumentParser(description="从VOC2012分割数据集选择指定数量图片")
    ap.add_argument("--voc_root", type=str, default=DEFAULT_VOC_ROOT,
                    help="VOC2012根目录")
    ap.add_argument("--split", type=str, default="val", choices=["val", "train", "trainval", "seg_all", "all"],
                    help="数据集划分: val(1449), train(1464), trainval(2913), seg_all(有标注2913), all(全部图片17125)")
    ap.add_argument("--num", type=int, default=100,
                    help="选取的图片数（默认100）")
    ap.add_argument("--out_dir", type=str, default=os.path.join(_PROJECT_ROOT, "utils"),
                    help="输出目录")
    ap.add_argument("--out_name", type=str, default=None,
                    help="输出文件名（默认 voc2012_<split>_<num>.txt）")
    ap.add_argument("--exclude", type=str, nargs='*', default=[],
                    help="排除列表文件路径（支持多个）")
    ap.add_argument("--include", type=str, nargs='*', default=[],
                    help="必须包含的样本列表（支持多个），优先写入输出，剩余配额再从池中补充。")
    ap.add_argument("--shuffle", action="store_true",
                    help="是否随机打乱（默认按原顺序）")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子（仅shuffle时有效）")
    args = ap.parse_args()

    # 读取图片列表
    all_names = load_image_list(args.voc_root, args.split)
    print(f"[Info] VOC2012({args.split})共 {len(all_names)} 张图片")

    # 读取排除和包含列表
    exclude_set, _ = read_name_list(args.exclude, "排除")
    include_set, include_ordered = read_name_list(args.include, "强制包含")

    # 过滤：排除已有 + 已在 include 中的
    available = [n for n in all_names if n not in exclude_set and n not in include_set]
    print(f"[Info] 排除后剩余 {len(available)} 张可选（不含 include）")

    # 打乱（可选）
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(available)

    # 先放入 include，再用剩余配额从池中补充
    remaining = args.num - len(include_ordered)
    selected = list(include_ordered) + available[:max(0, remaining)]
    print(f"[Info] 其中强制包含：{len(include_ordered)} 张")
    
    # 输出
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name or f"voc2012_{args.split}_{args.num}.txt"
    out_path = out_dir / out_name

    with open(out_path, "w", encoding="utf-8") as f:
        for name in selected:
            f.write(f"{name}\n")

    print(f"[Done] 写入 {len(selected)} 张（目标 {args.num}）")
    print(f"  输出文件: {out_path}")
    
    if len(selected) < args.num:
        print(f"[Warn] 可选样本不足，仅选取 {len(selected)} 张")


if __name__ == "__main__":
    main()
