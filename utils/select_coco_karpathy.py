#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 COCO Karpathy test split 中选取固定子集，生成：
  - coco_selected_pathname{N}.txt   每行: image_id  (如 COCO_val2014_000000462565)
  - coco_selected_caption{N}.json   JSON 数组: [{image, image_id, caption}, ...]

用法:
  python select_coco_karpathy.py \
      --karpathy_json /path/to/coco_karpathy_test.json \
      --num 500 --seed 42
"""

import os
import json
import argparse
import numpy as np


def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser(description="Select COCO Karpathy test subset.")
    ap.add_argument(
        "--karpathy_json",
        default=os.path.join(
            _script_dir, "..", "..", "..",
            "CAPO_ret", "dataset", "coco2014", "coco_karpathy_test.json"),
        help="Karpathy test split JSON",
    )
    ap.add_argument("--num", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=_script_dir)
    args = ap.parse_args()

    with open(args.karpathy_json, "r") as f:
        all_entries = json.load(f)
    print(f"Karpathy test split 共 {len(all_entries)} 张图片")

    rng = np.random.RandomState(args.seed)
    if 0 < args.num < len(all_entries):
        indices = rng.choice(len(all_entries), args.num, replace=False)
        indices.sort()
        selected = [all_entries[i] for i in indices]
    else:
        selected = all_entries

    os.makedirs(args.out_dir, exist_ok=True)
    pathname_file = os.path.join(args.out_dir, f"coco_selected_pathname{args.num}.txt")
    caption_file = os.path.join(args.out_dir, f"coco_selected_caption{args.num}.json")

    meta = []
    with open(pathname_file, "w") as fp:
        for i, entry in enumerate(selected):
            fname = os.path.basename(entry["image"])
            image_id = os.path.splitext(fname)[0]
            fp.write(f"{image_id}\n")
            meta.append({
                "index": i,
                "image": entry["image"],
                "image_id": image_id,
                "caption": entry["caption"],
            })

    with open(caption_file, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[Done] 选取 {len(selected)} 张图片 (seed={args.seed})")
    print(f"  pathname : {pathname_file}")
    print(f"  caption  : {caption_file}")


if __name__ == "__main__":
    main()
