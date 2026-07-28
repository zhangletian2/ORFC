#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 MMBench 数据集中随机采样 N 道题，生成固定的 index list 文件。

用法：
    python sample_mmbench.py \
        --data_dir /path/to/MMBench \
        --split validation --data_subdir en \
        --num_samples 1000 --seed 42 \
        --output ../utils/mmbench_en_val_1000.txt

输出格式（每行一条）：
    <index> <answer> <num_options> <category>
"""

import argparse
import random


def main():
    ap = argparse.ArgumentParser(description="MMBench 随机采样器")
    ap.add_argument("--data_dir", required=True, help="MMBench 数据集目录")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--data_subdir", default="en")
    ap.add_argument("--num_samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True, help="输出 list 文件路径")
    args = ap.parse_args()

    from datasets import load_dataset

    kwargs = {}
    if args.data_subdir:
        kwargs["data_dir"] = args.data_subdir
    ds = load_dataset(args.data_dir, split=args.split, **kwargs)
    total = len(ds)
    n = min(args.num_samples, total)

    random.seed(args.seed)
    selected = sorted(random.sample(range(total), n))

    with open(args.output, "w") as f:
        for i in selected:
            s = ds[i]
            idx = s["index"]
            ans = s.get("answer", "")
            n_opts = sum(
                1 for k in ("A", "B", "C", "D")
                if s.get(k) is not None
                and str(s[k]).strip()
                and str(s[k]).strip().lower() != "nan"
            )
            cat = s.get("category", "")
            f.write(f"{idx}\t{ans}\t{n_opts}\t{cat}\n")

    print(f"Sampled {n}/{total} from {args.split} (seed={args.seed})")
    print(f"  → {args.output}")


if __name__ == "__main__":
    main()
