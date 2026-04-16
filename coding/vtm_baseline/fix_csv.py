#!/usr/bin/env python
"""从已生成的文件中补写_stats.csv（无需重新编码）"""
import os, csv
import numpy as np
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))

CSV_HEADER = ["filename","model","layer","qp","bitstream_bytes","bits","bpfp","encode_s","decode_s","total_s"]

# 缺失的文件列表
MISSING_FILES = {
    ("clip_vitl14", "blk11"): [
        "ILSVRC2012_val_00000003", "ILSVRC2012_val_00000007", "ILSVRC2012_val_00000008",
        "ILSVRC2012_val_00000011", "ILSVRC2012_val_00000092", "ILSVRC2012_val_00000093",
        "ILSVRC2012_val_00000098", "ILSVRC2012_val_00000105", "ILSVRC2012_val_00000111",
        "ILSVRC2012_val_00000122", "ILSVRC2012_val_00000125", "ILSVRC2012_val_00000127",
        "ILSVRC2012_val_00000130", "ILSVRC2012_val_00000137", "ILSVRC2012_val_00000164",
        "ILSVRC2012_val_00000172", "ILSVRC2012_val_00000175", "ILSVRC2012_val_00000181",
        "ILSVRC2012_val_00000191", "ILSVRC2012_val_00000195", "ILSVRC2012_val_00000198",
        "ILSVRC2012_val_00000199", "ILSVRC2012_val_00000207", "ILSVRC2012_val_00000218",
        "ILSVRC2012_val_00000225", "ILSVRC2012_val_00000227", "ILSVRC2012_val_00000231",
        "ILSVRC2012_val_00000238", "ILSVRC2012_val_00000249", "ILSVRC2012_val_00000253",
        "ILSVRC2012_val_00000256", "ILSVRC2012_val_00000301",
    ],
    ("clip_vitl14", "blk17"): [
        "ILSVRC2012_val_00000074",
    ],
    ("dinov2_vitl14", "blk17"): [
        "ILSVRC2012_val_00000153", "ILSVRC2012_val_00000272", "ILSVRC2012_val_00000287",
        "ILSVRC2012_val_00000288", "ILSVRC2012_val_00000289", "ILSVRC2012_val_00000297",
        "ILSVRC2012_val_00000298", "ILSVRC2012_val_00000312", "ILSVRC2012_val_00000318",
        "ILSVRC2012_val_00000319", "ILSVRC2012_val_00000330", "ILSVRC2012_val_00000342",
        "ILSVRC2012_val_00000346", "ILSVRC2012_val_00001066", "ILSVRC2012_val_00001067",
    ],
}

def main():
    feat_root = Path(_PROJECT_ROOT) / "features" / "test"
    tmp_dir = Path(_SCRIPT_DIR) / "_vtm_tmp"
    qp = 7
    H, W = 257, 1024  # 特征形状

    for (model, layer), files in MISSING_FILES.items():
        out_dir = feat_root / model / "decoded" / "vtm" / str(qp) / layer
        stats_csv = out_dir.parent / "_stats.csv"
        
        print(f"\n[处理] model={model} layer={layer} qp={qp}")
        
        rows = []
        for stem in files:
            # 检查 .npy 文件是否已存在
            npy_file = out_dir / f"{stem}.npy"
            if not npy_file.exists():
                print(f"  [跳过] {stem}.npy 不存在")
                continue
            
            # 查找 bitstream 文件
            bitstream = tmp_dir / f"{stem}.qp{qp}.vvc"
            if not bitstream.exists():
                print(f"  [跳过] {stem}.qp{qp}.vvc bitstream 不存在")
                continue
            
            bs_bytes = os.path.getsize(bitstream)
            bits = bs_bytes * 8
            bpfp = bits / (H * W)
            
            rows.append([
                stem, model, layer, qp,
                bs_bytes, bits, f"{bpfp:.6f}",
                "0.0", "0.0", "0.0"  # 时间信息已丢失，填0
            ])
            print(f"  [添加] {stem}: BPFP={bpfp:.4f}")
        
        # 追加写入 _stats.csv
        if rows:
            write_header = not stats_csv.exists()
            with open(stats_csv, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(CSV_HEADER)
                w.writerows(rows)
            print(f"  [完成] 已追加 {len(rows)} 条记录到 {stats_csv}")

    print("\n[全部完成]")

if __name__ == "__main__":
    main()
