#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))

"""
Summarize per-layer average BPFP (from _stats.csv) and recomputed MSE
by reading original & decoded features.

Directory layout assumed:
feat_root/
  ├─ {model}/
  │    ├─ blk05/*.npy          # originals
  │    ├─ blk11/*.npy
  │    ├─ blk17/*.npy
  │    ├─ blk23/*.npy
  │    └─ decoded/chen/{qp}/
  │         ├─ _stats.csv
  │         ├─ blk05/*.npy     # decoded
  │         ├─ blk11/*.npy
  │         ├─ blk17/*.npy
  │         └─ blk23/*.npy
"""

def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = (a.astype(np.float64) - b.astype(np.float64)).ravel()
    return float(np.mean(d * d)) if d.size else 0.0

def summarize_for_model_qp(feat_root: Path, model: str, qp: int, layers, stats_name: str):
    qp_dir = feat_root / model / "decoded" / "chen" / str(qp)
    stats_csv = qp_dir / stats_name
    if not stats_csv.exists():
        print(f"[WARN] stats not found: {stats_csv}")
        return

    df = pd.read_csv(stats_csv)
    # 只保留本 model / qp 的记录（以防同目录混入其它）
    if "model" in df.columns: df = df[df["model"] == model]
    if "qp" in df.columns: df = df[df["qp"] == qp]
    if df.empty:
        print(f"[WARN] empty stats in {stats_csv}")
        return

    print(f"\n=== Model: {model} | QP: {qp} ===")
    header_printed = False

    for layer in layers:
        df_l = df[df["layer"] == layer]
        if df_l.empty:
            print(f"[INFO] no entries for layer {layer} in {stats_csv}")
            continue

        # 平均 BPFP（直接用 CSV 字段）
        avg_bpfp = float(df_l["bpfp"].mean())

        # 重新读取原始与解码特征，计算平均 MSE
        orig_dir = feat_root / model / layer
        dec_dir  = qp_dir / layer
        filenames = df_l["filename"].tolist()

        mses = []
        misses = 0
        for name in tqdm(filenames, desc=f"{model}/QP{qp}/{layer}", unit="file", leave=False):
            orig_p = orig_dir / name
            dec_p  = dec_dir  / name
            if not orig_p.exists() or not dec_p.exists():
                misses += 1
                continue
            try:
                a = np.load(orig_p)
                b = np.load(dec_p)
            except Exception as e:
                misses += 1
                continue

            if a.shape != b.shape:
                # 尽量不报错，直接跳过坏样本
                misses += 1
                continue
            mses.append(mse(a, b))

        avg_mse = float(np.mean(mses)) if len(mses) > 0 else float("nan")
        n_used  = len(mses)
        n_all   = len(filenames)

        if not header_printed:
            print(f"{'layer':<8} {'avg_bpfp':>12} {'avg_mse':>12} {'used/total':>12}")
            header_printed = True
        print(f"{layer:<8} {avg_bpfp:12.6f} {avg_mse:12.6f} {n_used:5d}/{n_all:<6d}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Summarize per-layer BPFP & MSE from _stats.csv and decoded features")
    parser.add_argument("--feat_root", type=str, default=os.path.join(_PROJECT_ROOT, "features", "test"))
    parser.add_argument("--models", nargs="+", default=["dinov2_vitl14", "clip_vitl14"])
    parser.add_argument("--layers", nargs="+", default=["blk05", "blk11", "blk17", "blk23"])
    parser.add_argument("--qps", type=int, nargs="+", default=[0, 12, 22, 32, 42])
    parser.add_argument("--stats_name", type=str, default="_stats.csv")
    args = parser.parse_args()

    feat_root = Path(args.feat_root)
    for model in args.models:
        for qp in args.qps:
            summarize_for_model_qp(feat_root, model, qp, args.layers, args.stats_name)

if __name__ == "__main__":
    main()
