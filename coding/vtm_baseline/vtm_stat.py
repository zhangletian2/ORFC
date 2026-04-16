#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从前述 vtm_featcodec_adapter.py 生成的 _stats.csv 读取 BPFP，
并与解码特征、原始特征对齐，计算每个 (model, qp, layer) 的平均 BPFP 和平均 MSE。

目录结构默认：
{feat_root}/{model}/{layer}/*.npy                     # 原始特征
{feat_root}/{model}/decoded/vtm/{qp}/{layer}/*.npy    # 解码特征
{feat_root}/{model}/decoded/vtm/{qp}/_stats.csv       # 对应CSV（上一版脚本输出）

使用示例：
python compute_layer_bpfp_mse_from_stats.py \
  --feat_root $PROJECT_ROOT/features/test \
  --models dinov2_vitl14 clip_vitl14 \
  --layers blk05 blk11 blk17 blk23 \
  --qps 0 12 22 32 42
"""

import os
import csv
import argparse
from pathlib import Path
import numpy as np

CSV_HEADER = [
    "filename","model","layer","qp",
    "bitstream_bytes","bits","bpfp",
    "encode_s","decode_s","total_s",
    "orig_bytes","comp_ratio"
]

def load_feat(path: Path) -> np.ndarray:
    arr = np.load(path)
    # 兼容 (1,257,1024) -> (257,1024)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    # 若还有不一致，最后兜底拉平
    return arr.astype(np.float32)

def compute_avg_for_layer(feat_root: Path, model: str, layer: str, qp: int) -> tuple:
    """
    返回 (count, avg_bpfp, avg_mse)。只统计两边文件都存在且能对齐的样本。
    BPFP 来自 {feat_root}/{model}/decoded/vtm/{qp}/_stats.csv 对应 layer 的行。
    """
    stats_csv = feat_root / model / "decoded" / "vtm" / str(qp) / "_stats.csv"
    if not stats_csv.exists():
        print(f"[WARN] stats csv missing: {stats_csv}")
        return 0, float("nan"), float("nan")

    # 读取该 qp 的所有记录，过滤到指定 layer
    rows = []
    with open(stats_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        # 简单校验列名（不强制完全一致，只要包含需要的列）
        need_cols = {"filename","layer","bpfp"}
        if not need_cols.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(f"_stats.csv 缺少必要字段：{need_cols}，实际列：{reader.fieldnames}")
        for r in reader:
            if r["layer"] == layer:
                rows.append(r)

    if not rows:
        return 0, float("nan"), float("nan")

    # 建立原始/解码路径
    orig_dir = feat_root / model / layer
    dec_dir  = feat_root / model / "decoded" / "vtm" / str(qp) / layer
    if not orig_dir.exists() or not dec_dir.exists():
        print(f"[WARN] missing dir: {orig_dir} or {dec_dir}")
        return 0, float("nan"), float("nan")

    cnt = 0
    sum_bpfp = 0.0
    sum_mse  = 0.0

    for r in rows:
        stem = r["filename"]
        bpfp_str = r.get("bpfp", "")
        if bpfp_str == "":
            # 没有 bpfp（理论上不会，因为 _stats.csv 一定有），跳过
            continue
        try:
            bpfp_val = float(bpfp_str)
        except:
            continue

        orig_fp = orig_dir / f"{stem}.npy"
        dec_fp  = dec_dir  / f"{stem}.npy"
        if not (orig_fp.exists() and dec_fp.exists()):
            # 文件缺失就略过该样本
            continue

        try:
            o = load_feat(orig_fp)
            d = load_feat(dec_fp)
            if o.shape != d.shape:
                # 兜底：拉平比较
                o = o.reshape(-1)
                d = d.reshape(-1)
                if o.shape != d.shape:
                    # 仍然对不上，跳过
                    continue
            mse = float(np.mean((o - d) ** 2))
        except Exception as e:
            print(f"[ERR] {stem}: {e}")
            continue

        cnt += 1
        sum_bpfp += bpfp_val
        sum_mse  += mse

    if cnt == 0:
        return 0, float("nan"), float("nan")

    return cnt, (sum_bpfp / cnt), (sum_mse / cnt)

def parse_layer_qps(layer_qps_str: str) -> dict:
    """
    解析 layer:qp1,qp2,... 格式的字符串
    例如: "blk05:25,30,35" -> {"blk05": [25, 30, 35]}
    """
    result = {}
    for item in layer_qps_str.split():
        if ":" not in item:
            continue
        layer, qps_str = item.split(":", 1)
        qps = [int(q.strip()) for q in qps_str.split(",") if q.strip()]
        result[layer] = qps
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_root", required=True, type=str)
    ap.add_argument("--models", nargs="+", default=["dinov2_vitl14","clip_vitl14"])
    ap.add_argument("--layers", nargs="+", default=["blk05","blk11","blk17","blk23"])
    ap.add_argument("--qps",    nargs="+", type=int, default=[0,12,22,32,42])
    ap.add_argument("--layer_qps", type=str, default="",
                    help="按layer指定QP，格式: 'blk05:25,30,35 blk11:29,34 ...'，指定后忽略--layers和--qps")
    args = ap.parse_args()

    feat_root = Path(args.feat_root)

    # 解析 layer-qp 映射
    if args.layer_qps:
        layer_qp_map = parse_layer_qps(args.layer_qps)
    else:
        # 使用传统方式：所有 layer 使用相同的 qps
        layer_qp_map = {layer: args.qps for layer in args.layers}

    print(f"[START] feat_root={feat_root}")
    print(f"[CONFIG] layer_qp_map={layer_qp_map}")
    
    for model in args.models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")
        
        # 按 layer 分组输出
        for layer, qps in sorted(layer_qp_map.items()):
            print(f"\n  Layer: {layer}")
            print(f"  {'-'*50}")
            for qp in sorted(qps):
                cnt, avg_bpfp, avg_mse = compute_avg_for_layer(feat_root, model, layer, qp)
                if cnt > 0:
                    print(f"    QP={qp:2d} | files={cnt:5d} | avg_BPFP={avg_bpfp:.6f} | avg_MSE={avg_mse:.6e}")
                else:
                    print(f"    QP={qp:2d} | [NO DATA]")

    print(f"\n{'='*60}")
    print("[DONE]")

if __name__ == "__main__":
    main()
