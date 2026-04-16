#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--decoded_root", type=str)
    args = parser.parse_args()
    # 根路径
    root = args.root
    decoded_root = args.decoded_root

    # 四个层目录
    layers = ["blk05", "blk11", "blk17", "blk23"]
    for layer in layers:
        orig_dir = os.path.join(root, layer)
        recon_dir = os.path.join(decoded_root, layer)
        print(f"\nProcessing layer: {layer}")
        os.makedirs(recon_dir, exist_ok=True)

        # 获取所有 .npy 文件（重建目录和原始目录文件名一致）
        recon_files = sorted([f for f in os.listdir(recon_dir) if f.endswith(".npy")])

        for fname in tqdm(recon_files):
            orig_path = os.path.join(orig_dir, fname)
            recon_path = os.path.join(recon_dir, fname)

            if not os.path.exists(orig_path):
                print(f"⚠️ 原始特征缺失: {orig_path}")
                continue

            # 读取原始和重建特征
            orig_feat = np.load(orig_path)    # [257, 1024]
            recon_feat = np.load(recon_path)  # [256, 1024]
            if orig_feat.shape != (257, 1024) or recon_feat.shape != (256, 1024):
                print(f"❌ 尺寸异常: {fname}, orig {orig_feat.shape}, recon {recon_feat.shape}")
                continue

            # 拼接CLS
            cls_token = orig_feat[0:1, :]           # [1, 1024]
            merged = np.concatenate([cls_token, recon_feat], axis=0)  # [257, 1024]

            # 覆盖保存
            np.save(recon_path, merged)

    print("\n✅ 所有层处理完成，CLS 已拼接并覆盖保存。")
