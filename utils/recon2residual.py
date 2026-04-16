#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np


_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ORIG_ROOT = os.path.join(_PROJECT_ROOT, "features", "test", "dinov2_vitl14")
ORIG_FEAT_ROOT = os.path.join(ORIG_ROOT, "")  # 原始特征根：.../blkxx/*.npy

RECON_ROOT = os.path.join(
    ORIG_ROOT,
    "split_decoded"
)  # 重建特征根：.../split_decoded/vq_embxx_dimxx_lmb1/blkxx/*.npy

OUT_ROOT = os.path.join(
    ORIG_ROOT,
    "split_decoded_residual"
)  # 输出根：.../split_decoded_residual/vq_embxx_dimxx_lmb1/blkxx/*.npy

BLKS = ["blk05", "blk11", "blk17", "blk23"]

# EMBS = [8, 16, 32, 512, 2048]
# DIMS = [64, 64, 64, 64, 32]
EMBS = [2048]
DIMS = [32]
LMB = 1  # 固定 lmb1


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    assert len(EMBS) == len(DIMS), "EMBS 和 DIMS 长度必须一致（一一对应）"

    # 以原始特征为基准：遍历每个 blk 下的所有 npy 文件
    for blk in BLKS:
        orig_blk_dir = os.path.join(ORIG_FEAT_ROOT, blk)
        if not os.path.isdir(orig_blk_dir):
            print(f"[WARN] 原始特征目录不存在，跳过: {orig_blk_dir}")
            continue

        orig_files = sorted(glob.glob(os.path.join(orig_blk_dir, "*.npy")))
        if not orig_files:
            print(f"[WARN] 原始特征目录无 .npy 文件: {orig_blk_dir}")
            continue

        print(f"[INFO] 处理 {blk}，原始文件数: {len(orig_files)}")

        for emb, dim in zip(EMBS, DIMS):
            tag = f"vq_emb{emb}_dim{dim}_lmb{LMB}"
            recon_blk_dir = os.path.join(RECON_ROOT, tag, blk)
            out_blk_dir = os.path.join(OUT_ROOT, tag, blk)
            ensure_dir(out_blk_dir)

            missing = 0
            shape_mismatch = 0
            ok = 0

            for orig_path in orig_files:
                fname = os.path.basename(orig_path)
                recon_path = os.path.join(recon_blk_dir, fname)
                out_path = os.path.join(out_blk_dir, fname)

                if not os.path.isfile(recon_path):
                    missing += 1
                    # 如需更详细日志可取消注释：
                    # print(f"[MISS] {recon_path}")
                    continue

                orig = np.load(orig_path)
                recon = np.load(recon_path).squeeze()

                if orig.shape != recon.shape:
                    shape_mismatch += 1
                    print(f"[SHAPE] {blk}/{tag}/{fname}: orig {orig.shape} != recon {recon.shape}")
                    continue

                # residual = orig - recon
                residual = orig - recon

                # 保存为 float32 可节省空间；如果你希望保持原 dtype，可改成 residual.astype(orig.dtype, copy=False)
                if residual.dtype != np.float32:
                    residual = residual.astype(np.float32, copy=False)

                np.save(out_path, residual)
                ok += 1

            print(f"[DONE] {tag}/{blk}: ok={ok}, missing={missing}, shape_mismatch={shape_mismatch}")


if __name__ == "__main__":
    main()
