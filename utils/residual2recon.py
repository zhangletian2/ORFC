#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np


_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ORIG_ROOT = os.path.join(_PROJECT_ROOT, "features", "test", "dinov2_vitl14")
ORIG_FEAT_ROOT = os.path.join(ORIG_ROOT, "")  # 原始特征根：.../blkxx/*.npy

RECON_FEAT_ROOT = os.path.join(
    ORIG_ROOT,
    "split_decoded"
)  # VQ重建特征根：.../split_decoded/vq_embxx_dimxx_lmb1/blkxx/*.npy

RECON_RESIDUAL_ROOT = os.path.join(
    ORIG_ROOT,
    "split_decoded"
)  # 残差重建特征根：.../split_decoded/residual_trunc_mse_lmbda1/blkxx/*.npy

OUT_ROOT = os.path.join(
    ORIG_ROOT,
    "split_decoded"
)  # 输出根：.../split_decoded/vq_embxx_dimxx_lmb1_residual_trunc_mse_lmbda1/blkxx/*.npy

BLKS = ["blk05", "blk11", "blk17", "blk23"]

# EMBS = [8, 16, 32, 512, 2048]
# DIMS = [64, 64, 64, 64, 32]
EMBS = [2048]
DIMS = [32]
VQ_LMB = [1]  
HP_LMB = [0.1, 1, 10]
def main():
    assert len(EMBS) == len(DIMS), "EMBS 和 DIMS 长度必须一致（一一对应）"    
    print(f"BLK\tVQ_EMB\tVQ_DIM\tVQ_LMB\tHY_LMB\tMSE")
    for emb, dim, vq_lmb in zip(EMBS, DIMS, VQ_LMB):
        for hp_lmb in HP_LMB:
            for blk in BLKS:
                vq_blk_dir = os.path.join(RECON_FEAT_ROOT, f"vq_emb{emb}_dim{dim}_lmb{vq_lmb}", blk)
                vq_files = glob.glob(os.path.join(vq_blk_dir, "*.npy"))
                recon_residual_blk_dir = os.path.join(RECON_FEAT_ROOT, f"residual_trunc_mse_lmbda{hp_lmb}", blk)
                tag = f"vq_emb{emb}_dim{dim}_lmb{vq_lmb}"
                out_blk_dir = os.path.join(RECON_FEAT_ROOT, f"{tag}_residual_trunc_mse_lmbda{hp_lmb}", blk)
                os.makedirs(out_blk_dir, exist_ok=True)

                missing = 0
                shape_mismatch = 0
                ok = 0
                mse = 0.0
                for vq_path in vq_files:
                    fname = os.path.basename(vq_path)
                    recon_residual_path = os.path.join(recon_residual_blk_dir, fname)
                    out_path = os.path.join(out_blk_dir, fname)
                    orig_path = os.path.join(ORIG_FEAT_ROOT, blk, fname)
                    orig = np.load(orig_path)

                    if not os.path.isfile(recon_residual_path):
                        missing += 1
                        print(f"[MISS] {recon_residual_path}")
                        continue

                    vq = np.load(vq_path).squeeze()
                    recon_residual = np.load(recon_residual_path).squeeze()

                    if vq.shape != recon_residual.shape:
                        shape_mismatch += 1
                        print(f"[SHAPE] {blk}/{tag}/{fname}: vq {vq.shape} != recon_residual {recon_residual.shape}")
                        continue

                    recon = vq + recon_residual
                    mse += np.mean((recon - orig) ** 2)

                    if recon.dtype != np.float32:
                        recon = recon.astype(np.float32, copy=False)

                    np.save(out_path, recon)
                    ok += 1
                print(f"{blk}\t{emb}\t{dim}\t{vq_lmb}\t{hp_lmb}\t{mse / ok:.4f}")


if __name__ == "__main__":
    main()
