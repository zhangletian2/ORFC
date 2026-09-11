#!/usr/bin/env python
"""NYU Depth RMSE for residual codecs (base-only vs base+residual)
and optional single-stage Soft-PQ checkpoints at matched rate.
"""
import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ORFCV2 = Path(__file__).resolve().parent
ORFC = ORFCV2.parent / "orfc"
PROJECT = ORFCV2.parents[1]  # .../featcodec/ORFC
sys.path.insert(0, str(ORFC))
sys.path.insert(0, str(ORFCV2))
sys.path.insert(0, str(PROJECT / "tools"))
sys.path.insert(0, str(PROJECT / "backbone" / "dinov2"))

from opq import batch_normalize_gpu, batch_inv_normalize_gpu  # noqa: E402
from residual_pq import load_residual_codec  # noqa: E402
from soft_pq import load_codec  # noqa: E402
from dinov2_depth_pipeline import (  # noqa: E402
    preprocess_image, parse_split_file, _load_backbone, _load_depth_head,
    decode_depth, load_depth_gt, compute_depth_metrics,
)


def _quantize_one(feat_td, codec, norm_mode, device, base_only=False, n_prefix=1):
    """feat_td [T, D] numpy -> reconstructed [T, D] numpy."""
    X = torch.from_numpy(feat_td).float().unsqueeze(0).to(device)
    Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
    if base_only:
        Y_hat = codec.base_forward(Y) if hasattr(codec, "base_forward") else codec(Y)[0]
    else:
        Y_hat = codec(Y)[0]
    return batch_inv_normalize_gpu(Y_hat, Mu, Std).squeeze(0).cpu()


@torch.no_grad()
def eval_codec(codec, feats, layer_idx, sample_meta, backbone, head,
               device, norm_mode, base_only=False, n_prefix=1):
    rmses = []
    for feat, (ori, pad, gt_path) in zip(feats, sample_meta):
        rec = _quantize_one(feat, codec, norm_mode, device,
                            base_only=base_only, n_prefix=n_prefix)
        pred = decode_depth(backbone, head, rec.unsqueeze(0), layer_idx,
                            pad, ori, device)
        gt = load_depth_gt(gt_path)
        rmses.append(compute_depth_metrics(pred, gt)["rmse"])
    return float(np.mean(rmses))


@torch.no_grad()
def eval_anchor(feats, layer_idx, sample_meta, backbone, head, device):
    rmses = []
    for feat, (ori, pad, gt_path) in zip(feats, sample_meta):
        ft = torch.from_numpy(feat).float().unsqueeze(0).to(device)
        pred = decode_depth(backbone, head, ft, layer_idx, pad, ori, device)
        gt = load_depth_gt(gt_path)
        rmses.append(compute_depth_metrics(pred, gt)["rmse"])
    return float(np.mean(rmses))


def load_feats(feat_dir, samples):
    out = []
    for rgb_rel, _, _ in samples:
        name = rgb_rel.replace("/", "_").rsplit(".", 1)[0]
        arr = np.load(str(Path(feat_dir) / f"{name}.npy"))
        if arr.ndim == 3:
            arr = arr.squeeze(0)
        out.append(arr.astype(np.float32))
    return out


def prepare_nyu_samples(data_root, split_file):
    samples = parse_split_file(split_file)
    data_root = Path(data_root)
    sample_meta = []
    for rgb_rel, depth_rel, _ in samples:
        _, ori, pad = preprocess_image(str(data_root / rgb_rel))
        sample_meta.append((ori, pad, str(data_root / depth_rel)))
    return samples, sample_meta


@torch.no_grad()
def evaluate_nyu_residual(codec, feats, layer_idx, sample_meta,
                          backbone, head, device, norm_mode,
                          skip_anchor=False, anchor_rmse=None, n_prefix=1):
    """Return {anchor_rmse, base_rmse, full_rmse, delta_vs_anchor, delta_vs_base}."""
    if skip_anchor and anchor_rmse is not None:
        anchor = float(anchor_rmse)
    else:
        t0 = time.time()
        anchor = eval_anchor(feats, layer_idx, sample_meta, backbone, head, device)
        print(f"  Anchor            RMSE={anchor:.4f}  ({time.time()-t0:.1f}s)")
    t0 = time.time()
    rmse_base = eval_codec(codec, feats, layer_idx, sample_meta, backbone, head,
                           device, norm_mode, base_only=True, n_prefix=n_prefix)
    print(f"  base only         RMSE={rmse_base:.4f}  "
          f"Δ={rmse_base-anchor:+.4f}  ({time.time()-t0:.1f}s)")
    t0 = time.time()
    rmse_full = eval_codec(codec, feats, layer_idx, sample_meta, backbone, head,
                           device, norm_mode, base_only=False, n_prefix=n_prefix)
    print(f"  base+residual     RMSE={rmse_full:.4f}  "
          f"Δ={rmse_full-anchor:+.4f}  Δvs_base={rmse_full-rmse_base:+.4f}  "
          f"({time.time()-t0:.1f}s)")
    return {
        'anchor_rmse': float(anchor),
        'base_rmse': float(rmse_base),
        'full_rmse': float(rmse_full),
        'delta_rmse_vs_anchor': float(rmse_full - anchor),
        'delta_rmse_vs_base': float(rmse_full - rmse_base),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", required=True)
    p.add_argument("--res_ckpt", required=True)
    p.add_argument("--k4_ckpt", default="")
    p.add_argument("--feat_root", default=str(PROJECT / "features" / "nyu_depth_80" / "dinov2_vitl14"))
    p.add_argument("--data_root", default="/data4/workspace/zlt/featcodec/CoFAI/data/NYU")
    p.add_argument("--split_file", default=str(PROJECT / "utils" / "nyu_test_80.txt"))
    p.add_argument("--weights_root", default="/data4/workspace/zlt/cache/torch/hub/checkpoints")
    p.add_argument("--norm_mode", default="per_image")
    p.add_argument("--n_prefix", type=int, default=1,
                   help="1 for dinov2 (CLS); 5 for dinov3 (CLS+4 reg)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_idx = int(args.layer[-2:])
    samples, sample_meta = prepare_nyu_samples(args.data_root, args.split_file)

    feats = load_feats(Path(args.feat_root) / args.layer, samples)
    print(f"{args.layer}: {len(feats)} feats  shape={feats[0].shape}")

    ns = SimpleNamespace(model="vitl14", weights_root=args.weights_root, device=device)
    backbone, _ = _load_backbone(ns)
    head = _load_depth_head(ns)

    codec = load_residual_codec(args.res_ckpt, device=device)
    codec.eval()
    evaluate_nyu_residual(
        codec, feats, layer_idx, sample_meta, backbone, head,
        device, args.norm_mode, n_prefix=args.n_prefix)
    del codec
    torch.cuda.empty_cache()

    if args.k4_ckpt:
        k4 = load_codec(args.k4_ckpt, device=device)
        k4.eval()
        t0 = time.time()
        rmse_k4 = eval_codec(k4, feats, layer_idx, sample_meta, backbone, head,
                             device, args.norm_mode, base_only=False,
                             n_prefix=args.n_prefix)
        print(f"  single-stage ckpt RMSE={rmse_k4:.4f}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
