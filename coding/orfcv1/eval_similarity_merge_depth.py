#!/usr/bin/env python
"""NYU test80 RMSE for similarity-merge (copy-mean and frozen r_ψ).

No retraining.  Matching is recomputed on NYU features (cosine greedy).
``r`` is set by merge fraction of the NYU patch count, not the ImageNet
r=128 absolute count: NYU tokens are a 35×46 (or similar) grid, not 16×16.

Usage:
    CUDA_VISIBLE_DEVICES=6 python -u eval_similarity_merge_depth.py \
        --layers blk05 blk10 blk15 blk20 --merge_fracs 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ORFC = HERE.parent / "orfc"
ORFCV2 = HERE.parent / "orfcv2"
PROJECT = HERE.parents[1]
for p in (str(HERE), str(ORFC), str(ORFCV2),
          str(PROJECT / "tools"), str(PROJECT / "backbone" / "dinov2")):
    if p not in sys.path:
        sys.path.insert(0, p)

from opq import batch_normalize_gpu, batch_inv_normalize_gpu  # noqa: E402
from similarity_merge import (  # noqa: E402
    SimilarityMergeCodec, load_merge_codec, pair_stats,
)
from eval_residual_depth import (  # noqa: E402
    load_feats, prepare_nyu_samples, eval_anchor,
)
from dinov2_depth_pipeline import (  # noqa: E402
    _load_backbone, _load_depth_head, decode_depth, load_depth_gt,
    compute_depth_metrics,
)

import warnings  # noqa: E402
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

MERGE_DIR = HERE / "results" / "similarity_merge" / "dinov2_vitl14"
CKPT_R128 = "{layer}_r128_h256_lr0.0003_ep100_s42.pt"
CKPT_R64 = "{layer}_r64_h256_lr0.0003_ep100_s42.pt"


def _feat_hw(feat, pad_shape, n_prefix=1):
    n_patch = feat.shape[0] - n_prefix
    pad_h, pad_w = pad_shape
    h, w = pad_h // 14, pad_w // 14
    if h * w != n_patch:
        raise ValueError(
            f"pad {pad_h}x{pad_w} -> {h}x{w}={h * w} != n_patch={n_patch}")
    return h, w


def _r_from_frac(n_patch, frac):
    """``frac`` is r / T_patch (0.5 = ImageNet r=128/256 perfect matching)."""
    r = int(round(float(frac) * n_patch))
    r = max(1, min(r, n_patch // 2))
    return r


def _ckpt_for(layer, frac):
    if abs(float(frac) - 0.25) < 1e-6:
        p = MERGE_DIR / CKPT_R64.format(layer=layer)
        if p.is_file():
            return p
    p = MERGE_DIR / CKPT_R128.format(layer=layer)
    return p if p.is_file() else None


@torch.no_grad()
def reconstruct_merge(feats, pads, codec, r, n_prefix, norm_mode, device,
                      batch_size=4):
    """Return reconstructed [T,D] tensors and matching diagnostics."""
    recs = []
    cos_acc = 0.0
    dist_acc = 0.0
    n = 0
    for start in range(0, len(feats), batch_size):
        end = min(start + batch_size, len(feats))
        chunk = feats[start:end]
        shapes = {a.shape for a in chunk}
        if len(shapes) != 1:
            for feat, pad in zip(chunk, pads[start:end]):
                rec, d = _one(feat, pad, codec, r, n_prefix, norm_mode, device)
                recs.append(rec)
                cos_acc += d["mean_matched_cos"]
                dist_acc += d["mean_grid_l2"]
                n += 1
            continue
        h, w = _feat_hw(chunk[0], pads[start], n_prefix)
        codec.r = r
        codec.grid = (h, w)
        X = torch.from_numpy(np.stack(chunk)).float().to(device)
        Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
        Y_hat, aux = _forward_with_aux(codec, Y)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        diag = _diag_from_aux(Y, aux, h, w, n_prefix)
        for i in range(X_hat.shape[0]):
            recs.append(X_hat[i].cpu())
        cos_acc += diag["mean_matched_cos"] * X_hat.shape[0]
        dist_acc += diag["mean_grid_l2"] * X_hat.shape[0]
        n += X_hat.shape[0]
        del X, Y, Mu, Std, Y_hat, X_hat
    stats = {
        "mean_matched_cos": float(cos_acc / max(n, 1)),
        "mean_grid_l2": float(dist_acc / max(n, 1)),
        "r": int(r),
    }
    return recs, stats


def _forward_with_aux(codec, Y):
    seq, aux = codec.encode_pairs(Y)
    Y_hat = codec.decode_pairs(seq, aux)
    return Y_hat, aux


def _diag_from_aux(Y, aux, h, w, n_prefix):
    p = n_prefix
    patch = Y[:, p:, :]
    x = F.normalize(patch, dim=-1)
    B = patch.shape[0]
    bi = torch.arange(B, device=Y.device)[:, None]
    left, right = aux["left"], aux["right"]
    cos = (x[bi, left] * x[bi, right]).sum(-1).mean().item()
    dist = pair_stats(left, right, h, w)
    return {"mean_matched_cos": float(cos), "mean_grid_l2": float(dist)}


def _one(feat, pad, codec, r, n_prefix, norm_mode, device):
    h, w = _feat_hw(feat, pad, n_prefix)
    codec.r = r
    codec.grid = (h, w)
    X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
    Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
    Y_hat, aux = _forward_with_aux(codec, Y)
    rec = batch_inv_normalize_gpu(Y_hat, Mu, Std).squeeze(0).cpu()
    diag = _diag_from_aux(Y, aux, h, w, n_prefix)
    return rec, diag


@torch.no_grad()
def eval_rmse_from_recs(recs, layer_idx, sample_meta, backbone, head, device):
    rmses = []
    for rec, (ori, pad, gt_path) in zip(recs, sample_meta):
        feat = rec if torch.is_tensor(rec) else torch.from_numpy(rec)
        feat = feat.float().unsqueeze(0).to(device)
        pred = decode_depth(backbone, head, feat, layer_idx, pad, ori, device)
        gt = load_depth_gt(gt_path)
        rmses.append(compute_depth_metrics(pred, gt)["rmse"])
    return float(np.mean(rmses))


def eval_arm(name, codec, feats, pads, r, layer_idx, sample_meta,
             backbone, head, device, n_prefix, norm_mode, batch_size):
    t0 = time.time()
    recs, stats = reconstruct_merge(
        feats, pads, codec, r, n_prefix, norm_mode, device,
        batch_size=batch_size)
    rmse = eval_rmse_from_recs(recs, layer_idx, sample_meta, backbone, head,
                               device)
    elapsed = time.time() - t0
    row = {"name": name, "rmse": rmse, "seconds": elapsed, **stats}
    print(f"  {name:<22} RMSE={rmse:.4f}  r={r}  "
          f"cos={stats['mean_matched_cos']:.3f}  "
          f"gridL2={stats['mean_grid_l2']:.2f}  ({elapsed:.1f}s)")
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", nargs="+", default=["blk05", "blk10", "blk15", "blk20"])
    p.add_argument("--merge_fracs", type=float, nargs="+", default=[0.5],
                   help="Fraction of patch tokens involved in a pair "
                        "(0.5 = ImageNet r=128/256; 0.25 = r=64/256)")
    p.add_argument("--feat_root", default=str(
        PROJECT / "features" / "nyu_depth_80" / "dinov2_vitl14"))
    p.add_argument("--data_root", default="/data4/workspace/zlt/featcodec/CoFAI/data/NYU")
    p.add_argument("--split_file", default=str(PROJECT / "utils" / "nyu_test_80.txt"))
    p.add_argument("--weights_root",
                   default="/data4/workspace/zlt/cache/torch/hub/checkpoints")
    p.add_argument("--norm_mode", default="per_image")
    p.add_argument("--n_prefix", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--skip_trained", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'#' * 70}")
    print(f"# NYU merge depth  layers={args.layers}  fracs={args.merge_fracs}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  device={device}")
    print(f"{'#' * 70}")

    samples, sample_meta = prepare_nyu_samples(args.data_root, args.split_file)
    if args.max_images > 0:
        samples = samples[:args.max_images]
        sample_meta = sample_meta[:args.max_images]
    pads = [pad for (_, pad, _) in sample_meta]
    print(f"  NYU samples: {len(samples)}")

    ns = SimpleNamespace(model="vitl14", weights_root=args.weights_root,
                         device=device)
    print("  Loading DINOv2 + NYU depth head...")
    backbone, _ = _load_backbone(ns)
    head = _load_depth_head(ns)

    all_results = {
        "config": vars(args),
        "n_images": len(samples),
        "layers": {},
    }

    for layer in args.layers:
        layer_idx = int(layer[-2:])
        print(f"\n{'=' * 60}\n  {layer}  (replay from block {layer_idx})\n"
              f"{'=' * 60}")
        feats = load_feats(Path(args.feat_root) / layer, samples)
        T, D = feats[0].shape
        n_patch = T - args.n_prefix
        h, w = _feat_hw(feats[0], pads[0], args.n_prefix)
        print(f"  feat {T}x{D}  grid={h}x{w}  n_patch={n_patch}")

        t0 = time.time()
        anchor = eval_anchor(feats, layer_idx, sample_meta, backbone, head,
                             device)
        print(f"  {'anchor':<22} RMSE={anchor:.4f}  ({time.time() - t0:.1f}s)")
        layer_row = {"T": T, "D": D, "grid": [h, w], "n_patch": n_patch,
                     "anchor_rmse": float(anchor), "arms": []}

        copy_codec = SimilarityMergeCodec(
            D, n_prefix=args.n_prefix, r=1, grid=(h, w),
            hidden=256, use_decoder=False, match_mode="cosine").to(device)
        copy_codec.eval()

        for frac in args.merge_fracs:
            r = _r_from_frac(n_patch, frac)
            print(f"  -- merge_frac={frac}  r={r}  "
                  f"(coded patch={n_patch - r})")
            row = eval_arm(
                f"copy_frac{frac}", copy_codec, feats, pads, r, layer_idx,
                sample_meta, backbone, head, device, args.n_prefix,
                args.norm_mode, args.batch_size)
            row["merge_frac"] = float(frac)
            row["delta_rmse"] = float(row["rmse"] - anchor)
            print(f"    ΔRMSE={row['delta_rmse']:+.4f}")
            layer_row["arms"].append(row)

            if args.skip_trained:
                continue
            ckpt = _ckpt_for(layer, frac)
            if ckpt is None:
                print(f"  trained skipped (no ckpt for {layer} frac={frac})")
                continue
            print(f"  loading {ckpt.name}")
            trained, meta = load_merge_codec(str(ckpt), device=device)
            trained.use_decoder = True
            trained.match_mode = "cosine"
            trained.eval()
            row_t = eval_arm(
                f"trained_frac{frac}", trained, feats, pads, r, layer_idx,
                sample_meta, backbone, head, device, args.n_prefix,
                args.norm_mode, args.batch_size)
            row_t["merge_frac"] = float(frac)
            row_t["delta_rmse"] = float(row_t["rmse"] - anchor)
            row_t["ckpt"] = ckpt.name
            row_t["ckpt_r"] = int(meta.get("r", -1))
            print(f"    ΔRMSE={row_t['delta_rmse']:+.4f}")
            layer_row["arms"].append(row_t)
            del trained
            torch.cuda.empty_cache()

        del copy_codec, feats
        torch.cuda.empty_cache()
        all_results["layers"][layer] = layer_row

    out_dir = MERGE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "nyu_depth_" + "_".join(args.layers) + "_f" + \
        "-".join(str(x) for x in args.merge_fracs)
    if args.max_images:
        tag += f"_n{args.max_images}"
    out_path = out_dir / f"{tag}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {out_path}")
    print("\nSummary  (RMSE, lower better; Δ vs unquantized NYU replay)")
    for layer, row in all_results["layers"].items():
        print(f"  {layer}  anchor={row['anchor_rmse']:.4f}")
        for arm in row["arms"]:
            print(f"    {arm['name']:<22} {arm['rmse']:.4f}  "
                  f"Δ={arm['delta_rmse']:+.4f}")
    return all_results


if __name__ == "__main__":
    main()
