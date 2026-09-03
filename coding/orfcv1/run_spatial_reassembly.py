#!/usr/bin/env python
"""Stage 1: train convolutional spatial reassembly with ΔL_ref.  No R, no PQ.

Fixed 2× down/up, bilinear-init kernels, softmax reassembly, CLS identity.

Usage (blk05, GPU 2):
    CUDA_VISIBLE_DEVICES=2 python -u run_spatial_reassembly.py --layer blk05 \
        --epochs 100 --lr 3e-4 --save_codec
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
_ORFC_DIR = os.path.normpath(os.path.join(HERE, "..", "orfc"))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if _ORFC_DIR not in sys.path:
    sys.path.insert(0, _ORFC_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from run_multilayer_calibrator import (  # noqa: E402
    set_seed, preload_features, load_gt, evaluate_accuracy,
)
from backbone.wrapper import Dinov2Wrapper  # noqa: E402
from soft_pq import FrozenTail  # noqa: E402
from run_soft_pq import evaluate_delta_l_ref  # noqa: E402
from opq import batch_normalize_gpu, batch_inv_normalize_gpu  # noqa: E402

from spatial_reassembly import (  # noqa: E402
    SpatialReassemblyCodec, train_spatial_reassembly, save_spatial_reassembly,
    load_spatial_reassembly, reshape_map, flatten_map,
)

import warnings  # noqa: E402
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")


class BilinearRoundtripCodec(nn.Module):
    """``F.interpolate`` bilinear down/up at the same scale.  CLS identity."""

    def __init__(self, D, n_prefix=1, scale=2, grid=None,
                 scale_h=None, scale_w=None):
        super().__init__()
        self.D = int(D)
        self.n_prefix = int(n_prefix)
        if scale_h is None and scale_w is None:
            if isinstance(scale, (tuple, list)):
                scale_h, scale_w = int(scale[0]), int(scale[1])
            else:
                scale_h = scale_w = int(scale)
        else:
            scale_h = int(scale if scale_h is None else scale_h)
            scale_w = int(scale if scale_w is None else scale_w)
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale = scale_h if scale_h == scale_w else (scale_h, scale_w)
        self.grid = grid

    def coded_tokens(self, T_full=None):
        if T_full is None:
            return None
        from spatial_reassembly import infer_patch_hw
        n_patch = int(T_full) - self.n_prefix
        H, W = infer_patch_hw(n_patch, self.grid)
        return self.n_prefix + (H // self.scale_h) * (W // self.scale_w)

    def forward(self, Y, **_kwargs):
        p = self.n_prefix
        from spatial_reassembly import infer_patch_hw
        patch = Y[:, p:, :]
        H, W = infer_patch_hw(patch.shape[1], self.grid)
        x = reshape_map(patch, H, W)
        Hm, Wm = H // self.scale_h, W // self.scale_w
        y = F.interpolate(x, size=(Hm, Wm), mode="bilinear", align_corners=False)
        xhat = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        body = flatten_map(xhat)
        if p == 0:
            return body, None
        return torch.cat([Y[:, :p, :], body], dim=1), None


@torch.no_grad()
def _roundtrip(features, codec, norm_mode, device, n_prefix, batch_size):
    all_xhat = []
    mse_sum = 0.0
    energy_sum = 0.0
    n_elem = 0
    codec.eval()
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
        Y_hat, _ = codec(Y)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        p = n_prefix
        diff = Y[:, p:, :] - Y_hat[:, p:, :]
        mse_sum += (diff ** 2).sum().item()
        energy_sum += (Y[:, p:, :] ** 2).sum().item()
        n_elem += diff.numel()
        for i in range(X_hat.shape[0]):
            all_xhat.append(X_hat[i].cpu().numpy())
        del X, Y, Mu, Std, Y_hat, X_hat, diff
    stats = {
        "mse_patch": float(mse_sum / max(n_elem, 1)),
        "rel_energy_err": float(mse_sum / max(energy_sum, 1e-12)),
    }
    return all_xhat, stats


def _eval(name, codec, features, basenames, gt, wrapper, layer_idx, device,
          norm_mode, n_prefix, batch_size, T_full):
    codec = codec.to(device).eval()
    wrapper.backbone.to(device)
    if wrapper.head is not None:
        wrapper.head.to(device)
    torch.cuda.empty_cache()

    t0 = time.time()
    xhat, stats = _roundtrip(
        features, codec, norm_mode, device, n_prefix, batch_size)
    acc = evaluate_accuracy(xhat, basenames, gt, wrapper, layer_idx, device)
    del xhat
    t_acc = time.time() - t0

    tail = FrozenTail(list(wrapper.backbone.blocks[layer_idx + 1:]),
                      wrapper.backbone.norm, device=device)
    t1 = time.time()
    dl = evaluate_delta_l_ref(features, tail, norm_mode, device,
                              codec=codec, batch_size=batch_size)
    t_dl = time.time() - t1
    tail.to("cpu")
    torch.cuda.empty_cache()

    coded = None
    if hasattr(codec, "coded_tokens"):
        coded = codec.coded_tokens(T_full)
    row = {
        "name": name,
        "acc": float(acc),
        "delta_l": float(dl),
        "coded_tokens": coded,
        **stats,
        "t_acc_s": t_acc,
        "t_dl_s": t_dl,
    }
    print(f"  Acc={row['acc']:.4f}  ΔL_ref={row['delta_l']:.1f}  "
          f"MSE_p={row['mse_patch']:.6f}  relE={row['rel_energy_err']:.4f}  "
          f"coded={row['coded_tokens']}/{T_full}")
    return row


def _eval_identity(features, basenames, gt, wrapper, layer_idx, device):
    wrapper.backbone.to(device)
    if wrapper.head is not None:
        wrapper.head.to(device)
    acc = evaluate_accuracy(features, basenames, gt, wrapper, layer_idx, device)
    row = {"name": "identity", "acc": float(acc), "delta_l": 0.0}
    print(f"  Acc={row['acc']:.4f}  ΔL_ref=0  (original features)")
    return row


def _rebuild_train_tail(wrapper, layer_idx, device):
    for i, blk in enumerate(wrapper.backbone.blocks):
        if i <= layer_idx:
            blk.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()
    return FrozenTail(list(wrapper.backbone.blocks[layer_idx + 1:]),
                      wrapper.backbone.norm, device=device)


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--layer", type=str, default="blk05")
    p.add_argument("--scale", type=int, default=2,
                   help="isotropic scale if scale_h/scale_w omitted")
    p.add_argument("--scale_h", type=int, default=None)
    p.add_argument("--scale_w", type=int, default=None)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--Cr", type=int, default=64)
    p.add_argument("--n_prefix", type=int, default=1)
    p.add_argument("--norm_mode", type=str, default="per_image",
                   choices=["per_image", "split_cls_patch"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_images", type=int, default=5000)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--save_codec", action="store_true")
    p.add_argument("--skip_identity", action="store_true")
    p.add_argument("--skip_bilinear", action="store_true")
    p.add_argument("--skip_init", action="store_true")
    p.add_argument("--feat_root", type=str,
                   default=os.path.join(PROJECT_ROOT, "features"))
    p.add_argument("--train_subset", type=str, default="train")
    p.add_argument("--test_subset", type=str, default="test")
    p.add_argument("--backbone", type=str, default="dinov2_vitl14")
    p.add_argument("--gt_path", type=str,
                   default=os.path.join(PROJECT_ROOT, "utils",
                                        "imagenet_selected_label500.txt"))
    p.add_argument("--forward_mode", type=str, default="integer",
                   choices=["integer", "generic", "down_pool", "down_win",
                            "up_kern", "up_win"])
    p.add_argument("--init_eps", type=float, default=0.0,
                   help="K_0=(1-eps)K_bili+eps/k^2; 0 keeps -20 off-support floor")
    p.add_argument("--n_groups", type=int, default=1,
                   help="channel groups G; D must be divisible by G")
    p.add_argument("--logit_span", type=float, default=0.0,
                   help="softmax pre-norm M: l'=(l-mean)/max(1,(max-min)/M); "
                        "0 disables")
    p.add_argument("--ckpt", type=str, default=None,
                   help="warm-start from a spatial_reassembly .pt")
    args = p.parse_args()
    scale_h = args.scale if args.scale_h is None else args.scale_h
    scale_w = args.scale if args.scale_w is None else args.scale_w

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    layer_idx = int(args.layer[-2:])

    print(f"\n{'#' * 70}")
    print(f"# Spatial reassembly stage 1 (no PQ)  {args.layer}  "
          f"scale={scale_h}x{scale_w} k={args.k} Cr={args.Cr} G={args.n_groups}")
    print(f"# mode={args.forward_mode}  init_eps={args.init_eps}  "
          f"G={args.n_groups}  spanM={args.logit_span:g}  "
          f"ep={args.epochs} lr={args.lr}")
    if args.ckpt:
        print(f"# warm-start ckpt={args.ckpt}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    train_dir = Path(args.feat_root) / args.train_subset / args.backbone / args.layer
    test_dir = Path(args.feat_root) / args.test_subset / args.backbone / args.layer
    train_files = sorted(train_dir.glob("*.npy"))
    test_files = sorted(test_dir.glob("*.npy"))
    if not train_files or not test_files:
        raise FileNotFoundError(f"features missing: {train_dir} / {test_dir}")
    features_train, _ = preload_features(train_files, num_workers=4)
    features_test, basenames_test = preload_features(test_files, num_workers=4)
    gt_test = load_gt(args.gt_path)

    T_full, D = features_train[0].shape
    T_patch = T_full - args.n_prefix
    H = W = int(round(T_patch ** 0.5))
    if H % scale_h or W % scale_w:
        raise ValueError(
            f"grid {H}x{W} not divisible by scale={scale_h}x{scale_w}")
    Hm, Wm = H // scale_h, W // scale_w
    Tm = args.n_prefix + Hm * Wm
    print(f"\nData: train={len(features_train)} test={len(features_test)}  "
          f"D={D} T={T_full}  {H}x{W} -> {Hm}x{Wm}  Tm={Tm}")

    if 0 < args.max_train_images < len(features_train):
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(features_train), args.max_train_images,
                         replace=False)
        features_train_sub = [features_train[i] for i in idx]
    else:
        features_train_sub = features_train
    n_val = min(args.n_val, len(features_train))
    rng_val = np.random.RandomState(args.seed + 1)
    val_idx = rng_val.choice(len(features_train), n_val, replace=False)
    val_features = [features_train[i] for i in val_idx]

    print(f"\nLoading DINOv2 ({args.backbone})...")
    wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone,
                            device=device)

    results = {
        "config": vars(args), "D": D, "T_full": T_full, "T_patch": T_patch,
        "Tm": Tm, "Hm": Hm, "Wm": Wm, "scale": args.scale,
        "scale_h": scale_h, "scale_w": scale_w, "n_groups": args.n_groups,
        "logit_span": args.logit_span, "ckpt": args.ckpt,
        "side_bits": 0.0,
    }

    if not args.skip_identity:
        print(f"\n{'=' * 60}\n  [identity] original features\n{'=' * 60}")
        results["identity"] = _eval_identity(
            features_test, basenames_test, gt_test, wrapper, layer_idx, device)

    if args.ckpt:
        merge_init, ckpt_meta = load_spatial_reassembly(
            args.ckpt, device=str(device))
        mismatches = []
        if int(merge_init.n_groups) != int(args.n_groups):
            mismatches.append(
                f"n_groups ckpt={merge_init.n_groups} vs cli={args.n_groups}")
        if int(merge_init.k) != int(args.k):
            mismatches.append(f"k ckpt={merge_init.k} vs cli={args.k}")
        if int(merge_init.scale_h) != int(scale_h) or int(merge_init.scale_w) != int(scale_w):
            mismatches.append(
                f"scale ckpt={merge_init.scale_h}x{merge_init.scale_w} "
                f"vs cli={scale_h}x{scale_w}")
        if int(merge_init.D) != int(D):
            mismatches.append(f"D ckpt={merge_init.D} vs data={D}")
        if mismatches:
            raise ValueError("ckpt mismatch: " + "; ".join(mismatches))
        merge_init.logit_span = float(args.logit_span)
        merge_init.set_forward_mode(args.forward_mode)
        print(f"\n  loaded ckpt  G={merge_init.n_groups}  "
              f"eps={merge_init.init_eps}  spanM={merge_init.logit_span:g}")
    else:
        merge_init = SpatialReassemblyCodec(
            D, n_prefix=args.n_prefix, scale_h=scale_h, scale_w=scale_w,
            k=args.k, Cr=args.Cr, grid=(H, W), init_eps=args.init_eps,
            n_groups=args.n_groups, logit_span=args.logit_span).to(device)
        merge_init.set_forward_mode(args.forward_mode)
        merge_init._init_bilinear()

    if not args.skip_bilinear:
        print(f"\n{'=' * 60}\n  [bilinear] F.interpolate down+up\n{'=' * 60}")
        bili = BilinearRoundtripCodec(
            D, n_prefix=args.n_prefix, scale_h=scale_h, scale_w=scale_w,
            grid=(H, W)).to(device)
        results["bilinear"] = _eval(
            "bilinear", bili, features_test, basenames_test, gt_test,
            wrapper, layer_idx, device, args.norm_mode, args.n_prefix,
            args.batch_size, T_full)
        del bili
        torch.cuda.empty_cache()

    if not args.skip_init:
        init_label = ("warm-start + span" if args.ckpt
                      else f"bilinear-init reassembly (untrained, {args.forward_mode})")
        print(f"\n{'=' * 60}\n  [init] {init_label}\n"
              f"{'=' * 60}")
        results["init"] = _eval(
            "init", merge_init, features_test, basenames_test, gt_test,
            wrapper, layer_idx, device, args.norm_mode, args.n_prefix,
            args.batch_size, T_full)

    if args.epochs > 0:
        print(f"\n{'=' * 60}\n  [train] spatial reassembly ΔL_ref  "
              f"{args.epochs} ep  n={len(features_train_sub)}\n{'=' * 60}")
        tail_train = _rebuild_train_tail(wrapper, layer_idx, device)
        t0 = time.time()
        codec, hist = train_spatial_reassembly(
            features_train=features_train_sub, tail=tail_train, D=D,
            n_prefix=args.n_prefix, scale=args.scale,
            scale_h=scale_h, scale_w=scale_w, k=args.k, Cr=args.Cr,
            grid=(H, W), norm_mode=args.norm_mode, epochs=args.epochs,
            lr=args.lr, batch_size=args.batch_size, device=device,
            seed=args.seed, val_features=val_features, verbose=True,
            grad_clip=args.grad_clip, forward_mode=args.forward_mode,
            init_eps=args.init_eps, n_groups=args.n_groups,
            logit_span=args.logit_span, init_codec=merge_init)
        print(f"  training: {time.time() - t0:.1f}s")
        results["history"] = hist

        print(f"\n{'=' * 60}\n  [trained] spatial reassembly, no PQ\n{'=' * 60}")
        results["trained"] = _eval(
            "trained", codec, features_test, basenames_test, gt_test,
            wrapper, layer_idx, device, args.norm_mode, args.n_prefix,
            args.batch_size, T_full)
    else:
        codec = merge_init
        results["history"] = []
        results["trained"] = None

    out_dir = Path(HERE) / "results" / "spatial_reassembly" / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = "" if args.forward_mode == "integer" else f"_{args.forward_mode}"
    scale_tag = f"s{scale_h}" if scale_h == scale_w else f"s{scale_h}x{scale_w}"
    eps_tag = "" if args.init_eps == 0 else f"_eps{args.init_eps}"
    g_tag = "" if args.n_groups == 1 else f"_G{args.n_groups}"
    span_tag = "" if args.logit_span <= 0 else f"_span{args.logit_span:g}"
    ft_tag = "_ft" if args.ckpt else ""
    tag = (f"{args.layer}_{scale_tag}_k{args.k}{g_tag}{mode_tag}{eps_tag}"
           f"{span_tag}{ft_tag}_lr{args.lr}_ep{args.epochs}_s{args.seed}")
    if args.save_codec and codec is not None:
        save_spatial_reassembly(
            codec, out_dir / f"{tag}.pt",
            meta_extra={"layer": args.layer, "scale": args.scale,
                        "scale_h": scale_h, "scale_w": scale_w,
                        "forward_mode": args.forward_mode,
                        "init_eps": args.init_eps,
                        "n_groups": args.n_groups,
                        "logit_span": args.logit_span,
                        "ckpt": args.ckpt})
    out_path = out_dir / f"{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")
    if results.get("identity"):
        print(f"  identity Acc={results['identity']['acc']:.4f}")
    if results.get("bilinear"):
        print(f"  bilinear Acc={results['bilinear']['acc']:.4f}  "
              f"ΔL={results['bilinear']['delta_l']:.1f}")
    if results.get("init"):
        print(f"  init      Acc={results['init']['acc']:.4f}  "
              f"ΔL={results['init']['delta_l']:.1f}")
    if results.get("trained"):
        print(f"  trained   Acc={results['trained']['acc']:.4f}  "
              f"ΔL={results['trained']['delta_l']:.1f}")
    return results


if __name__ == "__main__":
    main()
