#!/usr/bin/env python
"""Standalone VOC mIoU + NYU RMSE for Haar-joint + ORFC.

Does not go through residual-PQ.  The codec is HaarORFCWrapper:

    Haar-only : Haar encode → decode (no PQ)
    full      : Haar encode → ORFC → Haar decode

Unmatched patches (T % 4) pass through Haar as identity tokens, like CLS.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -u eval_haar_orfc_tasks.py \\
        --orfc_ckpt checkpoints/dinov2_vitl14/<stem>.pt
    python -u launch_haar_orfc_jointopq_tasks.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ORFC = HERE.parent / "orfc"
ORFCV2 = HERE.parent / "orfcv2"
PROJECT = HERE.parents[1]
for p in (str(ORFC), str(HERE), str(ORFCV2)):
    if p not in sys.path:
        sys.path.insert(0, p)

from backbone.wrapper import Dinov2Wrapper  # noqa: E402
from run_multilayer_calibrator import (  # noqa: E402
    evaluate_accuracy, load_gt, preload_features,
)
from run_soft_pq import CodecSegmentationEvaluator, evaluate_delta_l_ref  # noqa: E402
from soft_pq import (  # noqa: E402
    FrozenTail, batch_inv_normalize_gpu, batch_normalize_gpu, load_codec,
)

from eval_residual_depth import (  # noqa: E402
    eval_anchor, eval_codec, load_feats, prepare_nyu_samples,
)
from dinov2_depth_pipeline import _load_backbone, _load_depth_head  # noqa: E402
from run_global_residual import HaarORFCWrapper, load_global_detail  # noqa: E402


def resolve_haar_ckpt(orfc_ckpt, haar_ckpt=None):
    if haar_ckpt:
        return Path(haar_ckpt)
    meta = torch.load(orfc_ckpt, map_location="cpu")
    if isinstance(meta, dict) and meta.get("haar_ckpt"):
        p = Path(meta["haar_ckpt"])
        if p.is_file():
            return p
    cand = Path(orfc_ckpt).with_name(Path(orfc_ckpt).stem + "_haar.pt")
    if cand.is_file():
        return cand
    raise FileNotFoundError(
        f"no Haar ckpt for {orfc_ckpt}; pass --haar_ckpt")


def load_wrappers(orfc_ckpt, haar_ckpt, device, modes):
    haar, _ = load_global_detail(str(haar_ckpt), device=device)
    orfc = None
    if "full" in modes:
        orfc = load_codec(str(orfc_ckpt), device=device)
    out = {}
    if "haar_only" in modes:
        out["haar_only"] = HaarORFCWrapper(
            haar, None, freeze_haar=True, freeze_orfc=True)
        out["haar_only"].eval()
    if "full" in modes:
        out["full"] = HaarORFCWrapper(
            haar, orfc, freeze_haar=True, freeze_orfc=True)
        out["full"].eval()
    return out


def eval_seg(codec, args, device, layer_idx, D):
    ev = CodecSegmentationEvaluator(
        codec=codec, norm_mode=args.norm_mode, layer_idx=layer_idx,
        voc_root=args.voc_root, weights_root=args.weights_root,
        device=device, feat_dim=D, model_name=args.backbone)
    seg_feat_dir = Path(args.seg_feat_root) / args.backbone / args.layer
    t0 = time.time()
    out = ev.evaluate(
        seg_feat_dir=str(seg_feat_dir),
        image_list=args.seg_image_list, verbose=True)
    return {
        "miou": float(out["miou"]),
        "acc": float(out["acc"]),
        "t_s": time.time() - t0,
    }


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--orfc_ckpt", required=True)
    p.add_argument("--haar_ckpt", default="")
    p.add_argument("--layer", default="blk05")
    p.add_argument("--backbone", default="dinov2_vitl14")
    p.add_argument("--norm_mode", default="per_image")
    p.add_argument("--modes", default="haar_only,full",
                   help="comma list: haar_only, full")
    p.add_argument("--tasks", default="cls,seg,depth",
                   help="comma list: cls, seg, depth")
    p.add_argument("--feat_root",
                   default=str(PROJECT / "features"))
    p.add_argument("--test_subset", default="test")
    p.add_argument("--gt_path",
                   default=str(PROJECT / "utils"
                               / "imagenet_selected_label500.txt"))
    p.add_argument("--weights_root",
                   default=str(PROJECT / "pretrained"),
                   help="VOC backbone + linear seg head")
    p.add_argument("--voc_root",
                   default=str(PROJECT / "data" / "VOCdevkit" / "VOC2012"))
    p.add_argument("--seg_feat_root",
                   default=str(PROJECT / "features" / "voc2012_100"))
    p.add_argument("--seg_image_list",
                   default=str(PROJECT / "utils" / "voc2012_val_100.txt"))
    p.add_argument("--nyu_feat_root",
                   default=str(PROJECT / "features" / "nyu_depth_80"
                               / "dinov2_vitl14"))
    p.add_argument("--nyu_data_root",
                   default="/data4/workspace/zlt/featcodec/CoFAI/data/NYU")
    p.add_argument("--nyu_split_file",
                   default=str(PROJECT / "utils" / "nyu_test_80.txt"))
    p.add_argument("--nyu_weights_root",
                   default="/data4/workspace/zlt/cache/torch/hub/checkpoints",
                   help="DINOv2 pretrain + NYU depth head (eval_residual_depth)")
    p.add_argument("--result_dir",
                   default=str(HERE / "results" / "haar_orfc_jointopq"
                               / "tasks"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    haar_path = resolve_haar_ckpt(args.orfc_ckpt, args.haar_ckpt or None)
    orfc_path = Path(args.orfc_ckpt)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    layer_idx = int(args.layer[-2:])
    D = 1024

    print(f"\n{'#' * 70}")
    print(f"# Haar-ORFC task eval  layer={args.layer}")
    print(f"# ORFC  {orfc_path}")
    print(f"# Haar  {haar_path}")
    print(f"# modes={modes}  tasks={tasks}")
    print(f"# leftover patches pass through like CLS")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    wrappers = load_wrappers(orfc_path, haar_path, device, modes)
    results = {
        "layer": args.layer,
        "orfc_ckpt": str(orfc_path),
        "haar_ckpt": str(haar_path),
        "norm_mode": args.norm_mode,
        "modes": modes,
        "tasks": tasks,
    }
    for mode in modes:
        results[mode] = {}

    if "cls" in tasks:
        print(f"\n{'=' * 60}")
        print("  [ImageNet Acc & ΔL_ref]")
        print(f"{'=' * 60}")
        test_dir = Path(args.feat_root) / args.test_subset / args.backbone / args.layer
        files = sorted(test_dir.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"no test features in {test_dir}")
        features, basenames = preload_features(files, num_workers=4)
        gt = load_gt(args.gt_path)
        dino = Dinov2Wrapper(
            head_layers=1, model_name=args.backbone,
            weights_root=args.weights_root, device=device)
        tail = FrozenTail(
            list(dino.backbone.blocks[layer_idx + 1:]),
            dino.backbone.norm, device=device)
        for mode in modes:
            t0 = time.time()
            all_xhat = []
            batch_size = 32
            codec = wrappers[mode]
            for start in range(0, len(features), batch_size):
                end = min(start + batch_size, len(features))
                X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
                Y, Mu, Std = batch_normalize_gpu(X, mode=args.norm_mode)
                Y_hat, _ = codec(Y)
                X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
                all_xhat.extend(X_hat.cpu().numpy())
            acc = evaluate_accuracy(all_xhat, basenames, gt, dino, layer_idx, device)
            del all_xhat
            delta_l = evaluate_delta_l_ref(
                features, tail, args.norm_mode, device, codec=codec, batch_size=8)
            row = {
                "acc": float(acc),
                "delta_l": float(delta_l),
                "t_s": time.time() - t0,
                "n": len(basenames),
            }
            results[mode]["cls"] = row
            print(f"  {mode:10s}  Acc={row['acc']:.4f}  ΔL_ref={row['delta_l']:.1f}  "
                  f"({row['t_s']:.1f}s)")
        del dino, tail, features
        torch.cuda.empty_cache()

    if "depth" in tasks:
        print(f"\n{'=' * 60}")
        print("  [NYU Depth RMSE]")
        print(f"{'=' * 60}")
        samples, sample_meta = prepare_nyu_samples(
            args.nyu_data_root, args.nyu_split_file)
        nyu_feats = load_feats(Path(args.nyu_feat_root) / args.layer, samples)
        print(f"  NYU test80: {len(nyu_feats)}  shape={tuple(nyu_feats[0].shape)}")
        ns = SimpleNamespace(
            model="vitl14", weights_root=args.nyu_weights_root, device=device)
        backbone, _ = _load_backbone(ns)
        head = _load_depth_head(ns)
        t0 = time.time()
        anchor = eval_anchor(
            nyu_feats, layer_idx, sample_meta, backbone, head, device)
        print(f"  Anchor RMSE={anchor:.4f}  ({time.time() - t0:.1f}s)")
        results["anchor_rmse"] = float(anchor)
        for mode in modes:
            t0 = time.time()
            rmse = eval_codec(
                wrappers[mode], nyu_feats, layer_idx, sample_meta,
                backbone, head, device, args.norm_mode, base_only=False)
            row = {
                "rmse": float(rmse),
                "delta_vs_anchor": float(rmse - anchor),
                "t_s": time.time() - t0,
            }
            results[mode]["depth"] = row
            print(f"  {mode:10s}  RMSE={row['rmse']:.4f}  "
                  f"Δvs_anchor={row['delta_vs_anchor']:+.4f}  "
                  f"({row['t_s']:.1f}s)")
        del backbone, head, nyu_feats
        torch.cuda.empty_cache()

    if "seg" in tasks:
        print(f"\n{'=' * 60}")
        print("  [VOC2012 mIoU]")
        print(f"{'=' * 60}")
        for mode in modes:
            seg = eval_seg(wrappers[mode], args, device, layer_idx, D)
            results[mode]["seg"] = seg
            print(f"  {mode:10s}  mIoU={seg['miou']:.4f}  "
                  f"aAcc={seg['acc']:.4f}  ({seg['t_s']:.1f}s)")
            torch.cuda.empty_cache()

    if "haar_only" in results and "full" in results:
        ho, fu = results["haar_only"], results["full"]
        if "cls" in ho and "cls" in fu:
            results["delta_acc"] = fu["cls"]["acc"] - ho["cls"]["acc"]
            results["delta_delta_l"] = fu["cls"]["delta_l"] - ho["cls"]["delta_l"]
            print(f"  Δ(Acc)    = {results['delta_acc']:+.4f}")
            print(f"  Δ(ΔL_ref) = {results['delta_delta_l']:+.1f}")
        if "seg" in ho and "seg" in fu:
            results["delta_miou"] = fu["seg"]["miou"] - ho["seg"]["miou"]
            print(f"  Δ(mIoU)   = {results['delta_miou']:+.4f}")
        if "depth" in ho and "depth" in fu:
            results["delta_rmse"] = fu["depth"]["rmse"] - ho["depth"]["rmse"]
            print(f"  Δ(RMSE)   = {results['delta_rmse']:+.4f}")

    out_dir = Path(args.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{orfc_path.stem}_tasks.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
