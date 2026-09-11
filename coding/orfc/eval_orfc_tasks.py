#!/usr/bin/env python
"""Standalone ImageNet Acc + VOC mIoU + NYU RMSE for a FeatureCodec checkpoint.

Usage:
    CUDA_VISIBLE_DEVICES=4 python -u eval_orfc_tasks.py \\
        --ckpt checkpoints/dinov2_vitl14/blk05_K2_emb32_bt1024_ws_tau0.5_lr0.0003_ep300_n5000_s42.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
ORFCV2 = HERE.parent / "orfcv2"
PROJECT = HERE.parents[1]
for p in (str(HERE), str(ORFCV2)):
    if p not in sys.path:
        sys.path.insert(0, p)

from backbone.wrapper import Dinov2Wrapper  # noqa: E402
from run_multilayer_calibrator import (  # noqa: E402
    evaluate_accuracy, load_gt, preload_features,
)
from run_soft_pq import CodecSegmentationEvaluator, evaluate_delta_l_ref  # noqa: E402
from soft_pq import FrozenTail, load_codec, soft_pq_encode_decode  # noqa: E402

from eval_residual_depth import (  # noqa: E402
    eval_anchor, eval_codec, load_feats, prepare_nyu_samples,
)
from dinov2_depth_pipeline import _load_backbone, _load_depth_head  # noqa: E402


def eval_cls(codec, args, device, layer_idx):
    test_dir = Path(args.feat_root) / args.test_subset / args.backbone / args.layer
    files = sorted(test_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no test features in {test_dir}")
    features, basenames = preload_features(files, num_workers=4)
    gt = load_gt(args.gt_path)
    wrapper = Dinov2Wrapper(
        head_layers=1, model_name=args.backbone,
        weights_root=args.weights_root, device=device)
    t0 = time.time()
    xhat = soft_pq_encode_decode(features, codec, args.norm_mode, device)
    acc = evaluate_accuracy(xhat, basenames, gt, wrapper, layer_idx, device)
    del xhat
    tail = FrozenTail(
        list(wrapper.backbone.blocks[layer_idx + 1:]),
        wrapper.backbone.norm, device=device)
    delta_l = evaluate_delta_l_ref(
        features, tail, args.norm_mode, device, codec=codec, batch_size=8)
    t_s = time.time() - t0
    print(f"  cls  Acc={acc:.4f}  ΔL_ref={delta_l:.1f}  ({t_s:.1f}s)")
    del wrapper, tail, features
    torch.cuda.empty_cache()
    return {"acc": float(acc), "delta_l": float(delta_l), "t_s": t_s, "n": len(basenames)}


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
    row = {
        "miou": float(out["miou"]),
        "acc": float(out["acc"]),
        "t_s": time.time() - t0,
    }
    print(f"  seg  mIoU={row['miou']:.4f}  aAcc={row['acc']:.4f}  ({row['t_s']:.1f}s)")
    torch.cuda.empty_cache()
    return row


def eval_depth(codec, args, device, layer_idx):
    samples, sample_meta = prepare_nyu_samples(
        args.nyu_data_root, args.nyu_split_file)
    feats = load_feats(Path(args.nyu_feat_root) / args.layer, samples)
    print(f"  NYU test80: {len(feats)}  shape={tuple(feats[0].shape)}")
    ns = SimpleNamespace(
        model="vitl14", weights_root=args.nyu_weights_root, device=device)
    backbone, _ = _load_backbone(ns)
    head = _load_depth_head(ns)
    t0 = time.time()
    anchor = eval_anchor(feats, layer_idx, sample_meta, backbone, head, device)
    print(f"  Anchor RMSE={anchor:.4f}  ({time.time() - t0:.1f}s)")
    t0 = time.time()
    rmse = eval_codec(
        codec, feats, layer_idx, sample_meta, backbone, head,
        device, args.norm_mode, base_only=False)
    row = {
        "anchor_rmse": float(anchor),
        "rmse": float(rmse),
        "delta_vs_anchor": float(rmse - anchor),
        "t_s": time.time() - t0,
    }
    print(f"  depth RMSE={row['rmse']:.4f}  Δvs_anchor={row['delta_vs_anchor']:+.4f}  "
          f"({row['t_s']:.1f}s)")
    del backbone, head, feats
    torch.cuda.empty_cache()
    return row


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--layer", default="blk05")
    p.add_argument("--backbone", default="dinov2_vitl14")
    p.add_argument("--norm_mode", default="per_image")
    p.add_argument("--tasks", default="cls,seg,depth")
    p.add_argument("--feat_root", default=str(PROJECT / "features"))
    p.add_argument("--test_subset", default="test")
    p.add_argument("--gt_path",
                   default=str(PROJECT / "utils" / "imagenet_selected_label500.txt"))
    p.add_argument("--weights_root", default=str(PROJECT / "pretrained"))
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
                   default="/data4/workspace/zlt/cache/torch/hub/checkpoints")
    p.add_argument("--result_dir", default="")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.ckpt)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    layer_idx = int(args.layer[-2:])
    print(f"\n{'#' * 70}")
    print(f"# ORFC task eval  layer={args.layer}  {ckpt.name}")
    print(f"# tasks={tasks}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")
    codec = load_codec(str(ckpt), device=device)
    codec.eval()
    meta = torch.load(str(ckpt), map_location="cpu")
    D = int(meta.get("D", 1024))
    K = int(meta.get("K", -1))
    print(f"  K={K}  G={meta.get('G')}  d={meta.get('d')}  D={D}")

    results = {
        "layer": args.layer,
        "ckpt": str(ckpt),
        "K": K,
        "G": meta.get("G"),
        "d": meta.get("d"),
        "D": D,
        "norm_mode": args.norm_mode,
        "tasks": tasks,
    }
    if "cls" in tasks:
        print(f"\n{'=' * 60}\n  [ImageNet Acc]\n{'=' * 60}")
        results["cls"] = eval_cls(codec, args, device, layer_idx)
    if "depth" in tasks:
        print(f"\n{'=' * 60}\n  [NYU Depth RMSE]\n{'=' * 60}")
        results["depth"] = eval_depth(codec, args, device, layer_idx)
    if "seg" in tasks:
        print(f"\n{'=' * 60}\n  [VOC2012 mIoU]\n{'=' * 60}")
        results["seg"] = eval_seg(codec, args, device, layer_idx, D)

    out_dir = Path(args.result_dir) if args.result_dir else (
        HERE / "results" / "soft_pq" / args.backbone)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ckpt.stem}_tasks.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")
    if "cls" in results:
        print(f"  Acc  = {results['cls']['acc']:.4f}")
    if "seg" in results:
        print(f"  mIoU = {results['seg']['miou']:.4f}")
    if "depth" in results:
        print(f"  RMSE = {results['depth']['rmse']:.4f}  "
              f"(anchor {results['depth']['anchor_rmse']:.4f})")


if __name__ == "__main__":
    main()
