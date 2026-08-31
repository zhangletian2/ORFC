#!/usr/bin/env python
"""VOC2012 val-100 mIoU for similarity-merge copy-mean (no training).

Each slide window is merged independently (same as ORFC quantise-per-slide).
Typical 512 crop → 37×37 patches; ``r`` follows merge_frac (0.5 = r=128/256).

Usage:
    CUDA_VISIBLE_DEVICES=6 python -u eval_similarity_merge_seg.py \
        --layers blk05 blk10 blk15 blk20 --merge_fracs 0.5 0.25
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
ORFC = HERE.parent / "orfc"
PROJECT = HERE.parents[1]
for p in (str(HERE), str(ORFC),
          str(PROJECT / "tools"), str(PROJECT / "backbone" / "dinov2")):
    if p not in sys.path:
        sys.path.insert(0, p)

from opq import batch_normalize_gpu, batch_inv_normalize_gpu  # noqa: E402
from similarity_merge import SimilarityMergeCodec  # noqa: E402
from backbone.wrapper import (  # noqa: E402
    SegmentationEvaluator, load_seg_head, _DINOV2_REGISTRY,
)

import warnings  # noqa: E402
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=UserWarning)


def _r_from_frac(n_patch, frac):
    r = int(round(float(frac) * n_patch))
    return max(1, min(r, n_patch // 2))


@torch.no_grad()
def merge_slide(tokens_np, grid, r, codec, n_prefix, norm_mode, device):
    codec.r = int(r)
    codec.grid = tuple(grid)
    X = torch.from_numpy(tokens_np).float().unsqueeze(0).to(device)
    Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
    Y_hat, _ = codec(Y)
    rec = batch_inv_normalize_gpu(Y_hat, Mu, Std).squeeze(0)
    return rec


def hist_from_pred_gt(pred, gt, n_cls, ignore):
    mask = gt != ignore
    return np.bincount(
        n_cls * gt[mask].astype(int) + pred[mask].astype(int),
        minlength=n_cls ** 2,
    ).reshape(n_cls, n_cls)


def miou_from_hist(hist):
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    return float(np.nanmean(iou)), float(np.diag(hist).sum() / hist.sum())


@torch.no_grad()
def eval_layer(layer_idx, feat_dir, val_list, voc_root, backbone, head,
               helper, codec, merge_frac, n_prefix, norm_mode, device,
               test_pipeline):
    from mmcv.parallel import collate

    hist = np.zeros((helper.NUM_CLASSES, helper.NUM_CLASSES), dtype=np.int64)
    missing = 0
    n_slides = 0
    r_used = None
    grid_used = None

    for name in val_list:
        feat_path = feat_dir / f"{name}.npy"
        if not feat_path.is_file():
            missing += 1
            continue
        img_path = os.path.join(voc_root, "JPEGImages", f"{name}.jpg")
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)[:, :, ::-1]
        data = test_pipeline(dict(img=img_np))
        data = collate([data], samples_per_gpu=1)
        h_img, w_img = data["img"][0].shape[2], data["img"][0].shape[3]
        crops = helper.get_slide_crops(
            h_img, w_img, helper.CROP_SIZE, helper.STRIDE)
        features = np.load(feat_path)
        if features.shape[0] != len(crops):
            raise RuntimeError(
                f"{name}: slides {features.shape[0]} vs crops {len(crops)}")

        recs = []
        for s, (y1, x1, y2, x2) in enumerate(crops):
            tokens = features[s].astype(np.float32)
            if tokens.ndim == 3:
                tokens = tokens.squeeze(0)
            n_patch = tokens.shape[0] - n_prefix
            fh = math.ceil((y2 - y1) / helper.PATCH_SIZE)
            fw = math.ceil((x2 - x1) / helper.PATCH_SIZE)
            if fh * fw != n_patch:
                raise ValueError(
                    f"{name} s{s}: pad grid {fh}x{fw}={fh * fw} "
                    f"!= n_patch={n_patch}")
            if merge_frac is None:
                rec = torch.from_numpy(tokens).float().to(device)
            else:
                r = _r_from_frac(n_patch, merge_frac)
                r_used = r
                grid_used = [fh, fw]
                rec = merge_slide(
                    tokens, (fh, fw), r, codec, n_prefix, norm_mode, device)
            recs.append(rec.unsqueeze(0))
            n_slides += 1

        preds_logits = helper.slide_inference_decode(
            backbone, head, recs, crops, (h_img, w_img))
        gt_path = os.path.join(voc_root, "SegmentationClass", f"{name}.png")
        gt = np.array(Image.open(gt_path))
        ori_h, ori_w = gt.shape[:2]
        if (h_img, w_img) != (ori_h, ori_w):
            preds_logits = F.interpolate(
                preds_logits, size=(ori_h, ori_w),
                mode="bilinear", align_corners=False)
        seg_pred = preds_logits.argmax(dim=1).squeeze(0).cpu().numpy()
        hist += hist_from_pred_gt(
            seg_pred, gt, helper.NUM_CLASSES, helper.IGNORE_INDEX)
        del recs, preds_logits

    miou, acc = miou_from_hist(hist)
    return {
        "miou": miou,
        "acc": acc,
        "missing": missing,
        "n_slides": n_slides,
        "r": r_used,
        "grid": grid_used,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", nargs="+",
                   default=["blk05", "blk10", "blk15", "blk20"])
    p.add_argument("--merge_fracs", type=float, nargs="+", default=[0.5, 0.25])
    p.add_argument("--feat_root", default=str(
        PROJECT / "features" / "voc2012_100" / "dinov2_vitl14"))
    p.add_argument("--voc_root", default=str(
        PROJECT / "data" / "VOCdevkit" / "VOC2012"))
    p.add_argument("--image_list", default=str(
        PROJECT / "utils" / "voc2012_val_100.txt"))
    p.add_argument("--weights_root",
                   default="/data4/workspace/zlt/cache/torch/hub/checkpoints")
    p.add_argument("--norm_mode", default="per_image")
    p.add_argument("--n_prefix", type=int, default=1)
    p.add_argument("--model_name", default="dinov2_vitl14")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'#' * 70}")
    print(f"# VOC merge copy-mean  layers={args.layers}  "
          f"fracs={args.merge_fracs}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  device={device}")
    print(f"{'#' * 70}")

    with open(args.image_list) as f:
        val_list = [ln.strip() for ln in f if ln.strip()]
    print(f"  images: {len(val_list)}  feat_root={args.feat_root}")

    import mmcv
    from mmseg.datasets.pipelines import Compose

    reg = _DINOV2_REGISTRY[args.model_name]
    cfg = mmcv.Config.fromfile(reg["config"])
    cfg.data_root = args.voc_root

    class _LoadImage:
        def __call__(self, results):
            results["filename"] = results["ori_filename"] = None
            img = results["img"]
            results["img_shape"] = img.shape
            results["ori_shape"] = img.shape
            return results

    test_pipeline = Compose([_LoadImage()] + cfg.data.test.pipeline[1:])

    print("  Loading backbone + VOC linear head...")
    from dinov2.models import vision_transformer as vits
    vit_builder = getattr(vits, reg["vit_fn"])
    backbone = vit_builder(**reg["vit_kwargs"])
    backbone.load_state_dict(
        torch.load(os.path.join(args.weights_root, reg["pretrain"]),
                   map_location="cpu"),
        strict=True)
    backbone = backbone.to(device).eval()
    head = load_seg_head(
        os.path.join(args.weights_root, reg["seg_head"]),
        in_channels=reg["embed_dim"], num_classes=21, device=device)

    helper = SegmentationEvaluator(
        codec_list=[], calibrator=None, layer_idx=0,
        voc_root=args.voc_root, weights_root=args.weights_root,
        device=device, feat_dim=reg["embed_dim"],
        model_name=args.model_name)

    codec = SimilarityMergeCodec(
        reg["embed_dim"], n_prefix=args.n_prefix, r=1, grid=37,
        hidden=256, use_decoder=False, match_mode="cosine").to(device)
    codec.eval()

    results = {"config": vars(args), "n_images": len(val_list), "layers": {}}

    for layer in args.layers:
        layer_idx = int(layer[-2:])
        helper.layer_idx = layer_idx
        feat_dir = Path(args.feat_root) / layer
        print(f"\n{'=' * 60}\n  {layer}\n{'=' * 60}")
        layer_row = {"arms": []}

        t0 = time.time()
        anchor = eval_layer(
            layer_idx, feat_dir, val_list, args.voc_root, backbone, head,
            helper, codec, None, args.n_prefix, args.norm_mode, device,
            test_pipeline)
        print(f"  {'anchor':<16} mIoU={anchor['miou']:.4f}  "
              f"aAcc={anchor['acc']:.4f}  ({time.time() - t0:.1f}s)")
        layer_row["anchor_miou"] = anchor["miou"]
        layer_row["anchor_acc"] = anchor["acc"]
        layer_row["arms"].append({"name": "anchor", **anchor, "seconds": time.time() - t0})

        for frac in args.merge_fracs:
            t0 = time.time()
            row = eval_layer(
                layer_idx, feat_dir, val_list, args.voc_root, backbone, head,
                helper, codec, frac, args.n_prefix, args.norm_mode, device,
                test_pipeline)
            row["name"] = f"copy_frac{frac}"
            row["merge_frac"] = float(frac)
            row["delta_miou"] = float(row["miou"] - anchor["miou"])
            row["seconds"] = time.time() - t0
            print(f"  {row['name']:<16} mIoU={row['miou']:.4f}  "
                  f"Δ={row['delta_miou']:+.4f}  r={row['r']}  "
                  f"({row['seconds']:.1f}s)")
            layer_row["arms"].append(row)

        results["layers"][layer] = layer_row

    out_dir = HERE / "results" / "similarity_merge" / "dinov2_vitl14"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "voc100_" + "_".join(args.layers) + "_f" + "-".join(
        str(x) for x in args.merge_fracs)
    out_path = out_dir / f"{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {out_path}")
    print("\nSummary  (mIoU, higher better)")
    for layer, row in results["layers"].items():
        print(f"  {layer}  anchor={row['anchor_miou']:.4f}")
        for arm in row["arms"]:
            if arm["name"] == "anchor":
                continue
            print(f"    {arm['name']:<16} {arm['miou']:.4f}  "
                  f"Δ={arm['delta_miou']:+.4f}")


if __name__ == "__main__":
    main()
