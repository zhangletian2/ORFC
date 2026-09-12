#!/usr/bin/env python
"""Eval-only pass over trained stage-2 ckpts, reporting prefix vs patch MSE.

Reuses run_bilinear_orfc_joint's own argument parser and loaders so the
feature split, normalization and tail construction match training exactly.
No training, no checkpoint writes.
"""
import json
import sys
from pathlib import Path

import torch

import run_bilinear_orfc_joint as J
from bilinear_residual import (
    load_residual_codec, load_spatial_weights, freeze_module,
    BilinearSpatialCodec,
)
from run_bilinear_residual import eval_cascade
from soft_pq import load_codec


def build_argv(layer, K):
    return [
        "run_bilinear_orfc_joint.py",
        "--layer", layer, "--K", str(K),
        "--embedding_dim", "32", "--bottleneck_dim", "1024",
        "--backbone", "dinov3_vitl16", "--n_prefix", "5",
        "--norm_mode", "split_per_reg_cls_patch",
        "--residual_ablation", "main", "--cls_mode", "conv2",
        "--epochs", "100", "--lr", "3e-4", "--spatial_lr_scale", "0.1",
        "--lmbda", "0.5", "--tau_start", "2.0", "--tau_end", "2.0",
        "--tau_schedule", "constant", "--batch_size", "32",
        "--max_train_images", "5000", "--n_val", "200", "--seed", "42",
        "--skip_baselines", "--skip_test_acc",
    ]


def main():
    layer, K = sys.argv[1], int(sys.argv[2])
    sys.argv = build_argv(layer, K)
    args = J.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    J.set_seed(args.seed)
    layer_idx = int(args.layer[-2:])

    features_train, features_test, _bn = J.load_features(args)
    train_feat, val_feat, _ti, _vi = J.make_split(features_train, args)
    D = int(train_feat[0].shape[1])

    # stage-2 spatial+residual live in the spatial ckpt written after training
    spat_ckpt = J.spatial_out_path(args)
    orfc_ckpt = J.orfc_ckpt_path(args)
    for p in (spat_ckpt, orfc_ckpt):
        if not Path(p).is_file():
            raise FileNotFoundError(p)

    residual, meta = load_residual_codec(str(spat_ckpt), device=device)
    spatial = BilinearSpatialCodec(
        D, n_prefix=args.n_prefix, scale=2,
        down=meta.get("spatial_down", "conv2"),
        up=meta.get("spatial_up", "conv2"),
        cls_mode=meta.get("cls_mode", "conv2")).to(device)
    load_spatial_weights(spatial, meta)
    codec = load_codec(str(orfc_ckpt), device=device)
    for m in (spatial, residual, codec):
        freeze_module(m)

    wrapper = J.build_wrapper(args, layer_idx, device) if hasattr(
        J, "build_wrapper") else None
    if wrapper is None:
        import timm
        from types import SimpleNamespace
        ck = ("/data4/workspace/zlt/cache/torch/hub/checkpoints/"
              "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
        m = timm.create_model("vit_large_patch16_dinov3", pretrained=False,
                              img_size=224, dynamic_img_size=True)
        sd = torch.load(ck, map_location="cpu", weights_only=True)
        sd.pop("mask_token", None)
        m.load_state_dict(sd, strict=False)
        m.eval().to(device)
        wrapper = SimpleNamespace(backbone=m, head=None, _is_dinov3=True,
                                  _token_hw=(14, 14))
    tail = J._full_tail(wrapper, layer_idx, device)

    out = {"layer": layer, "K": K}
    for name, feats in (("val", val_feat), ("test", features_test)):
        out[name] = eval_cascade(
            f"s2 {name}", feats, spatial, codec, residual, tail,
            args.norm_mode, device, args.batch_size,
            ablation=J.joint_ablation(args), quantize=False,
            n_prefix=args.n_prefix)
    print("JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
