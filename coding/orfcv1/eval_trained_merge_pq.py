#!/usr/bin/env python
"""NYU test80 RMSE + VOC100 mIoU for trained ToMe merge+PQ checkpoints.

Skips ImageNet cls (already in the train json).  Merge ``r`` is
``merge_frac * T_patch`` on each target grid (same as frozen ToMe eval).

Usage:
    CUDA_VISIBLE_DEVICES=4 python -u eval_trained_merge_pq.py --layer blk05
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
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ORFC = HERE.parent / "orfc"
ORFCV2 = HERE.parent / "orfcv2"
PROJECT = HERE.parents[1]
for p in (str(HERE), str(ORFC), str(ORFCV2),
          str(PROJECT / "tools"), str(PROJECT / "backbone" / "dinov2")):
    if p not in sys.path:
        sys.path.insert(0, p)

from similarity_merge import load_merge_pq_codec, merge_side_bits  # noqa: E402
from eval_frozen_orfc_merge import (  # noqa: E402
    eval_feats_rate, eval_seg_codec,
)
from eval_similarity_merge_seg import _r_from_frac  # noqa: E402
from eval_residual_depth import (  # noqa: E402
    load_feats, prepare_nyu_samples, eval_anchor, eval_codec,
)
from dinov2_depth_pipeline import _load_backbone, _load_depth_head  # noqa: E402
from backbone.wrapper import (  # noqa: E402
    SegmentationEvaluator, load_seg_head, _DINOV2_REGISTRY,
)
from run_multilayer_calibrator import set_seed  # noqa: E402

import warnings  # noqa: E402
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=UserWarning)

CKPT_DIR = HERE / "results" / "similarity_merge_pq" / "dinov2_vitl14"


class AdaptiveTrainedMergePQ(nn.Module):
    """Reuse trained R+PQ; reset merge r/grid per image (ToMe frac)."""

    def __init__(self, codec, merge_frac, n_prefix=1):
        super().__init__()
        self.inner = codec
        self.merge = codec.merge
        self.merge_frac = float(merge_frac)
        self.n_prefix = int(n_prefix)
        self.forced_grid = None

    @property
    def pq(self):
        return self.inner.pq

    @property
    def transform(self):
        return self.inner.transform

    def _prepare(self, T):
        n_patch = T - self.n_prefix
        if self.forced_grid is not None:
            h, w = self.forced_grid
            if h * w != n_patch:
                raise ValueError(
                    f"forced_grid {h}x{w}={h * w} != n_patch={n_patch}")
        else:
            h = int(round(math.sqrt(n_patch)))
            if h * h != n_patch:
                raise ValueError(
                    f"n_patch={n_patch} not square; set forced_grid")
            w = h
        r = _r_from_frac(n_patch, self.merge_frac)
        self.merge.r = r
        self.merge.grid = (h, w)
        return r, (h, w), n_patch

    def forward(self, Y, **kwargs):
        self._prepare(Y.shape[1])
        return self.inner(Y, **kwargs)


def discover_ckpts(ckpt_dir, layer, skip_stems=None):
    skip = set(skip_stems or [])
    out = []
    for p in sorted(Path(ckpt_dir).glob(f"{layer}_tome_*.pt")):
        if p.stem in skip:
            continue
        out.append(p)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", required=True)
    p.add_argument("--ckpt_dir", default=str(CKPT_DIR))
    p.add_argument("--merge_frac", type=float, default=0.25)
    p.add_argument("--n_prefix", type=int, default=1)
    p.add_argument("--norm_mode", default="per_image")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_depth", action="store_true")
    p.add_argument("--skip_seg", action="store_true")
    p.add_argument("--nyu_feat_root", default=str(
        PROJECT / "features" / "nyu_depth_80" / "dinov2_vitl14"))
    p.add_argument("--nyu_data_root",
                   default="/data4/workspace/zlt/featcodec/CoFAI/data/NYU")
    p.add_argument("--nyu_split", default=str(PROJECT / "utils" / "nyu_test_80.txt"))
    p.add_argument("--seg_feat_root", default=str(
        PROJECT / "features" / "voc2012_100" / "dinov2_vitl14"))
    p.add_argument("--voc_root", default=str(
        PROJECT / "data" / "VOCdevkit" / "VOC2012"))
    p.add_argument("--seg_list", default=str(
        PROJECT / "utils" / "voc2012_val_100.txt"))
    p.add_argument("--weights_root",
                   default="/data4/workspace/zlt/cache/torch/hub/checkpoints")
    p.add_argument("--backbone", default="dinov2_vitl14")
    p.add_argument("--skip_stem", nargs="*", default=[],
                   help="Checkpoint stems to skip (in-progress runs).")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    layer_idx = int(args.layer[-2:])
    jobs = discover_ckpts(args.ckpt_dir, args.layer, args.skip_stem)
    if not jobs:
        raise FileNotFoundError(
            f"no {args.layer}_tome_*.pt in {args.ckpt_dir}")

    print(f"\n{'#' * 70}")
    print(f"# Trained ToMe merge+PQ  {args.layer}  frac={args.merge_frac}")
    for ckpt in jobs:
        print(f"#   {ckpt.name}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {device}")
    print(f"{'#' * 70}")

    results = {"config": vars(args), "runs": []}
    match_mode = "tome"

    nyu_feats = nyu_meta = nyu_hw = nyu_anchor = None
    d_backbone = d_head = None
    if not args.skip_depth:
        print("\nLoading NYU test80...")
        samples, nyu_meta = prepare_nyu_samples(
            args.nyu_data_root, args.nyu_split)
        nyu_feats = load_feats(Path(args.nyu_feat_root) / args.layer, samples)
        pads0 = nyu_meta[0][1]
        n_patch = nyu_feats[0].shape[0] - args.n_prefix
        nyu_hw = (pads0[0] // 14, pads0[1] // 14)
        print(f"  NYU {len(nyu_feats)}  T={nyu_feats[0].shape}  grid={nyu_hw}")
        ns = SimpleNamespace(model="vitl14", weights_root=args.weights_root,
                             device=device)
        d_backbone, _ = _load_backbone(ns)
        d_head = _load_depth_head(ns)
        t0 = time.time()
        nyu_anchor = eval_anchor(nyu_feats, layer_idx, nyu_meta,
                                 d_backbone, d_head, device)
        print(f"  NYU anchor RMSE={nyu_anchor:.4f}  ({time.time() - t0:.1f}s)")
        results["nyu_anchor_rmse"] = float(nyu_anchor)
        results["nyu_grid"] = list(nyu_hw)

    seg_helper = seg_backbone = seg_head = test_pipeline = val_list = None
    if not args.skip_seg:
        print("\nLoading VOC head...")
        import mmcv
        from mmseg.datasets.pipelines import Compose
        with open(args.seg_list) as f:
            val_list = [ln.strip() for ln in f if ln.strip()]
        reg = _DINOV2_REGISTRY[args.backbone]
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
        from dinov2.models import vision_transformer as vits
        vit_builder = getattr(vits, reg["vit_fn"])
        seg_backbone = vit_builder(**reg["vit_kwargs"])
        seg_backbone.load_state_dict(
            torch.load(os.path.join(args.weights_root, reg["pretrain"]),
                       map_location="cpu"),
            strict=True)
        seg_backbone = seg_backbone.to(device).eval()
        seg_head = load_seg_head(
            os.path.join(args.weights_root, reg["seg_head"]),
            in_channels=reg["embed_dim"], num_classes=21, device=device)
        seg_helper = SegmentationEvaluator(
            codec_list=[], calibrator=None, layer_idx=layer_idx,
            voc_root=args.voc_root, weights_root=args.weights_root,
            device=device, feat_dim=reg["embed_dim"],
            model_name=args.backbone)
        print(f"  VOC images={len(val_list)}")

    for ckpt in jobs:
        print(f"\n{'=' * 60}\n  {ckpt.name}\n{'=' * 60}")
        inner, meta = load_merge_pq_codec(str(ckpt), device=device)
        inner.eval()
        frac = float(meta.get("merge_frac", args.merge_frac))
        wrap = AdaptiveTrainedMergePQ(
            inner, frac, n_prefix=args.n_prefix).to(device)
        wrap.eval()
        row = {
            "ckpt": ckpt.name,
            "K": int(meta["K"]),
            "layer": args.layer,
            "lmbda": float(meta.get("lmbda", 0.0)),
            "merge_frac": frac,
            "match_mode": meta.get("match_mode", match_mode),
        }
        json_path = ckpt.with_suffix(".json")
        if json_path.is_file():
            trained = json.loads(json_path.read_text()).get("trained")
            if trained:
                row["cls_trained"] = trained

        if not args.skip_depth:
            T_nyu, D_nyu = nyu_feats[0].shape
            T_patch_n = T_nyu - args.n_prefix
            r_n = _r_from_frac(T_patch_n, frac)
            Tm_n = T_nyu - r_n
            side_n = merge_side_bits(r_n, T_patch_n, match_mode)
            wrap.forced_grid = tuple(nyu_hw)
            print(f"  [depth] grid={nyu_hw}  r={r_n}  Tm={Tm_n}")
            t0 = time.time()
            rmse_m = eval_codec(
                wrap, nyu_feats, layer_idx, nyu_meta, d_backbone, d_head,
                device, args.norm_mode, base_only=False)
            rate_m = eval_feats_rate(
                wrap, nyu_feats, args.norm_mode, device, args.batch_size,
                Tm=Tm_n, T_full=T_nyu, D=D_nyu, side_bits=side_n)
            print(f"    RMSE={rmse_m:.4f}  Δ={rmse_m - nyu_anchor:+.4f}  "
                  f"BPFP={rate_m['bpfp']:.4f}  BPFP+side="
                  f"{rate_m['bpfp_with_match_side']:.4f}  "
                  f"({time.time() - t0:.1f}s)")
            row["depth_merge_rmse"] = float(rmse_m)
            row["depth_merge_delta"] = float(rmse_m - nyu_anchor)
            row["depth_merge_rate"] = rate_m

        if not args.skip_seg:
            feat_dir = Path(args.seg_feat_root) / args.layer
            print("  [seg]")
            t0 = time.time()
            s_m = eval_seg_codec(
                wrap, layer_idx, feat_dir, val_list, args.voc_root,
                seg_backbone, seg_head, seg_helper, args.n_prefix,
                args.norm_mode, device, test_pipeline, set_grid=True)
            print(f"    mIoU={s_m['miou']:.4f}  r={s_m['r']}  "
                  f"BPFP={s_m['bpfp']:.4f}  BPFP+side="
                  f"{s_m['bpfp_with_match_side']:.4f}  "
                  f"({time.time() - t0:.1f}s)")
            row["seg_merge"] = s_m

        del wrap, inner
        torch.cuda.empty_cache()
        results["runs"].append(row)

    out_dir = Path(args.ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_tasks_{args.layer}_tome.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")
    return results


if __name__ == "__main__":
    main()
