#!/usr/bin/env python
"""Frozen full-token ORFC + copy-mean merge.  Eval only, no training.

Merge first (keep 192/256 => r/T_patch = 0.25), then the loaded ORFC
R+PQ+prior on the short sequence, then copy-mean expand.

Tasks on blk05: ImageNet cls / NYU depth / VOC100 seg.

Usage:
    CUDA_VISIBLE_DEVICES=6 python -u eval_frozen_orfc_merge.py \
        --layer blk05 --merge_frac 0.25 --skip_seg
    CUDA_VISIBLE_DEVICES=6 python -u eval_frozen_orfc_merge.py \
        --layer blk05 --Ks 4 8 16 --merge_frac 0.25
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

from opq import batch_normalize_gpu, batch_inv_normalize_gpu  # noqa: E402
from soft_pq import load_codec as load_orfc_codec  # noqa: E402
from similarity_merge import (  # noqa: E402
    SimilarityMergeCodec, SimilarityMergePQCodec, merge_side_bits,
)
from run_similarity_merge_pq import _eval_acc_dl_rate, _bpfp  # noqa: E402
from run_soft_pq import evaluate_rate, _codec_labels, _histogram_pmf  # noqa: E402
from run_multilayer_calibrator import (  # noqa: E402
    set_seed, preload_features, load_gt, evaluate_accuracy,
)
from backbone.wrapper import Dinov2Wrapper  # noqa: E402
from eval_residual_depth import (  # noqa: E402
    load_feats, prepare_nyu_samples, eval_anchor, eval_codec,
)
from dinov2_depth_pipeline import (  # noqa: E402
    _load_backbone, _load_depth_head,
)
from eval_similarity_merge_seg import (  # noqa: E402
    hist_from_pred_gt, miou_from_hist, _r_from_frac,
)
from backbone.wrapper import (  # noqa: E402
    SegmentationEvaluator, load_seg_head, _DINOV2_REGISTRY,
)

import warnings  # noqa: E402
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=UserWarning)

CKPT_DIR = ORFC / "checkpoints" / "dinov2_vitl14"

# ShareR / CSV e32 grid, with the *working* low-rate K4 (not collapsed runs):
#   blk05 K4: λ=0, lr=5e-4, ep=300  (CoFAI; λ=0.5 ep=100 is ~10pt worse on VOC)
#   blk10/15 K4: λ=0.5 ep=100 s42
#   blk20 K4: λ=0, lr=5e-4, ep=300  (CSV Acc 82.6%; lr=3e-4 ep=300 collapsed)
SHARER_CKPTS = [
    ("blk05", 4, "blk05_K4_emb32_bt1024_ws_lmbda0.0_tau0.5_lr0.0005_ep300_n5000_s42.pt"),
    ("blk05", 8, "blk05_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"),
    ("blk05", 16, "blk05_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"),
    ("blk05", 64, "blk05_K64_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0005_ep100_n5000_s42.pt"),
    ("blk05", 256, "blk05_K256_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"),
    ("blk10", 4, "blk10_K4_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"),
    ("blk10", 8, "blk10_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"),
    ("blk10", 16, "blk10_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"),
    ("blk10", 64, "blk10_K64_emb32_bt1024_ws_tau0.5_lr0.0003_ep100_n5000_s42.pt"),
    ("blk10", 256, "blk10_K256_emb32_bt1024_ws_tau0.5_lr0.0005_ep100_n5000_s42.pt"),
    ("blk10", 256, "blk10_K256_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"),
    ("blk15", 4, "blk15_K4_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"),
    ("blk15", 8, "blk15_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"),
    ("blk15", 16, "blk15_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"),
    ("blk15", 64, "blk15_K64_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0005_ep100_n5000_s42.pt"),
    ("blk15", 256, "blk15_K256_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"),
    ("blk20", 4, "blk20_K4_emb32_bt1024_ws_tau0.5_lr0.0005_ep300_n5000_s42.pt"),
    ("blk20", 8, "blk20_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"),
    ("blk20", 16, "blk20_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0005_ep100_n5000_s42.pt"),
    ("blk20", 32, "blk20_K32_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"),
    ("blk20", 64, "blk20_K64_emb32_bt1024_ws_tau0.5_lr0.0003_ep100_n5000_s42.pt"),
    ("blk20", 256, "blk20_K256_emb32_bt1024_ws_tau0.5_lr0.0005_ep100_n5000_s42.pt"),
]


def layer_ckpt_jobs(layer, ckpt_dir, ks=None):
    rows = [(K, name) for ly, K, name in SHARER_CKPTS if ly == layer]
    if ks:
        want = set(ks)
        rows = [(K, name) for K, name in rows if K in want]
    if not rows:
        raise FileNotFoundError(f"no shareR ckpts for {layer} ks={ks}")
    out = []
    for K, name in rows:
        p = Path(ckpt_dir) / name
        out.append((K, p))
    return out


def eval_feats_rate(codec, feats, norm_mode, device, batch_size,
                    Tm, T_full, D, side_bits):
    ref = feats[:min(len(feats), 1000)]
    labels = _codec_labels(ref, codec, norm_mode, device, batch_size)
    pmf = _histogram_pmf(labels, codec.pq.G, codec.pq.K)
    rate = evaluate_rate(
        feats, codec, norm_mode, device, batch_size=batch_size, train_pmf=pmf)
    rbpt = rate.get("rans_train_bpt") or rate.get("rans_bpt")
    return {
        "rate": rate,
        "rans_bpt": None if rbpt is None else float(rbpt),
        "bpfp": _bpfp(rbpt, Tm, T_full, D, side_bits=0.0),
        "bpfp_with_match_side": _bpfp(rbpt, Tm, T_full, D, side_bits=side_bits),
        "side_bits": float(side_bits),
        "coded_tokens": int(Tm),
    }


class AdaptiveMergeORFC(nn.Module):
    """Copy-mean merge then frozen ORFC, T-agnostic."""

    def __init__(self, orfc, merge_frac, n_prefix=1, grid=None,
                 match_mode="cosine"):
        super().__init__()
        self.orfc = orfc.eval()
        for p in self.orfc.parameters():
            p.requires_grad_(False)
        D = int(orfc.pq.D)
        self.merge = SimilarityMergeCodec(
            D, n_prefix=n_prefix, r=1, grid=grid, hidden=256,
            use_decoder=False, match_mode=match_mode)
        self.wrap = SimilarityMergePQCodec(
            self.merge, orfc.pq, orfc.transform)
        self.merge_frac = float(merge_frac)
        self.n_prefix = int(n_prefix)
        if grid is None:
            self.forced_grid = None
        elif isinstance(grid, int):
            self.forced_grid = (int(grid), int(grid))
        else:
            self.forced_grid = tuple(grid)

    @property
    def pq(self):
        return self.orfc.pq

    @property
    def transform(self):
        return self.orfc.transform

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
        return self.wrap(Y, **kwargs)


def _orfc_ckpt(layer, K):
    jobs = layer_ckpt_jobs(layer, CKPT_DIR, ks=[K])
    return jobs[0][1]


@torch.no_grad()
def reconstruct_np(tokens_np, codec, norm_mode, n_prefix, device):
    X = torch.from_numpy(np.asarray(tokens_np)).float().unsqueeze(0).to(device)
    if X.ndim == 4:
        X = X.squeeze(1)
    Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
    Y_hat = codec(Y)[0]
    return batch_inv_normalize_gpu(Y_hat, Mu, Std).squeeze(0)


def eval_seg_codec(codec, layer_idx, feat_dir, val_list, voc_root,
                   backbone, head, helper, n_prefix, norm_mode, device,
                   test_pipeline, set_grid=True):
    from mmcv.parallel import collate
    from PIL import Image

    hist = np.zeros((helper.NUM_CLASSES, helper.NUM_CLASSES), dtype=np.int64)
    missing = 0
    n_slides = 0
    r_last = None
    bpfp_list = []
    bpfp_side_list = []
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
        recs = []
        for s, (y1, x1, y2, x2) in enumerate(crops):
            tokens = features[s].astype(np.float32)
            if tokens.ndim == 3:
                tokens = tokens.squeeze(0)
            fh = math.ceil((y2 - y1) / helper.PATCH_SIZE)
            fw = math.ceil((x2 - x1) / helper.PATCH_SIZE)
            if set_grid and hasattr(codec, "forced_grid"):
                codec.forced_grid = (fh, fw)
                r_last, _, _ = codec._prepare(tokens.shape[0])
            recs.append(reconstruct_np(
                tokens, codec, norm_mode, n_prefix, device).unsqueeze(0))
            T, D = int(tokens.shape[0]), int(tokens.shape[-1])
            if set_grid and hasattr(codec, "merge"):
                r = int(codec.merge.r)
                Tm = T - r
                mode = getattr(codec.merge, "match_mode", "cosine")
                side = merge_side_bits(r, max(T - n_prefix, 1), mode)
            else:
                Tm, side = T, 0.0
            rt = eval_feats_rate(
                codec, [tokens], norm_mode, device, batch_size=1,
                Tm=Tm, T_full=T, D=D, side_bits=side)
            if rt["bpfp"] is not None:
                bpfp_list.append(rt["bpfp"])
                bpfp_side_list.append(rt["bpfp_with_match_side"])
            n_slides += 1
        preds_logits = helper.slide_inference_decode(
            backbone, head, recs, crops, (h_img, w_img))
        gt = np.array(Image.open(
            os.path.join(voc_root, "SegmentationClass", f"{name}.png")))
        ori_h, ori_w = gt.shape[:2]
        if (h_img, w_img) != (ori_h, ori_w):
            import torch.nn.functional as F
            preds_logits = F.interpolate(
                preds_logits, size=(ori_h, ori_w),
                mode="bilinear", align_corners=False)
        pred = preds_logits.argmax(dim=1).squeeze(0).cpu().numpy()
        hist += hist_from_pred_gt(
            pred, gt, helper.NUM_CLASSES, helper.IGNORE_INDEX)
        del recs, preds_logits
    miou, acc = miou_from_hist(hist)
    return {"miou": miou, "acc": acc, "missing": missing,
            "n_slides": n_slides, "r": r_last,
            "bpfp": float(np.mean(bpfp_list)) if bpfp_list else None,
            "bpfp_with_match_side": (
                float(np.mean(bpfp_side_list)) if bpfp_side_list else None),
            }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", default="blk05")
    p.add_argument("--Ks", type=int, nargs="*", default=None,
                   help="Subset of K values. Default: all shareR e32 ckpts.")
    p.add_argument("--merge_frac", type=float, default=0.25)
    p.add_argument("--n_prefix", type=int, default=1)
    p.add_argument("--norm_mode", default="per_image")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt_dir", default=str(CKPT_DIR))
    p.add_argument("--match_mode", default="cosine", choices=("cosine", "tome"))
    p.add_argument("--eval_orfc", action="store_true",
                   help="Also eval full-token ORFC (on by default for cosine).")
    p.add_argument("--skip_orfc", action="store_true",
                   help="Skip full-token ORFC even for cosine (merge only).")
    p.add_argument("--skip_cls", action="store_true")
    p.add_argument("--skip_depth", action="store_true")
    p.add_argument("--skip_seg", action="store_true")
    p.add_argument("--feat_root", default=os.path.join(PROJECT, "features"))
    p.add_argument("--gt_path", default=os.path.join(
        PROJECT, "utils", "imagenet_selected_label500.txt"))
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
    args = p.parse_args()
    eval_orfc = ((args.eval_orfc or args.match_mode == "cosine")
                 and not args.skip_orfc)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    layer_idx = int(args.layer[-2:])
    jobs = layer_ckpt_jobs(args.layer, args.ckpt_dir, ks=args.Ks)
    print(f"\n{'#' * 70}")
    print(f"# Frozen ORFC + copy-mean merge  {args.layer}  "
          f"frac={args.merge_frac}  match={args.match_mode}  "
          f"eval_orfc={eval_orfc}  n_ckpt={len(jobs)}")
    for K, ckpt in jobs:
        print(f"#   K={K:<3d}  {ckpt.name}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {device}")
    print(f"{'#' * 70}")

    results = {"config": vars(args), "runs": []}

    # -------- classification data (shared) --------
    features_train = features_test = basenames_test = gt_test = None
    wrapper = None
    T_full = D = None
    if not args.skip_cls:
        print("\nLoading ImageNet features...")
        train_dir = Path(args.feat_root) / "train" / args.backbone / args.layer
        test_dir = Path(args.feat_root) / "test" / args.backbone / args.layer
        train_files = sorted(train_dir.glob("*.npy"))
        test_files = sorted(test_dir.glob("*.npy"))
        features_train, _ = preload_features(train_files, num_workers=4)
        features_test, basenames_test = preload_features(test_files, num_workers=4)
        gt_test = load_gt(args.gt_path)
        T_full, D = features_train[0].shape
        print(f"  cls train={len(features_train)} test={len(features_test)}  "
              f"T={T_full} D={D}")
        wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone,
                                device=device)
        if 0 < 1000 < len(features_train):
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(len(features_train), 1000, replace=False)
            train_rate = [features_train[i] for i in idx]
        else:
            train_rate = features_train

    # -------- NYU --------
    nyu_feats = nyu_meta = None
    d_backbone = d_head = None
    if not args.skip_depth:
        print("\nLoading NYU test80...")
        samples, nyu_meta = prepare_nyu_samples(args.nyu_data_root, args.nyu_split)
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
    else:
        nyu_hw = None
        nyu_anchor = None

    # -------- VOC --------
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

    for K, ckpt in jobs:
        if not ckpt.is_file():
            print(f"\n[skip] missing {ckpt}")
            continue
        print(f"\n{'=' * 60}\n  K={K}  {ckpt.name}\n{'=' * 60}")
        orfc = load_orfc_codec(str(ckpt), device=device)
        orfc.eval()
        row = {"ckpt": ckpt.name, "K": K, "layer": args.layer}

        if not args.skip_cls:
            T_patch = T_full - args.n_prefix
            r = _r_from_frac(T_patch, args.merge_frac)
            Tm = T_full - r
            side = merge_side_bits(r, T_patch, args.match_mode)
            if eval_orfc:
                print(f"  [cls] ORFC full T={T_full}")
                t0 = time.time()
                m_orfc = _eval_acc_dl_rate(
                    orfc, features_test, basenames_test, gt_test, wrapper,
                    layer_idx, device, args.norm_mode, args.batch_size,
                    train_rate, Tm=T_full, T_full=T_full, D=D, side_bits=0.0)
                print(f"    Acc={m_orfc['acc']:.4f}  ΔL={m_orfc['delta_l']:.1f}  "
                      f"BPFP={m_orfc['bpfp']:.4f}  ({time.time() - t0:.1f}s)")
                row["cls_orfc"] = m_orfc
            wrap = AdaptiveMergeORFC(
                orfc, args.merge_frac, args.n_prefix, grid=(16, 16),
                match_mode=args.match_mode).to(device)
            wrap.eval()
            print(f"  [cls] merge+ORFC  r={r}  Tm={Tm}  match={args.match_mode}")
            t0 = time.time()
            m_m = _eval_acc_dl_rate(
                wrap, features_test, basenames_test, gt_test, wrapper,
                layer_idx, device, args.norm_mode, args.batch_size,
                train_rate, Tm=Tm, T_full=T_full, D=D, side_bits=side)
            print(f"    Acc={m_m['acc']:.4f}  ΔL={m_m['delta_l']:.1f}  "
                  f"BPFP={m_m['bpfp']:.4f}  BPFP+side={m_m['bpfp_with_match_side']:.4f}  "
                  f"({time.time() - t0:.1f}s)")
            row["cls_merge"] = m_m
            del wrap

        if not args.skip_depth:
            T_nyu, D_nyu = nyu_feats[0].shape
            T_patch_n = T_nyu - args.n_prefix
            r_n = _r_from_frac(T_patch_n, args.merge_frac)
            Tm_n = T_nyu - r_n
            side_n = merge_side_bits(r_n, T_patch_n, args.match_mode)
            if eval_orfc:
                print(f"  [depth] ORFC")
                t0 = time.time()
                rmse_o = eval_codec(
                    orfc, nyu_feats, layer_idx, nyu_meta, d_backbone, d_head,
                    device, args.norm_mode, base_only=False)
                rate_o = eval_feats_rate(
                    orfc, nyu_feats, args.norm_mode, device, args.batch_size,
                    Tm=T_nyu, T_full=T_nyu, D=D_nyu, side_bits=0.0)
                print(f"    RMSE={rmse_o:.4f}  Δ={rmse_o - nyu_anchor:+.4f}  "
                      f"BPFP={rate_o['bpfp']:.4f}  ({time.time() - t0:.1f}s)")
                row["depth_orfc_rmse"] = float(rmse_o)
                row["depth_orfc_delta"] = float(rmse_o - nyu_anchor)
                row["depth_orfc_rate"] = rate_o
            wrap = AdaptiveMergeORFC(
                orfc, args.merge_frac, args.n_prefix,
                match_mode=args.match_mode).to(device)
            wrap.forced_grid = tuple(nyu_hw)
            wrap.eval()
            print(f"  [depth] merge+ORFC  grid={nyu_hw}  r={r_n}  Tm={Tm_n}  "
                  f"match={args.match_mode}")
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
            del wrap

        if not args.skip_seg:
            feat_dir = Path(args.seg_feat_root) / args.layer
            if eval_orfc:
                print(f"  [seg] ORFC")
                t0 = time.time()
                s_o = eval_seg_codec(
                    orfc, layer_idx, feat_dir, val_list, args.voc_root,
                    seg_backbone, seg_head, seg_helper, args.n_prefix,
                    args.norm_mode, device, test_pipeline, set_grid=False)
                print(f"    mIoU={s_o['miou']:.4f}  BPFP={s_o['bpfp']:.4f}  "
                      f"({time.time() - t0:.1f}s)")
                row["seg_orfc"] = s_o
            wrap = AdaptiveMergeORFC(
                orfc, args.merge_frac, args.n_prefix,
                match_mode=args.match_mode).to(device)
            wrap.eval()
            print(f"  [seg] merge+ORFC  match={args.match_mode}")
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
            del wrap

        del orfc
        torch.cuda.empty_cache()
        results["runs"].append(row)

    tasks = []
    if not args.skip_cls:
        tasks.append("cls")
    if not args.skip_depth:
        tasks.append("depth")
    if not args.skip_seg:
        tasks.append("seg")
    ks_tag = ("shareR" if not args.Ks
              else "K" + "-".join(str(k) for k in args.Ks))
    stem = ("frozen_tome_merge" if args.match_mode == "tome"
            else "frozen_orfc_merge")
    out_dir = HERE / "results" / "similarity_merge" / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"{stem}_{args.layer}_f{args.merge_frac}_"
        f"{ks_tag}_{'-'.join(tasks) or 'all'}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {out_path}")
    print("\nSummary")
    for row in results["runs"]:
        print(f"  K={row['K']}  {row['ckpt']}")
        if "cls_merge" in row:
            o = row.get("cls_orfc")
            orfc_s = (f"{o['acc']:.4f} / {o['bpfp']:.4f}" if o
                      else "skip")
            print(f"    cls   ORFC {orfc_s}   "
                  f"merge {row['cls_merge']['acc']:.4f} / "
                  f"{row['cls_merge']['bpfp_with_match_side']:.4f}")
        if "depth_merge_rmse" in row:
            bm = row["depth_merge_rate"]["bpfp_with_match_side"]
            if "depth_orfc_rmse" in row:
                bo = row["depth_orfc_rate"]["bpfp"]
                print(f"    depth ORFC {row['depth_orfc_rmse']:.4f} / {bo:.4f}  "
                      f"merge {row['depth_merge_rmse']:.4f} / {bm:.4f}")
            else:
                print(f"    depth merge {row['depth_merge_rmse']:.4f} / {bm:.4f}")
        if "seg_merge" in row:
            sm = row["seg_merge"]
            if "seg_orfc" in row:
                print(f"    seg   ORFC {row['seg_orfc']['miou']:.4f} / "
                      f"{row['seg_orfc']['bpfp']:.4f}  "
                      f"merge {sm['miou']:.4f} / {sm['bpfp_with_match_side']:.4f}")
            else:
                print(f"    seg   merge {sm['miou']:.4f} / "
                      f"{sm['bpfp_with_match_side']:.4f}")


if __name__ == "__main__":
    main()
