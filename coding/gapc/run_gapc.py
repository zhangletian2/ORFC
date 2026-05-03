#!/usr/bin/env python
"""GAPC experiment runner — train sweep, test report.

Strict reproduction of
    Dhaouadi et al., "Enhancing Split ViT Inference Through Sparsity-Driven
    Compression", VTC2025-Spring.

Experiment context (aligned with ``ORFC/coding/orfc/run_soft_pq.py``):
    * Backbone : DINOv2 ViT-L/14 (24 blocks, T=257 tokens, D=1024)
    * Split    : blk05 / blk10 / blk15 / blk20
    * Task     : ImageNet classification (utils/imagenet_selected_label500.txt)
    * Data     : features/{train,test}/dinov2_vitl14/blk{XX}/*.npy
    * Metric   : bpfp (bits per feature), bpt, compression ratio, accuracy

Methodology
-----------
GAPC has zero learned parameters.  Nevertheless, to match the methodology of
``run_soft_pq.py`` (and to guarantee the test set is not used for operating-
point selection), this script:

    1. Sweeps thresholds / keep-ratios on the *train* split — the operating
       curve reported for calibration / tuning.
    2. Independently evaluates every operating point on the *test* split and
       reports accuracy, bpfp, compression ratio there — the final metrics.

Per-point evaluation is streaming and overlapped:
    * Batched GPU inference (``wrapper.forward_from_tokens``) for accuracy.
    * CPU thread-pool zlib compression (GIL-free).
    * No per-image reconstruction ever materialised in host memory.

Usage
-----
    # Full GAPC sweep on one layer (one GPU), reports train & test curves
    python run_gapc.py --layer blk10 \
        --threshold_list 0.005 0.02 0.05 0.075 0.1

    # Baseline Top-k / Random Zero
    python run_gapc.py --layer blk10 --mode topk \
        --keep_ratio_list 0.1 0.25 0.5

    # Smoke test
    python run_gapc.py --layer blk20 --threshold_list 0.5 \
        --max_train_images 50 --max_test_images 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

GAPC_ROOT = os.path.dirname(os.path.abspath(__file__))
ORFC_DIR = os.path.normpath(os.path.join(GAPC_ROOT, "..", "orfc"))
PROJECT_ROOT = os.path.normpath(os.path.join(GAPC_ROOT, "..", "..", ".."))

sys.path.insert(0, ORFC_DIR)
from run_multilayer_calibrator import (               # noqa: E402
    set_seed, preload_features, load_gt,
)
from backbone.wrapper import Dinov2Wrapper, ClipWrapper  # noqa: E402

from gapc_codec import (                               # noqa: E402
    GAPCCodec,
    run_sweep_point,
    zip_bits_reference,
)
from gapc_seg import (                                 # noqa: E402
    evaluate_gapc_seg_point,
    load_seg_backbone_and_head,
    preload_seg_data,
)

warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
import logging  # noqa: E402
from mmcv.utils import get_logger                      # noqa: E402
logger = get_logger('mmcv')
logger.setLevel(logging.WARNING)


# ================================================================
#                    Sweep helpers
# ================================================================

def _build_codec(args, D: int, threshold: float) -> GAPCCodec:
    """Build a GAPC codec at a given threshold (paper Algorithm 1)."""
    return GAPCCodec(D=D, threshold=float(threshold),
                     seed=args.seed, quant_bits=args.quant_bits)


def _point_dict(codec: GAPCCodec, split: str, acc: float, agg: dict,
                wall_s: float) -> dict:
    return {
        "split": split,
        "quant_bits": int(codec.quant_bits),
        "threshold": codec.threshold,
        "acc": float(acc),
        "avg_kept_cols": float(agg["avg_kept_cols"]),
        "avg_kept_frac": float(agg["avg_kept_cols"] / agg["feat_dim"]),
        "avg_bpt": float(agg["avg_bpt"]),
        "avg_bpfp": float(agg["avg_bpfp"]),
        "avg_cr": float(agg["avg_cr"]),
        "pooled_cr": float(agg["pooled_cr"]),
        "avg_bits_ref": float(agg["avg_bits_ref"]),
        "avg_bits_reduced": float(agg["avg_bits_reduced"]),
        "avg_bits_sideinfo": float(agg["avg_bits_sideinfo"]),
        "avg_bits_total": float(agg["avg_bits_total"]),
        "num_images": int(agg["num_images"]),
        "feat_dim": int(agg["feat_dim"]),
        "num_tokens": int(agg["num_tokens"]),
        "wall_s": float(wall_s),
    }


def _print_point(p: dict, header: str = ""):
    prefix = f"{header} " if header else ""
    print(f"  {prefix}θ={p['threshold']:.4g}  "
          f"kept={p['avg_kept_cols']:.1f}/{p['feat_dim']} "
          f"({p['avg_kept_frac']*100:.1f}%)  "
          f"acc={p['acc']:.4f}  "
          f"bpfp={p['avg_bpfp']:.4f}  "
          f"bpt={p['avg_bpt']:.1f}  "
          f"CR={p['avg_cr']:.2f}  "
          f"({p['wall_s']:.1f}s)")


# ================================================================
#                    Main
# ================================================================

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    layer_idx = int(args.layer[-2:])

    # Sanity: at least one task has to be enabled.
    if args.skip_cls and not args.eval_seg:
        raise ValueError(
            "--skip_cls was set but --eval_seg is off: nothing to run. "
            "Enable --eval_seg to run the segmentation-only pipeline.")

    values = list(args.threshold_list)

    print(f"\n{'#' * 70}")
    qb_tag = ("paper-fp32" if args.quant_bits == 32
              else f"GAPC+{args.quant_bits}b (extension)")
    print(f"# GAPC experiment  [{qb_tag}]")
    print(f"# layer={args.layer} (idx={layer_idx})  backbone={args.backbone}")
    tasks = []
    if not args.skip_cls:
        tasks.append("cls")
    if args.eval_seg:
        tasks.append("seg")
    print(f"# tasks={','.join(tasks) or '<none>'}  zip_level={args.zip_level}  "
          f"quant_bits={args.quant_bits}  "
          f"mask_overhead={'on' if not args.no_mask_overhead else 'off'}")
    print(f"# chunk_images={args.chunk_images}  zip_workers={args.zip_workers}")
    print(f"# thresholds: {values}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    # Placeholders populated only when the cls pipeline runs.
    features_train: list = []
    features_test: list = []
    basenames_train: list = []
    basenames_test: list = []
    acc_ref_train = float("nan")
    acc_ref_test = float("nan")
    avg_bits_ref_train = 0.0
    avg_bpfp_ref_train = 0.0
    avg_bits_ref_test = 0.0
    avg_bpfp_ref_test = 0.0
    sweep_train: list = []
    sweep_test: list = []
    wrapper = None
    D = None
    T = None
    is_clip = args.backbone.startswith("clip")

    # ================================================================
    #  (Cls) Classification task — optional (skip with --skip_cls)
    # ================================================================
    if not args.skip_cls:
        # ---- Load train + test features (aligned with run_soft_pq) ----
        train_dir = Path(args.feat_root) / args.train_subset / args.backbone / args.layer
        test_dir = Path(args.feat_root) / "test" / args.backbone / args.layer
        train_files = sorted(train_dir.glob("*.npy"))
        test_files = sorted(test_dir.glob("*.npy"))
        if args.max_train_images > 0 and len(train_files) > args.max_train_images:
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(len(train_files), args.max_train_images,
                             replace=False)
            train_files = [train_files[i] for i in sorted(idx)]
        if args.max_test_images > 0 and len(test_files) > args.max_test_images:
            test_files = test_files[:args.max_test_images]

        print(f"\nData (cls): train={len(train_files)}  test={len(test_files)}")
        print(f"  train_dir = {train_dir}")
        print(f"  test_dir  = {test_dir}")

        features_train, basenames_train = preload_features(
            train_files, num_workers=8)
        features_test, basenames_test = preload_features(
            test_files, num_workers=8)

        # Strict no-overlap guarantee between cls train and test samples.
        train_bn_set = set(basenames_train)
        test_bn_set = set(basenames_test)
        cls_overlap = train_bn_set & test_bn_set
        if cls_overlap:
            sample = sorted(cls_overlap)[:5]
            raise RuntimeError(
                f"[cls] train/test sample overlap is not allowed but found "
                f"{len(cls_overlap)} shared basenames, e.g. {sample}.  "
                f"Check --feat_root {args.feat_root}.")

        gt_train = load_gt(args.train_gt_path)
        gt_test = load_gt(args.gt_path)
        cov_tr = sum(1 for bn in basenames_train if bn in gt_train)
        cov_te = sum(1 for bn in basenames_test if bn in gt_test)
        print(f"  GT coverage: train={cov_tr}/{len(basenames_train)}  "
              f"test={cov_te}/{len(basenames_test)}")
        if cov_tr == 0:
            raise RuntimeError(
                f"No train basename found in --train_gt_path "
                f"({args.train_gt_path}). Check the GT file.")
        D = features_test[0].shape[1]
        T = features_test[0].shape[0]
        print(f"  D={D}, T={T}")

        # ---- Load backbone (classification tail+head on device) ----
        if is_clip:
            print(f"\nLoading CLIP (classnames={args.classnames})...")
            wrapper = ClipWrapper(args.classnames, device=device)
        else:
            print(f"\nLoading DINOv2 ({args.backbone})...")
            wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone,
                                    device=device)

        # ---- (A) Reference — uncompressed accuracy & ZIP(raw) bits ----
        print(f"\n{'=' * 60}")
        print(f"  [Cls Reference] Uncompressed acc + ZIP(raw) rate")
        print(f"{'=' * 60}")
        t0 = time.time()
        ident = GAPCCodec(D=D, threshold=-float("inf"),
                          seed=args.seed, quant_bits=32).to(device)
        acc_ref_train, _, _ = run_sweep_point(
            features_train, basenames_train, gt_train, wrapper, layer_idx, ident,
            device, chunk_images=args.chunk_images,
            zip_level=args.zip_level,
            count_mask=not args.no_mask_overhead,
            zip_workers=args.zip_workers,
        )
        acc_ref_test, _, _ = run_sweep_point(
            features_test, basenames_test, gt_test, wrapper, layer_idx, ident,
            device, chunk_images=args.chunk_images,
            zip_level=args.zip_level,
            count_mask=not args.no_mask_overhead,
            zip_workers=args.zip_workers,
        )
        avg_bits_ref_train, avg_bpfp_ref_train = zip_bits_reference(
            features_train, zip_level=args.zip_level,
            zip_workers=args.zip_workers)
        avg_bits_ref_test, avg_bpfp_ref_test = zip_bits_reference(
            features_test, zip_level=args.zip_level,
            zip_workers=args.zip_workers)
        print(f"  * train:  acc={acc_ref_train:.4f}  ZIP(raw) bpfp={avg_bpfp_ref_train:.4f}")
        print(f"  * test :  acc={acc_ref_test:.4f}   ZIP(raw) bpfp={avg_bpfp_ref_test:.4f}")
        print(f"  (ref time: {time.time() - t0:.1f}s)")

        # ---- (B) Sweep on train, then report on test at same operating points ----
        print(f"\n{'=' * 60}")
        print(f"  [Cls Sweep on TRAIN]   points={len(values)}")
        print(f"{'=' * 60}")
        for v in values:
            codec = _build_codec(args, D, v).to(device)

            # (1) train — operating-point curve
            t0 = time.time()
            acc_tr, agg_tr, _ = run_sweep_point(
                features_train, basenames_train, gt_train, wrapper, layer_idx,
                codec, device, chunk_images=args.chunk_images,
                zip_level=args.zip_level,
                count_mask=not args.no_mask_overhead,
                zip_workers=args.zip_workers,
            )
            p_tr = _point_dict(codec, "train", acc_tr, agg_tr, time.time() - t0)
            sweep_train.append(p_tr)
            _print_point(p_tr, header="train")

            # (2) test — final metrics at the *same* operating point
            t0 = time.time()
            acc_te, agg_te, _ = run_sweep_point(
                features_test, basenames_test, gt_test, wrapper, layer_idx,
                codec, device, chunk_images=args.chunk_images,
                zip_level=args.zip_level,
                count_mask=not args.no_mask_overhead,
                zip_workers=args.zip_workers,
            )
            p_te = _point_dict(codec, "test", acc_te, agg_te, time.time() - t0)
            sweep_test.append(p_te)
            _print_point(p_te, header="test ")
            print()
    else:
        print("\n[Cls] Skipped by --skip_cls.")

    # ---- (C) Segmentation evaluation (optional, VOC2012 mIoU + bpfp) ----
    # Train/test split (disjoint): sweep on voc2012_5000 (TRAIN); report on
    # voc2012_100 (TEST) at the *same* operating points -- aligned with the
    # classification methodology above and with run_soft_pq.py.
    sweep_seg_train: list = []
    sweep_seg_test: list = []
    seg_ref_train = None
    seg_ref_test = None
    if args.eval_seg:
        if is_clip:
            print(f"\n[Segmentation] Skipped (no seg head for CLIP).")
        else:
            seg_train_dir = str(
                Path(args.seg_train_feat_root) / args.backbone / args.layer
            )
            seg_test_dir = str(
                Path(args.seg_test_feat_root) / args.backbone / args.layer
            )
            print(f"\n{'=' * 60}")
            print(f"  [Segmentation] VOC2012 mIoU + bpfp sweep")
            print(f"  train feats : {seg_train_dir}")
            print(f"  train list  : {args.seg_train_image_list}")
            print(f"               (max_seg_train_images="
                  f"{args.max_seg_train_images})")
            print(f"  test  feats : {seg_test_dir}")
            print(f"  test  list  : {args.seg_test_image_list}")
            print(f"               (max_seg_test_images="
                  f"{args.max_seg_test_images})")
            print(f"{'=' * 60}")

            # Offload classification backbone (when it was instantiated) to
            # save GPU memory (aligned with run_soft_pq.py).
            if wrapper is not None:
                wrapper.backbone.cpu()
                if wrapper.head is not None:
                    wrapper.head.cpu()
                torch.cuda.empty_cache()
                weights_root = wrapper.weights_root
            else:
                # --skip_cls path: no cls wrapper was built.  Resolve the
                # weights_root via a throw-away Dinov2Wrapper on CPU so we can
                # reuse the same DINOv2 weights for the seg graph.
                tmp_w = Dinov2Wrapper(head_layers=1,
                                      model_name=args.backbone,
                                      device="cpu")
                weights_root = tmp_w.weights_root
                del tmp_w

            seg_data_train = preload_seg_data(
                seg_feat_dir=seg_train_dir,
                image_list=args.seg_train_image_list,
                voc_root=args.voc_root,
                model_name=args.backbone,
                max_images=args.max_seg_train_images,
                verbose=True,
            )
            seg_data_test = preload_seg_data(
                seg_feat_dir=seg_test_dir,
                image_list=args.seg_test_image_list,
                voc_root=args.voc_root,
                model_name=args.backbone,
                max_images=args.max_seg_test_images,
                verbose=True,
            )

            # Strict no-overlap guarantee between seg train and test samples.
            train_names = {r["name"] for r in seg_data_train}
            test_names = {r["name"] for r in seg_data_test}
            seg_overlap = train_names & test_names
            if seg_overlap:
                sample = sorted(seg_overlap)[:5]
                raise RuntimeError(
                    f"[seg] train/test sample overlap is not allowed but "
                    f"found {len(seg_overlap)} shared basenames, e.g. "
                    f"{sample}.  Check --seg_train_image_list / "
                    f"--seg_test_image_list.")

            if not seg_data_train or not seg_data_test:
                print(f"  [seg] empty train or test split -- skipping.")
            else:
                # Resolve feat_dim if --skip_cls path (no cls features loaded).
                if D is None:
                    # seg record's "features" is [num_slides, 1+N, D].
                    feat0 = seg_data_train[0]["features"]
                    D = int(feat0.shape[-1])
                    T = int(feat0.shape[-2])
                    print(f"  [seg] inferred D={D}, tokens/slide={T}  "
                          f"(--skip_cls path)")

                print(f"  Loading seg backbone + VOC head (shared across "
                      f"{len(values)} operating points)...")
                seg_backbone, seg_head = load_seg_backbone_and_head(
                    model_name=args.backbone,
                    weights_root=weights_root,
                    device=device,
                )

                def _record(codec_seg, out, wall):
                    return {
                        "quant_bits": int(codec_seg.quant_bits),
                        "threshold": float(codec_seg.threshold),
                        "miou": float(out["miou"]),
                        "acc": float(out["acc"]),
                        "avg_bpfp": float(out["rate"]["avg_bpfp"]),
                        "avg_bpt": float(out["rate"]["avg_bpt"]),
                        "avg_cr": float(out["rate"]["avg_cr"]),
                        "pooled_cr": float(out["rate"]["pooled_cr"]),
                        "avg_kept_cols": float(out["rate"]["avg_kept_cols"]),
                        "avg_kept_frac":
                            float(out["rate"]["avg_kept_cols"]
                                  / out["rate"]["feat_dim"]),
                        "avg_bits_ref": float(out["rate"]["avg_bits_ref"]),
                        "avg_bits_reduced":
                            float(out["rate"]["avg_bits_reduced"]),
                        "avg_bits_sideinfo":
                            float(out["rate"]["avg_bits_sideinfo"]),
                        "avg_bits_total": float(out["rate"]["avg_bits_total"]),
                        "num_slides": int(out["rate"]["num_images"]),
                        "wall_s": float(wall),
                    }

                def _run_pt(codec_seg, seg_data):
                    t0 = time.time()
                    out, _ = evaluate_gapc_seg_point(
                        codec=codec_seg,
                        seg_data=seg_data,
                        backbone=seg_backbone, head=seg_head,
                        layer_idx=layer_idx, device=device, feat_dim=D,
                        zip_level=args.zip_level,
                        count_mask=not args.no_mask_overhead,
                        zip_workers=args.zip_workers,
                        verbose=True,
                    )
                    return out, time.time() - t0

                # ---- Reference (uncompressed) on TRAIN + TEST ----
                print(f"  [seg] reference (no compression) ...")
                ident_seg = GAPCCodec(
                    D=D, threshold=-float("inf"),
                    seed=args.seed, quant_bits=32,
                ).to(device)
                ref_tr, wall_tr = _run_pt(ident_seg, seg_data_train)
                ref_te, wall_te = _run_pt(ident_seg, seg_data_test)
                seg_ref_train = {
                    "miou": float(ref_tr["miou"]),
                    "acc": float(ref_tr["acc"]),
                    "avg_bpfp": float(ref_tr["rate"]["avg_bpfp"]),
                    "avg_cr": float(ref_tr["rate"]["avg_cr"]),
                    "num_slides": int(ref_tr["rate"]["num_images"]),
                    "wall_s": float(wall_tr),
                }
                seg_ref_test = {
                    "miou": float(ref_te["miou"]),
                    "acc": float(ref_te["acc"]),
                    "avg_bpfp": float(ref_te["rate"]["avg_bpfp"]),
                    "avg_cr": float(ref_te["rate"]["avg_cr"]),
                    "num_slides": int(ref_te["rate"]["num_images"]),
                    "wall_s": float(wall_te),
                }
                print(f"  * [seg-ref] train: mIoU={seg_ref_train['miou']:.4f}"
                      f"  bpfp={seg_ref_train['avg_bpfp']:.4f}  "
                      f"CR={seg_ref_train['avg_cr']:.2f}  "
                      f"slides={seg_ref_train['num_slides']}  "
                      f"({wall_tr:.1f}s)")
                print(f"  * [seg-ref] test : mIoU={seg_ref_test['miou']:.4f}"
                      f"  bpfp={seg_ref_test['avg_bpfp']:.4f}  "
                      f"CR={seg_ref_test['avg_cr']:.2f}  "
                      f"slides={seg_ref_test['num_slides']}  "
                      f"({wall_te:.1f}s)")

                # ---- Sweep on TRAIN; at each op point also report TEST ----
                for v in values:
                    codec_seg = _build_codec(args, D, v).to(device)
                    out_tr, wall_tr = _run_pt(codec_seg, seg_data_train)
                    p_tr = _record(codec_seg, out_tr, wall_tr)
                    sweep_seg_train.append(p_tr)
                    out_te, wall_te = _run_pt(codec_seg, seg_data_test)
                    p_te = _record(codec_seg, out_te, wall_te)
                    sweep_seg_test.append(p_te)

                    print(f"  seg train θ={codec_seg.threshold:.4g}  "
                          f"kept={p_tr['avg_kept_cols']:.1f}/{D} "
                          f"({p_tr['avg_kept_frac']*100:.1f}%)  "
                          f"mIoU={p_tr['miou']:.4f}  "
                          f"bpfp={p_tr['avg_bpfp']:.4f}  "
                          f"CR={p_tr['avg_cr']:.2f}  "
                          f"({wall_tr:.1f}s)")
                    print(f"  seg test  θ={codec_seg.threshold:.4g}  "
                          f"kept={p_te['avg_kept_cols']:.1f}/{D} "
                          f"({p_te['avg_kept_frac']*100:.1f}%)  "
                          f"mIoU={p_te['miou']:.4f}  "
                          f"bpfp={p_te['avg_bpfp']:.4f}  "
                          f"CR={p_te['avg_cr']:.2f}  "
                          f"({wall_te:.1f}s)")
                    print()

                # Free seg graph before exiting the experiment.
                del seg_backbone, seg_head
                torch.cuda.empty_cache()

    # ---- (D) Summary + JSON save ----
    results = {
        "config": vars(args),
        "layer_idx": layer_idx,
        "feat_dim": D,
        "num_tokens": T,
        "num_train_images": len(features_train),
        "num_test_images": len(features_test),
        "acc_ref_train": float(acc_ref_train),
        "acc_ref_test": float(acc_ref_test),
        "avg_bpfp_ref_train": float(avg_bpfp_ref_train),
        "avg_bpfp_ref_test": float(avg_bpfp_ref_test),
        "avg_bits_ref_train": float(avg_bits_ref_train),
        "avg_bits_ref_test": float(avg_bits_ref_test),
        "sweep_train": sweep_train,
        "sweep_test": sweep_test,
        "sweep_seg_train": sweep_seg_train,
        "sweep_seg_test": sweep_seg_test,
        "seg_ref_train": seg_ref_train,
        "seg_ref_test": seg_ref_test,
    }
    out_dir = os.path.join(GAPC_ROOT, "results", args.backbone)
    os.makedirs(out_dir, exist_ok=True)
    mask_tag = "_nomask" if args.no_mask_overhead else ""
    qbit_tag = f"_q{args.quant_bits}" if args.quant_bits != 32 else ""
    # Tag reflects which tasks actually ran, and -- for seg -- which sample
    # sizes; for cls-only the old size tuple is kept for backward compat.
    if args.skip_cls and args.eval_seg:
        n_seg_tr = sweep_seg_train[0]["num_slides"] if sweep_seg_train else 0
        n_seg_te = sweep_seg_test[0]["num_slides"] if sweep_seg_test else 0
        tag = (f"{args.layer}_seg_zip{args.zip_level}{qbit_tag}"
               f"_segtr{n_seg_tr}_segte{n_seg_te}"
               f"_s{args.seed}{mask_tag}")
    else:
        seg_tag = "_seg" if args.eval_seg and sweep_seg_train else ""
        tag = (f"{args.layer}_gapc_zip{args.zip_level}{qbit_tag}"
               f"_ntr{len(features_train)}_nte{len(features_test)}"
               f"{seg_tag}_s{args.seed}{mask_tag}")
    out_path = os.path.join(out_dir, f"{tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ---- Pretty summary table (train & test side-by-side) ----
    print(f"\n{'=' * 72}")
    print(f"  Summary: {args.layer} / {args.backbone}")
    if sweep_train:
        print(f"  [cls] n_train={len(features_train)}  "
              f"n_test={len(features_test)}")
        print(f"  [cls] ref acc  train={acc_ref_train:.4f}   "
              f"test={acc_ref_test:.4f}")
        print(f"  [cls] ref bpfp train={avg_bpfp_ref_train:.4f}  "
              f"test={avg_bpfp_ref_test:.4f}")
        print(f"  {'point':>12} {'kept%':>7} {'acc_tr':>7} {'acc_te':>7} "
              f"{'bpfp_tr':>8} {'bpfp_te':>8} {'CR_tr':>6} {'CR_te':>6}")
        for p_tr, p_te in zip(sweep_train, sweep_test):
            head = f"θ={p_tr['threshold']:.4g}"
            print(f"  {head:>12} "
                  f"{p_tr['avg_kept_frac']*100:>7.1f} "
                  f"{p_tr['acc']:>7.4f} {p_te['acc']:>7.4f} "
                  f"{p_tr['avg_bpfp']:>8.4f} {p_te['avg_bpfp']:>8.4f} "
                  f"{p_tr['avg_cr']:>6.2f} {p_te['avg_cr']:>6.2f}")

    # ---- Segmentation summary (train sweep + test report) ----
    if sweep_seg_train:
        print(f"\n  [seg] VOC2012 mIoU ({len(sweep_seg_train)} points  "
              f"train={sweep_seg_train[0]['num_slides']} slides, "
              f"test={sweep_seg_test[0]['num_slides']} slides)")
        if seg_ref_train is not None:
            print(f"  [seg] ref mIoU  train={seg_ref_train['miou']:.4f}   "
                  f"test={seg_ref_test['miou']:.4f}")
            print(f"  [seg] ref bpfp train={seg_ref_train['avg_bpfp']:.4f}  "
                  f"test={seg_ref_test['avg_bpfp']:.4f}")
        print(f"  {'point':>12} {'kept%':>7} "
              f"{'mIoU_tr':>8} {'mIoU_te':>8} "
              f"{'bpfp_tr':>8} {'bpfp_te':>8} "
              f"{'CR_tr':>6} {'CR_te':>6}")
        for p_tr, p_te in zip(sweep_seg_train, sweep_seg_test):
            head = f"θ={p_tr['threshold']:.4g}"
            print(f"  {head:>12} "
                  f"{p_tr['avg_kept_frac']*100:>7.1f} "
                  f"{p_tr['miou']:>8.4f} {p_te['miou']:>8.4f} "
                  f"{p_tr['avg_bpfp']:>8.4f} {p_te['avg_bpfp']:>8.4f} "
                  f"{p_tr['avg_cr']:>6.2f} {p_te['avg_cr']:>6.2f}")

    print(f"\nResults saved: {out_path}")
    print(f"{'=' * 72}")
    return results


# ================================================================
#                    CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GAPC experiment runner "
                    "(strict paper impl., train sweep / test report)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--layer", type=str, default="blk10",
                        choices=[f"blk{i:02d}" for i in range(40)])
    parser.add_argument("--backbone", type=str, default="dinov2_vitl14")
    parser.add_argument("--threshold_list", type=float, nargs="+",
                        default=[0.005, 0.02, 0.05, 0.075, 0.1],
                        help="GAPC thresholds to sweep (Algorithm 1).")
    parser.add_argument("--skip_cls", action="store_true",
                        help="Skip the ImageNet classification pipeline "
                             "(data loading, cls backbone, cls sweep).  "
                             "Requires --eval_seg so there is something to "
                             "run.  Use when cls results are already cached "
                             "and only the segmentation task needs to run.")

    parser.add_argument("--zip_level", type=int, default=6,
                        help="zlib DEFLATE compression level 0..9.")
    parser.add_argument("--quant_bits", type=int, default=32,
                        choices=[32, 16, 8, 4],
                        help="Scalar quantisation bit-width of the kept "
                             "columns.  32 = paper-strict (no quant, fp32); "
                             "16 = IEEE fp16 round-to-nearest; "
                             "8 / 4 = per-column asymmetric int quant with "
                             "fp16 side-info (scale, zero-point).  Forward "
                             "pass performs matching fake quant+dequant so "
                             "the reported accuracy matches the bit-stream.")
    parser.add_argument("--no_mask_overhead", action="store_true",
                        help="Skip the per-image D-bit mask overhead in the "
                             "bpfp/CR tally (sensitivity study only).")

    parser.add_argument("--chunk_images", type=int, default=64,
                        help="GPU batch size (streaming mask + tail forward).")
    parser.add_argument("--zip_workers", type=int, default=8,
                        help="CPU thread-pool size for parallel zlib.")

    parser.add_argument("--train_subset", type=str, default="train",
                        help="Train-set feature subdir under features/.")
    parser.add_argument("--max_train_images", type=int, default=500,
                        help="0 = use all available (aligned with run_soft_pq).")
    parser.add_argument("--max_test_images", type=int, default=0,
                        help="0 = use all available.")

    parser.add_argument("--feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features"))
    parser.add_argument("--classnames", type=str,
                        default=os.path.join(PROJECT_ROOT, "utils",
                                             "classnames.txt"))
    parser.add_argument("--gt_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "utils",
                                             "imagenet_selected_label500.txt"),
                        help="Ground-truth labels for the TEST split.")
    parser.add_argument("--train_gt_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "utils",
                                             "imagenet_selected_label5000.txt"),
                        help="Ground-truth labels for the TRAIN split "
                             "(disjoint from --gt_path in this codebase).")
    parser.add_argument("--seed", type=int, default=42)

    # ---- Segmentation (VOC2012 mIoU + bpfp) ----
    parser.add_argument("--eval_seg", action="store_true",
                        help="Also evaluate VOC2012 segmentation mIoU at "
                             "every sweep operating point (no effect for "
                             "CLIP backbones — they have no seg head).  "
                             "Train sweep on voc2012_5000; test report on "
                             "voc2012_100 (disjoint).")

    # --- seg TRAIN (used to choose operating points, bpfp, mIoU curve) ---
    parser.add_argument(
        "--seg_train_feat_root", type=str,
        default=os.path.join(PROJECT_ROOT, "features", "voc2012_5000"),
        help="Root of VOC seg *train* features: "
             "<root>/<backbone>/<layer>/*.npy, each [num_slides, 1+N, D].")
    parser.add_argument(
        "--seg_train_image_list", type=str,
        default=os.path.join(PROJECT_ROOT, "utils", "voc2012_all_5000.txt"),
        help="Basenames used for the seg train sweep.")
    parser.add_argument(
        "--max_seg_train_images", type=int, default=500,
        help="0 = use all names from --seg_train_image_list.  Default 500 "
             "matches the classification train sample size.")

    # --- seg TEST (final reporting -- disjoint from TRAIN) ---
    parser.add_argument(
        "--seg_test_feat_root", type=str,
        default=os.path.join(PROJECT_ROOT, "features", "voc2012_100"),
        help="Root of VOC seg *test* features (held out).")
    parser.add_argument(
        "--seg_test_image_list", type=str,
        default=os.path.join(PROJECT_ROOT, "utils", "voc2012_val_100.txt"),
        help="Basenames used for the seg test report.")
    parser.add_argument(
        "--max_seg_test_images", type=int, default=0,
        help="0 = use all names from --seg_test_image_list.")

    parser.add_argument(
        "--voc_root", type=str,
        default=os.path.join(PROJECT_ROOT, "data", "VOCdevkit", "VOC2012"),
        help="VOC2012 devkit root (for JPEGImages, SegmentationClass).")

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
