#!/usr/bin/env python
"""VOC mIoU + NYU RMSE for bilinear main ORFC ± residual guidance.

Same protocol as ``eval_haar_orfc_tasks.py`` (VOC val-100, NYU test-80).

Usage:
    CUDA_VISIBLE_DEVICES=4 python -u eval_bilinear_residual_tasks.py \\
        --mode both
    CUDA_VISIBLE_DEVICES=5 python -u eval_bilinear_residual_tasks.py \\
        --mode main --tasks seg,depth
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
ORFC = HERE.parent / "orfc"
ORFCV2 = HERE.parent / "orfcv2"
PROJECT = HERE.parents[1]
for p in (str(ORFC), str(HERE), str(ORFCV2)):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_soft_pq import CodecSegmentationEvaluator  # noqa: E402
from soft_pq import load_codec  # noqa: E402

from bilinear_residual import (  # noqa: E402
    BilinearORFCWrapper, BilinearSpatialCodec, freeze_module,
    load_residual_codec,
)
from eval_haar_orfc_tasks import eval_seg  # noqa: E402
from eval_residual_depth import (  # noqa: E402
    eval_anchor, eval_codec, load_feats, prepare_nyu_samples,
)
from dinov2_depth_pipeline import _load_backbone, _load_depth_head  # noqa: E402
from run_bilinear_residual import (  # noqa: E402
    init_path, main_ckpt_path, residual_ckpt_path,
)


def parse_eval_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", default="both",
                   choices=["main", "fixed", "recon", "both"])
    p.add_argument("--layer", default="blk05")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--c", type=int, default=4)
    p.add_argument("--backbone", default="dinov2_vitl14")
    p.add_argument("--norm_mode", default="per_image")
    p.add_argument("--tasks", default="seg,depth")
    p.add_argument("--main_ckpt", default="")
    p.add_argument("--residual_ckpt", default="")
    p.add_argument("--ckpt_dir", default=str(HERE / "checkpoints"))
    p.add_argument("--residual_dir", default=str(
        HERE / "results" / "bilinear_residual"))
    p.add_argument("--result_dir", default=str(
        HERE / "results" / "bilinear_residual" / "tasks"))
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
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--residual_epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--residual_lr", type=float, default=3e-4)
    p.add_argument("--lmbda", type=float, default=0.5)
    p.add_argument("--tau_start", type=float, default=2.0)
    p.add_argument("--tau_end", type=float, default=2.0)
    p.add_argument("--tau_schedule", default="constant")
    p.add_argument("--embedding_dim", type=int, default=32)
    p.add_argument("--bottleneck_dim", type=int, default=1024)
    p.add_argument("--max_train_images", type=int, default=5000)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warm_start_opq", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--init_ckpt", default="")
    p.add_argument("--scale", type=int, default=2)
    return p.parse_args()


def load_wrapper(args, device):
    ns = SimpleNamespace(**vars(args))
    ns.result_dir = args.residual_dir
    main_ckpt = Path(args.main_ckpt) if args.main_ckpt else main_ckpt_path(ns)
    if not main_ckpt.is_file():
        raise FileNotFoundError(main_ckpt)
    orfc = freeze_module(load_codec(str(main_ckpt), device=device))
    D = int(getattr(orfc.transform, "D", 1024))
    spatial = freeze_module(
        BilinearSpatialCodec(D, n_prefix=1, scale=args.scale).to(device))
    residual = None
    residual_ckpt = None
    if args.mode != "main":
        residual_ckpt = (Path(args.residual_ckpt) if args.residual_ckpt
                         else residual_ckpt_path(ns, args.mode))
        if not residual_ckpt.is_file():
            raise FileNotFoundError(residual_ckpt)
        residual, _ = load_residual_codec(str(residual_ckpt), device=device)
        residual = freeze_module(residual)
    wrapper = BilinearORFCWrapper(spatial, orfc, residual=residual).to(device)
    wrapper.eval()
    return wrapper, main_ckpt, residual_ckpt


def out_stem(args, main_ckpt, residual_ckpt):
    if args.mode == "main":
        return f"{Path(main_ckpt).stem}_tasks"
    return f"{Path(residual_ckpt).stem}_tasks"


def main():
    args = parse_eval_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    layer_idx = int(args.layer[-2:])
    wrapper, main_ckpt, residual_ckpt = load_wrapper(args, device)

    print(f"\n{'#' * 70}")
    print(f"# Bilinear residual task eval  layer={args.layer}  mode={args.mode}")
    print(f"# ORFC  {main_ckpt}")
    if residual_ckpt:
        print(f"# residual  {residual_ckpt}")
    print(f"# tasks={tasks}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}", flush=True)

    results = {
        "layer": args.layer,
        "mode": args.mode,
        "orfc_ckpt": str(main_ckpt),
        "residual_ckpt": None if residual_ckpt is None else str(residual_ckpt),
        "norm_mode": args.norm_mode,
        "tasks": tasks,
        args.mode: {},
    }

    if "depth" in tasks:
        print(f"\n{'=' * 60}")
        print("  [NYU Depth RMSE]")
        print(f"{'=' * 60}", flush=True)
        samples, sample_meta = prepare_nyu_samples(
            args.nyu_data_root, args.nyu_split_file)
        nyu_feats = load_feats(Path(args.nyu_feat_root) / args.layer, samples)
        print(f"  NYU test80: {len(nyu_feats)}  "
              f"shape={tuple(nyu_feats[0].shape)}", flush=True)
        ns = SimpleNamespace(
            model="vitl14", weights_root=args.nyu_weights_root, device=device)
        backbone, _ = _load_backbone(ns)
        head = _load_depth_head(ns)
        t0 = time.time()
        anchor = eval_anchor(
            nyu_feats, layer_idx, sample_meta, backbone, head, device)
        print(f"  Anchor RMSE={anchor:.4f}  ({time.time() - t0:.1f}s)",
              flush=True)
        results["anchor_rmse"] = float(anchor)
        t0 = time.time()
        rmse = eval_codec(
            wrapper, nyu_feats, layer_idx, sample_meta,
            backbone, head, device, args.norm_mode, base_only=False)
        row = {
            "rmse": float(rmse),
            "delta_vs_anchor": float(rmse - anchor),
            "t_s": time.time() - t0,
        }
        results[args.mode]["depth"] = row
        print(f"  {args.mode:10s}  RMSE={row['rmse']:.4f}  "
              f"Δvs_anchor={row['delta_vs_anchor']:+.4f}  "
              f"({row['t_s']:.1f}s)", flush=True)
        del backbone, head, nyu_feats
        torch.cuda.empty_cache()

    if "seg" in tasks:
        print(f"\n{'=' * 60}")
        print("  [VOC2012 mIoU]")
        print(f"{'=' * 60}", flush=True)
        seg = eval_seg(wrapper, args, device, layer_idx, 1024)
        results[args.mode]["seg"] = seg
        print(f"  {args.mode:10s}  mIoU={seg['miou']:.4f}  "
              f"aAcc={seg['acc']:.4f}  ({seg['t_s']:.1f}s)", flush=True)
        torch.cuda.empty_cache()

    out_dir = Path(args.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_stem(args, main_ckpt, residual_ckpt)}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
