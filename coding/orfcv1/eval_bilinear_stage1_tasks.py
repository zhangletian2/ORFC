#!/usr/bin/env python
"""Stage-1 conv2 E/U task eval (no ORFC / no PQ) + encode/decode timing.

Odd grids: replicate pad on encode, crop on decode (inside
``BilinearSpatialCodec``).  Three reconstructions:

    main    hat X = U(E(X))
    recon0  hat X = X0 + F_φ(X0, 0)
    full    hat X = X0 + F_φ(X0, G)

Usage:
    CUDA_VISIBLE_DEVICES=2 python -u eval_bilinear_stage1_tasks.py \\
        --layer blk20 --ablations main,recon0,full
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

from opq import batch_inv_normalize_gpu, batch_normalize_gpu  # noqa: E402

from bilinear_residual import (  # noqa: E402
    BilinearORFCWrapper, BilinearSpatialCodec, combine_residual,
    freeze_module, load_residual_codec, load_spatial_weights,
    residual_from_main,
)
from eval_haar_orfc_codec_timing import (  # noqa: E402
    flatten_slides, load_task_images, summarise_image_times,
)
from eval_haar_orfc_tasks import eval_seg  # noqa: E402
from eval_residual_depth import (  # noqa: E402
    eval_anchor, eval_codec, load_feats, prepare_nyu_samples,
)
from dinov2_depth_pipeline import _load_backbone, _load_depth_head  # noqa: E402
from run_bilinear_residual import residual_ckpt_path  # noqa: E402


ABLATION_ORDER = ("main", "recon0", "full")


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--layer", default="blk20")
    p.add_argument("--ablations", default="main,recon0,full")
    p.add_argument("--tasks", default="seg,depth")
    p.add_argument("--backbone", default="dinov2_vitl14")
    p.add_argument("--norm_mode", default="per_image")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--c", type=int, default=4)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--spatial_down", default="conv2")
    p.add_argument("--spatial_up", default="conv2")
    p.add_argument("--cls_mode", default="learned",
                   choices=["learned", "identity", "conv2"])
    p.add_argument("--residual_decoder", default="conv")
    p.add_argument("--residual_epochs", type=int, default=30)
    p.add_argument("--residual_lr", type=float, default=3e-4)
    p.add_argument("--max_train_images", type=int, default=5000)
    p.add_argument("--n_val", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_warmup", type=int, default=5)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--result_dir", default=str(
        HERE / "results" / "bilinear_residual"))
    p.add_argument("--out_dir", default=str(
        HERE / "results" / "bilinear_residual" / "tasks"))
    p.add_argument("--checkpoint", default=None,
                   help="explicit stage-1 checkpoint; overrides generated path")
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
    return p.parse_args()


def ckpt_ns(args, ablation):
    return SimpleNamespace(
        layer=args.layer, K=args.K, c=args.c, backbone=args.backbone,
        residual_decoder=args.residual_decoder,
        spatial_down=args.spatial_down, spatial_up=args.spatial_up,
        residual_orfc=False, residual_quantize=False,
        residual_ablation=ablation,
        residual_lr=args.residual_lr, residual_epochs=args.residual_epochs,
        max_train_images=args.max_train_images, n_val=args.n_val,
        seed=args.seed, result_dir=args.result_dir, scale=args.scale,
        cls_mode=getattr(args, "cls_mode", "learned"),
    )


def load_stack(args, ablation, device):
    ns = ckpt_ns(args, ablation)
    ckpt = (Path(args.checkpoint) if args.checkpoint else
            residual_ckpt_path(ns, "both"))
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    residual, meta = load_residual_codec(str(ckpt), device=device)
    D = int(meta["D"])
    down = meta.get("spatial_down", args.spatial_down)
    up = meta.get("spatial_up", args.spatial_up)
    spatial = BilinearSpatialCodec(
        D, n_prefix=1, scale=args.scale, down=down, up=up,
        cls_mode=meta.get("cls_mode", "learned")).to(device)
    load_spatial_weights(spatial, meta)
    residual = freeze_module(residual)
    spatial = freeze_module(spatial)
    wrapper = BilinearORFCWrapper(
        spatial, orfc=None, residual=residual,
        residual_quantize=False, residual_ablation=ablation).to(device)
    wrapper.eval()
    return {
        "ablation": ablation,
        "ckpt": ckpt,
        "spatial": spatial,
        "residual": residual,
        "wrapper": wrapper,
        "D": D,
        "coded_imagenet": spatial.coded_tokens(257),
    }


def encode_stage1(Y, spatial, residual, ablation, n_prefix=1):
    """Sender: ``seq`` plus optional unquantized ``G``.  No PQ."""
    seq, aux = spatial.encode(Y)
    G = None
    if ablation == "full" and residual is not None:
        Y0 = spatial.decode(seq, aux)
        G = residual.encode_residual(
            residual_from_main(Y, Y0, n_prefix=n_prefix))
    return seq, aux, G


def decode_stage1(seq, aux, G, spatial, residual, ablation, n_prefix=1):
    """Receiver: only ``seq`` / ``G`` / grid size.  No original ``Y``."""
    Y0 = spatial.decode(seq, aux)
    if ablation == "main" or residual is None:
        return Y0
    X0 = Y0[:, n_prefix:]
    if ablation == "recon0" or G is None:
        G_in = X0.new_zeros(X0.shape[0], X0.shape[1], residual.c)
    else:
        G_in = G
    return combine_residual(
        Y0, residual.decode_residual(G_in, X0), n_prefix)


@torch.no_grad()
def measure_stage1(slides, owners, spatial, residual, ablation, device,
                   n_warmup_images, n_prefix=1):
    enc = {"t": [], "tok": [], "parts": {"gpu": []}}
    dec = {"t": [], "parts": {"gpu": []}}
    spatial.eval()
    if residual is not None:
        residual.eval()
    for feat in slides:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
        Y, Mu, Std = batch_normalize_gpu(X, mode="per_image", n_prefix=n_prefix)
        seq, aux, G = encode_stage1(
            Y, spatial, residual, ablation, n_prefix=n_prefix)
        Tm = int(seq.shape[1])
        torch.cuda.synchronize()
        t_enc = time.perf_counter() - t0

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        Y_hat = decode_stage1(
            seq, aux, G, spatial, residual, ablation, n_prefix=n_prefix)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        torch.cuda.synchronize()
        t_dec = time.perf_counter() - t1

        enc["t"].append(t_enc)
        enc["tok"].append(Tm)
        enc["parts"]["gpu"].append(t_enc)
        dec["t"].append(t_dec)
        dec["parts"]["gpu"].append(t_dec)
        del X, Y, Mu, Std, seq, G, Y_hat, X_hat
    out = summarise_image_times(enc, dec, owners, n_warmup_images)
    out["avg_coded_tokens_per_image"] = out.pop("avg_tokens_per_image")
    keep_src = [sl.shape[0] for sl, own in zip(slides, owners)
                if own >= n_warmup_images]
    out["avg_src_tokens_per_image"] = round(
        float(np.mean(keep_src) * (out["n_slides_per_image"] or 1)), 1)
    out["pq"] = False
    out["odd_grid"] = "replicate_pad"
    return out


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ablations = [a.strip() for a in args.ablations.split(",") if a.strip()]
    for a in ablations:
        if a not in ABLATION_ORDER:
            raise ValueError(f"unknown ablation {a!r}")
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    layer_idx = int(args.layer[-2:])

    print(f"\n{'#' * 70}")
    print(f"# Stage-1 conv E/U tasks  layer={args.layer}  no ORFC / no PQ")
    print(f"# odd grid: replicate pad + crop")
    print(f"# ablations={ablations}  tasks={tasks}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}", flush=True)

    stacks = {ab: load_stack(args, ab, device) for ab in ablations}
    for ab, st in stacks.items():
        print(f"  loaded {ab:6s}  {st['ckpt'].name}  "
              f"ImageNet coded={st['coded_imagenet']}", flush=True)

    results = {
        "layer": args.layer,
        "protocol": "stage1_no_pq",
        "odd_grid": "replicate_pad",
        "spatial_down": args.spatial_down,
        "spatial_up": args.spatial_up,
        "residual_decoder": args.residual_decoder,
        "norm_mode": args.norm_mode,
        "ablations": ablations,
        "tasks": tasks,
        "ckpts": {ab: str(stacks[ab]["ckpt"]) for ab in ablations},
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
        ns_d = SimpleNamespace(
            model="vitl14", weights_root=args.nyu_weights_root, device=device)
        backbone, _ = _load_backbone(ns_d)
        head = _load_depth_head(ns_d)
        t0 = time.time()
        anchor = eval_anchor(
            nyu_feats, layer_idx, sample_meta, backbone, head, device)
        print(f"  Anchor RMSE={anchor:.4f}  ({time.time() - t0:.1f}s)",
              flush=True)
        results["anchor_rmse"] = float(anchor)
        for ab in ablations:
            t0 = time.time()
            rmse = eval_codec(
                stacks[ab]["wrapper"], nyu_feats, layer_idx, sample_meta,
                backbone, head, device, args.norm_mode, base_only=False)
            row = {
                "rmse": float(rmse),
                "delta_vs_anchor": float(rmse - anchor),
                "t_s": time.time() - t0,
            }
            results.setdefault(ab, {})["depth"] = row
            print(f"  {ab:8s}  RMSE={row['rmse']:.4f}  "
                  f"Δvs_anchor={row['delta_vs_anchor']:+.4f}  "
                  f"({row['t_s']:.1f}s)", flush=True)
        del backbone, head, nyu_feats
        torch.cuda.empty_cache()

    if "seg" in tasks:
        print(f"\n{'=' * 60}")
        print("  [VOC2012 mIoU]")
        print(f"{'=' * 60}", flush=True)
        for ab in ablations:
            seg = eval_seg(
                stacks[ab]["wrapper"], args, device, layer_idx,
                stacks[ab]["D"])
            results.setdefault(ab, {})["seg"] = seg
            print(f"  {ab:8s}  mIoU={seg['miou']:.4f}  "
                  f"aAcc={seg['acc']:.4f}  ({seg['t_s']:.1f}s)", flush=True)
        torch.cuda.empty_cache()

    time_tasks = [t for t in tasks if t in ("seg", "depth")]
    if time_tasks:
        print(f"\n{'=' * 60}")
        print("  [encode/decode time, no PQ]")
        print(f"{'=' * 60}", flush=True)
        results["timing"] = {}
        for task in time_tasks:
            images = load_task_images(task, args.layer, args)
            slides, owners = flatten_slides(images)
            print(f"  {task}: {len(images)} images, {len(slides)} slides, "
                  f"shape={slides[0].shape}", flush=True)
            results["timing"][task] = {}
            for ab in ablations:
                st = stacks[ab]
                stats = measure_stage1(
                    slides, owners, st["spatial"], st["residual"], ab,
                    device, args.n_warmup)
                results["timing"][task][ab] = stats
                print(
                    f"  {ab:8s}  {task:5s}  "
                    f"enc={stats['enc_ms_mean']:.3f}±{stats['enc_ms_std']:.3f} ms  "
                    f"dec={stats['dec_ms_mean']:.3f}±{stats['dec_ms_std']:.3f} ms  "
                    f"coded={stats['avg_coded_tokens_per_image']}",
                    flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.layer}_bili65_{args.spatial_down}_u{args.spatial_up}"
        f"_stage1_pad_tasks"
    )
    out_path = out_dir / f"{stem}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
