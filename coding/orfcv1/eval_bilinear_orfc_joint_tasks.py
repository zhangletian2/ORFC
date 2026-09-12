#!/usr/bin/env python
"""VOC mIoU + NYU RMSE + rANS rate + encode/decode time.

Compares conv2 spatial+ORFC joint with original ORFC on full tokens.
Default ablation is recon0; pass ``--residual_ablation main`` for U(E(X)).

    CUDA_VISIBLE_DEVICES=2 python -u eval_bilinear_orfc_joint_tasks.py \\
        --layer blk20 --K 4
    CUDA_VISIBLE_DEVICES=2 python -u eval_bilinear_orfc_joint_tasks.py \\
        --layer blk20 --K 4 --residual_ablation main
    CUDA_VISIBLE_DEVICES=2 python -u eval_bilinear_orfc_joint_tasks.py \\
        --layer blk05 --K 2 --orfc_only
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
from soft_pq import load_codec  # noqa: E402

from bilinear_residual import (  # noqa: E402
    RESIDUAL_ABLATIONS, BilinearORFCWrapper, BilinearSpatialCodec,
    apply_residual, freeze_module, load_residual_codec, load_spatial_weights,
)
from eval_haar_orfc_codec_timing import (  # noqa: E402
    codebook_decode, flatten_slides, load_task_images, log2_pmf_cost,
    pmf_from_codec, pq_labels, prepare_cdfs, rans_pack, rans_unpack,
    rotation, summarise_image_times,
)
from eval_haar_orfc_tasks import eval_seg  # noqa: E402
from eval_residual_depth import (  # noqa: E402
    eval_anchor, eval_codec, load_feats, prepare_nyu_samples,
)
from dinov2_depth_pipeline import _load_backbone, _load_depth_head  # noqa: E402
from run_bilinear_orfc_joint import (  # noqa: E402
    joint_stem, orfc_ckpt_path, spatial_out_path,
)

NORM_BITS = 32.0
D_FEAT = 1024
PLAIN_ORFC_NAMES = (
    "emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt",
    "emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0005_ep100_n5000_s42.pt",
    "emb32_bt1024_ws_tau0.5_lr0.0003_ep100_n5000_s42.pt",
)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--layer", default="blk20")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--backbone", default="dinov2_vitl14")
    p.add_argument("--norm_mode", default="per_image")
    p.add_argument("--tasks", default="seg,depth")
    p.add_argument("--n_warmup", type=int, default=5)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--lmbda", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--spatial_lr_scale", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max_train_images", type=int, default=5000)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embedding_dim", type=int, default=32)
    p.add_argument("--bottleneck_dim", type=int, default=1024)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--ckpt_dir", default=str(HERE / "checkpoints"))
    p.add_argument("--orfc_ckpt_dir", default=str(ORFC / "checkpoints"))
    p.add_argument("--linear4_task_dir", default=str(
        HERE / "results" / "linear4_orfc_jointopq" / "tasks"))
    p.add_argument("--result_dir", default=str(
        HERE / "results" / "bilinear_orfc_jointopq" / "tasks"))
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
    p.add_argument("--skip_orfc", action="store_true")
    p.add_argument("--orfc_only", action="store_true",
                   help="Eval original ORFC only (no spatial joint ckpt)")
    p.add_argument("--residual_ablation", default="recon0",
                   choices=list(RESIDUAL_ABLATIONS))
    return p.parse_args()


def joint_ns(args):
    return SimpleNamespace(
        layer=args.layer, K=args.K, embedding_dim=args.embedding_dim,
        bottleneck_dim=args.bottleneck_dim, lmbda=args.lmbda, lr=args.lr,
        spatial_lr_scale=args.spatial_lr_scale, epochs=args.epochs,
        max_train_images=args.max_train_images, n_val=args.n_val,
        seed=args.seed, backbone=args.backbone, ckpt_dir=args.ckpt_dir,
        tau_start=2.0, tau_end=2.0, tau_schedule="constant", scale=args.scale,
        residual_ablation=getattr(args, "residual_ablation", "recon0"),
        cls_mode=getattr(args, "cls_mode", "learned"),
    )


def find_plain_orfc(layer, K, ckpt_dir, backbone):
    d = Path(ckpt_dir) / backbone
    for name in PLAIN_ORFC_NAMES:
        path = d / f"{layer}_K{K}_{name}"
        if path.is_file():
            return path
    return None


def empirical_pmf(codec, slides, device, n_probe=21):
    """Histogram prior when the checkpoint has no learned log_prior."""
    labels = []
    for feat in slides[:n_probe]:
        X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
        Y, _, _ = batch_normalize_gpu(X, mode="per_image")
        codec(Y)
        labels.append(codec.pq._last_labels.cpu())
        del X, Y
    lab = torch.cat(labels, dim=1).numpy()
    G, Kp = codec.pq.G, codec.pq.K
    pmf = []
    for g in range(G):
        c = np.zeros(Kp, dtype=np.float64)
        np.add.at(c, lab[g], 1)
        c += 1.0
        pmf.append(c / c.sum())
    return pmf


def codec_cdfs(codec, slides, device, n_warmup):
    pmf = pmf_from_codec(codec)
    source = "learned"
    if pmf is None:
        pmf = empirical_pmf(
            codec, slides, device, n_probe=max(16, n_warmup + 5))
        source = "empirical"
        print(f"    prior=empirical (checkpoint has no log_prior)", flush=True)
    cdfs, sizes = prepare_cdfs(pmf, codec.pq.G, codec.pq.K)
    return cdfs, sizes, source


def load_recon0_joint(args, device):
    ablation = getattr(args, "residual_ablation", "recon0")
    ns = joint_ns(args)
    orfc_path = orfc_ckpt_path(ns)
    spat_path = spatial_out_path(ns)
    if not orfc_path.is_file() or not spat_path.is_file():
        raise FileNotFoundError(f"{orfc_path} / {spat_path}")
    orfc = freeze_module(load_codec(str(orfc_path), device=device))
    residual, meta = load_residual_codec(str(spat_path), device=device)
    D = int(meta.get("D", getattr(orfc.transform, "D", D_FEAT) if orfc.transform else D_FEAT))
    spatial = BilinearSpatialCodec(
        D, n_prefix=1, scale=args.scale,
        down=meta.get("spatial_down", "conv2"),
        up=meta.get("spatial_up", "conv2"),
        cls_mode=meta.get("cls_mode", "learned")).to(device)
    load_spatial_weights(spatial, meta)
    residual.set_mode("fixed" if ablation == "main" else "recon")
    wrapper = BilinearORFCWrapper(
        spatial, orfc=orfc, residual=residual,
        residual_quantize=False, residual_ablation=ablation).to(device)
    wrapper.eval()
    return {
        "wrapper": wrapper, "spatial": freeze_module(spatial),
        "orfc": orfc, "residual": freeze_module(residual),
        "orfc_ckpt": orfc_path, "spatial_ckpt": spat_path, "D": D,
        "ablation": ablation,
    }


def attach_rate(stats, bits_slide, owners, n_warmup, extra_bits=NORM_BITS):
    n_img = max(owners) + 1 if owners else 0
    acc = [0.0] * n_img
    for bits, owner in zip(bits_slide, owners):
        acc[owner] += bits
    keep = [i for i in range(n_img) if i >= n_warmup]
    rans = np.array([acc[i] for i in keep], dtype=np.float64)
    src = stats.get("avg_src_tokens_per_image") or stats.get("avg_tokens_per_image") or 0
    stats["rans_bits_mean"] = round(float(rans.mean()), 2) if keep else None
    stats["rans_bits_std"] = round(float(rans.std()), 2) if keep else None
    stats["norm_bits_per_image"] = extra_bits
    stats["guidance_bits_per_image"] = 0.0
    stats["grouping_bits_per_image"] = 0.0
    stats["bits_per_image"] = (
        round(float(rans.mean()) + extra_bits, 2) if keep else None)
    if src:
        stats["bpfp"] = (stats["bits_per_image"] or 0.0) / (float(src) * D_FEAT)
    return stats


def finish_stats(enc, dec, owners, n_warmup, slides):
    out = summarise_image_times(enc, dec, owners, n_warmup)
    coded_key = "avg_tokens_per_image"
    if "avg_coded_tokens_per_image" in out:
        coded_key = "avg_coded_tokens_per_image"
    else:
        out["avg_coded_tokens_per_image"] = out.get(coded_key)
    keep_src = [sl.shape[0] for sl, own in zip(slides, owners)
                if own >= n_warmup]
    out["avg_src_tokens_per_image"] = round(
        float(np.mean(keep_src) * (out["n_slides_per_image"] or 1)), 1)
    return attach_rate(out, enc["bits"], owners, n_warmup)


@torch.no_grad()
def measure_recon0(slides, owners, spatial, orfc, residual, device,
                   cdfs, sizes, n_warmup, ablation="recon0"):
    pq = orfc.pq
    C, G, K, d = pq.codebooks, pq.G, pq.K, pq.d
    D = C.shape[0] * d
    Rmat = rotation(orfc)
    cost = log2_pmf_cost(pq)
    lmbda = pq.lmbda if pq.use_rate else None
    n_prefix = spatial.n_prefix
    enc = {"t": [], "tok": [], "bits": [], "parts": {"gpu": [], "rans": []}}
    dec = {"t": [], "parts": {"rans": [], "gpu": []}}
    checked = False
    spatial.eval(); orfc.eval(); residual.eval()
    for feat in slides:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
        Y, Mu, Std = batch_normalize_gpu(X, mode="per_image", n_prefix=n_prefix)
        seq, aux = spatial.encode(Y)
        Tm = int(seq.shape[1])
        flat = seq.reshape(-1, D)
        Z = flat if Rmat is None else flat @ Rmat
        labels = pq_labels(Z, C, d, G, cost, lmbda)
        labels_np = labels.cpu().numpy()
        torch.cuda.synchronize()
        t_gpu = time.perf_counter() - t0
        bitstream, idx_list = rans_pack(labels_np, cdfs, sizes, G)
        t_enc = time.perf_counter() - t0
        t1 = time.perf_counter()
        dec_np = rans_unpack(bitstream, idx_list, cdfs, sizes, G, Tm)
        t_rans_dec = time.perf_counter() - t1
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        dec_labels = torch.from_numpy(dec_np).to(device)
        seq_hat = codebook_decode(dec_labels, C, d, Rmat, D).view(1, Tm, D)
        Y0 = spatial.decode(seq_hat, aux)
        Y_hat, _, _, _ = apply_residual(
            Y, Y0, residual, n_prefix=n_prefix, quantize=False,
            ablation=ablation)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        torch.cuda.synchronize()
        t_gpu_dec = time.perf_counter() - t2
        t_dec = time.perf_counter() - t1
        if not checked:
            if not np.array_equal(labels_np, dec_np):
                raise RuntimeError("spatial+ORFC rANS round-trip mismatch")
            checked = True
        enc["t"].append(t_enc)
        enc["tok"].append(Tm)
        enc["bits"].append(len(bitstream) * 8.0)
        enc["parts"]["gpu"].append(t_gpu)
        enc["parts"]["rans"].append(t_enc - t_gpu)
        dec["t"].append(t_dec)
        dec["parts"]["rans"].append(t_rans_dec)
        dec["parts"]["gpu"].append(t_gpu_dec)
        del X, Y, Mu, Std, seq, Y0, Y_hat, X_hat
    return finish_stats(enc, dec, owners, n_warmup, slides)


@torch.no_grad()
def measure_plain_orfc(slides, owners, codec, device, cdfs, sizes, n_warmup):
    pq = codec.pq
    C, G, K, d = pq.codebooks, pq.G, pq.K, pq.d
    D = C.shape[0] * d
    R = rotation(codec)
    cost = log2_pmf_cost(pq)
    lmbda = pq.lmbda if pq.use_rate else None
    enc = {"t": [], "tok": [], "bits": [], "parts": {"gpu": [], "rans": []}}
    dec = {"t": [], "parts": {"rans": [], "gpu": []}}
    checked = False
    codec.eval()
    for feat in slides:
        N = feat.shape[0]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
        Y, Mu, Std = batch_normalize_gpu(X, mode="per_image")
        flat = Y.reshape(-1, D)
        Z = flat if R is None else flat @ R
        labels = pq_labels(Z, C, d, G, cost, lmbda)
        labels_np = labels.cpu().numpy()
        torch.cuda.synchronize()
        t_gpu = time.perf_counter() - t0
        bitstream, idx_list = rans_pack(labels_np, cdfs, sizes, G)
        t_enc = time.perf_counter() - t0
        t1 = time.perf_counter()
        dec_np = rans_unpack(bitstream, idx_list, cdfs, sizes, G, N)
        t_rans_dec = time.perf_counter() - t1
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        dec_labels = torch.from_numpy(dec_np).to(device)
        Y_hat = codebook_decode(dec_labels, C, d, R, D).view(1, N, D)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        torch.cuda.synchronize()
        t_gpu_dec = time.perf_counter() - t2
        t_dec = time.perf_counter() - t1
        if not checked:
            if not np.array_equal(labels_np, dec_np):
                raise RuntimeError("ORFC rANS round-trip mismatch")
            checked = True
        enc["t"].append(t_enc)
        enc["tok"].append(N)
        enc["bits"].append(len(bitstream) * 8.0)
        enc["parts"]["gpu"].append(t_gpu)
        enc["parts"]["rans"].append(t_enc - t_gpu)
        dec["t"].append(t_dec)
        dec["parts"]["rans"].append(t_rans_dec)
        dec["parts"]["gpu"].append(t_gpu_dec)
        del X, Y, Mu, Std, X_hat
    return finish_stats(enc, dec, owners, n_warmup, slides)


def run_tasks(name, codec, args, device, layer_idx, D, nyu_pack, results):
    bucket = results.setdefault(name, {})
    if "depth" in args.tasks.split(","):
        nyu_feats, sample_meta, backbone, head, anchor = nyu_pack
        t0 = time.time()
        rmse = eval_codec(
            codec, nyu_feats, layer_idx, sample_meta,
            backbone, head, device, args.norm_mode, base_only=False)
        row = {
            "rmse": float(rmse),
            "delta_vs_anchor": float(rmse - anchor),
            "t_s": time.time() - t0,
        }
        bucket["depth"] = row
        print(f"  {name:12s}  RMSE={row['rmse']:.4f}  "
              f"Δvs_anchor={row['delta_vs_anchor']:+.4f}  "
              f"({row['t_s']:.1f}s)", flush=True)
    if "seg" in args.tasks.split(","):
        seg = eval_seg(codec, args, device, layer_idx, D)
        bucket["seg"] = seg
        print(f"  {name:12s}  mIoU={seg['miou']:.4f}  "
              f"aAcc={seg['acc']:.4f}  ({seg['t_s']:.1f}s)", flush=True)
        torch.cuda.empty_cache()


def run_timing(name, measure_fn, args, results, prior_source=None):
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()
             in ("seg", "depth")]
    results.setdefault(name, {})["timing"] = {}
    if prior_source:
        results[name]["rans_prior"] = prior_source
    print(f"\n  [{name} encode/decode + rANS]", flush=True)
    for task in tasks:
        images = load_task_images(task, args.layer, args)
        slides, owners = flatten_slides(images)
        print(f"    {task}: {len(images)} images, {len(slides)} slides, "
              f"shape={slides[0].shape}", flush=True)
        stats = measure_fn(slides, owners)
        results[name]["timing"][task] = stats
        print(
            f"    {task:5s}  enc={stats['enc_ms_mean']:.3f}±"
            f"{stats['enc_ms_std']:.3f} ms  "
            f"dec={stats['dec_ms_mean']:.3f}±{stats['dec_ms_std']:.3f} ms  "
            f"bits/img={stats['bits_per_image']:.1f}  "
            f"coded={stats.get('avg_coded_tokens_per_image')}",
            flush=True)


def linear4_snapshot(args):
    stem = (
        f"{args.layer}_linear4_jointopq_K{args.K}_emb{args.embedding_dim}"
        f"_bt{args.bottleneck_dim}_ws_lmbda{args.lmbda}_tau2.0_te2.0_tscon"
        f"_hlr{args.spatial_lr_scale:g}_bv_lr{args.lr}_ep{args.epochs}"
        f"_n{args.max_train_images}_nval{args.n_val}_s{args.seed}"
    )
    path = Path(args.linear4_task_dir) / f"{stem}_tasks.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    full = data.get("full") or {}
    return {
        "path": str(path),
        "seg": full.get("seg"),
        "depth": full.get("depth"),
        "anchor_rmse": data.get("anchor_rmse"),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_idx = int(args.layer[-2:])
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    args.tasks = ",".join(tasks)

    print(f"\n{'#' * 70}")
    if args.orfc_only:
        print(f"# original ORFC only  {args.layer}  K={args.K}")
    else:
        print(f"# {args.residual_ablation}+ORFC joint vs original ORFC  "
              f"{args.layer}  K={args.K}")
    print(f"# tasks={tasks}  warmup={args.n_warmup}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}", flush=True)

    bili = None
    if not args.orfc_only:
        bili = load_recon0_joint(args, device)
        print(f"  bilinear  {bili['orfc_ckpt'].name}", flush=True)
    skip_plain = args.skip_orfc and not args.orfc_only
    plain_path = None if skip_plain else find_plain_orfc(
        args.layer, args.K, args.orfc_ckpt_dir, args.backbone)
    plain = None
    if plain_path is not None:
        plain = freeze_module(load_codec(str(plain_path), device=device))
        print(f"  ORFC      {plain_path.name}", flush=True)
    elif args.orfc_only:
        raise FileNotFoundError(
            f"plain ORFC missing: {args.layer} K={args.K}")
    elif not args.skip_orfc:
        print(f"  ORFC      MISSING K={args.K} (skip original)", flush=True)

    results = {
        "layer": args.layer,
        "K": args.K,
        "protocol": "voc100_nyu80_rans",
        "norm_mode": args.norm_mode,
        "tasks": tasks,
        "bilinear_ckpt": None if bili is None else str(bili["orfc_ckpt"]),
        "spatial_ckpt": None if bili is None else str(bili["spatial_ckpt"]),
        "orfc_ckpt": str(plain_path) if plain_path else None,
        "residual_ablation": None if args.orfc_only else args.residual_ablation,
        "orfc_only": bool(args.orfc_only),
        "linear4": None if args.orfc_only else linear4_snapshot(args),
    }

    nyu_pack = None
    if "depth" in tasks:
        print(f"\n{'=' * 60}\n  [NYU Depth RMSE]\n{'=' * 60}", flush=True)
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
        nyu_pack = (nyu_feats, sample_meta, backbone, head, anchor)

    if "depth" in tasks or "seg" in tasks:
        print(f"\n{'=' * 60}\n  [tasks]\n{'=' * 60}", flush=True)
        if bili is not None:
            run_tasks("bilinear", bili["wrapper"], args, device, layer_idx,
                      bili["D"], nyu_pack, results)
        if plain is not None:
            run_tasks("orfc", plain, args, device, layer_idx,
                      D_FEAT, nyu_pack, results)

    if nyu_pack is not None:
        del nyu_pack
        torch.cuda.empty_cache()

    probe_images = load_task_images(
        "depth" if "depth" in tasks else tasks[0], args.layer, args)
    probe_slides, _ = flatten_slides(probe_images)
    if bili is not None:
        cdfs_b, sizes_b, src_b = codec_cdfs(
            bili["orfc"], probe_slides, device, args.n_warmup)
        run_timing(
            "bilinear",
            lambda slides, owners: measure_recon0(
                slides, owners, bili["spatial"], bili["orfc"], bili["residual"],
                device, cdfs_b, sizes_b, args.n_warmup,
                ablation=args.residual_ablation),
            args, results, prior_source=src_b)
    if plain is not None:
        cdfs_o, sizes_o, src_o = codec_cdfs(
            plain, probe_slides, device, args.n_warmup)
        run_timing(
            "orfc",
            lambda slides, owners: measure_plain_orfc(
                slides, owners, plain, device, cdfs_o, sizes_o, args.n_warmup),
            args, results, prior_source=src_o)
    del probe_images, probe_slides

    out_dir = Path(args.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (plain_path.stem if args.orfc_only else bili["orfc_ckpt"].stem)
    out_path = out_dir / f"{stem}_tasks.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
