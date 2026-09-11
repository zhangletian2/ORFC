#!/usr/bin/env python
"""Per-image codec encode/decode time on VOC seg and NYU depth test features.

Includes rANS; excludes backbone and task heads.

Haar+ORFC encode:
    normalize → greedy groups → Haar analysis → rotate → PQ → rANS
Haar+ORFC decode:
    rANS → codebook lookup → inv-rotate → Haar synthesis → denorm
    (groups are side-info: GPU→CPU on encode, CPU→GPU on decode)

ORFC encode/decode: same as orfc/eval/eval_timing.py on the raw tokens.

Usage:
    CUDA_VISIBLE_DEVICES=2 python -u eval_haar_orfc_codec_timing.py \\
        --K 16 --layers blk05 --tasks seg --n_warmup 2 --max_images 8
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
import torch.nn.functional as F_fn

HERE = Path(__file__).resolve().parent
ORFC = HERE.parent / "orfc"
PROJECT = HERE.parents[1]
for p in (str(ORFC), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from compressai import ans  # noqa: E402
from compressai._CXX import pmf_to_quantized_cdf  # noqa: E402
from opq import batch_inv_normalize_gpu, batch_normalize_gpu  # noqa: E402

from run_global_residual import global_groups, load_global_detail  # noqa: E402
from soft_pq import load_codec  # noqa: E402

LAYERS = ("blk05", "blk10", "blk15", "blk20")
KS = (4, 8, 16, 64, 256, 512)
D_FEAT = 1024
EMB = 32


def haar_ckpt_pair(layer, K, args):
    ns = SimpleNamespace(
        layer=layer, K=K, embedding_dim=EMB, bottleneck_dim=1024,
        lmbda=args.lmbda, ste="softmax", tau_start=2.0, tau_end=2.0,
        tau_schedule="constant", lr=args.lr, haar_lr_scale=args.haar_lr_scale,
        epochs=args.epochs, max_train_images=5000, n_val=200, seed=42,
        backbone=args.backbone,
    )
    from run_haar_orfc_joint_from_opq import joint_stem
    stem = joint_stem(ns)
    ckpt_dir = Path(args.haar_ckpt_dir) / args.backbone
    return ckpt_dir / f"{stem}.pt", ckpt_dir / f"{stem}_haar.pt"


def find_orfc_ckpt(layer, K, args):
    ckpt_dir = Path(args.orfc_ckpt_dir) / args.backbone
    names = [
        f"{layer}_K{K}_emb{EMB}_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt",
        f"{layer}_K{K}_emb{EMB}_bt1024_ws_lmbda0.5_tau0.5_lr0.0005_ep100_n5000_s42.pt",
        f"{layer}_K{K}_emb{EMB}_bt1024_ws_tau0.5_lr0.0003_ep100_n5000_s42.pt",
    ]
    for name in names:
        path = ckpt_dir / name
        if path.is_file():
            return path
    return None


def load_task_images(task, layer, args):
    """List of images, each a list of [T, D] float32 slides."""
    if task == "seg":
        feat_dir = Path(args.seg_feat_root) / args.backbone / layer
    elif task == "depth":
        feat_dir = Path(args.nyu_feat_root) / layer
    else:
        raise ValueError(task)
    files = sorted(feat_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(feat_dir)
    if args.max_images > 0:
        files = files[: args.max_images]
    images = []
    for path in files:
        arr = np.load(path)
        if arr.ndim == 2:
            slides = [arr.astype(np.float32)]
        else:
            slides = [arr[s].astype(np.float32) for s in range(arr.shape[0])]
        images.append(slides)
    return images


def flatten_slides(images):
    slides, owners = [], []
    for i, img in enumerate(images):
        for sl in img:
            slides.append(sl)
            owners.append(i)
    return slides, owners


def prepare_cdfs(pmf_list, G, K, precision=16):
    cdfs, sizes = [], []
    for g in range(G):
        p = torch.from_numpy(np.asarray(pmf_list[g])).float()
        p = torch.cat([p, (1.0 - p.sum()).clamp_min(0).unsqueeze(0)])
        cdfs.append(pmf_to_quantized_cdf(p.tolist(), precision))
        sizes.append(int(K) + 2)
    return cdfs, sizes


def log2_pmf_cost(pq):
    if not pq.use_rate:
        return None
    with torch.no_grad():
        log_p = F_fn.log_softmax(pq.log_prior, dim=-1)
        pf = pq.prior_floor
        if pf > 0:
            p_val = log_p.exp()
            p_val = (1.0 - pf) * p_val + pf / pq.K
            return -(p_val + 1e-30).log() / math.log(2)
        return -log_p / math.log(2)


def pq_labels(Z, C, d, G, log2_cost, lmbda):
    N = Z.shape[0]
    sub_g = Z.reshape(N, G, d).permute(1, 0, 2)
    dists_sq = torch.cdist(sub_g, C).pow(2)
    cost = dists_sq if log2_cost is None else dists_sq + log2_cost.unsqueeze(1) / lmbda
    return cost.argmin(dim=-1)


def rans_pack(labels_np, cdfs, sizes, G):
    N = labels_np.shape[1]
    sym = labels_np.T.ravel().tolist()
    idx_list = np.tile(np.arange(G, dtype=np.int32), N).tolist()
    bitstream = ans.RansEncoder().encode_with_indexes(
        sym, idx_list, cdfs, sizes, [0] * G)
    return bitstream, idx_list


def rans_unpack(bitstream, idx_list, cdfs, sizes, G, N):
    decoded = ans.RansDecoder().decode_with_indexes(
        bitstream, idx_list, cdfs, sizes, [0] * G)
    return np.asarray(decoded, dtype=np.int64).reshape(N, G).T


def rotation(codec):
    tf = codec.transform
    if tf is not None and hasattr(tf, "get_rotation"):
        with torch.no_grad():
            return tf.get_rotation()
    return None


def codebook_decode(labels, C, d, R, D):
    G, N = labels.shape
    Z_hat_g = torch.gather(
        C.unsqueeze(1).expand(-1, N, -1, -1), 2,
        labels.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, d),
    ).squeeze(2)
    Z_hat = Z_hat_g.permute(1, 0, 2).reshape(N, D)
    return Z_hat if R is None else Z_hat @ R.t()


def summarise_image_times(enc_slide, dec_slide, owners, n_warmup_images):
    n_img = max(owners) + 1 if owners else 0
    enc_img = [0.0] * n_img
    dec_img = [0.0] * n_img
    n_slides = [0] * n_img
    tok = [0] * n_img
    for t_enc, t_dec, owner, n_tok in zip(
            enc_slide["t"], dec_slide["t"], owners, enc_slide["tok"]):
        enc_img[owner] += t_enc
        dec_img[owner] += t_dec
        n_slides[owner] += 1
        tok[owner] += n_tok
    keep = [i for i in range(n_img) if i >= n_warmup_images]
    enc = np.array([enc_img[i] for i in keep], dtype=np.float64)
    dec = np.array([dec_img[i] for i in keep], dtype=np.float64)
    toks = np.array([tok[i] for i in keep], dtype=np.float64)
    parts = {}
    for key in enc_slide.get("parts", {}):
        acc = [0.0] * n_img
        for val, owner in zip(enc_slide["parts"][key], owners):
            acc[owner] += val
        arr = np.array([acc[i] for i in keep], dtype=np.float64)
        parts[f"enc_{key}_ms"] = round(float(arr.mean()) * 1000, 4)
    for key in dec_slide.get("parts", {}):
        acc = [0.0] * n_img
        for val, owner in zip(dec_slide["parts"][key], owners):
            acc[owner] += val
        arr = np.array([acc[i] for i in keep], dtype=np.float64)
        parts[f"dec_{key}_ms"] = round(float(arr.mean()) * 1000, 4)
    return {
        "n_images": int(len(keep)),
        "n_slides_per_image": int(np.mean([n_slides[i] for i in keep])) if keep else 0,
        "avg_tokens_per_image": round(float(toks.mean()), 1) if keep else 0,
        "enc_ms_mean": round(float(enc.mean()) * 1000, 4) if keep else None,
        "enc_ms_std": round(float(enc.std()) * 1000, 4) if keep else None,
        "dec_ms_mean": round(float(dec.mean()) * 1000, 4) if keep else None,
        "dec_ms_std": round(float(dec.std()) * 1000, 4) if keep else None,
        **parts,
    }


@torch.no_grad()
def measure_orfc(slides, owners, codec, device, cdfs, sizes, n_warmup_images):
    pq = codec.pq
    C, G, K, d = pq.codebooks, pq.G, pq.K, pq.d
    D = C.shape[0] * d
    R = rotation(codec)
    cost = log2_pmf_cost(pq)
    lmbda = pq.lmbda if pq.use_rate else None
    enc = {"t": [], "tok": [], "parts": {"gpu": [], "rans": []}}
    dec = {"t": [], "parts": {"rans": [], "gpu": []}}
    checked = False
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
        enc["parts"]["gpu"].append(t_gpu)
        enc["parts"]["rans"].append(t_enc - t_gpu)
        dec["t"].append(t_dec)
        dec["parts"]["rans"].append(t_rans_dec)
        dec["parts"]["gpu"].append(t_gpu_dec)
        del X, Y, Mu, Std, X_hat
    return summarise_image_times(enc, dec, owners, n_warmup_images)


@torch.no_grad()
def measure_haar(slides, owners, haar, orfc, device, cdfs, sizes, n_warmup_images):
    pq = orfc.pq
    C, G, K, d = pq.codebooks, pq.G, pq.K, pq.d
    D = haar.D
    R = rotation(orfc)
    cost = log2_pmf_cost(pq)
    lmbda = pq.lmbda if pq.use_rate else None
    n_prefix = haar.n_prefix
    enc = {"t": [], "tok": [], "parts": {"group": [], "haar_pq": [], "rans": []}}
    dec = {"t": [], "parts": {"rans": [], "gpu": []}}
    checked = False
    for feat in slides:
        N = feat.shape[0]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
        Y, Mu, Std = batch_normalize_gpu(X, mode="per_image")
        groups = global_groups(Y, n_prefix=n_prefix)
        torch.cuda.synchronize()
        t_group = time.perf_counter() - t0
        seq, aux = haar.encode(Y, groups)
        Tm = seq.shape[1]
        flat = seq.reshape(-1, D)
        Z = flat if R is None else flat @ R
        labels = pq_labels(Z, C, d, G, cost, lmbda)
        labels_np = labels.cpu().numpy()
        groups_cpu = groups.cpu()
        leftover_cpu = aux["leftover_idx"].cpu()
        n_patch = int(aux["n_patch"])
        torch.cuda.synchronize()
        t_haar_pq = time.perf_counter() - t0
        bitstream, idx_list = rans_pack(labels_np, cdfs, sizes, G)
        t_enc = time.perf_counter() - t0
        t1 = time.perf_counter()
        dec_np = rans_unpack(bitstream, idx_list, cdfs, sizes, G, Tm)
        t_rans_dec = time.perf_counter() - t1
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        dec_labels = torch.from_numpy(dec_np).to(device)
        seq_hat = codebook_decode(dec_labels, C, d, R, D).view(1, Tm, D)
        aux_dec = {
            "groups": groups_cpu.to(device),
            "n_patch": n_patch,
            "leftover_idx": leftover_cpu.to(device),
        }
        Y_hat = haar.decode(seq_hat, aux_dec)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        torch.cuda.synchronize()
        t_gpu_dec = time.perf_counter() - t2
        t_dec = time.perf_counter() - t1
        if not checked:
            if not np.array_equal(labels_np, dec_np):
                raise RuntimeError("Haar+ORFC rANS round-trip mismatch")
            checked = True
        enc["t"].append(t_enc)
        enc["tok"].append(Tm)
        enc["parts"]["group"].append(t_group)
        enc["parts"]["haar_pq"].append(t_haar_pq - t_group)
        enc["parts"]["rans"].append(t_enc - t_haar_pq)
        dec["t"].append(t_dec)
        dec["parts"]["rans"].append(t_rans_dec)
        dec["parts"]["gpu"].append(t_gpu_dec)
        del X, Y, Mu, Std, X_hat, seq, groups
    out = summarise_image_times(enc, dec, owners, n_warmup_images)
    out["avg_coded_tokens_per_image"] = out.pop("avg_tokens_per_image")
    out["avg_src_tokens_per_image"] = round(
        float(np.mean([sl.shape[0] for sl, own in zip(slides, owners)
                       if own >= n_warmup_images])
              * (out["n_slides_per_image"] or 1)), 1)
    return out


def pmf_from_codec(codec):
    pq = codec.pq
    if pq.use_rate:
        return [pq.get_prior_pmf()[g] for g in range(pq.G)]
    return None


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--layers", default=",".join(LAYERS))
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--tasks", default="seg,depth")
    p.add_argument("--codecs", default="haar,orfc")
    p.add_argument("--backbone", default="dinov2_vitl14")
    p.add_argument("--lmbda", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--haar_lr_scale", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--n_warmup", type=int, default=5)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--haar_ckpt_dir", default=str(HERE / "checkpoints"))
    p.add_argument("--orfc_ckpt_dir", default=str(ORFC / "checkpoints"))
    p.add_argument("--seg_feat_root",
                   default=str(PROJECT / "features" / "voc2012_100"))
    p.add_argument("--nyu_feat_root",
                   default=str(PROJECT / "features" / "nyu_depth_80"
                               / "dinov2_vitl14"))
    p.add_argument("--result_dir",
                   default=str(HERE / "results" / "haar_orfc_jointopq" / "timing"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    codecs = [x.strip() for x in args.codecs.split(",") if x.strip()]
    K = int(args.K)
    out_dir = Path(args.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"# codec timing  K={K}  layers={layers}  tasks={tasks}")
    print(f"# codecs={codecs}  warmup_images={args.n_warmup}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}", flush=True)

    rows = []
    for layer in layers:
        haar_orfc_path, haar_path = haar_ckpt_pair(layer, K, args)
        orfc_path = find_orfc_ckpt(layer, K, args)
        orfc_note = None
        if "haar" in codecs:
            if not haar_orfc_path.is_file() or not haar_path.is_file():
                raise FileNotFoundError(haar_orfc_path)
        if "orfc" in codecs and orfc_path is None:
            if haar_orfc_path.is_file():
                orfc_path = haar_orfc_path
                orfc_note = "no standalone ORFC ckpt; timed Haar-joint ORFC weights on full tokens"
                print(f"  [{layer}] ORFC K={K} missing → proxy {haar_orfc_path.name}")
            else:
                print(f"  [{layer}] SKIP orfc, no ckpt")
        haar = orfc_h = orfc_plain = None
        if "haar" in codecs:
            haar, _ = load_global_detail(str(haar_path), device=device)
            orfc_h = load_codec(str(haar_orfc_path), device=device)
            haar.eval(); orfc_h.eval()
        if "orfc" in codecs and orfc_path is not None:
            orfc_plain = load_codec(str(orfc_path), device=device)
            orfc_plain.eval()

        for task in tasks:
            images = load_task_images(task, layer, args)
            slides, owners = flatten_slides(images)
            print(f"\n  {layer} {task}: {len(images)} images, "
                  f"{len(slides)} slides, shape={slides[0].shape}", flush=True)
            if "haar" in codecs:
                pmf = pmf_from_codec(orfc_h)
                if pmf is None:
                    raise RuntimeError("Haar ORFC has no learned prior")
                cdfs, sizes = prepare_cdfs(pmf, orfc_h.pq.G, orfc_h.pq.K)
                t0 = time.time()
                stats = measure_haar(
                    slides, owners, haar, orfc_h, device, cdfs, sizes,
                    args.n_warmup)
                stats.update({
                    "codec": "haar_orfc", "layer": layer, "K": K, "task": task,
                    "orfc_ckpt": str(haar_orfc_path),
                    "haar_ckpt": str(haar_path),
                    "elapsed_s": round(time.time() - t0, 2),
                })
                rows.append(stats)
                print(f"    Haar+ORFC  enc {stats['enc_ms_mean']:.2f}±"
                      f"{stats['enc_ms_std']:.2f} ms  dec {stats['dec_ms_mean']:.2f}±"
                      f"{stats['dec_ms_std']:.2f} ms  "
                      f"group {stats.get('enc_group_ms', 0):.1f}  "
                      f"coded {stats.get('avg_coded_tokens_per_image')}",
                      flush=True)
            if "orfc" in codecs and orfc_plain is not None:
                pmf = pmf_from_codec(orfc_plain)
                if pmf is None:
                    labels = []
                    for feat in slides[: max(16, args.n_warmup + 5)]:
                        X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
                        Y, _, _ = batch_normalize_gpu(X, mode="per_image")
                        orfc_plain(Y)
                        labels.append(orfc_plain.pq._last_labels.cpu())
                    lab = torch.cat(labels, dim=1).numpy()
                    G, Kp = orfc_plain.pq.G, orfc_plain.pq.K
                    pmf = []
                    for g in range(G):
                        c = np.zeros(Kp, dtype=np.float64)
                        np.add.at(c, lab[g], 1)
                        c += 1.0
                        pmf.append(c / c.sum())
                cdfs, sizes = prepare_cdfs(pmf, orfc_plain.pq.G, orfc_plain.pq.K)
                t0 = time.time()
                stats = measure_orfc(
                    slides, owners, orfc_plain, device, cdfs, sizes,
                    args.n_warmup)
                stats.update({
                    "codec": "orfc", "layer": layer, "K": K, "task": task,
                    "orfc_ckpt": str(orfc_path),
                    "orfc_note": orfc_note,
                    "elapsed_s": round(time.time() - t0, 2),
                })
                rows.append(stats)
                print(f"    ORFC       enc {stats['enc_ms_mean']:.2f}±"
                      f"{stats['enc_ms_std']:.2f} ms  dec {stats['dec_ms_mean']:.2f}±"
                      f"{stats['dec_ms_std']:.2f} ms  "
                      f"tok {stats.get('avg_tokens_per_image')}",
                      flush=True)
            torch.cuda.empty_cache()
        del haar, orfc_h, orfc_plain
        torch.cuda.empty_cache()

    out_path = out_dir / f"K{K}_timing.json"
    payload = {
        "K": K,
        "layers": layers,
        "tasks": tasks,
        "n_warmup_images": args.n_warmup,
        "unit": "ms per image (sum of slides); rANS included; backbone/task excluded",
        "rows": rows,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
