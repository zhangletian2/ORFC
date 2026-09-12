#!/usr/bin/env python
"""Evaluate absorbed merged weights vs original joint vs plain ORFC.

Three configurations per (layer, K):
  1. absorbed  — R merged into Conv2d, CLS uses dense R from checkpoint
  2. original  — original joint training (Cayley R)
  3. orfc      — plain ORFC baseline (no spatial codec)

Metrics: VOC mIoU, NYU RMSE, rANS bitrate, FLOPs, encode/decode timing.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -u eval_absorbed_v2.py \
        --layer blk20 --K 64 --residual_ablation main
    p.add_argument("--max_images", type=int, default=0)
"""

from __future__ import annotations
import argparse, json, sys, time, os
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

from opq import batch_inv_normalize_gpu, batch_normalize_gpu
from soft_pq import load_codec, SoftPQ, FeatureCodec

from bilinear_residual import (
    RESIDUAL_ABLATIONS, BilinearORFCWrapper, BilinearSpatialCodec,
    apply_residual, freeze_module, load_residual_codec, load_spatial_weights,
)
from eval_haar_orfc_codec_timing import (
    codebook_decode, flatten_slides, load_task_images, log2_pmf_cost,
    pmf_from_codec, pq_labels, prepare_cdfs, rans_pack, rans_unpack,
    rotation, summarise_image_times,
)
from eval_haar_orfc_tasks import eval_seg
from eval_residual_depth import (
    eval_anchor, eval_codec, load_feats, prepare_nyu_samples,
)
from dinov2_depth_pipeline import _load_backbone, _load_depth_head
from run_bilinear_orfc_joint import (
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
    p = argparse.ArgumentParser()
    p.add_argument("--layer", default="blk20")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--backbone", default="dinov2_vitl14")
    p.add_argument("--norm_mode", default="per_image")
    p.add_argument("--n_prefix", type=int, default=1)
    p.add_argument("--tasks", default="seg,depth")
    p.add_argument("--n_warmup", type=int, default=5)
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
    p.add_argument("--result_dir", default=str(
        HERE / "results" / "absorbed_eval"))
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
    p.add_argument("--residual_ablation", default="main",
                   choices=list(RESIDUAL_ABLATIONS))
    p.add_argument("--max_images", type=int, default=0)
    return p.parse_args()


def joint_ns(args):
    return SimpleNamespace(
        layer=args.layer, K=args.K, embedding_dim=args.embedding_dim,
        bottleneck_dim=args.bottleneck_dim, lmbda=args.lmbda, lr=args.lr,
        spatial_lr_scale=args.spatial_lr_scale, epochs=args.epochs,
        max_train_images=args.max_train_images, n_val=args.n_val,
        seed=args.seed, backbone=args.backbone, ckpt_dir=args.ckpt_dir,
        tau_start=2.0, tau_end=2.0, tau_schedule="constant", scale=args.scale,
        residual_ablation=getattr(args, "residual_ablation", "main"),
        cls_mode=getattr(args, "cls_mode", "learned"),
    )


def find_plain_orfc(layer, K, ckpt_dir, backbone):
    d = Path(ckpt_dir) / backbone
    for name in PLAIN_ORFC_NAMES:
        path = d / f"{layer}_K{K}_{name}"
        if path.is_file():
            return path
    return None



class AbsorbedFeatureCodec(torch.nn.Module):
    """FeatureCodec wrapper that applies dense R only to prefix tokens.

    For absorbed models: patches go through conv2 (R already absorbed),
    the prefix group (CLS, plus dinov3's 4 register tokens) needs explicit
    R rotation before/after PQ.  ``n_prefix`` sizes that group.
    """
    def __init__(self, codec, R_dense, n_prefix=1):
        super().__init__()
        self.codec = codec
        self.register_buffer("R_dense", R_dense)
        self.n_prefix = n_prefix

    def forward(self, Y_norm):
        # Slice on the token axis, not the flattened (B*T) axis: after
        # reshape(B*T, D) the prefix of image b sits at rows b*T .. b*T+p-1,
        # so a flat [:n_prefix*B] slice only coincides with the prefix when
        # B == 1.  Working in [B, T, D] keeps this correct for any batch size
        # and any prefix length (dinov2 n_prefix=1, dinov3 n_prefix=5).
        B, T, D = Y_norm.shape
        p = min(int(self.n_prefix), T)
        rotate = p > 0 and self.R_dense is not None
        if rotate:
            Z = torch.cat(
                [Y_norm[:, :p] @ self.R_dense, Y_norm[:, p:]], dim=1)
        else:
            Z = Y_norm
        Z_hat, usage = self.codec.pq._quantise(Z.reshape(B * T, D))
        Z_hat = Z_hat.reshape(B, T, D)
        if rotate:
            Y_hat = torch.cat(
                [Z_hat[:, :p] @ self.R_dense.t(), Z_hat[:, p:]], dim=1)
        else:
            Y_hat = Z_hat
        return Y_hat, usage

    @property
    def pq(self):
        return self.codec.pq

    @property
    def transform(self):
        return None

    @property
    def use_rate(self):
        return self.codec.pq.use_rate


# ── Load absorbed merged checkpoint ──
def load_absorbed(args, device):
    """Load absorbed checkpoint: conv2 with R merged + PQ + dense R for CLS."""
    ns = joint_ns(args)
    orfc_path = orfc_ckpt_path(ns)
    absorbed_orfc_path = Path(str(orfc_path).replace(".pt", "_absorbed.pt"))
    absorbed_spat_path = Path(str(orfc_path).replace(".pt", "_absorbed_spatial.pt"))

    if not absorbed_orfc_path.is_file() or not absorbed_spat_path.is_file():
        raise FileNotFoundError(
            f"Absorbed ckpts not found:\n  {absorbed_orfc_path}\n  {absorbed_spat_path}")

    orfc_meta = torch.load(str(absorbed_orfc_path), map_location="cpu")
    spat_meta = torch.load(str(absorbed_spat_path), map_location="cpu")

    D = int(orfc_meta.get("D", D_FEAT))
    cls_mode = spat_meta.get("cls_mode", orfc_meta.get("cls_mode", "learned"))
    R_dense = orfc_meta.get("R_dense", None)
    if R_dense is not None:
        R_dense = R_dense.float().to(device)
    # Build PQ codec without transform
    sd = orfc_meta["state_dict"]
    G = int(orfc_meta.get("G", 32))
    K = int(orfc_meta.get("K", args.K))
    d = int(orfc_meta.get("embedding_dim", args.embedding_dim))
    pq = SoftPQ(G, K, d, lmbda=args.lmbda)
    pq.load_state_dict({k.replace("pq.", ""): v for k, v in sd.items()
                        if k.startswith("pq.")}, strict=False)
    if "codebooks" in sd:
        pq.load_state_dict(sd, strict=False)
    codec = FeatureCodec(pq, transform=None).to(device)
    codec.eval()
    freeze_module(codec)

    # Wrap codec with CLS R rotation for absorbed model
    if R_dense is not None:
        absorbed_codec = AbsorbedFeatureCodec(codec, R_dense, n_prefix=args.n_prefix).to(device)
        absorbed_codec.eval()
    else:
        absorbed_codec = codec

    # Build absorbed spatial (cls_mode comes from the checkpoint)
    spatial = BilinearSpatialCodec(
        D, n_prefix=args.n_prefix, scale=args.scale, down="conv2", up="conv2",
        cls_mode=cls_mode).to(device)
    load_spatial_weights(spatial, spat_meta)
    freeze_module(spatial)

    # Build residual
    residual, _ = load_residual_codec(str(absorbed_spat_path), device=device)
    ablation = args.residual_ablation
    residual.set_mode("fixed" if ablation == "main" else "recon")
    freeze_module(residual)

    # Wrapper for task eval (seg, depth) — uses absorbed_codec with CLS R
    wrapper = BilinearORFCWrapper(
        spatial, orfc=absorbed_codec, residual=residual,
        residual_quantize=False, residual_ablation=ablation).to(device)
    wrapper.eval()

    return {
        "wrapper": wrapper, "spatial": spatial,
        "codec": absorbed_codec, "residual": residual,
        "R_dense": R_dense, "D": D, "cls_mode": cls_mode,
        "orfc_path": absorbed_orfc_path,
        "spatial_path": absorbed_spat_path,
        "ablation": ablation,
    }


# ── Load original joint checkpoint (with Cayley R) ──
def load_original_joint(args, device):
    ablation = args.residual_ablation
    ns = joint_ns(args)
    orfc_path = orfc_ckpt_path(ns)
    spat_path = spatial_out_path(ns)
    if not orfc_path.is_file() or not spat_path.is_file():
        raise FileNotFoundError(f"{orfc_path} / {spat_path}")
    orfc = freeze_module(load_codec(str(orfc_path), device=device))
    residual, meta = load_residual_codec(str(spat_path), device=device)
    D = int(meta.get("D", getattr(orfc.transform, "D", D_FEAT)
                      if orfc.transform else D_FEAT))
    spatial = BilinearSpatialCodec(
        D, n_prefix=args.n_prefix, scale=args.scale,
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
        "D": D, "ablation": ablation,
    }


# ── Rate measurement helpers ──
def empirical_pmf(codec, slides, device, n_probe=21, norm_mode="per_image", n_prefix=0):
    labels = []
    for feat in slides[:n_probe]:
        X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
        Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
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


def codec_cdfs(codec, slides, device, n_warmup, norm_mode="per_image", n_prefix=0):
    pmf = pmf_from_codec(codec)
    source = "learned"
    if pmf is None:
        pmf = empirical_pmf(codec, slides, device, n_probe=max(16, n_warmup + 5),
                            norm_mode=norm_mode, n_prefix=n_prefix)
        source = "empirical"
    cdfs, sizes = prepare_cdfs(pmf, codec.pq.G, codec.pq.K)
    return cdfs, sizes, source


def attach_rate(stats, bits_slide, owners, n_warmup, extra_bits=NORM_BITS):
    n_img = max(owners) + 1 if owners else 0
    acc = [0.0] * n_img
    for bits, owner in zip(bits_slide, owners):
        acc[owner] += bits
    keep = [i for i in range(n_img) if i >= n_warmup]
    rans = np.array([acc[i] for i in keep], dtype=np.float64)
    src = stats.get("avg_src_tokens_per_image") or 0
    stats["rans_bits_mean"] = round(float(rans.mean()), 2) if keep else None
    stats["bits_per_image"] = (
        round(float(rans.mean()) + extra_bits, 2) if keep else None)
    if src:
        stats["bpfp"] = (stats["bits_per_image"] or 0.0) / (float(src) * D_FEAT)
    return stats


def finish_stats(enc, dec, owners, n_warmup, slides):
    out = summarise_image_times(enc, dec, owners, n_warmup)
    if "avg_coded_tokens_per_image" not in out:
        out["avg_coded_tokens_per_image"] = out.get("avg_tokens_per_image")
    keep_src = [sl.shape[0] for sl, own in zip(slides, owners)
                if own >= n_warmup]
    out["avg_src_tokens_per_image"] = round(
        float(np.mean(keep_src) * (out.get("n_slides_per_image") or 1)), 1)
    return attach_rate(out, enc["bits"], owners, n_warmup)


# ── Measure: absorbed (R in conv2, dense R for CLS) ──
@torch.no_grad()
def measure_absorbed(slides, owners, spatial, codec, residual, R_dense,
                     device, cdfs, sizes, n_warmup, ablation="main",
                     norm_mode="per_image"):
    pq = codec.pq
    C, G, K, d = pq.codebooks, pq.G, pq.K, pq.d
    D = C.shape[0] * d
    cost = log2_pmf_cost(pq)
    lmbda = pq.lmbda if pq.use_rate else None
    n_prefix = spatial.n_prefix
    enc = {"t": [], "tok": [], "bits": [], "parts": {"gpu": [], "rans": []}}
    dec = {"t": [], "parts": {"rans": [], "gpu": []}}
    checked = False
    for feat in slides:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
        Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
        seq, aux = spatial.encode(Y)  # absorbed conv2: patches already in R-rotated space
        Tm = int(seq.shape[1])
        flat = seq.reshape(-1, D)
        # CLS needs R rotation (not absorbed into Conv2d), patches don't
        if R_dense is not None and n_prefix > 0:
            # flat[0:n_prefix] = CLS tokens, apply R
            flat_cls = flat[:n_prefix] @ R_dense
            flat_patch = flat[n_prefix:]
            Z = torch.cat([flat_cls, flat_patch], dim=0)
        else:
            Z = flat
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
        # Decode PQ: no R^T for patches (absorbed), R^T for CLS
        Z_hat_g = torch.gather(
            C.unsqueeze(1).expand(-1, Tm, -1, -1), 2,
            dec_labels.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, d),
        ).squeeze(2)
        Z_hat = Z_hat_g.permute(1, 0, 2).reshape(Tm, D)
        if R_dense is not None and n_prefix > 0:
            cls_hat = Z_hat[:n_prefix] @ R_dense.t()
            patch_hat = Z_hat[n_prefix:]
            seq_hat = torch.cat([cls_hat, patch_hat], dim=0).unsqueeze(0)
        else:
            seq_hat = Z_hat.unsqueeze(0)
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
                raise RuntimeError("absorbed rANS round-trip mismatch")
            checked = True
        enc["t"].append(t_enc); enc["tok"].append(Tm)
        enc["bits"].append(len(bitstream) * 8.0)
        enc["parts"]["gpu"].append(t_gpu)
        enc["parts"]["rans"].append(t_enc - t_gpu)
        dec["t"].append(t_dec)
        dec["parts"]["rans"].append(t_rans_dec)
        dec["parts"]["gpu"].append(t_gpu_dec)
        del X, Y, Mu, Std, seq, Y0, Y_hat, X_hat
    return finish_stats(enc, dec, owners, n_warmup, slides)


# ── Measure: original joint (Cayley R) ──
@torch.no_grad()
def measure_original(slides, owners, spatial, orfc, residual, device,
                     cdfs, sizes, n_warmup, ablation="main",
                     norm_mode="per_image"):
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
    for feat in slides:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
        Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
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
                raise RuntimeError("original rANS round-trip mismatch")
            checked = True
        enc["t"].append(t_enc); enc["tok"].append(Tm)
        enc["bits"].append(len(bitstream) * 8.0)
        enc["parts"]["gpu"].append(t_gpu)
        enc["parts"]["rans"].append(t_enc - t_gpu)
        dec["t"].append(t_dec)
        dec["parts"]["rans"].append(t_rans_dec)
        dec["parts"]["gpu"].append(t_gpu_dec)
        del X, Y, Mu, Std, seq, Y0, Y_hat, X_hat
    return finish_stats(enc, dec, owners, n_warmup, slides)


# ── Measure: plain ORFC ──
@torch.no_grad()
def measure_plain_orfc(slides, owners, codec, device, cdfs, sizes, n_warmup,
                       norm_mode="per_image", n_prefix=0):
    pq = codec.pq
    C, G, K, d = pq.codebooks, pq.G, pq.K, pq.d
    D = C.shape[0] * d
    R = rotation(codec)
    cost = log2_pmf_cost(pq)
    lmbda = pq.lmbda if pq.use_rate else None
    enc = {"t": [], "tok": [], "bits": [], "parts": {"gpu": [], "rans": []}}
    dec = {"t": [], "parts": {"rans": [], "gpu": []}}
    checked = False
    for feat in slides:
        N = feat.shape[0]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
        Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
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
        enc["t"].append(t_enc); enc["tok"].append(N)
        enc["bits"].append(len(bitstream) * 8.0)
        enc["parts"]["gpu"].append(t_gpu)
        enc["parts"]["rans"].append(t_enc - t_gpu)
        dec["t"].append(t_dec)
        dec["parts"]["rans"].append(t_rans_dec)
        dec["parts"]["gpu"].append(t_gpu_dec)
        del X, Y, Mu, Std, X_hat
    return finish_stats(enc, dec, owners, n_warmup, slides)


# ── FLOPs computation ──
def compute_flops(D, C_latent, G, d, K, n_prefix, T_in=257, scale=2):
    """Compute FLOPs for encode/decode pipeline, per-image.

    Cayley solve is one-time (precomputable), listed separately.
    R matmul is per-image.  The prefix bypasses the convs, so it adds no
    conv positions.
    """
    H_in = int((T_in - n_prefix) ** 0.5)  # 16
    H_out = H_in // scale  # 8
    n_pos = H_out * H_out
    T_coded = n_prefix + H_out * H_out  # 1 + 64 = 65

    # Conv2d analysis [encode]: D_in * C_out * k^2 * n_pos * 2
    conv2_enc = D * C_latent * 4 * n_pos * 2
    # ConvTranspose2d synthesis [decode]: same
    conv2_dec = conv2_enc
    # PQ quantize [encode]: T * G * K * d * 2 (distance computation)
    pq_enc = T_coded * G * K * d * 2
    # PQ lookup [decode]: T * G * d (gather, negligible)
    pq_dec = T_coded * G * d

    # R rotation per-image (R precomputed)
    R_enc = T_coded * D * D * 2       # seq @ R
    R_dec = T_coded * D * D * 2       # Z_hat @ R^T
    # The prefix bypasses the convs, so it always needs its own dense R
    R_cls_enc = n_prefix * D * D * 2
    R_cls_dec = n_prefix * D * D * 2
    cayley_solve = 2 * D ** 3         # one-time

    # Plain ORFC (all 257 tokens)
    R_orfc_enc = T_in * D * D * 2
    R_orfc_dec = T_in * D * D * 2
    pq_orfc_enc = T_in * G * K * d * 2
    pq_orfc_dec = T_in * G * d

    result = {}
    result["original"] = {
        "encode": {"conv2": conv2_enc, "R_matmul": R_enc, "pq": pq_enc,
                   "total": conv2_enc + R_enc + pq_enc},
        "decode": {"pq_lookup": pq_dec, "R_matmul": R_dec, "conv2": conv2_dec,
                   "total": pq_dec + R_dec + conv2_dec},
        "cayley_once": cayley_solve,
        "per_image_total": conv2_enc + R_enc + pq_enc + pq_dec + R_dec + conv2_dec,
    }
    result["absorbed"] = {
        "encode": {"conv2": conv2_enc, "R_cls": R_cls_enc, "pq": pq_enc,
                   "total": conv2_enc + R_cls_enc + pq_enc},
        "decode": {"pq_lookup": pq_dec, "R_cls": R_cls_dec, "conv2": conv2_dec,
                   "total": pq_dec + R_cls_dec + conv2_dec},
        "per_image_total": conv2_enc + R_cls_enc + pq_enc + pq_dec + R_cls_dec + conv2_dec,
    }
    result["plain_orfc"] = {
        "encode": {"R_matmul": R_orfc_enc, "pq": pq_orfc_enc,
                   "total": R_orfc_enc + pq_orfc_enc},
        "decode": {"pq_lookup": pq_orfc_dec, "R_matmul": R_orfc_dec,
                   "total": pq_orfc_dec + R_orfc_dec},
        "cayley_once": cayley_solve,
        "per_image_total": R_orfc_enc + pq_orfc_enc + pq_orfc_dec + R_orfc_dec,
    }
    return result


# ── Task eval helpers ──
def run_tasks(name, codec_wrapper, args, device, layer_idx, D, nyu_pack):
    bucket = {}
    tasks = [t.strip() for t in args.tasks.split(",")]
    if "depth" in tasks and nyu_pack is not None:
        nyu_feats, sample_meta, backbone, head, anchor = nyu_pack
        t0 = time.time()
        rmse = eval_codec(
            codec_wrapper, nyu_feats, layer_idx, sample_meta,
            backbone, head, device, args.norm_mode, base_only=False)
        bucket["depth"] = {
            "rmse": float(rmse),
            "delta_vs_anchor": float(rmse - anchor),
            "t_s": time.time() - t0,
        }
        print(f"  {name:12s}  RMSE={rmse:.4f}  "
              f"Δanchor={rmse - anchor:+.4f}", flush=True)
    if "seg" in tasks:
        seg_args = SimpleNamespace(
            seg_feat_root=args.seg_feat_root,
            seg_image_list=args.seg_image_list,
            voc_root=args.voc_root, weights_root=args.weights_root,
            backbone=args.backbone, norm_mode=args.norm_mode,
            layer=args.layer,
        )
        seg = eval_seg(codec_wrapper, seg_args, device, layer_idx, D)
        bucket["seg"] = seg
        print(f"  {name:12s}  mIoU={seg['miou']:.4f}  "
              f"aAcc={seg['acc']:.4f}", flush=True)
        torch.cuda.empty_cache()
    return bucket


def run_timing(name, measure_fn, args, task="depth"):
    images = load_task_images(task, args.layer, args)
    slides, owners = flatten_slides(images)
    stats = measure_fn(slides, owners)
    print(f"  {name:12s}  enc={stats['enc_ms_mean']:.3f}ms  "
          f"dec={stats['dec_ms_mean']:.3f}ms  "
          f"bits/img={stats.get('bits_per_image','?')}", flush=True)
    return stats


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_idx = int(args.layer[-2:])
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print(f"\n{'#' * 70}")
    print(f"# Absorbed vs Original vs ORFC  {args.layer}  K={args.K}  "
          f"ablation={args.residual_ablation}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}", flush=True)

    # Load all three configurations
    print("\n[Loading checkpoints]", flush=True)
    absorbed = load_absorbed(args, device)
    print(f"  absorbed:  {absorbed['orfc_path'].name}", flush=True)

    original = load_original_joint(args, device)
    print(f"  original:  OK (Cayley R)", flush=True)

    plain_path = find_plain_orfc(args.layer, args.K,
                                 args.orfc_ckpt_dir, args.backbone)
    plain = None
    if plain_path:
        plain = freeze_module(load_codec(str(plain_path), device=device))
        print(f"  orfc:      {plain_path.name}", flush=True)
    else:
        print(f"  orfc:      NOT FOUND K={args.K}", flush=True)

    results = {
        "layer": args.layer, "K": args.K,
        "ablation": args.residual_ablation,
        "timestamp": datetime.now().isoformat(),
    }

    # ── FLOPs ──
    print(f"\n{'=' * 60}\n  [FLOPs per-image]\n{'=' * 60}", flush=True)
    flops = compute_flops(D_FEAT, D_FEAT, 32, args.embedding_dim, args.K, 1)
    print(f"  {'config':>12s} {'encode':>12s} {'decode':>12s} {'total':>12s} {'cayley(1x)':>12s}")
    print(f"  {'-'*54}")
    for cfg in ["original", "absorbed", "plain_orfc"]:
        data = flops[cfg]
        enc_g = data["encode"]["total"] / 1e9
        dec_g = data["decode"]["total"] / 1e9
        tot_g = data["per_image_total"] / 1e9
        cay_g = data.get("cayley_once", 0) / 1e9
        cay_s = f"{cay_g:.3f}" if cay_g > 0 else "-"
        print(f"  {cfg:>12s} {enc_g:>12.3f} {dec_g:>12.3f} {tot_g:>12.3f} {cay_s:>12s}")
    a = flops["absorbed"]
    print(f"\n  Absorbed encode breakdown:")
    for k, v in a["encode"].items():
        if k != "total":
            print(f"    {k:>12s}: {v/1e6:>10.1f} MFLOPs")
    print(f"  Absorbed decode breakdown:")
    for k, v in a["decode"].items():
        if k != "total":
            print(f"    {k:>12s}: {v/1e6:>10.1f} MFLOPs")
    results["flops"] = flops

    # ── Depth + Seg tasks ──
    nyu_pack = None
    if "depth" in tasks:
        print(f"\n{'=' * 60}\n  [NYU Depth]\n{'=' * 60}", flush=True)
        samples, sample_meta = prepare_nyu_samples(
            args.nyu_data_root, args.nyu_split_file)
        nyu_feats = load_feats(Path(args.nyu_feat_root) / args.layer, samples)
        ns_d = SimpleNamespace(
            model="vitl14", weights_root=args.nyu_weights_root, device=device)
        backbone, _ = _load_backbone(ns_d)
        head = _load_depth_head(ns_d)
        anchor = eval_anchor(
            nyu_feats, layer_idx, sample_meta, backbone, head, device)
        print(f"  Anchor RMSE={anchor:.4f}", flush=True)
        results["anchor_rmse"] = float(anchor)
        nyu_pack = (nyu_feats, sample_meta, backbone, head, anchor)

    if "depth" in tasks or "seg" in tasks:
        print(f"\n{'=' * 60}\n  [Task Evaluation]\n{'=' * 60}", flush=True)
        results["absorbed"] = run_tasks(
            "absorbed", absorbed["wrapper"], args, device, layer_idx,
            absorbed["D"], nyu_pack)
        results["original"] = run_tasks(
            "original", original["wrapper"], args, device, layer_idx,
            original["D"], nyu_pack)
        if plain is not None:
            results["orfc"] = run_tasks(
                "orfc", plain, args, device, layer_idx, D_FEAT, nyu_pack)

    if nyu_pack is not None:
        del nyu_pack
        torch.cuda.empty_cache()

    # ── Timing + Rate ──
    print(f"\n{'=' * 60}\n  [Timing + Rate]\n{'=' * 60}", flush=True)
    task_for_timing = "depth" if "depth" in tasks else tasks[0]
    probe_images = load_task_images(task_for_timing, args.layer, args)
    probe_slides, _ = flatten_slides(probe_images)

    # Absorbed timing
    cdfs_a, sizes_a, _ = codec_cdfs(absorbed["codec"], probe_slides, device, args.n_warmup,
        norm_mode=args.norm_mode, n_prefix=args.n_prefix)
    results.setdefault("absorbed", {})["timing"] = run_timing(
        "absorbed",
        lambda sl, ow: measure_absorbed(
            sl, ow, absorbed["spatial"], absorbed["codec"], absorbed["residual"],
            absorbed["R_dense"], device, cdfs_a, sizes_a, args.n_warmup,
            ablation=args.residual_ablation, norm_mode=args.norm_mode),
        args, task=task_for_timing)

    # Original timing
    cdfs_o, sizes_o, _ = codec_cdfs(original["orfc"], probe_slides, device, args.n_warmup,
        norm_mode=args.norm_mode, n_prefix=args.n_prefix)
    results.setdefault("original", {})["timing"] = run_timing(
        "original",
        lambda sl, ow: measure_original(
            sl, ow, original["spatial"], original["orfc"], original["residual"],
            device, cdfs_o, sizes_o, args.n_warmup,
            ablation=args.residual_ablation, norm_mode=args.norm_mode),
        args, task=task_for_timing)

    # Plain ORFC timing
    if plain is not None:
        cdfs_p, sizes_p, _ = codec_cdfs(plain, probe_slides, device, args.n_warmup,
        norm_mode=args.norm_mode, n_prefix=args.n_prefix)
        results.setdefault("orfc", {})["timing"] = run_timing(
            "orfc",
            lambda sl, ow: measure_plain_orfc(
                sl, ow, plain, device, cdfs_p, sizes_p, args.n_warmup,
                norm_mode=args.norm_mode, n_prefix=args.n_prefix),
            args, task=task_for_timing)

    # ── Summary table ──
    print(f"\n{'=' * 70}")
    print(f"  Summary: {args.layer} K={args.K} ablation={args.residual_ablation}")
    print(f"{'=' * 70}")
    print(f"{'config':>12s} {'RMSE':>8s} {'mIoU':>8s} {'bits/img':>10s} {'GFLOPs':>8s} {'enc_ms':>8s} {'dec_ms':>8s}")
    print("-" * 70)
    for cfg in ["absorbed", "original", "orfc"]:
        r = results.get(cfg, {})
        rmse = r.get("depth", {}).get("rmse", "-")
        miou = r.get("seg", {}).get("miou", "-")
        bits = r.get("timing", {}).get("bits_per_image", "-")
        gflops = flops.get(cfg, flops.get("plain_orfc", {})).get("per_image_total", 0) / 1e9
        enc = r.get("timing", {}).get("enc_ms_mean", "-")
        dec = r.get("timing", {}).get("dec_ms_mean", "-")
        rmse_s = f"{rmse:.4f}" if isinstance(rmse, float) else rmse
        miou_s = f"{miou:.4f}" if isinstance(miou, float) else miou
        bits_s = f"{bits:.1f}" if isinstance(bits, float) else bits
        enc_s = f"{enc:.3f}" if isinstance(enc, float) else enc
        dec_s = f"{dec:.3f}" if isinstance(dec, float) else dec
        print(f"{cfg:>12s} {rmse_s:>8s} {miou_s:>8s} {bits_s:>10s} {gflops:>8.2f} {enc_s:>8s} {dec_s:>8s}")

    # Save
    out_dir = Path(args.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ns = joint_ns(args)
    stem = joint_stem(ns)
    out_path = out_dir / f"{stem}_absorbed_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
