#!/usr/bin/env python
"""DINOv3 ViT-L/16 ImageNet Soft-PQ (ΔL_ref) — ORFC checkpoint layout.

Train on ``features/train/dinov3_vitl16/blkXX`` (T=201 = 5 prefix + 14×14).
Default: split_reg_cls_patch (reg own μ/σ; CLS shares with patch), n_prefix=5,
λ=0.5, e32, ep100, lr=3e-4, OPQ warm-start, OrthogonalTransform.

    python run_soft_pq_dinov3.py --layer blk05 --K 4 --gpu 2
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

ORFC_ROOT = os.path.dirname(os.path.abspath(__file__))
# coding/orfc → ORFC repo root (features/ is a symlink into featcodec/features)
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))
COFAI_ROOT = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "CoFAI"))
for p in (ORFC_ROOT, COFAI_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PROJECT_ROOT", COFAI_ROOT)

from run_multilayer_calibrator import preload_features, set_seed
from opq import batch_normalize_gpu, batched_assign, learn_opq_rotation
from soft_pq import (
    OrthogonalTransform,
    save_codec,
    train_soft_pq,
)
from backbone.dinov3_tail import build_dinov3_tail
from cofai.backbone.timm import Dinov3TimmBackbone

DEFAULT_CKPT = os.path.join(
    COFAI_ROOT, "weights", "dinov3", "backbone",
    "dinov3_vitl16_pretrain_lvd1689m.safetensors",
)


def _gpu_mem_mb(device):
    if device.type != "cuda":
        return 0.0, 0.0
    alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return alloc, peak


def _log_mem(tag, device):
    alloc, peak = _gpu_mem_mb(device)
    print(f"  [mem] {tag}: alloc={alloc:.0f} MiB  peak={peak:.0f} MiB")


def infer_token_hw(T, n_prefix):
    n_patch = int(T) - int(n_prefix)
    if n_patch <= 0:
        raise ValueError(f"T={T} n_prefix={n_prefix}: no patch tokens")
    side = int(round(n_patch ** 0.5))
    if side * side != n_patch:
        raise ValueError(
            f"T={T} n_prefix={n_prefix} → {n_patch} patches, not square"
        )
    return (side, side)


def ckpt_stem(args):
    bt_tag = f"bt{args.bottleneck_dim}" if args.bottleneck_dim > 0 else "noBt"
    ws_tag = "ws" if args.warm_start_opq else "km"
    norm_tag = "" if args.norm_mode == "per_image" else f"_{args.norm_mode}"
    return (
        f"{args.layer}_K{args.K}_emb{args.embedding_dim}"
        f"_{bt_tag}_{ws_tag}_lmbda{args.lmbda}{norm_tag}"
        f"_tau{args.tau_start}_lr{args.lr}_ep{args.epochs}"
        f"_n{args.max_train_images}_s{args.seed}"
    )


def _log_norm_groups(features, mode, n_prefix, device, n_img=8):
    """Sanity: after split_reg_cls_patch, patch RMS must not collapse."""
    n = min(n_img, len(features))
    X = torch.from_numpy(np.stack(features[:n])).float().to(device)
    with torch.no_grad():
        Y, _, std = batch_normalize_gpu(X, mode=mode, n_prefix=n_prefix)

    def rms(t):
        return t.float().pow(2).mean().sqrt().item()

    pfx = int(n_prefix)
    print(
        f"  post-norm RMS  cls={rms(Y[:, :1]):.4f}  "
        f"reg={rms(Y[:, 1:pfx]):.4f}  patch={rms(Y[:, pfx:]):.4f}  "
        f"σ_reg={std[:, 1:pfx].mean().item():.1f}  "
        f"σ_cp={std[:, pfx:].mean().item():.3f}"
    )
    del X, Y, std


def run_experiment(args):
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    set_seed(args.seed)
    layer_idx = int(args.layer[-2:])

    print(f"\n{'#' * 70}")
    print("# DINOv3 ViT-L/16 Soft-PQ  (ImageNet 5k, ΔL_ref)")
    print(f"# layer={args.layer} (idx={layer_idx})  K={args.K}  e={args.embedding_dim}")
    print(f"# norm={args.norm_mode}  n_prefix={args.n_prefix}  λ={args.lmbda}")
    print(f"# epochs={args.epochs}  lr={args.lr}  bs={args.batch_size}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    train_dir = Path(args.feat_root) / args.train_subset / args.backbone / args.layer
    train_files = sorted(train_dir.glob("*.npy"))
    if not train_files:
        raise FileNotFoundError(f"No train features in {train_dir}")
    print(f"\nData: {len(train_files)} npy  dir={train_dir}")
    features_train, _ = preload_features(train_files, num_workers=4)

    D = int(features_train[0].shape[1])
    T = int(features_train[0].shape[0])
    token_hw = infer_token_hw(T, args.n_prefix)
    bt_dim = args.bottleneck_dim
    Dp = bt_dim if bt_dim > 0 else D
    G = Dp // args.embedding_dim
    print(f"  D={D} T={T} token_hw={token_hw}  G={G}  "
          f"bits/token={G * math.log2(args.K):.0f}")

    if args.max_train_images > 0 and len(features_train) > args.max_train_images:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(features_train), args.max_train_images, replace=False)
        features_train = [features_train[i] for i in idx]
        print(f"  subset max_train={len(features_train)}")

    _log_norm_groups(features_train, args.norm_mode, args.n_prefix, device)

    n_val = min(args.n_val, len(features_train))
    rng_val = np.random.RandomState(args.seed + 1)
    val_idx = rng_val.choice(len(features_train), n_val, replace=False)
    val_features = [features_train[i] for i in val_idx]
    train_sub = features_train
    print(f"  train={len(train_sub)}  val={len(val_features)} (val overlaps train, ORFC style)")
    del features_train

    print(f"\nLoading DINOv3-L/16 backbone (tail from blk{layer_idx + 1})...")
    t_bb = time.time()
    backbone = Dinov3TimmBackbone(
        model_size="large",
        img_size=224,
        patch_size=16,
        dynamic_size=False,
        slot=24,
        n_last_blocks=1,
        pretrained=False,
        ckpt_path=args.backbone_ckpt,
        device=str(device),
        cast_dtype="float32",
    ).eval()
    backbone.model.to(device)
    tail = build_dinov3_tail(backbone, layer_idx, token_hw, device)
    n_tail = len(getattr(tail, "blocks", []))
    print(f"  tail blocks={n_tail} (+ norm)  load={time.time() - t_bb:.1f}s")
    for i, blk in enumerate(backbone.model.blocks):
        if i <= layer_idx:
            blk.cpu()
    del backbone
    torch.cuda.empty_cache()
    _log_mem("after tail load", device)

    print(f"\n{'=' * 60}\n  [OPQ warm-start]\n{'=' * 60}")
    t0 = time.time()
    all_vectors = []
    for start in range(0, len(train_sub), 200):
        end = min(start + 200, len(train_sub))
        X = torch.from_numpy(np.stack(train_sub[start:end])).float().to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(
                X, mode=args.norm_mode, n_prefix=args.n_prefix,
            )
        all_vectors.append(Y.reshape(-1, D).cpu().numpy())
        del X, Y
    full_vectors = np.concatenate(all_vectors, axis=0)
    del all_vectors
    max_flat = max(args.kmeans_max_samples // G, 1)
    if full_vectors.shape[0] > max_flat:
        rng2 = np.random.RandomState(args.seed)
        full_vectors = full_vectors[
            rng2.choice(full_vectors.shape[0], max_flat, replace=False)
        ]
    print(f"  OPQ sample: {full_vectors.shape[0]} vectors")
    R_std, codebooks_std, hist_std = learn_opq_rotation(
        full_vectors, G, args.embedding_dim, args.K,
        max_iter_opq=20, max_iter_kmeans=100,
        device=device, verbose=False,
    )
    del full_vectors
    torch.cuda.empty_cache()
    print(f"  OPQ done: MSE={hist_std[-1][0]:.8f} ({time.time() - t0:.1f}s)")

    opq_usage = None
    if args.lmbda > 0 and args.warm_start_opq:
        R_t = torch.from_numpy(R_std).float().to(device)
        cb_t = torch.from_numpy(np.stack(codebooks_std)).float().to(device)
        opq_usage = np.zeros((G, args.K), dtype=np.float64)
        for start in range(0, len(train_sub), 32):
            end = min(start + 32, len(train_sub))
            X = torch.from_numpy(np.stack(train_sub[start:end])).float().to(device)
            with torch.no_grad():
                Y, _, _ = batch_normalize_gpu(
                    X, mode=args.norm_mode, n_prefix=args.n_prefix,
                )
                flat = Y.reshape(-1, D) @ R_t
                sub = flat.reshape(-1, G, args.embedding_dim).permute(1, 0, 2).contiguous()
                labels = batched_assign(sub, cb_t, device=device)[1].cpu().numpy()
                for g in range(G):
                    np.add.at(opq_usage[g], labels[g], 1)
            del X, Y
        del R_t, cb_t
        torch.cuda.empty_cache()

    R_ws = R_std.copy()
    C_ws = [c.copy() for c in codebooks_std]
    if np.linalg.det(R_ws) < 0:
        R_ws[:, -1] *= -1
        C_ws[-1][:, -1] *= -1
        print("  det(R)<0: flipped last col")
    transform = OrthogonalTransform(D) if bt_dim == D else None
    if bt_dim > 0 and bt_dim != D:
        raise SystemExit("this runner only supports OrthogonalTransform (bt==D)")

    print(f"\n{'=' * 60}")
    print(f"  [SoftPQ] ΔL_ref  epochs={args.epochs}  λ={args.lmbda}")
    print(f"{'=' * 60}")
    _log_mem("before train_soft_pq", device)
    t_spq = time.time()
    codec, history = train_soft_pq(
        features_train=train_sub,
        tail=tail,
        G=G,
        K=args.K,
        d=args.embedding_dim,
        norm_mode=args.norm_mode,
        n_prefix=args.n_prefix,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        val_features=val_features,
        transform=transform,
        R_init=R_ws,
        codebooks_init=C_ws,
        kmeans_max_samples=args.kmeans_max_samples,
        use_mse_loss=False,
        lmbda=args.lmbda,
        prior_init_counts=opq_usage,
        grad_clip=args.grad_clip,
        prior_floor=args.prior_floor,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        tau_schedule=args.tau_schedule,
    )
    train_s = time.time() - t_spq
    _log_mem("after train_soft_pq", device)
    print(f"  train_soft_pq wall={train_s:.1f}s")
    if history:
        ep0 = history[0]
        print(
            f"  ep0 D={ep0['loss_distortion']:.1f}  "
            f"R={ep0.get('rate_bits', 0):.2f}b/t  "
            f"val={ep0.get('val_loss')}  "
            f"time={ep0['time']:.1f}s"
        )

    if not args.no_save:
        codec_dir = os.path.join(ORFC_ROOT, "checkpoints", args.backbone)
        os.makedirs(codec_dir, exist_ok=True)
        stem = ckpt_stem(args)
        ckpt_path = os.path.join(codec_dir, f"{stem}.pt")
        save_codec(codec, ckpt_path)
        meta_path = os.path.join(codec_dir, f"{stem}.json")
        with open(meta_path, "w") as f:
            json.dump({
                "layer": args.layer,
                "K": args.K,
                "embedding_dim": args.embedding_dim,
                "norm_mode": args.norm_mode,
                "n_prefix": args.n_prefix,
                "lmbda": args.lmbda,
                "epochs": args.epochs,
                "lr": args.lr,
                "token_hw": list(token_hw),
                "T": T,
                "D": D,
                "history": history,
                "train_wall_s": train_s,
            }, f, indent=2, default=str)
        print(f"  saved {ckpt_path}")

    tail.to("cpu")
    del tail, codec
    torch.cuda.empty_cache()
    _, peak = _gpu_mem_mb(device)
    print(f"  peak GPU memory = {peak:.0f} MiB")
    return history


def main():
    p = argparse.ArgumentParser(description="DINOv3 ImageNet Soft-PQ")
    p.add_argument("--layer", type=str, default="blk05")
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--embedding_dim", type=int, default=32)
    p.add_argument(
        "--norm_mode", type=str, default="split_reg_cls_patch",
        choices=[
            "per_image", "per_token_ln",
            "split_cls_patch", "split_reg_cls_patch",
        ],
    )
    p.add_argument("--n_prefix", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lmbda", type=float, default=0.5)
    p.add_argument("--bottleneck_dim", type=int, default=1024)
    p.add_argument("--warm_start_opq", action="store_true", default=True)
    p.add_argument("--no_warm_start", dest="warm_start_opq", action="store_false")
    p.add_argument("--tau_start", type=float, default=0.5)
    p.add_argument("--tau_end", type=float, default=0.005)
    p.add_argument("--tau_schedule", type=str, default="exponential")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--prior_floor", type=float, default=0.0)
    p.add_argument("--max_train_images", type=int, default=5000)
    p.add_argument("--kmeans_max_samples", type=int, default=2_000_000)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--feat_root", type=str,
                   default=os.path.join(PROJECT_ROOT, "features"))
    p.add_argument("--train_subset", type=str, default="train")
    p.add_argument("--backbone", type=str, default="dinov3_vitl16")
    p.add_argument("--backbone_ckpt", type=str, default=DEFAULT_CKPT)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_save", action="store_true")
    args = p.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
