#!/usr/bin/env python
"""
Per-group classification-loss ablation (restore-one-group).

For each group g, replace quantised sub-vectors with the originals
(zero-out that group's quantisation error) and measure how much L_cls drops.

    ΔL_cls[g] = L_cls(all quantised) − L_cls(group g restored)

Positive ΔL_cls[g]  ⟹  group g matters for classification.

Three codec conditions:
  Identity  — no rotation, k-means codebooks
  OPQ       — OPQ rotation + codebooks, no gradient training
  Trained   — DOPQ checkpoint (L_ref-optimised)

Pseudo-labels = argmax(logits(X_orig)) from the teacher model.

Outputs JSON compatible with plot_fig2_cls.py.

Usage:
    python compute_cls_sensitivity.py --gpu 0
    python compute_cls_sensitivity.py --gpu 0 --codec_path checkpoints/dinov2_vitl14/blk20_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt
"""

import os, sys, argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

ORFC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

from run_multilayer_calibrator import set_seed, preload_features
from opq import (
    batch_normalize_gpu, batch_inv_normalize_gpu,
    batched_assign, learn_opq_rotation,
)
from backbone.wrapper import Dinov2Wrapper
from soft_pq import (
    SoftPQ, OrthogonalTransform, FeatureCodec,
    FrozenTail, train_soft_pq, load_codec,
)

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
import logging
from mmcv.utils import get_logger
get_logger('mmcv').setLevel(logging.WARNING)


# ================================================================
#     Helpers
# ================================================================

def collect_normalised(features, norm_mode, device, max_vecs=500_000):
    D = features[0].shape[1]
    all_vecs = []
    for s in range(0, len(features), 200):
        e = min(s + 200, len(features))
        X = torch.from_numpy(np.stack(features[s:e])).float().to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
        all_vecs.append(Y.reshape(-1, D).cpu())
        del X, Y
    flat = torch.cat(all_vecs, dim=0)
    if flat.shape[0] > max_vecs:
        idx = np.random.choice(flat.shape[0], max_vecs, replace=False)
        flat = flat[idx]
    return flat


def learn_opq(features_train, D, G, d, K, norm_mode, device, seed=42):
    flat = collect_normalised(features_train, norm_mode, device)
    max_flat = 2_000_000 // G
    if flat.shape[0] > max_flat:
        rng = np.random.RandomState(seed)
        flat = flat[rng.choice(flat.shape[0], max_flat, replace=False)]
    R, codebooks, _ = learn_opq_rotation(
        flat.numpy(), G, d, K,
        max_iter_opq=20, max_iter_kmeans=100,
        device=device, verbose=False)
    del flat

    R_t = torch.from_numpy(R).float().to(device)
    cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
    usage = np.zeros((G, K), dtype=np.float64)
    for s in range(0, len(features_train), 200):
        e = min(s + 200, len(features_train))
        X = torch.from_numpy(
            np.stack(features_train[s:e])).float().to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
            flat = Y.reshape(-1, D) @ R_t
            sub = flat.reshape(-1, G, d).permute(1, 0, 2).contiguous()
            labels = torch.cdist(sub, cb_t).argmin(dim=-1)
            for g in range(G):
                np.add.at(usage[g], labels[g].cpu().numpy(), 1)
        del X, Y, flat, sub, labels
    del R_t, cb_t
    torch.cuda.empty_cache()
    return R, codebooks, usage


def build_codec_identity(features_train, G, K, d, norm_mode, device):
    flat = collect_normalised(features_train, norm_mode, device)
    pq = SoftPQ(G, K, d, lmbda=0.5).to(device)
    pq.init_from_kmeans(flat, device=device)
    pq.temperature = 0.0
    codec = FeatureCodec(pq, transform=None).to(device).eval()
    del flat
    torch.cuda.empty_cache()
    return codec


def build_codec_opq(R, codebooks, G, K, d, D, device, usage=None):
    transform = OrthogonalTransform(D)
    R_ws = R.copy()
    C_ws = [c.copy() for c in codebooks]
    if np.linalg.det(R_ws) < 0:
        R_ws[:, -1] *= -1
        C_ws[-1][:, -1] *= -1
    transform.init_from_opq(R_ws)
    pq = SoftPQ(G, K, d, lmbda=0.5).to(device)
    pq.init_codebooks(C_ws)
    if usage is not None:
        pq.init_prior_from_freq(usage)
    pq.temperature = 0.0
    codec = FeatureCodec(pq, transform=transform).to(device).eval()
    return codec


# ================================================================
#     Classification-loss ablation (restore-one-group)
# ================================================================

def _logits_from_features(X_hat, tail, head):
    """X_hat [B,T,C] → logits [B, num_classes] (no grad)."""
    x = tail.forward_nograd(X_hat)
    cls_tok = x[:, 0]
    lin_in = torch.cat([cls_tok, x[:, 1:].mean(1)], dim=1)
    return head(lin_in)


def cls_ablation(codec, tail, head, features, norm_mode,
                 device, batch_size=8):
    """
    Restore-one-group ablation for L_cls.

    For each group g, replace Z_hat[:, g, :] with Z[:, g, :] (original)
    and compute ΔL_cls[g] = L_cls_base − L_cls_restored_g.

    Pseudo-labels = argmax(logits(X_orig)).
    """
    pq = codec.pq
    G, d = pq.G, pq.d
    D = G * d
    delta_L = np.zeros(G)
    n = 0
    codec.eval()

    with torch.no_grad():
        for s in range(0, len(features), batch_size):
            e = min(s + batch_size, len(features))
            X = torch.from_numpy(np.stack(features[s:e])).float().to(device)
            B, T, C = X.shape
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            flat = Y.reshape(B * T, C)

            teacher_logits = _logits_from_features(X, tail, head)
            labels = teacher_logits.argmax(dim=1)

            Z = codec.transform.encode(flat) if codec.transform else flat
            Z_hat, _ = pq._quantise(Z)

            Y_hat_base = (codec.transform.decode(Z_hat)
                          if codec.transform else Z_hat)
            X_hat_base = batch_inv_normalize_gpu(
                Y_hat_base.reshape(B, T, C), Mu, Std)
            logits_base = _logits_from_features(X_hat_base, tail, head)
            L_base = F.cross_entropy(logits_base, labels, reduction='sum')

            Z_g = Z.reshape(-1, G, d)
            Z_hat_g = Z_hat.reshape(-1, G, d)

            for g in range(G):
                oracle = Z_hat_g.clone()
                oracle[:, g, :] = Z_g[:, g, :]
                Y_hat_g = (codec.transform.decode(oracle.reshape(-1, D))
                           if codec.transform else oracle.reshape(-1, D))
                X_hat_g = batch_inv_normalize_gpu(
                    Y_hat_g.reshape(B, T, C), Mu, Std)
                logits_g = _logits_from_features(X_hat_g, tail, head)
                L_g = F.cross_entropy(logits_g, labels, reduction='sum')
                delta_L[g] += (L_base - L_g).item()

            n += 1
            del X, Y, Mu, Std, flat, Z, Z_hat, Y_hat_base, X_hat_base
            torch.cuda.empty_cache()

    return delta_L / max(n, 1)


def stats(arr):
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'cv': float(np.std(arr) / (np.mean(arr) + 1e-10)),
        'values': arr.tolist(),
    }


# ================================================================
#     Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Per-group sensitivity w.r.t. classification task loss")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--layer", type=str, default="blk20")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--norm_mode", type=str, default="per_image")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_diag", type=int, default=200)
    parser.add_argument("--max_train_images", type=int, default=5000)
    parser.add_argument("--codec_path", type=str, default="",
                        help="Trained codec checkpoint (skip training)")
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lmbda", type=float, default=0.5)
    parser.add_argument("--feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features"))
    parser.add_argument("--backbone", type=str, default="dinov2_vitl14")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    set_seed(args.seed)

    layer_idx = int(args.layer[-2:])
    D = 1024
    G = D // args.embedding_dim
    d = args.embedding_dim
    K = args.K
    norm_mode = args.norm_mode

    print(f"\n{'#' * 70}")
    print(f"# Classification-loss ablation (restore-one-group)")
    print(f"# layer={args.layer}, K={K}, G={G}, d={d}")
    print(f"# n_diag={args.n_diag}, max_train={args.max_train_images}")
    print(f"# {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'#' * 70}")

    # ── Load features ──
    feat_dir = Path(args.feat_root) / "train" / args.backbone / args.layer
    feat_files = sorted(feat_dir.glob("*.npy"))
    print(f"\nLoading train features: {len(feat_files)} files from {feat_dir}")
    features_all, _ = preload_features(feat_files, num_workers=8)

    if args.max_train_images > 0 and len(features_all) > args.max_train_images:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(features_all), args.max_train_images, replace=False)
        features_all = [features_all[i] for i in idx]
        print(f"Subsampled to {len(features_all)} train images")

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(features_all))
    diag_idx = perm[:args.n_diag]
    train_idx = perm[args.n_diag:]
    features_diag = [features_all[i] for i in diag_idx]
    features_tr = [features_all[i] for i in train_idx]
    del features_all
    print(f"Split: diagnostic={len(features_diag)}, training={len(features_tr)}")

    # ── Load DINOv2 ──
    print("Loading DINOv2 ViT-L/14 ...")
    wrapper = Dinov2Wrapper(head_layers=1, device=device)

    # ── Build Frozen-Tail (blocks + norm) ──
    wrapper.backbone.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()
    tail_blocks = list(wrapper.backbone.blocks[layer_idx + 1:])
    tail = FrozenTail(tail_blocks, wrapper.backbone.norm, device=device)

    # ── Classification head on GPU (frozen, but allows gradient flow) ──
    head = wrapper.head.to(device)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    print("Frozen-Tail + classification head ready\n")

    # ── Learn OPQ ──
    print(f"Learning OPQ (K={K}, G={G}, d={d}) ...")
    R_opq, codebooks_opq, opq_usage = learn_opq(
        features_tr, D, G, d, K, norm_mode, device, args.seed)
    print("OPQ done\n")

    all_results = {}

    # ── Condition 1: Identity ──
    print(f"{'=' * 60}")
    print("  Building Identity codec ...")
    t0 = time.time()
    codec_id = build_codec_identity(features_tr, G, K, d, norm_mode, device)
    abl_id = cls_ablation(
        codec_id, tail, head, features_diag, norm_mode, device,
        batch_size=args.batch_size)
    all_results['identity'] = {
        'condition': 'Identity (no rotation)',
        'ablation_cls': stats(abl_id),
    }
    print(f"  Identity CV={all_results['identity']['ablation_cls']['cv']:.4f}  "
          f"({time.time()-t0:.1f}s)")
    del codec_id
    torch.cuda.empty_cache()

    # ── Condition 2: OPQ ──
    print(f"\n{'=' * 60}")
    print("  Building OPQ codec ...")
    t0 = time.time()
    codec_opq = build_codec_opq(
        R_opq, codebooks_opq, G, K, d, D, device, opq_usage)
    abl_opq = cls_ablation(
        codec_opq, tail, head, features_diag, norm_mode, device,
        batch_size=args.batch_size)
    all_results['opq'] = {
        'condition': 'OPQ (untrained)',
        'ablation_cls': stats(abl_opq),
    }
    print(f"  OPQ CV={all_results['opq']['ablation_cls']['cv']:.4f}  "
          f"({time.time()-t0:.1f}s)")
    del codec_opq
    torch.cuda.empty_cache()

    # ── Condition 3: Trained (DOPQ) ──
    codec_trained = None
    if args.codec_path and os.path.exists(args.codec_path):
        print(f"\n{'=' * 60}")
        print(f"  Loading trained codec: {args.codec_path}")
        codec_trained = load_codec(args.codec_path, device=device)
        codec_trained.eval()
    elif args.train_epochs > 0:
        print(f"\n{'=' * 60}")
        print(f"  Training DOPQ codec ({args.train_epochs} epochs) ...")
        transform = OrthogonalTransform(D)
        R_ws = R_opq.copy()
        C_ws = [c.copy() for c in codebooks_opq]
        if np.linalg.det(R_ws) < 0:
            R_ws[:, -1] *= -1
            C_ws[-1][:, -1] *= -1
        n_val = min(50, len(features_tr))
        rng_v = np.random.RandomState(args.seed + 1)
        val_idx = rng_v.choice(len(features_tr), n_val, replace=False)
        val_feat = [features_tr[i] for i in val_idx]
        codec_trained, _ = train_soft_pq(
            features_train=features_tr, tail=tail,
            G=G, K=K, d=d, norm_mode=norm_mode,
            epochs=args.train_epochs, lr=args.lr,
            batch_size=32, device=device, seed=args.seed,
            val_features=val_feat, verbose=True,
            transform=transform, R_init=R_ws,
            codebooks_init=C_ws, lmbda=args.lmbda,
            prior_init_counts=opq_usage, grad_clip=1.0,
            tau_start=0.5, tau_end=0.005)

    if codec_trained is not None:
        t0 = time.time()
        label = ("Trained (ckpt)" if args.codec_path
                 else f"Trained ({args.train_epochs} ep)")
        abl_tr = cls_ablation(
            codec_trained, tail, head, features_diag, norm_mode, device,
            batch_size=args.batch_size)
        all_results['trained'] = {
            'condition': label,
            'ablation_cls': stats(abl_tr),
        }
        print(f"  Trained CV={all_results['trained']['ablation_cls']['cv']:.4f}  "
              f"({time.time()-t0:.1f}s)")
        del codec_trained
        torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("  Classification-loss ablation (restore-one-group) CV summary:")
    print(f"  {'Condition':<30} {'CV':>8} {'Mean':>12}")
    print(f"  {'─' * 52}")
    for key in all_results:
        r = all_results[key]
        s = r['ablation_cls']
        print(f"  {r['condition']:<30} {s['cv']:>8.4f} {s['mean']:>12.4f}")

    # ── Save ──
    out_dir = os.path.join(ORFC_ROOT, 'results', 'analysis_intro_v2')
    os.makedirs(out_dir, exist_ok=True)
    trained_tag = "ckpt" if args.codec_path else f"ep{args.train_epochs}"
    tag = (f"cls_sensitivity_{args.layer}_K{K}_emb{d}"
           f"_ndiag{args.n_diag}_{trained_tag}")
    out_path = os.path.join(out_dir, f'{tag}.json')
    output = {
        'config': {
            'metric': 'ablation_cls',
            'layer': args.layer, 'K': K, 'embedding_dim': d,
            'G': G, 'D': D,
            'n_diag': args.n_diag,
            'n_train': len(features_tr),
            'codec_path': args.codec_path or None,
            'train_epochs': args.train_epochs if not args.codec_path else 0,
            'norm_mode': norm_mode, 'seed': args.seed,
        },
        'conditions': all_results,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    return out_path


if __name__ == '__main__':
    json_path = main()
