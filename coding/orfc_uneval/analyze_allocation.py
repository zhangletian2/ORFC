#!/usr/bin/env python
"""
Analyze DP bit allocation extremity across min_bits values.

Runs OPQ -> cls_ablation importance -> DP allocation (Stage 1+2 only).
Optionally computes init MSE after k-means init (no training).

Usage:
    # Allocation analysis only (fast, ~1 min)
    python analyze_allocation.py --layer blk05 --K 16 --seed 42

    # With init MSE for seed comparison (~2-3 min)
    python analyze_allocation.py --layer blk05 --K 16 --seed 42 --eval_init_mse
"""

import os, sys, argparse, json, math, time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import Counter

UNEVAL_ROOT = os.path.dirname(os.path.abspath(__file__))
CODING_ROOT = os.path.normpath(os.path.join(UNEVAL_ROOT, ".."))
ORFC_ROOT = os.path.join(CODING_ROOT, "orfc")
VAQ_ROOT = os.path.join(CODING_ROOT, "vaq")
PROJECT_ROOT = os.path.normpath(os.path.join(CODING_ROOT, ".."))

for _p in (UNEVAL_ROOT, ORFC_ROOT, VAQ_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_multilayer_calibrator import set_seed, preload_features
from opq import (
    batch_normalize_gpu, batch_inv_normalize_gpu, batched_kmeans,
)
from compute_cls_sensitivity import (
    learn_opq, build_codec_opq, cls_ablation, collect_normalised, stats,
)
from allocate_bits import allocate_bits_importance

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
import logging
from mmcv.utils import get_logger
get_logger('mmcv').setLevel(logging.WARNING)


# ================================================================
#   K-means (inlined to avoid importing run_uneval_pq's heavy deps)
# ================================================================

def _kmeans_single_group(data_g, k, device, max_iter=50, seed=42):
    data_g = np.ascontiguousarray(data_g.astype(np.float32, copy=False))
    n, dim = data_g.shape
    if k <= 1:
        return data_g.mean(axis=0, keepdims=True).astype(np.float32)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, k, replace=n < k)
    centroids = torch.from_numpy(data_g[idx]).float().to(device)
    chunk = max(1, min(n, 768 * 1024**2 // max(k * 4, 1)))

    for _ in range(max_iter):
        sums = torch.zeros(k, dim, dtype=torch.float32, device=device)
        counts = torch.zeros(k, dtype=torch.float32, device=device)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            batch = torch.from_numpy(data_g[s:e]).float().to(device)
            with torch.no_grad():
                labels = torch.cdist(batch, centroids).argmin(dim=1)
            sums.scatter_add_(0, labels[:, None].expand(-1, dim), batch)
            counts.scatter_add_(0, labels,
                                torch.ones_like(labels, dtype=torch.float32))
            del batch, labels
        new_c = sums / counts.clamp_min(1.0)[:, None]
        empty = counts == 0
        if empty.any():
            ei = empty.nonzero(as_tuple=False).flatten()
            repl = rng.choice(n, int(ei.numel()), replace=n < int(ei.numel()))
            new_c[ei] = torch.from_numpy(data_g[repl]).float().to(device)
        shift = (new_c - centroids).norm(dim=1).max().item()
        centroids = new_c
        if shift < 1e-4:
            break
    return centroids.cpu().numpy().astype(np.float32)


# ================================================================
#   PCA rotation (consecutive, no interleave)
# ================================================================

def learn_pca_for_analysis(features_train, D, G, d, K, norm_mode, device,
                           seed=42):
    """PCA consecutive rotation + k-means codebooks (for cls_ablation)."""
    flat = collect_normalised(features_train, norm_mode, device)
    max_flat = 2_000_000 // G
    if flat.shape[0] > max_flat:
        rng = np.random.RandomState(seed)
        flat = flat[rng.choice(flat.shape[0], max_flat, replace=False)]

    X = flat.float().to(device)
    N = X.shape[0]

    with torch.no_grad():
        cov = (X.T @ X) / N
        eigvals, eigvecs = torch.linalg.eigh(cov)
        idx = torch.argsort(eigvals, descending=True)
        R = eigvecs[:, idx].contiguous()

    R_np = R.cpu().numpy().astype(np.float32)

    Z = X @ R
    sub_3d = Z.reshape(N, G, d).permute(1, 0, 2).contiguous()
    centroids_t = batched_kmeans(sub_3d, K, max_iter=100,
                                 device=device, verbose=False)
    codebooks = [centroids_t[g].cpu().numpy().astype(np.float32)
                 for g in range(G)]

    del X, Z, sub_3d, centroids_t, flat
    torch.cuda.empty_cache()

    R_t = torch.from_numpy(R_np).float().to(device)
    cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
    usage = np.zeros((G, K), dtype=np.float64)
    for s in range(0, len(features_train), 200):
        e = min(s + 200, len(features_train))
        Xb = torch.from_numpy(
            np.stack(features_train[s:e])).float().to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(Xb, mode=norm_mode)
            fl = Y.reshape(-1, D) @ R_t
            sub = fl.reshape(-1, G, d).permute(1, 0, 2).contiguous()
            labels = torch.cdist(sub, cb_t).argmin(dim=-1)
            for g in range(G):
                np.add.at(usage[g], labels[g].cpu().numpy(), 1)
    del R_t, cb_t
    torch.cuda.empty_cache()

    return R_np, codebooks, usage


# ================================================================
#   Init MSE computation (k-means init, no training)
# ================================================================

def compute_init_mse_batch(features_val, R_opq, alloc_dict, G, d,
                           norm_mode, device, seed=42):
    """Compute init MSE for multiple allocations, caching rotated vectors Z.

    alloc_dict: {min_bits: {'K_per_group': [...]}, ...}
    Returns dict {min_bits: mse}.
    """
    from soft_pq import OrthogonalTransform, FeatureCodec
    from vaq_soft import VAQSoftPQ

    D = G * d
    R_ws = R_opq.copy()
    if np.linalg.det(R_ws) < 0:
        R_ws[:, -1] *= -1

    flat = collect_normalised(features_val, norm_mode, device, max_vecs=500_000)
    R_t = torch.from_numpy(R_opq).float().to(device)
    Z = (flat.to(device) @ R_t).cpu().numpy()
    del flat, R_t
    torch.cuda.empty_cache()

    mse_results = {}
    for min_b, info in sorted(alloc_dict.items()):
        K_per_group = info['K_per_group']
        centroids = []
        for g in range(G):
            c_g = _kmeans_single_group(Z[:, g*d:(g+1)*d], K_per_group[g],
                                       device, max_iter=50, seed=seed+g)
            centroids.append(c_g)

        transform = OrthogonalTransform(D)
        transform.init_from_opq(R_ws)
        pq = VAQSoftPQ(K_per_group=K_per_group, d=d, lmbda=0.0)
        pq.init_from_centroids(centroids)
        codec = FeatureCodec(pq, transform=transform).to(device).eval()

        total_mse, n_tokens = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(features_val), 8):
                e = min(s + 8, len(features_val))
                X = torch.from_numpy(
                    np.stack(features_val[s:e])).float().to(device)
                Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
                Y_hat, _ = codec(Y)
                X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
                total_mse += ((X - X_hat) ** 2).sum().item()
                n_tokens += X.shape[0] * X.shape[1]
                del X, Y, Mu, Std, Y_hat, X_hat
        del codec, pq, transform
        torch.cuda.empty_cache()
        mse_results[min_b] = total_mse / n_tokens

    del Z
    return mse_results


# ================================================================
#   Allocation statistics
# ================================================================

def alloc_stats(bits_alloc, min_bits, max_bits):
    bits = np.array(bits_alloc)
    n_min = int((bits == min_bits).sum())
    n_max = int((bits == max_bits).sum())
    hist = dict(sorted(Counter(int(b) for b in bits).items()))
    return {
        'n_at_min': n_min,
        'n_at_max': n_max,
        'bits_std': round(float(bits.std()), 3),
        'bits_range': int(bits.max() - bits.min()),
        'unique_levels': len(np.unique(bits)),
        'histogram': hist,
    }


# ================================================================
#   Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze DP bit allocation extremity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--layer", type=str, required=True)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_bits_list", type=str, default="1,2,3,4",
                        help="Comma-separated min_bits values to sweep")
    parser.add_argument("--max_bits", type=int, default=10)
    parser.add_argument("--alloc_objective", type=str, default="rd")
    parser.add_argument("--rotation", type=str, default="opq",
                        choices=["opq", "pca"])
    parser.add_argument("--monotonic", action="store_true",
                        help="VAQ-style gaps constraint")
    parser.add_argument("--norm_mode", type=str, default="per_image")
    parser.add_argument("--max_train_images", type=int, default=5000)
    parser.add_argument("--n_diag", type=int, default=200)
    parser.add_argument("--ablation_batch_size", type=int, default=8)
    parser.add_argument("--feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features"))
    parser.add_argument("--backbone", type=str, default="dinov2_vitl14")
    parser.add_argument("--eval_init_mse", action="store_true",
                        help="Compute init MSE after k-means (slower)")
    args = parser.parse_args()

    min_bits_list = [int(x) for x in args.min_bits_list.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    layer_idx = int(args.layer[-2:])
    D = 1024
    G = D // args.embedding_dim
    d = args.embedding_dim
    K_ref = args.K
    bit_budget = int(G * math.log2(K_ref))

    print(f"\n{'=' * 70}")
    print(f"  Allocation Analysis")
    print(f"  layer={args.layer}  K_ref={K_ref}  e={d}  G={G}  seed={args.seed}")
    print(f"  rotation={args.rotation}  monotonic={args.monotonic}")
    print(f"  bit_budget={bit_budget}  max_bits={args.max_bits}")
    print(f"  min_bits_list={min_bits_list}")
    print(f"  eval_init_mse={args.eval_init_mse}")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'=' * 70}")

    # ---- Load features ----
    train_dir = (Path(args.feat_root) / "train" /
                 args.backbone / args.layer)
    train_files = sorted(train_dir.glob("*.npy"))
    print(f"\n  Loading {len(train_files)} training features ...")
    features_train, _ = preload_features(train_files, num_workers=8)

    if args.max_train_images > 0 and len(features_train) > args.max_train_images:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(features_train), args.max_train_images, replace=False)
        features_train = [features_train[i] for i in idx]

    rng_diag = np.random.RandomState(args.seed)
    perm = rng_diag.permutation(len(features_train))
    n_diag = min(args.n_diag, len(features_train) // 2)
    features_diag = [features_train[i] for i in perm[:n_diag]]
    features_tr = [features_train[i] for i in perm[n_diag:]]

    # Val split (for init MSE)
    n_val = min(200, len(features_tr))
    rng_val = np.random.RandomState(args.seed + 1)
    val_idx = rng_val.choice(len(features_tr), n_val, replace=False)
    features_val = [features_tr[i] for i in val_idx]

    # ---- Load backbone (for cls_ablation) ----
    from backbone.wrapper import Dinov2Wrapper
    from soft_pq import FrozenTail

    print(f"  Loading DINOv2 ({args.backbone}) ...")
    wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone,
                            device=device)
    wrapper.backbone.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()

    tail_blocks = list(wrapper.backbone.blocks[layer_idx + 1:])
    tail = FrozenTail(tail_blocks, wrapper.backbone.norm, device=device)
    head = wrapper.head.to(device)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)

    # ---- Stage 1: Rotation + importance ----
    if args.rotation == 'pca':
        print(f"\n  Learning PCA-consecutive (K={K_ref}, G={G}, d={d}) ...")
        t0 = time.time()
        R_opq, codebooks_opq, opq_usage = learn_pca_for_analysis(
            features_tr, D, G, d, K_ref, args.norm_mode, device, args.seed)
        print(f"  PCA done ({time.time() - t0:.1f}s)")
    else:
        print(f"\n  Learning OPQ (K={K_ref}, G={G}, d={d}) ...")
        t0 = time.time()
        R_opq, codebooks_opq, opq_usage = learn_opq(
            features_tr, D, G, d, K_ref, args.norm_mode, device, args.seed)
        print(f"  OPQ done ({time.time() - t0:.1f}s)")

    codec_opq = build_codec_opq(
        R_opq, codebooks_opq, G, K_ref, d, D, device, opq_usage)
    t0 = time.time()
    importance = cls_ablation(
        codec_opq, tail, head, features_diag, args.norm_mode, device,
        batch_size=args.ablation_batch_size)
    print(f"  cls_ablation done ({time.time() - t0:.1f}s)")

    imp_stats = stats(importance)
    print(f"  importance cv={imp_stats['cv']:.4f}  "
          f"min={imp_stats['min']:.4f}  max={imp_stats['max']:.4f}  "
          f"range={imp_stats['max'] - imp_stats['min']:.4f}")

    del codec_opq, tail
    head.cpu()
    del wrapper
    torch.cuda.empty_cache()

    # ---- Stage 2: Allocation for each min_bits ----
    allocations = {}
    uniform_bits = int(math.log2(K_ref))

    print(f"\n  {'minb':>4} | {'n_min':>5} {'n_max':>5} {'uniq':>4} | "
          f"{'std':>5} {'range':>5} | bits distribution")
    print(f"  {'-' * 75}")

    for min_b in min_bits_list:
        if bit_budget < G * min_b:
            print(f"  {min_b:>4} | INFEASIBLE  "
                  f"(budget {bit_budget} < {G}*{min_b}={G * min_b})")
            continue
        if bit_budget > G * args.max_bits:
            print(f"  {min_b:>4} | INFEASIBLE  "
                  f"(budget {bit_budget} > {G}*{args.max_bits})")
            continue

        bits_alloc = allocate_bits_importance(
            importance, bit_budget,
            min_bits=min_b, max_bits=args.max_bits,
            d=d, objective=args.alloc_objective,
            monotonic=args.monotonic,
        )
        K_per_group = [1 << b for b in bits_alloc]
        astats = alloc_stats(bits_alloc, min_b, args.max_bits)
        allocations[min_b] = {
            'bits_alloc': bits_alloc,
            'K_per_group': K_per_group,
            'stats': astats,
        }

        hist_str = " ".join(f"{k}bit:{v}" for k, v in astats['histogram'].items())
        print(f"  {min_b:>4} | {astats['n_at_min']:>5} {astats['n_at_max']:>5} "
              f"{astats['unique_levels']:>4} | {astats['bits_std']:>5.2f} "
              f"{astats['bits_range']:>5} | {hist_str}")

    # min_bits = uniform_bits is OPQ baseline (all groups = uniform_bits)
    if uniform_bits not in [a for a in allocations]:
        print(f"  {uniform_bits:>4} | {'(uniform OPQ baseline)':>55}")

    # ---- Optional: Init MSE ----
    if args.eval_init_mse and allocations:
        non_uniform = {k: v for k, v in allocations.items()
                       if k != uniform_bits}
        if non_uniform:
            print(f"\n  Computing init MSE (k-means, val={n_val} images) ...")
            t0 = time.time()
            mse_dict = compute_init_mse_batch(
                features_val, R_opq, non_uniform, G, d,
                args.norm_mode, device, seed=args.seed)
            for min_b, mse in sorted(mse_dict.items()):
                allocations[min_b]['init_mse'] = round(mse, 2)
                print(f"    min_bits={min_b}:  init_MSE={mse:.2f}")
            print(f"  Init MSE done ({time.time() - t0:.1f}s)")

    # ---- Save ----
    results = {
        'layer': args.layer, 'K_ref': K_ref, 'embedding_dim': d,
        'seed': args.seed, 'bit_budget': bit_budget,
        'rotation': args.rotation, 'monotonic': args.monotonic,
        'importance': imp_stats,
        'allocations': {str(k): v for k, v in allocations.items()},
    }
    out_dir = os.path.join(UNEVAL_ROOT, "results", "alloc_analysis")
    os.makedirs(out_dir, exist_ok=True)
    rot_tag = f"_{args.rotation}" if args.rotation != "opq" else ""
    mono_tag = "_mono" if args.monotonic else ""
    out_path = os.path.join(
        out_dir,
        f"{args.layer}_K{K_ref}_e{d}_s{args.seed}{rot_tag}{mono_tag}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
