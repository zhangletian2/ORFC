#!/usr/bin/env python
"""Subspace sensitivity diagnosis for Differentiable Orthogonal PQ.

Quantifies per-group (subspace) metrics under three transform conditions
(identity / OPQ / trained) to answer:

  Q1: Does L_ref provide differentiated signals across subspaces?
  Q2: Are L_ref signals different from MSE signals?
  Q3: Does the orthogonal rotation change the sensitivity profile?

Metrics per group g ∈ {0, ..., G-1}:
  - variance   : Var(z_g) in the rotated space
  - mse        : ||z_g − ẑ_g||² (quantization error)
  - sens_lref  : ||∂L_ref/∂C_g|| (task-loss gradient on codebooks)
  - sens_mse   : ||∂MSE/∂C_g||  (reconstruction-loss gradient)
  - ablation   : ΔL_ref when group g has zero quant error

Usage:
    python diagnose_subspace.py --gpu 4
    python diagnose_subspace.py --gpu 4 --train_epochs 20
"""

import os, sys, argparse, json, math, time
import numpy as np
import torch
from pathlib import Path

ORFC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

from run_multilayer_calibrator import set_seed, preload_features
from opq import batch_normalize_gpu, batch_inv_normalize_gpu, learn_opq_rotation
from backbone.wrapper import Dinov2Wrapper
from soft_pq import (
    SoftPQ, OrthogonalTransform, FeatureCodec,
    FrozenTail, train_soft_pq,
)

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
import logging
from mmcv.utils import get_logger
get_logger('mmcv').setLevel(logging.WARNING)


# ================================================================
#                    Codec builders
# ================================================================

def _collect_normalised(features, norm_mode, device, max_vecs=500_000):
    """Collect normalised flat vectors [N_total, D]."""
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


def build_codec_identity(features, G, K, d, norm_mode, device):
    """No rotation, k-means codebooks."""
    flat = _collect_normalised(features, norm_mode, device)
    pq = SoftPQ(G, K, d, lmbda=0.5).to(device)
    pq.init_from_kmeans(flat, device=device)
    pq.temperature = 0.0
    codec = FeatureCodec(pq, transform=None).to(device).eval()
    del flat
    torch.cuda.empty_cache()
    return codec


def build_codec_opq(R_std, codebooks_std, G, K, d, D, device,
                    opq_usage=None):
    """OPQ rotation + codebooks, no training."""
    transform = OrthogonalTransform(D)
    R_ws = R_std.copy()
    C_ws = [c.copy() for c in codebooks_std]
    if np.linalg.det(R_ws) < 0:
        R_ws[:, -1] *= -1
        C_ws[-1][:, -1] *= -1
    transform.init_from_opq(R_ws)
    pq = SoftPQ(G, K, d, lmbda=0.5).to(device)
    pq.init_codebooks(C_ws)
    if opq_usage is not None:
        pq.init_prior_from_freq(opq_usage)
    pq.temperature = 0.0
    codec = FeatureCodec(pq, transform=transform).to(device).eval()
    return codec


# ================================================================
#                    Per-group diagnostics
# ================================================================

def compute_variance(codec, features, norm_mode, device, batch_size=32):
    """Per-group feature variance in the (possibly rotated) space."""
    pq = codec.pq
    G, d = pq.G, pq.d
    D = G * d
    sum_sq = torch.zeros(G, device=device)
    sum_x = torch.zeros(G, device=device)
    n = 0
    codec.eval()
    with torch.no_grad():
        for s in range(0, len(features), batch_size):
            e = min(s + batch_size, len(features))
            X = torch.from_numpy(np.stack(features[s:e])).float().to(device)
            B, T, C = X.shape
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
            flat = Y.reshape(B * T, C)
            Z = codec.transform.encode(flat) if codec.transform else flat
            Z_g = Z.reshape(-1, G, d)  # [N, G, d]
            sum_sq += (Z_g ** 2).sum(dim=(0, 2))
            sum_x += Z_g.sum(dim=(0, 2))
            n += Z_g.shape[0] * d
            del X, Y, flat, Z
    var = (sum_sq / n) - (sum_x / n) ** 2
    return var.cpu().numpy()


def compute_mse(codec, features, norm_mode, device, batch_size=32):
    """Per-group quantization MSE."""
    pq = codec.pq
    G, d = pq.G, pq.d
    mse_sum = torch.zeros(G, device=device)
    n_tokens = 0
    codec.eval()
    with torch.no_grad():
        for s in range(0, len(features), batch_size):
            e = min(s + batch_size, len(features))
            X = torch.from_numpy(np.stack(features[s:e])).float().to(device)
            B, T, C = X.shape
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
            flat = Y.reshape(B * T, C)
            Z = codec.transform.encode(flat) if codec.transform else flat
            Z_hat, _ = pq._quantise(Z)
            Z_g = Z.reshape(-1, G, d)
            Z_hat_g = Z_hat.reshape(-1, G, d)
            mse_sum += ((Z_g - Z_hat_g) ** 2).sum(dim=(0, 2))
            n_tokens += B * T
            del X, Y, flat, Z, Z_hat
    return (mse_sum / n_tokens).cpu().numpy()


def compute_rate(codec, features, norm_mode, device, batch_size=32):
    """Per-group rate (bits/token) from learned prior."""
    pq = codec.pq
    G, K, d = pq.G, pq.K, pq.d
    rate_sum = torch.zeros(G, device=device)
    n_tokens = 0
    codec.eval()
    with torch.no_grad():
        for s in range(0, len(features), batch_size):
            e = min(s + batch_size, len(features))
            X = torch.from_numpy(np.stack(features[s:e])).float().to(device)
            B, T, C = X.shape
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
            Y_hat, _ = codec(Y)
            if pq._last_rate_per_group is not None:
                rate_sum += pq._last_rate_per_group * (B * T)
            n_tokens += B * T
            del X, Y, Y_hat
    if n_tokens == 0 or not pq.use_rate:
        return np.full(G, math.log2(K))
    return (rate_sum / n_tokens).cpu().numpy()


def _gradient_step(codec, tail, features, norm_mode, device, batch_size,
                   loss_type):
    """Accumulate per-group gradient norm for one loss type ('lref' or 'mse')."""
    pq = codec.pq
    G = pq.G
    grad_sum = torch.zeros(G, device=device)
    n = 0

    codec.train()
    old_tau = pq.temperature
    pq.temperature = 0.1

    for s in range(0, len(features), batch_size):
        e = min(s + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[s:e])).float().to(device)
        B, T, C = X.shape
        with torch.no_grad():
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
        Y_hat, _ = codec(Y)

        if loss_type == 'lref':
            with torch.no_grad():
                Y_teacher = tail.forward_nograd(X)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            F_hat = tail(X_hat)
            loss = ((Y_teacher - F_hat) ** 2).sum() / B
        else:
            loss = ((Y - Y_hat) ** 2).sum() / B

        grad_C = torch.autograd.grad(loss, pq.codebooks, retain_graph=False)[0]
        grad_sum += grad_C.reshape(G, -1).norm(dim=1).detach()
        n += 1
        del X, Y, Mu, Std, Y_hat, loss, grad_C
        torch.cuda.empty_cache()

    codec.eval()
    pq.temperature = old_tau
    return (grad_sum / max(n, 1)).cpu().numpy()


def compute_sensitivity(codec, tail, features, norm_mode, device,
                        batch_size=8):
    """Per-group gradient norm for L_ref and MSE losses."""
    sens_lref = _gradient_step(codec, tail, features, norm_mode, device,
                               batch_size, 'lref')
    sens_mse = _gradient_step(codec, tail, features, norm_mode, device,
                              batch_size, 'mse')
    return sens_lref, sens_mse


def compute_ablation(codec, tail, features, norm_mode, device, batch_size=8):
    """Per-group ablation: ΔL_ref when group g has zero quantization error."""
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
            Y_teacher = tail.forward_nograd(X)

            Z = codec.transform.encode(flat) if codec.transform else flat
            Z_hat, _ = pq._quantise(Z)

            # baseline
            Y_hat_base = (codec.transform.decode(Z_hat) if codec.transform
                          else Z_hat)
            X_hat_base = batch_inv_normalize_gpu(
                Y_hat_base.reshape(B, T, C), Mu, Std)
            L_base = ((Y_teacher -
                       tail.forward_nograd(X_hat_base)) ** 2).sum() / B

            Z_g = Z.reshape(-1, G, d)
            Z_hat_g = Z_hat.reshape(-1, G, d)

            for g in range(G):
                oracle = Z_hat_g.clone()
                oracle[:, g, :] = Z_g[:, g, :]
                Y_hat_g = (codec.transform.decode(oracle.reshape(-1, D))
                           if codec.transform else oracle.reshape(-1, D))
                X_hat_g = batch_inv_normalize_gpu(
                    Y_hat_g.reshape(B, T, C), Mu, Std)
                L_g = ((Y_teacher -
                        tail.forward_nograd(X_hat_g)) ** 2).sum() / B
                delta_L[g] += (L_base - L_g).item()

            n += 1
            del X, Y, Mu, Std, flat, Z, Z_hat, Y_hat_base, X_hat_base
            torch.cuda.empty_cache()
    return delta_L / max(n, 1)


# ================================================================
#                    Orchestration
# ================================================================

def _stats(arr):
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'cv': float(np.std(arr) / (np.mean(arr) + 1e-10)),
        'values': arr.tolist(),
    }


def run_diagnostics(codec, tail, features, norm_mode, device, name,
                    batch_size=8):
    """Run all diagnostics for one transform condition."""
    print(f"\n{'─' * 55}")
    print(f"  {name}")
    print(f"{'─' * 55}")

    t0 = time.time()

    print("  [1/5] Variance ...", end="", flush=True)
    var = compute_variance(codec, features, norm_mode, device, batch_size=32)
    print(f" {time.time()-t0:.1f}s")

    t1 = time.time()
    print("  [2/5] MSE ...", end="", flush=True)
    mse = compute_mse(codec, features, norm_mode, device, batch_size=32)
    print(f" {time.time()-t1:.1f}s")

    t2 = time.time()
    print("  [3/5] Rate ...", end="", flush=True)
    rate = compute_rate(codec, features, norm_mode, device, batch_size=32)
    print(f" {time.time()-t2:.1f}s")

    t3 = time.time()
    print("  [4/5] Sensitivity (L_ref + MSE gradients) ...", end="", flush=True)
    sens_lref, sens_mse = compute_sensitivity(
        codec, tail, features, norm_mode, device, batch_size=batch_size)
    print(f" {time.time()-t3:.1f}s")

    t4 = time.time()
    print("  [5/5] Ablation ΔL_ref ...", end="", flush=True)
    ablation = compute_ablation(
        codec, tail, features, norm_mode, device, batch_size=batch_size)
    print(f" {time.time()-t4:.1f}s")

    res = {
        'condition': name,
        'variance': _stats(var),
        'mse': _stats(mse),
        'rate': _stats(rate),
        'sens_lref': _stats(sens_lref),
        'sens_mse': _stats(sens_mse),
        'ablation': _stats(ablation),
    }

    pairs = [
        ('sens_lref', 'sens_mse', sens_lref, sens_mse),
        ('sens_lref', 'mse', sens_lref, mse),
        ('sens_lref', 'ablation', sens_lref, ablation),
        ('mse', 'ablation', mse, ablation),
        ('variance', 'mse', var, mse),
        ('variance', 'sens_lref', var, sens_lref),
    ]
    res['correlations'] = {}
    for n1, n2, a, b in pairs:
        key = f"{n1}_vs_{n2}"
        if np.std(a) < 1e-15 or np.std(b) < 1e-15:
            res['correlations'][key] = 0.0
        else:
            res['correlations'][key] = float(np.corrcoef(a, b)[0, 1])

    # ── Print table ──
    metrics = [
        ('Variance', 'variance'),
        ('MSE', 'mse'),
        ('Rate (b/g)', 'rate'),
        ('Sens L_ref', 'sens_lref'),
        ('Sens MSE', 'sens_mse'),
        ('Ablation ΔL', 'ablation'),
    ]
    print(f"\n  {'Metric':<14} {'Mean':>12} {'Std':>12} {'CV':>8} "
          f"{'Min':>12} {'Max':>12}")
    print(f"  {'─' * 70}")
    for label, key in metrics:
        s = res[key]
        print(f"  {label:<14} {s['mean']:>12.4f} {s['std']:>12.4f} "
              f"{s['cv']:>8.3f} {s['min']:>12.4f} {s['max']:>12.4f}")

    print(f"\n  Correlations:")
    for key, val in res['correlations'].items():
        a_name, b_name = key.split('_vs_')
        print(f"    {a_name:<12} ↔ {b_name:<12}  r = {val:+.4f}")

    print(f"  Total: {time.time()-t0:.1f}s")
    return res


# ================================================================
#                    Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Subspace sensitivity diagnosis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--n_diag", type=int, default=200,
                        help="Images for diagnostic evaluation")
    parser.add_argument("--n_train", type=int, default=2000,
                        help="Images for OPQ / optional training")
    parser.add_argument("--train_epochs", type=int, default=0,
                        help="Epochs for trained condition (0 = skip)")
    parser.add_argument("--layer", type=str, default="blk20")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features"))
    parser.add_argument("--backbone", type=str, default="dinov2_vitl14")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    set_seed(args.seed)

    D = 1024
    G = D // args.embedding_dim
    d = args.embedding_dim
    K = args.K
    norm_mode = "per_image"
    layer_idx = int(args.layer[-2:])

    print(f"\n{'#' * 60}")
    print(f"# Subspace Sensitivity Diagnosis")
    print(f"# layer={args.layer}, K={K}, G={G}, d={d}, D={D}")
    print(f"# n_diag={args.n_diag}, n_train={args.n_train}")
    print(f"# train_epochs={args.train_epochs}")
    print(f"{'#' * 60}")

    # ── Load features ──
    train_dir = (Path(args.feat_root) / "train" / args.backbone / args.layer)
    train_files = sorted(train_dir.glob("*.npy"))
    print(f"\nLoading features ({len(train_files)} files) ...")
    features_all, _ = preload_features(train_files, num_workers=16)

    rng = np.random.RandomState(args.seed)
    n_total = len(features_all)
    perm = rng.permutation(n_total)
    diag_idx = perm[:args.n_diag]
    train_idx = perm[args.n_diag:args.n_diag + args.n_train]
    features_diag = [features_all[i] for i in diag_idx]
    features_train = [features_all[i] for i in train_idx]
    print(f"  Diagnostic: {len(features_diag)}, "
          f"Training: {len(features_train)}")

    # ── DINOv2 tail ──
    print("Loading DINOv2 ...")
    dino_wrapper = Dinov2Wrapper(head_layers=1, device=device)
    tail_blocks = list(dino_wrapper.backbone.blocks[layer_idx + 1:])
    norm_layer = dino_wrapper.backbone.norm
    tail = FrozenTail(tail_blocks, norm_layer, device=device)
    for i, blk in enumerate(dino_wrapper.backbone.blocks):
        if i <= layer_idx:
            blk.cpu()
    if dino_wrapper.head is not None:
        dino_wrapper.head.cpu()
    torch.cuda.empty_cache()

    # ── Learn OPQ ──
    print("Learning OPQ rotation ...")
    flat_train = _collect_normalised(features_train, norm_mode, device)
    R_std, codebooks_std, _ = learn_opq_rotation(
        flat_train.numpy(), G, d, K,
        max_iter_opq=20, max_iter_kmeans=100,
        device=device, verbose=False)
    del flat_train

    R_t = torch.from_numpy(R_std).float().to(device)
    cb_t = torch.from_numpy(np.stack(codebooks_std)).float().to(device)
    opq_usage = np.zeros((G, K), dtype=np.float64)
    for s in range(0, len(features_train), 200):
        e = min(s + 200, len(features_train))
        X = torch.from_numpy(
            np.stack(features_train[s:e])).float().to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
            flat = Y.reshape(-1, D) @ R_t
            sub = flat.reshape(-1, G, d).permute(1, 0, 2).contiguous()
            labels = torch.cdist(sub, cb_t).argmin(dim=-1)  # [G, N]
            for g in range(G):
                np.add.at(opq_usage[g], labels[g].cpu().numpy(), 1)
        del X, Y, flat, sub, labels
    del R_t, cb_t
    torch.cuda.empty_cache()
    print(f"  OPQ done: G={G}, K={K}, d={d}")

    all_results = {}

    # ═══════ Condition 1: Identity ═══════
    print(f"\n{'═' * 55}")
    print("  Building Identity codec (no rotation, k-means) ...")
    codec_id = build_codec_identity(
        features_train, G, K, d, norm_mode, device)
    all_results['identity'] = run_diagnostics(
        codec_id, tail, features_diag, norm_mode, device,
        "Identity (no rotation)", batch_size=args.batch_size)
    del codec_id
    torch.cuda.empty_cache()

    # ═══════ Condition 2: OPQ ═══════
    print(f"\n{'═' * 55}")
    print("  Building OPQ codec (rotation + codebooks, untrained) ...")
    codec_opq = build_codec_opq(
        R_std, codebooks_std, G, K, d, D, device, opq_usage)
    all_results['opq'] = run_diagnostics(
        codec_opq, tail, features_diag, norm_mode, device,
        "OPQ (untrained)", batch_size=args.batch_size)

    # ═══════ Condition 3: Trained (optional) ═══════
    if args.train_epochs > 0:
        print(f"\n{'═' * 55}")
        print(f"  Training codec ({args.train_epochs} epochs, "
              f"lr=3e-4, λ=0.5, τ=0.5→0.005) ...")
        t_transform = OrthogonalTransform(D)
        R_ws = R_std.copy()
        C_ws = [c.copy() for c in codebooks_std]
        if np.linalg.det(R_ws) < 0:
            R_ws[:, -1] *= -1
            C_ws[-1][:, -1] *= -1

        n_val = min(50, len(features_train))
        rng_v = np.random.RandomState(args.seed + 1)
        val_idx = rng_v.choice(len(features_train), n_val, replace=False)
        val_feat = [features_train[i] for i in val_idx]

        codec_trained, _ = train_soft_pq(
            features_train=features_train, tail=tail,
            G=G, K=K, d=d, norm_mode=norm_mode,
            epochs=args.train_epochs, lr=3e-4,
            batch_size=32, device=device, seed=args.seed,
            val_features=val_feat, verbose=True,
            transform=t_transform, R_init=R_ws,
            codebooks_init=C_ws, lmbda=0.5,
            prior_init_counts=opq_usage, grad_clip=1.0,
            tau_start=0.5, tau_end=0.005)

        all_results['trained'] = run_diagnostics(
            codec_trained, tail, features_diag, norm_mode, device,
            f"Trained ({args.train_epochs} ep)", batch_size=args.batch_size)
        del codec_trained
        torch.cuda.empty_cache()

    del codec_opq
    torch.cuda.empty_cache()

    # ═══════ Summary ═══════
    conds = list(all_results.keys())
    print(f"\n{'═' * 70}")
    print("  SUMMARY: Coefficient of Variation (CV) across groups")
    print(f"{'═' * 70}")
    header = (f"  {'Condition':<25} {'Var':>7} {'MSE':>7} {'Rate':>7} "
              f"{'S_lref':>7} {'S_mse':>7} {'Ablat':>7}")
    print(header)
    print(f"  {'─' * 64}")
    for c in conds:
        r = all_results[c]
        print(f"  {r['condition']:<25} "
              f"{r['variance']['cv']:>7.3f} "
              f"{r['mse']['cv']:>7.3f} "
              f"{r['rate']['cv']:>7.3f} "
              f"{r['sens_lref']['cv']:>7.3f} "
              f"{r['sens_mse']['cv']:>7.3f} "
              f"{r['ablation']['cv']:>7.3f}")

    print(f"\n  Key Correlations")
    print(f"  {'─' * 64}")
    print(f"  {'Condition':<25} {'Sl↔Sm':>8} {'Sl↔MSE':>8} "
          f"{'Sl↔Abl':>8} {'Var↔Sl':>8}")
    print(f"  {'─' * 64}")
    for c in conds:
        co = all_results[c]['correlations']
        print(f"  {all_results[c]['condition']:<25} "
              f"{co['sens_lref_vs_sens_mse']:>+8.3f} "
              f"{co['sens_lref_vs_mse']:>+8.3f} "
              f"{co['sens_lref_vs_ablation']:>+8.3f} "
              f"{co['variance_vs_sens_lref']:>+8.3f}")

    # ── Interpretation guide ──
    print(f"\n  Interpretation:")
    id_r = all_results.get('identity', {})
    opq_r = all_results.get('opq', {})
    if id_r and opq_r:
        var_cv_id = id_r['variance']['cv']
        var_cv_opq = opq_r['variance']['cv']
        sl_cv_id = id_r['sens_lref']['cv']
        sl_cv_opq = opq_r['sens_lref']['cv']
        print(f"    Variance CV: identity={var_cv_id:.3f} → "
              f"OPQ={var_cv_opq:.3f} "
              f"({'equalised' if var_cv_opq < var_cv_id * 0.7 else 'similar'})")
        print(f"    L_ref sens CV: identity={sl_cv_id:.3f} → "
              f"OPQ={sl_cv_opq:.3f} "
              f"({'equalised' if sl_cv_opq < sl_cv_id * 0.7 else 'still non-uniform'})")
        corr_sl_sm = opq_r['correlations'].get('sens_lref_vs_sens_mse', 0)
        if abs(corr_sl_sm) < 0.7:
            print(f"    L_ref vs MSE gradient: r={corr_sl_sm:+.3f} → "
                  f"L_ref provides DIFFERENT signal from MSE")
        else:
            print(f"    L_ref vs MSE gradient: r={corr_sl_sm:+.3f} → "
                  f"signals are similar")

    # ── Save ──
    out_dir = os.path.join(ORFC_ROOT, "results", "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    tag = (f"subspace_{args.layer}_K{K}_emb{d}"
           f"_ndiag{args.n_diag}_ntrain{args.n_train}")
    if args.train_epochs > 0:
        tag += f"_ep{args.train_epochs}"
    out_path = os.path.join(out_dir, f"{tag}.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == '__main__':
    main()
