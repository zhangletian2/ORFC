#!/usr/bin/env python
"""
Unified intro-figure analysis for the DOPQ paper.

Three analysis modes producing data for three paper figures:

  Figure 1 — correlation (mode=correlation):
    Scatter/bar of (MSE, Acc) vs (L_ref, Acc) across multiple OPQ operating
    points + DOPQ comparisons.  Shows L_ref predicts task performance better
    than MSE; MSE-optimised codec is worse than L_ref-optimised.

  Figure 2 — sensitivity balance (mode=sensitivity):
    Per-group gradient sensitivity ||dL_ref/dC_g|| under three conditions
    (Identity / OPQ / Trained).  CV of sens_lref: Identity >> OPQ > Trained,
    demonstrating that DOPQ training equalises marginal task-utility.

  Figure 3 — signal divergence (mode=sensitivity, same data):
    Cross-metric correlations produced alongside Figure 2:
      - r(sens_lref, sens_mse) ~ 0 or negative
        -> MSE and L_ref give DIFFERENT optimisation signals.
      - r(mse, ablation) < 0 under Trained
        -> system learned to sacrifice precision on task-unimportant groups.

Usage:
    python analysis_intro_v2.py --mode sensitivity --layer blk20  # Fig 2+3
    python analysis_intro_v2.py --mode correlation --layer blk20  # Fig 1
    python analysis_intro_v2.py --mode all         --layer blk20  # All
"""

import os, sys, argparse, json, math, time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

ORFC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

from run_multilayer_calibrator import (
    set_seed, preload_features, load_gt, evaluate_accuracy,
)
from opq import (
    batch_normalize_gpu, batch_inv_normalize_gpu,
    batched_assign, learn_opq_rotation,
)
from backbone.wrapper import Dinov2Wrapper, SegmentationEvaluator
from soft_pq import (
    SoftPQ, OrthogonalTransform, FeatureCodec,
    FrozenTail, train_soft_pq, load_codec,
    soft_pq_encode_decode,
)

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
import logging
from mmcv.utils import get_logger
get_logger('mmcv').setLevel(logging.WARNING)


# ================================================================
#     Data loading
# ================================================================

def load_data(args, device, need_test=True):
    """Load features and DINOv2 wrapper.

    Returns:
        features_train, features_test, basenames_test, gt_test,
        wrapper, D, layer_idx
    (test-related outputs are None when need_test=False)
    """
    layer_idx = int(args.layer[-2:])
    feat_root = Path(args.feat_root)
    backbone = args.backbone

    train_dir = feat_root / "train" / backbone / args.layer
    train_files = sorted(train_dir.glob("*.npy"))
    print(f"  Train features: {len(train_files)} files from {train_dir}")
    features_train, _ = preload_features(train_files, num_workers=8)

    if args.max_train_images > 0 and len(features_train) > args.max_train_images:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(features_train),
                         args.max_train_images, replace=False)
        features_train = [features_train[i] for i in idx]
        print(f"  Subsampled to {len(features_train)} train images")

    features_test = basenames_test = gt_test = None
    if need_test:
        test_dir = feat_root / "test" / backbone / args.layer
        test_files = sorted(test_dir.glob("*.npy"))
        print(f"  Test features: {len(test_files)} files from {test_dir}")
        features_test, basenames_test = preload_features(
            test_files, num_workers=8)
        gt_test = load_gt(args.gt_path)

    D = features_train[0].shape[1]
    print(f"  Loading {backbone}...")
    wrapper = Dinov2Wrapper(head_layers=1, model_name=backbone, device=device)

    return (features_train, features_test, basenames_test, gt_test,
            wrapper, D, layer_idx)


def build_tail(wrapper, layer_idx, device):
    """Move tail blocks to GPU, rest to CPU. Return FrozenTail."""
    wrapper.backbone.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()
    tail_blocks = list(wrapper.backbone.blocks[layer_idx + 1:])
    tail = FrozenTail(tail_blocks, wrapper.backbone.norm, device=device)
    return tail


# ================================================================
#     OPQ helpers
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


def learn_opq(features_train, D, G, d, K, norm_mode, device, seed=42):
    """Learn OPQ rotation + codebooks + usage counts for prior init."""
    flat = _collect_normalised(features_train, norm_mode, device)
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


# ================================================================
#     Codec builders (for sensitivity mode)
# ================================================================

def build_codec_identity(features_train, G, K, d, norm_mode, device):
    """No rotation, k-means codebooks."""
    flat = _collect_normalised(features_train, norm_mode, device)
    pq = SoftPQ(G, K, d, lmbda=0.5).to(device)
    pq.init_from_kmeans(flat, device=device)
    pq.temperature = 0.0
    codec = FeatureCodec(pq, transform=None).to(device).eval()
    del flat
    torch.cuda.empty_cache()
    return codec


def build_codec_opq(R, codebooks, G, K, d, D, device, usage=None):
    """OPQ rotation + codebooks, no gradient training."""
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


def build_codec_trained(features_train, tail, R, codebooks, usage,
                        G, K, d, D, device,
                        epochs=50, lr=3e-4, lmbda=0.5, seed=42):
    """Train from OPQ warm-start with L_ref loss."""
    transform = OrthogonalTransform(D)
    R_ws = R.copy()
    C_ws = [c.copy() for c in codebooks]
    if np.linalg.det(R_ws) < 0:
        R_ws[:, -1] *= -1
        C_ws[-1][:, -1] *= -1

    n_val = min(50, len(features_train))
    rng_v = np.random.RandomState(seed + 1)
    val_idx = rng_v.choice(len(features_train), n_val, replace=False)
    val_feat = [features_train[i] for i in val_idx]

    codec, _ = train_soft_pq(
        features_train=features_train, tail=tail,
        G=G, K=K, d=d, norm_mode='per_image',
        epochs=epochs, lr=lr,
        batch_size=32, device=device, seed=seed,
        val_features=val_feat, verbose=True,
        transform=transform, R_init=R_ws,
        codebooks_init=C_ws, lmbda=lmbda,
        prior_init_counts=usage, grad_clip=1.0,
        tau_start=0.5, tau_end=0.005)
    return codec


# ================================================================
#     Per-group diagnostics
# ================================================================

def compute_variance(codec, features, norm_mode, device, batch_size=32):
    """Per-group feature variance in the (possibly rotated) space."""
    pq = codec.pq
    G, d = pq.G, pq.d
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
            Z_g = Z.reshape(-1, G, d)
            sum_sq += (Z_g ** 2).sum(dim=(0, 2))
            sum_x += Z_g.sum(dim=(0, 2))
            n += Z_g.shape[0] * d
            del X, Y, flat, Z
    var = (sum_sq / n) - (sum_x / n) ** 2
    return var.cpu().numpy()


def compute_mse_per_group(codec, features, norm_mode, device, batch_size=32):
    """Per-group quantization MSE using codec's own codebooks."""
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


def _gradient_step(codec, tail, features, norm_mode, device, batch_size,
                   loss_type):
    """Accumulate per-group gradient norm for 'lref' or 'mse' loss.

    Sets tau=0.1 temporarily so soft-assignment provides gradients through
    the hard-assignment bottleneck.
    """
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

        grad_C = torch.autograd.grad(
            loss, pq.codebooks, retain_graph=False)[0]
        grad_sum += grad_C.reshape(G, -1).norm(dim=1).detach()
        n += 1
        del X, Y, Mu, Std, Y_hat, loss, grad_C
        torch.cuda.empty_cache()

    codec.eval()
    pq.temperature = old_tau
    return (grad_sum / max(n, 1)).cpu().numpy()


def compute_gradient_sensitivity(codec, tail, features, norm_mode, device,
                                 batch_size=8):
    """Per-group gradient norm for L_ref and MSE losses."""
    sens_lref = _gradient_step(codec, tail, features, norm_mode, device,
                               batch_size, 'lref')
    sens_mse = _gradient_step(codec, tail, features, norm_mode, device,
                              batch_size, 'mse')
    return sens_lref, sens_mse


def compute_ablation(codec, tail, features, norm_mode, device, batch_size=8):
    """Per-group ablation: delta_L when group g has zero quantization error.

    Positive delta_L[g] means fixing group g's quant error reduces L_ref.
    Negative means the system compensates elsewhere (differential allocation).
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
            Y_teacher = tail.forward_nograd(X)

            Z = codec.transform.encode(flat) if codec.transform else flat
            Z_hat, _ = pq._quantise(Z)

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
#     Diagnostics orchestration
# ================================================================

def _stats(arr):
    """Summary statistics for a per-group array."""
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
    """Run all per-group diagnostics for one transform condition.

    Metrics computed:
      variance  — per-group feature variance in the rotated space
      mse       — per-group quantization MSE (codec's own codebooks)
      sens_lref — ||dL_ref/dC_g|| gradient norm
      sens_mse  — ||dMSE/dC_g|| gradient norm
      ablation  — delta L_ref when group g has zero quant error
    """
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")

    t0 = time.time()

    print("  [1/4] Variance ...", end="", flush=True)
    var = compute_variance(codec, features, norm_mode, device, batch_size=32)
    print(f" {time.time()-t0:.1f}s")

    t1 = time.time()
    print("  [2/4] Per-group MSE ...", end="", flush=True)
    mse = compute_mse_per_group(
        codec, features, norm_mode, device, batch_size=32)
    print(f" {time.time()-t1:.1f}s")

    t2 = time.time()
    print("  [3/4] Gradient sensitivity (L_ref + MSE) ...", end="",
          flush=True)
    sens_lref, sens_mse = compute_gradient_sensitivity(
        codec, tail, features, norm_mode, device, batch_size=batch_size)
    print(f" {time.time()-t2:.1f}s")

    t3 = time.time()
    print("  [4/4] Ablation delta_L_ref ...", end="", flush=True)
    ablation = compute_ablation(
        codec, tail, features, norm_mode, device, batch_size=batch_size)
    print(f" {time.time()-t3:.1f}s")

    res = {
        'condition': name,
        'variance': _stats(var),
        'mse': _stats(mse),
        'sens_lref': _stats(sens_lref),
        'sens_mse': _stats(sens_mse),
        'ablation': _stats(ablation),
    }

    corr_pairs = [
        ('sens_lref', 'sens_mse', sens_lref, sens_mse),
        ('sens_lref', 'mse', sens_lref, mse),
        ('sens_lref', 'ablation', sens_lref, ablation),
        ('mse', 'ablation', mse, ablation),
        ('variance', 'mse', var, mse),
        ('variance', 'sens_lref', var, sens_lref),
    ]
    res['correlations'] = {}
    for n1, n2, a, b in corr_pairs:
        key = f"{n1}_vs_{n2}"
        if np.std(a) < 1e-15 or np.std(b) < 1e-15:
            res['correlations'][key] = 0.0
        else:
            res['correlations'][key] = float(np.corrcoef(a, b)[0, 1])

    metrics = [
        ('Variance', 'variance'),
        ('MSE', 'mse'),
        ('Sens L_ref', 'sens_lref'),
        ('Sens MSE', 'sens_mse'),
        ('Ablation dL', 'ablation'),
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
        print(f"    {a_name:<12} <-> {b_name:<12}  r = {val:+.4f}")

    print(f"  Total: {time.time()-t0:.1f}s")
    return res


# ================================================================
#     Correlation helpers (for Figure 1)
# ================================================================

def evaluate_opq_mse(features, R, codebooks, G, d, norm_mode, device,
                     batch_size=200):
    """Feature-space MSE of standard OPQ encode/decode."""
    D = G * d
    R_t = torch.from_numpy(R).float().to(device)
    cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
    total_mse = 0.0
    N = len(features)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        X = torch.from_numpy(
            np.stack(features[start:end])).float().to(device)
        B = X.shape[0]
        with torch.no_grad():
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            flat = Y.reshape(-1, D) @ R_t
            z_3d = flat.reshape(-1, G, d).permute(1, 0, 2).contiguous()
            z_hat_3d, _ = batched_assign(z_3d, cb_t, device=device)
            flat_hat = z_hat_3d.permute(1, 0, 2).reshape(-1, D)
            Y_hat = (flat_hat @ R_t.T).reshape(B, -1, D)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            mse = ((X - X_hat) ** 2).sum().item() / B
            total_mse += mse * B
        del X, Y, Mu, Std, flat, z_3d, z_hat_3d, flat_hat, Y_hat, X_hat
    return total_mse / N


def evaluate_opq_l_ref(features, R, codebooks, G, d, tail,
                       norm_mode, device, batch_size=16):
    """L_ref of standard OPQ encode/decode."""
    D = G * d
    R_t = torch.from_numpy(R).float().to(device)
    cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
    total_dl = 0.0
    N = len(features)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        X = torch.from_numpy(
            np.stack(features[start:end])).float().to(device)
        B = X.shape[0]
        with torch.no_grad():
            Y_teacher = tail.forward_nograd(X)
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            flat = Y.reshape(-1, D) @ R_t
            z_3d = flat.reshape(-1, G, d).permute(1, 0, 2).contiguous()
            z_hat_3d, _ = batched_assign(z_3d, cb_t, device=device)
            flat_hat = z_hat_3d.permute(1, 0, 2).reshape(-1, D)
            Y_hat = (flat_hat @ R_t.T).reshape(B, -1, D)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            Y_student = tail.forward_nograd(X_hat)
            dl = ((Y_teacher - Y_student) ** 2).sum().item() / B
            total_dl += dl * B
        del X, Y_teacher, Y, Mu, Std, flat, z_3d, z_hat_3d
        del flat_hat, Y_hat, X_hat, Y_student
    return total_dl / N


def opq_encode_decode(features, R, codebooks, G, d, norm_mode, device,
                      batch_size=200):
    """Standard OPQ encode/decode -> list of [T,D] numpy arrays."""
    D = G * d
    R_t = torch.from_numpy(R).float().to(device)
    cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
    results = []
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(
            np.stack(features[start:end])).float().to(device)
        B = X.shape[0]
        with torch.no_grad():
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            flat = Y.reshape(-1, D) @ R_t
            z_3d = flat.reshape(-1, G, d).permute(1, 0, 2).contiguous()
            z_hat_3d, _ = batched_assign(z_3d, cb_t, device=device)
            flat_hat = z_hat_3d.permute(1, 0, 2).reshape(-1, D)
            Y_hat = (flat_hat @ R_t.T).reshape(B, -1, D)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        for i in range(B):
            results.append(X_hat[i].cpu().numpy())
        del X, Y, Mu, Std, flat, z_3d, z_hat_3d, flat_hat, Y_hat, X_hat
    return results


# ================================================================
#     Segmentation evaluators (for correlation mode)
# ================================================================

class OPQSegmentationEvaluator(SegmentationEvaluator):
    """SegmentationEvaluator with OPQ encode/decode."""

    def __init__(self, R, codebooks, embedding_dim, norm_mode, layer_idx,
                 voc_root, weights_root, device='cuda', feat_dim=1024,
                 model_name='dinov2_vitl14'):
        self.R_t = torch.from_numpy(R).float().to(device)
        self.cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
        self.num_groups = len(codebooks)
        self.embedding_dim = embedding_dim
        self.norm_mode = norm_mode
        self.layer_idx = layer_idx
        self.voc_root = voc_root
        self.weights_root = weights_root
        self.device = device
        self.feat_dim = feat_dim
        self.model_name = model_name

    @torch.no_grad()
    def quantize_tokens(self, tokens_np):
        y, mu, std = self.normalize(tokens_np, mode=self.norm_mode)
        Y = torch.from_numpy(y).float().to(self.device).unsqueeze(0)
        flat = Y.reshape(-1, self.feat_dim)
        Z = flat @ self.R_t
        z_3d = Z.reshape(-1, self.num_groups, self.embedding_dim) \
                .permute(1, 0, 2).contiguous()
        z_hat_3d, _ = batched_assign(z_3d, self.cb_t, device=self.device)
        flat_hat = z_hat_3d.permute(1, 0, 2).reshape(-1, self.feat_dim)
        Y_hat = (flat_hat @ self.R_t.T).reshape(1, -1, self.feat_dim)
        Mu = torch.from_numpy(mu).float().to(self.device).unsqueeze(0)
        Std = torch.from_numpy(std).float().to(self.device).unsqueeze(0)
        X_hat = Y_hat * Std + Mu
        return X_hat.squeeze(0)


class CodecSegmentationEvaluator(SegmentationEvaluator):
    """SegmentationEvaluator with FeatureCodec encode/decode."""

    def __init__(self, codec, norm_mode, layer_idx,
                 voc_root, weights_root, device='cuda', feat_dim=1024,
                 model_name='dinov2_vitl14'):
        self.codec = codec
        self.norm_mode = norm_mode
        self.layer_idx = layer_idx
        self.voc_root = voc_root
        self.weights_root = weights_root
        self.device = device
        self.feat_dim = feat_dim
        self.model_name = model_name

    @torch.no_grad()
    def quantize_tokens(self, tokens_np):
        X = torch.from_numpy(tokens_np).float().unsqueeze(0).to(self.device)
        Y, Mu, Std = batch_normalize_gpu(X, mode=self.norm_mode)
        Y_hat, _ = self.codec(Y)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        return X_hat.squeeze(0)


# ================================================================
#     Mode: sensitivity (Figure 2 + Figure 3)
# ================================================================

def run_sensitivity(args):
    """Run gradient-based sensitivity diagnostics for 3 conditions.

    Produces data for both Figure 2 (balance) and Figure 3 (signal
    divergence).  Each condition builds a full codec and evaluates
    per-group variance, quantization MSE, gradient sensitivity (L_ref
    and MSE), and ablation delta-L_ref.
    """
    device = torch.device("cuda")
    set_seed(args.seed)

    D = 1024
    G = D // args.embedding_dim
    d = args.embedding_dim
    K = args.K
    norm_mode = args.norm_mode

    print(f"\n{'#' * 70}")
    print(f"# Figure 2+3: Gradient sensitivity & signal divergence")
    print(f"# layer={args.layer}, K={K}, G={G}, d={d}, D={D}")
    print(f"# n_diag={args.n_diag}, max_train={args.max_train_images}")
    print(f"# train_epochs={args.train_epochs}")
    print(f"# {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'#' * 70}")

    (features_train, _, _, _, wrapper, _, layer_idx) = load_data(
        args, device, need_test=False)

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(features_train))
    diag_idx = perm[:args.n_diag]
    train_idx = perm[args.n_diag:]
    features_diag = [features_train[i] for i in diag_idx]
    features_tr = [features_train[i] for i in train_idx]
    print(f"  Split: diagnostic={len(features_diag)}, "
          f"training={len(features_tr)}")

    tail = build_tail(wrapper, layer_idx, device)

    print(f"\n  Learning OPQ (K={K}, G={G}, d={d})...")
    R_opq, codebooks_opq, opq_usage = learn_opq(
        features_tr, D, G, d, K, norm_mode, device, args.seed)
    print(f"  OPQ done")

    all_results = {}

    # ── Condition 1: Identity ──
    print(f"\n{'=' * 60}")
    print("  Building Identity codec (no rotation, k-means) ...")
    codec_id = build_codec_identity(
        features_tr, G, K, d, norm_mode, device)
    all_results['identity'] = run_diagnostics(
        codec_id, tail, features_diag, norm_mode, device,
        "Identity (no rotation)", batch_size=args.batch_size)
    del codec_id
    torch.cuda.empty_cache()

    # ── Condition 2: OPQ ──
    print(f"\n{'=' * 60}")
    print("  Building OPQ codec (rotation + codebooks, untrained) ...")
    codec_opq = build_codec_opq(
        R_opq, codebooks_opq, G, K, d, D, device, opq_usage)
    all_results['opq'] = run_diagnostics(
        codec_opq, tail, features_diag, norm_mode, device,
        "OPQ (untrained)", batch_size=args.batch_size)
    del codec_opq
    torch.cuda.empty_cache()

    # ── Condition 3: Trained ──
    codec_trained = None
    if args.codec_path and os.path.exists(args.codec_path):
        print(f"\n{'=' * 60}")
        print(f"  Loading trained codec: {args.codec_path}")
        codec_trained = load_codec(args.codec_path, device=device)
        codec_trained.eval()
        if codec_trained.transform is not None:
            print(f"  Loaded: G={codec_trained.pq.G}, K={codec_trained.pq.K},"
                  f" d={codec_trained.pq.d}")
            print(f"  orth_error={codec_trained.transform.orth_error():.2e}")
    elif args.train_epochs > 0:
        print(f"\n{'=' * 60}")
        print(f"  Training DOPQ codec ({args.train_epochs} epochs, "
              f"lr={args.lr}, lmbda={args.lmbda}) ...")
        codec_trained = build_codec_trained(
            features_tr, tail, R_opq, codebooks_opq, opq_usage,
            G, K, d, D, device,
            epochs=args.train_epochs, lr=args.lr,
            lmbda=args.lmbda, seed=args.seed)

    if codec_trained is not None:
        label = ("Trained (ckpt)" if args.codec_path
                 else f"Trained ({args.train_epochs} ep)")
        all_results['trained'] = run_diagnostics(
            codec_trained, tail, features_diag, norm_mode, device,
            label, batch_size=args.batch_size)
        del codec_trained
        torch.cuda.empty_cache()

    # ── Summary tables ──
    conds = list(all_results.keys())
    print(f"\n{'=' * 70}")
    print("  SUMMARY — Figure 2: Coefficient of Variation (CV)")
    print(f"{'=' * 70}")
    header = (f"  {'Condition':<25} {'Var':>7} {'MSE':>7} "
              f"{'S_lref':>7} {'S_mse':>7} {'Ablat':>7}")
    print(header)
    print(f"  {'─' * 60}")
    for c in conds:
        r = all_results[c]
        print(f"  {r['condition']:<25} "
              f"{r['variance']['cv']:>7.3f} "
              f"{r['mse']['cv']:>7.3f} "
              f"{r['sens_lref']['cv']:>7.3f} "
              f"{r['sens_mse']['cv']:>7.3f} "
              f"{r['ablation']['cv']:>7.3f}")

    print(f"\n  SUMMARY — Figure 3: Key Correlations")
    print(f"  {'─' * 64}")
    print(f"  {'Condition':<25} {'Sl<->Sm':>8} {'Sl<->MSE':>8} "
          f"{'Sl<->Abl':>8} {'MSE<->Abl':>8}")
    print(f"  {'─' * 64}")
    for c in conds:
        co = all_results[c]['correlations']
        print(f"  {all_results[c]['condition']:<25} "
              f"{co['sens_lref_vs_sens_mse']:>+8.3f} "
              f"{co['sens_lref_vs_mse']:>+8.3f} "
              f"{co['sens_lref_vs_ablation']:>+8.3f} "
              f"{co['mse_vs_ablation']:>+8.3f}")

    # ── Interpretation ──
    print(f"\n  Interpretation:")
    id_r = all_results.get('identity', {})
    opq_r = all_results.get('opq', {})
    tr_r = all_results.get('trained', {})
    if id_r and opq_r:
        sl_id = id_r['sens_lref']['cv']
        sl_opq = opq_r['sens_lref']['cv']
        print(f"    sens_lref CV: Identity={sl_id:.3f} -> OPQ={sl_opq:.3f}")
        if tr_r:
            sl_tr = tr_r['sens_lref']['cv']
            print(f"    sens_lref CV: OPQ={sl_opq:.3f} -> Trained={sl_tr:.3f}")
            if sl_tr < sl_opq < sl_id:
                print(f"    => Identity >> OPQ > Trained — clean 3-level"
                      f" progression (Figure 2 argument supported)")

        corr_sl_sm = opq_r['correlations'].get('sens_lref_vs_sens_mse', 0)
        print(f"    r(sens_lref, sens_mse) under OPQ: {corr_sl_sm:+.3f}"
              f" {'-> signals DIVERGE' if abs(corr_sl_sm) < 0.5 else ''}")

        if tr_r:
            mse_abl = tr_r['correlations'].get('mse_vs_ablation', 0)
            print(f"    r(mse, ablation) under Trained: {mse_abl:+.3f}"
                  f" {'-> anti-correlated (differential allocation)'if mse_abl < -0.3 else ''}")

    # ── Save ──
    out_dir = os.path.join(ORFC_ROOT, 'results', 'analysis_intro_v2')
    os.makedirs(out_dir, exist_ok=True)
    trained_tag = "ckpt" if args.codec_path else f"ep{args.train_epochs}"
    tag = (f"sensitivity_{args.layer}_K{K}_emb{d}"
           f"_ndiag{args.n_diag}_{trained_tag}")
    out_path = os.path.join(out_dir, f'{tag}.json')
    output = {
        'config': {
            'layer': args.layer, 'K': K, 'embedding_dim': d,
            'G': G, 'D': D,
            'n_diag': len(features_diag),
            'n_train': len(features_tr),
            'codec_path': args.codec_path or None,
            'train_epochs': args.train_epochs if not args.codec_path else 0,
            'lr': args.lr, 'lmbda': args.lmbda,
            'norm_mode': norm_mode, 'seed': args.seed,
        },
        'conditions': all_results,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {out_path}")
    return output


# ================================================================
#     Mode: correlation (Figure 1)
# ================================================================

def run_correlation(args):
    """Compute (MSE, L_ref, Acc) for OPQ at multiple (K, emb) points.

    Also loads existing DOPQ results from JSON for comparison.
    Produces data for Figure 1: L_ref is a better proxy than MSE.
    """
    device = torch.device("cuda")
    set_seed(args.seed)

    norm_mode = args.norm_mode

    print(f"\n{'#' * 70}")
    print(f"# Figure 1: MSE vs L_ref correlation with downstream Acc")
    print(f"# layer={args.layer}")
    print(f"# {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'#' * 70}")

    (features_train, features_test, basenames_test, gt_test,
     wrapper, D, layer_idx) = load_data(args, device, need_test=True)

    configs = [
        (4, 32), (8, 32), (16, 32), (32, 32), (64, 32),
        (256, 32), (64, 16), (256, 16), (64, 8), (256, 8),
    ]

    points = []

    for K, emb in configs:
        G = D // emb
        d = emb
        print(f"\n  --- OPQ K={K}, emb={emb}, G={G} ---")

        wrapper.backbone.cpu()
        if wrapper.head is not None:
            wrapper.head.cpu()
        torch.cuda.empty_cache()

        R, cb, _ = learn_opq(
            features_train, D, G, d, K, norm_mode, device, args.seed)

        mse = evaluate_opq_mse(
            features_test, R, cb, G, d, norm_mode, device)

        tail = build_tail(wrapper, layer_idx, device)
        l_ref = evaluate_opq_l_ref(
            features_test, R, cb, G, d, tail,
            norm_mode, device, batch_size=args.batch_size)
        tail.to('cpu')
        torch.cuda.empty_cache()

        wrapper.backbone.to(device)
        if wrapper.head is not None:
            wrapper.head.to(device)
        xhat = opq_encode_decode(
            features_test, R, cb, G, d, norm_mode, device)
        acc = evaluate_accuracy(
            xhat, basenames_test, gt_test, wrapper, layer_idx, device)
        del xhat

        rate = G * math.log2(K)
        pt = {
            'method': 'opq', 'K': K, 'emb': emb, 'G': G,
            'rate_bpt': float(rate),
            'mse': float(mse), 'l_ref': float(l_ref), 'acc': float(acc),
        }

        if args.eval_seg:
            wrapper.backbone.cpu()
            if wrapper.head is not None:
                wrapper.head.cpu()
            torch.cuda.empty_cache()
            seg_feat_dir = str(
                Path(args.seg_feat_root) / args.backbone / args.layer)
            opq_seg = OPQSegmentationEvaluator(
                R=R, codebooks=cb, embedding_dim=emb,
                norm_mode=norm_mode, layer_idx=layer_idx,
                voc_root=args.voc_root,
                weights_root=wrapper.weights_root, device=device,
                feat_dim=D, model_name=args.backbone)
            seg_res = opq_seg.evaluate(
                seg_feat_dir=seg_feat_dir,
                image_list=args.seg_image_list, verbose=False)
            pt['miou'] = float(seg_res['miou'])
            del opq_seg
            torch.cuda.empty_cache()

        points.append(pt)
        miou_s = f"  mIoU={pt['miou']:.4f}" if 'miou' in pt else ""
        print(f"    MSE={mse:.1f}  L_ref={l_ref:.1f}  "
              f"Acc={acc:.4f}{miou_s}  Rate={rate:.0f} bpt")

    # Load existing DOPQ results
    result_dir = os.path.join(
        ORFC_ROOT, 'results', 'soft_pq', args.backbone)
    n_dopq = 0
    if os.path.isdir(result_dir):
        for fname in sorted(os.listdir(result_dir)):
            if not fname.startswith(args.layer):
                continue
            fpath = os.path.join(result_dir, fname)
            with open(fpath) as fp:
                d_json = json.load(fp)
            cfg = d_json.get('config', {})
            dopq_pt = {
                'method': 'dopq',
                'K': cfg.get('K'),
                'emb': cfg.get('embedding_dim'),
                'lmbda': cfg.get('lmbda'),
                'mse_loss': cfg.get('mse_loss', False),
                'freeze_transform': cfg.get('freeze_transform', False),
                'std_opq_acc': d_json.get('std_opq_acc'),
                'std_opq_l_ref': d_json.get('std_opq_delta_l'),
                'soft_pq_acc': d_json.get('soft_pq_acc'),
                'soft_pq_l_ref': d_json.get('soft_pq_delta_l'),
                'std_opq_miou': d_json.get('std_opq_miou'),
                'soft_pq_miou': d_json.get('soft_pq_miou'),
                'rate_bpt': d_json.get('rate_info', {}).get('rans_bpt'),
                'source': fname,
            }
            points.append(dopq_pt)
            n_dopq += 1
    print(f"\n  Loaded {n_dopq} existing DOPQ results")

    output = {
        'config': {
            'layer': args.layer,
            'n_test': len(features_test),
            'seed': args.seed,
        },
        'points': points,
    }

    out_dir = os.path.join(ORFC_ROOT, 'results', 'analysis_intro_v2')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'correlation_{args.backbone}_{args.layer}.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")

    opq_pts = [p for p in points if p['method'] == 'opq']
    if len(opq_pts) >= 3:
        mse_arr = np.array([p['mse'] for p in opq_pts])
        lref_arr = np.array([p['l_ref'] for p in opq_pts])
        acc_arr = np.array([p['acc'] for p in opq_pts])
        print(f"\n  === OPQ Correlation Summary ===")
        print(f"    Pearson r(MSE, Acc)   = "
              f"{np.corrcoef(mse_arr, acc_arr)[0, 1]:+.4f}")
        print(f"    Pearson r(L_ref, Acc) = "
              f"{np.corrcoef(lref_arr, acc_arr)[0, 1]:+.4f}")
        if all('miou' in p for p in opq_pts):
            miou_arr = np.array([p['miou'] for p in opq_pts])
            print(f"    Pearson r(MSE, mIoU)  = "
                  f"{np.corrcoef(mse_arr, miou_arr)[0, 1]:+.4f}")
            print(f"    Pearson r(L_ref,mIoU) = "
                  f"{np.corrcoef(lref_arr, miou_arr)[0, 1]:+.4f}")

    dopq_pts = [p for p in points if p['method'] == 'dopq']
    if dopq_pts:
        print(f"\n  === DOPQ Summary ({len(dopq_pts)} configs) ===")
        for dp in dopq_pts:
            miou_s = ""
            if dp.get('soft_pq_miou') is not None:
                miou_s = f"  mIoU={dp['soft_pq_miou']:.4f}"
            lmbda_tag = f"  mse_loss" if dp.get('mse_loss') else ""
            print(f"    K={dp.get('K'):>3} emb={dp.get('emb'):>2} "
                  f"λ={dp.get('lmbda', '?')}"
                  f"  Acc={dp.get('soft_pq_acc', 0):.4f}"
                  f"  L_ref={dp.get('soft_pq_l_ref', 0):.1f}"
                  f"{miou_s}{lmbda_tag}")

    return output


# ================================================================
#     CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified intro-figure analysis for DOPQ paper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--mode", type=str, default="sensitivity",
                        choices=["sensitivity", "correlation", "all"])
    parser.add_argument("--layer", type=str, default="blk20")
    parser.add_argument("--K", type=int, default=16,
                        help="PQ codebook size (for sensitivity mode)")
    parser.add_argument("--embedding_dim", type=int, default=32,
                        help="PQ sub-dimension d (for sensitivity mode)")
    parser.add_argument("--norm_mode", type=str, default="per_image")
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--codec_path", type=str, default="",
                        help="Path to saved codec .pt (skip training)")
    parser.add_argument("--n_diag", type=int, default=200,
                        help="Images for diagnostic evaluation")
    parser.add_argument("--train_epochs", type=int, default=50,
                        help="Epochs for Trained condition (0 = skip)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lmbda", type=float, default=0.5)

    parser.add_argument("--max_train_images", type=int, default=5000)
    parser.add_argument("--feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features"))
    parser.add_argument("--backbone", type=str, default="dinov2_vitl14")
    parser.add_argument("--gt_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "utils",
                                "imagenet_selected_label500.txt"))
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--eval_seg", action="store_true",
                        help="Evaluate segmentation mIoU (VOC2012)")
    parser.add_argument("--seg_feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features",
                                "voc2012_100"))
    parser.add_argument("--voc_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "data",
                                "VOCdevkit", "VOC2012"))
    parser.add_argument("--seg_image_list", type=str,
                        default=os.path.join(PROJECT_ROOT, "utils",
                                "voc2012_val_100.txt"))

    args = parser.parse_args()

    if args.mode in ("sensitivity", "all"):
        run_sensitivity(args)
    if args.mode in ("correlation", "all"):
        run_correlation(args)


if __name__ == '__main__':
    main()
