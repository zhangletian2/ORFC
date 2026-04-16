#!/usr/bin/env python
"""
Differentiable Soft-PQ experiment runner.

Compares standard OPQ baseline vs Soft-PQ (ΔL_ref optimised) on
downstream classification accuracy.

Usage:
    python run_soft_pq.py --layer blk20 --K 64 --embedding_dim 32 --epochs 100

    # Quick smoke test:
    python run_soft_pq.py --layer blk20 --K 64 --epochs 5 --max_train_images 50
"""

import os, sys, argparse, json, math, time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

ORFC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))

from run_multilayer_calibrator import (
    set_seed, preload_features, load_gt, evaluate_accuracy,
)
from opq import (
    batch_normalize_gpu, batch_inv_normalize_gpu, batched_assign,
    learn_opq_rotation,
)
from backbone.wrapper import Dinov2Wrapper, ClipWrapper, SegmentationEvaluator
from soft_pq import (
    SoftPQ, FeatureTransform, OrthogonalTransform, FeatureCodec,
    FrozenTail, CLIPFrozenTail, train_soft_pq, soft_pq_encode_decode,
    save_codec, load_codec,
)

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
import logging
from mmcv.utils import get_logger
logger = get_logger('mmcv')
logger.setLevel(logging.WARNING)


# ================================================================
#                    Standard OPQ encode/decode
# ================================================================

def pq_encode_decode_features(features, codebooks, embedding_dim, norm_mode,
                              device, R=None, chunk_images=500):
    """GPU batch PQ encode/decode (standard OPQ path)."""
    num_groups = len(codebooks)
    N_img = len(features)
    T = features[0].shape[0]
    C = features[0].shape[1]
    cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
    R_t = torch.from_numpy(R).float().to(device) if R is not None else None
    all_xhat = []
    for start in range(0, N_img, chunk_images):
        end = min(start + chunk_images, N_img)
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        B = X.shape[0]
        with torch.no_grad():
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            flat = Y.reshape(-1, C)
            Z = flat @ R_t if R_t is not None else flat
            z_3d = Z.reshape(-1, num_groups, embedding_dim) \
                    .permute(1, 0, 2).contiguous()
            z_hat_3d, _ = batched_assign(z_3d, cb_t, device=device)
            flat_hat = z_hat_3d.permute(1, 0, 2).reshape(-1, C)
            Y_hat = flat_hat @ R_t.T if R_t is not None else flat_hat
            Y_hat = Y_hat.reshape(B, T, C)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        for i in range(B):
            all_xhat.append(X_hat[i].cpu().numpy())
        del X, Y, Mu, Std, Z, z_3d, z_hat_3d, flat_hat, Y_hat, X_hat
        torch.cuda.empty_cache()
    return all_xhat


# ================================================================
#                    ΔL_ref evaluation (no-grad)
# ================================================================

def evaluate_delta_l_ref(features, tail, norm_mode, device,
                         codec=None,
                         codebooks=None, R=None, embedding_dim=None,
                         batch_size=4):
    """Evaluate ΔL_ref = ||F(H) - F(Ĥ)||² for a set of features.

    Either provide codec (FeatureCodec / any module with same forward) or
    (codebooks, R, embedding_dim) for standard OPQ path.
    """
    N = len(features)
    total_loss = 0.0
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X = torch.from_numpy(
                np.stack(features[start:end])
            ).float().to(device)
            B = X.shape[0]
            Y_teacher = tail.forward_nograd(X)

            if codec is not None:
                Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
                Y_hat, _ = codec(Y)
                X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            else:
                C_dim = features[0].shape[1]
                num_groups = len(codebooks)
                cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
                R_t = torch.from_numpy(R).float().to(device) if R is not None else None
                Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
                flat = Y.reshape(-1, C_dim)
                Z = flat @ R_t if R_t is not None else flat
                z_3d = Z.reshape(-1, num_groups, embedding_dim) \
                        .permute(1, 0, 2).contiguous()
                z_hat_3d, _ = batched_assign(z_3d, cb_t, device=device)
                flat_hat = z_hat_3d.permute(1, 0, 2).reshape(-1, C_dim)
                Y_hat_flat = flat_hat @ R_t.T if R_t is not None else flat_hat
                Y_hat_r = Y_hat_flat.reshape(B, X.shape[1], C_dim)
                X_hat = batch_inv_normalize_gpu(Y_hat_r, Mu, Std)

            Y_student = tail.forward_nograd(X_hat)
            loss = ((Y_teacher - Y_student) ** 2).sum().item() / B
            total_loss += loss * B
            del X, X_hat, Y_teacher, Y_student
            torch.cuda.empty_cache()
    return total_loss / N


# ================================================================
#                    Rate evaluation (post-hoc)
# ================================================================

try:
    from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
    from compressai import ans as _ans
    _HAS_ANS = True
except (ImportError, ModuleNotFoundError):
    _HAS_ANS = False


def _codec_labels(features, codec, norm_mode, device, batch_size=32):
    """Run codec on features and return hard labels [G, N_total]."""
    codec.eval()
    pq = codec.pq
    all_labels = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            X = torch.from_numpy(
                np.stack(features[start:end])
            ).float().to(device)
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
            _ = codec(Y)
            all_labels.append(pq._last_labels.cpu())
            del X, Y
    return torch.cat(all_labels, dim=1).numpy()       # [G, N_total]


def _histogram_pmf(labels, G, K_per_group, smoothing=1.0):
    """Build list of G PMFs from [G, N] label array with Laplace smoothing.

    Args:
        K_per_group: int or list[int]. If int, all groups use the same K.
    Returns:
        list of G numpy arrays, each of shape [K_g].
    """
    if isinstance(K_per_group, int):
        K_per_group = [K_per_group] * G
    pmfs = []
    for g in range(G):
        k_g = K_per_group[g]
        counts = np.zeros(k_g, dtype=np.float64)
        np.add.at(counts, labels[g], 1)
        counts += smoothing
        pmfs.append(counts / counts.sum())
    return pmfs


def _rans_encode_bpt(labels_np, pmf_list, G, K_per_group, precision=16):
    """Encode labels with rANS, return actual bits per token.

    Args:
        pmf_list: list of G numpy arrays, each of shape [K_g].
        K_per_group: int or list[int].
    """
    if not _HAS_ANS:
        return None
    if isinstance(K_per_group, int):
        K_per_group = [K_per_group] * G
    encoder = _ans.RansEncoder()
    N = labels_np.shape[1]
    cdfs = []
    cdf_sizes = []
    for g in range(G):
        p = torch.from_numpy(pmf_list[g]).float()
        overflow = (1.0 - p.sum()).clamp_min(0)
        p = torch.cat([p, overflow.unsqueeze(0)])
        cdf = _pmf_to_quantized_cdf(p.tolist(), precision)
        cdfs.append(cdf)
        cdf_sizes.append(K_per_group[g] + 2)
    symbols = []
    cdf_indices = []
    for n in range(N):
        for g in range(G):
            symbols.append(int(labels_np[g, n]))
            cdf_indices.append(g)
    byte_string = encoder.encode_with_indexes(
        symbols, cdf_indices, cdfs,
        cdf_sizes, [0] * G,
    )
    total_bits = len(byte_string) * 8
    return total_bits / N


def evaluate_rate(features, codec, norm_mode, device, batch_size=32,
                  train_pmf=None):
    """Compute rate metrics on held-out features.

    Args:
        features: list of [T, D] numpy arrays (test / held-out).
        codec: FeatureCodec.
        train_pmf: list of G arrays (each [K_g]) or [G, K] numpy PMF fitted
            on training data. If None and codec has no learned prior, uniform
            PMF is used (= max rate).

    Returns dict with:
        xent_rate_bpt     - cross-entropy with *primary* PMF
        xent_train_bpt    - cross-entropy with train histogram PMF (always)
        empirical_entropy_bpt - H(test labels), theoretical lower bound
        max_rate_bpt      - sum(log2(K_g)) (fixed-length ceiling)
        rans_bpt          - actual rANS coded bits/token (primary PMF)
        rans_train_bpt    - actual rANS coded bits/token (train PMF)
    """
    pq = codec.pq
    G = pq.G
    K = pq.K

    test_labels = _codec_labels(features, codec, norm_mode, device,
                                batch_size=batch_size)  # [G, N]
    N = test_labels.shape[1]

    # --- primary PMF ---
    if pq.use_rate:
        primary_pmf = pq.get_prior_pmf()       # [G, K] numpy
    elif train_pmf is not None:
        primary_pmf = train_pmf
    else:
        primary_pmf = np.full((G, K), 1.0 / K)

    # --- cross-entropy with primary PMF ---
    xent_primary = 0.0
    for g in range(G):
        log2_p = np.log2(primary_pmf[g] + 1e-30)
        xent_primary += -log2_p[test_labels[g]].sum()
    xent_primary_bpt = xent_primary / N

    # --- cross-entropy with train histogram PMF ---
    xent_train_bpt = None
    if train_pmf is not None:
        xent_train = 0.0
        for g in range(G):
            log2_t = np.log2(train_pmf[g] + 1e-30)
            xent_train += -log2_t[test_labels[g]].sum()
        xent_train_bpt = xent_train / N

    # --- empirical entropy (test histogram, diagnostic only) ---
    test_pmf = _histogram_pmf(test_labels, G, K, smoothing=0)
    empirical_entropy = 0.0
    for g in range(G):
        pg = test_pmf[g]
        pg = pg[pg > 0]
        empirical_entropy += -np.sum(pg * np.log2(pg))

    # --- rANS real encoding ---
    rans_bpt = _rans_encode_bpt(test_labels, primary_pmf, G, K)
    rans_train_bpt = None
    if train_pmf is not None:
        rans_train_bpt = _rans_encode_bpt(test_labels, train_pmf, G, K)

    max_rate = G * math.log2(K)
    result = {
        'xent_rate_bpt': float(xent_primary_bpt),
        'empirical_entropy_bpt': float(empirical_entropy),
        'max_rate_bpt': float(max_rate),
    }
    if xent_train_bpt is not None:
        result['xent_train_bpt'] = float(xent_train_bpt)
    if rans_bpt is not None:
        result['rans_bpt'] = float(rans_bpt)
    if rans_train_bpt is not None:
        result['rans_train_bpt'] = float(rans_train_bpt)
    return result


# ================================================================
#                    Segmentation evaluators
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
#                    Main experiment
# ================================================================

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    layer_idx = int(args.layer[-2:])

    print(f"\n{'#' * 70}")
    freeze_parts = []
    if args.freeze_transform:
        freeze_parts.append("freeze_R")
    if args.freeze_codebooks:
        freeze_parts.append("freeze_C")
    freeze_str = ", ".join(freeze_parts) if freeze_parts else "all trainable"
    rot_info = f", rot={args.init_rotation}" if args.init_rotation != 'opq' else ""
    print(f"# Feature Codec Experiment (OrthogonalTransform + PQ)")
    print(f"# layer={args.layer} (idx={layer_idx}), K={args.K}, "
          f"emb={args.embedding_dim}, bt={args.bottleneck_dim}")
    print(f"# transform: {freeze_str}{rot_info}")
    tau_info = f", τ={args.tau_start}→{args.tau_end}" if args.tau_start > 0 else ""
    print(f"# epochs={args.epochs}, lr={args.lr}, λ={args.lmbda}{tau_info}")
    print(f"# max_train={args.max_train_images}, batch={args.batch_size}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    # ---- Load features ----
    train_dir = Path(args.feat_root) / args.train_subset / args.backbone / args.layer
    test_dir = Path(args.feat_root) / "test" / args.backbone / args.layer
    train_files = sorted(train_dir.glob("*.npy"))
    test_files = sorted(test_dir.glob("*.npy"))

    print(f"\nData: train={len(train_files)}, test={len(test_files)}")
    features_train, _ = preload_features(train_files, num_workers=8)
    features_test, basenames_test = preload_features(test_files, num_workers=8)
    gt_test = load_gt(args.gt_path)

    D = features_train[0].shape[1]
    T = features_train[0].shape[0]
    bt_dim = args.bottleneck_dim
    Dp = bt_dim if bt_dim > 0 else D
    num_groups = Dp // args.embedding_dim
    bits_per_token = num_groups * math.log2(args.K)

    print(f"  D={D}, D'={Dp}, T={T}, G={num_groups}, d={args.embedding_dim}")
    print(f"  bits/token={bits_per_token:.0f}")

    if args.max_train_images > 0 and len(features_train) > args.max_train_images:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(features_train), args.max_train_images, replace=False)
        features_train_sub = [features_train[i] for i in idx]
        print(f"  Training subset: {len(features_train_sub)} images")
    else:
        features_train_sub = features_train

    # Validation set (held-out from training pool)
    n_val = min(args.n_val, len(features_train))
    rng_val = np.random.RandomState(args.seed + 1)
    val_idx = rng_val.choice(len(features_train), n_val, replace=False)
    val_features = [features_train[i] for i in val_idx]

    # ---- Load backbone ----
    is_clip = args.backbone.startswith("clip")
    if is_clip:
        print(f"\nLoading CLIP ViT-L/14 (classnames={args.classnames})...")
        wrapper = ClipWrapper(args.classnames, device=device)
    else:
        print(f"\nLoading DINOv2 ({args.backbone})...")
        wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone, device=device)

    n_blocks = len(wrapper.backbone.blocks)
    tail_blocks = list(wrapper.backbone.blocks[layer_idx + 1:])
    n_tail = len(tail_blocks)
    print(f"  Tail: {n_tail} blocks (blk{layer_idx+1}..blk{layer_idx+n_tail})"
          f" + norm")

    results = {}
    results['config'] = vars(args)

    # ================================================================
    #   (A) Standard OPQ baseline
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  [Standard OPQ] baseline")
    print(f"{'=' * 60}")

    wrapper.backbone.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()

    t0 = time.time()
    all_vectors = []
    for start in range(0, len(features_train_sub), 200):
        end = min(start + 200, len(features_train_sub))
        X = torch.from_numpy(
            np.stack(features_train_sub[start:end])
        ).float().to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode=args.norm_mode)
        all_vectors.append(Y.reshape(-1, D).cpu().numpy())
        del X, Y
    full_vectors = np.concatenate(all_vectors, axis=0)
    del all_vectors

    opq_groups = D // args.embedding_dim
    max_flat = args.kmeans_max_samples // opq_groups
    if full_vectors.shape[0] > max_flat:
        rng2 = np.random.RandomState(args.seed)
        idx2 = rng2.choice(full_vectors.shape[0], max_flat, replace=False)
        full_vectors = full_vectors[idx2]

    R_std, codebooks_std, hist_std = learn_opq_rotation(
        full_vectors, opq_groups, args.embedding_dim, args.K,
        max_iter_opq=20, max_iter_kmeans=100,
        device=device, verbose=False,
    )
    # Compute alternative rotation matrix if requested
    R_alt = None
    if args.init_rotation == 'pca':
        cov = full_vectors.T @ full_vectors / max(full_vectors.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        R_alt = eigvecs[:, ::-1].T.copy().astype(np.float32)
    elif args.init_rotation == 'random_orth':
        rng_rot = np.random.RandomState(args.seed + 7)
        R_alt, _ = np.linalg.qr(rng_rot.randn(D, D).astype(np.float32))
    elif args.init_rotation == 'identity':
        R_alt = np.eye(D, dtype=np.float32)

    del full_vectors
    torch.cuda.empty_cache()
    std_time = time.time() - t0
    print(f"    OPQ done: MSE={hist_std[-1][0]:.8f} ({std_time:.1f}s)")
    if R_alt is not None:
        print(f"    Alt rotation: {args.init_rotation}")

    opq_usage_counts = None
    if args.lmbda > 0 and args.warm_start_opq:
        R_t = torch.from_numpy(R_std).float().to(device)
        cb_t = torch.from_numpy(np.stack(codebooks_std)).float().to(device)
        opq_usage_counts = np.zeros((opq_groups, args.K), dtype=np.float64)
        for start in range(0, len(features_train_sub), 200):
            end = min(start + 200, len(features_train_sub))
            X = torch.from_numpy(
                np.stack(features_train_sub[start:end])
            ).float().to(device)
            with torch.no_grad():
                Y, _, _ = batch_normalize_gpu(X, mode=args.norm_mode)
                flat = Y.reshape(-1, D) @ R_t
                sub = flat.reshape(-1, opq_groups, args.embedding_dim) \
                      .permute(1, 0, 2).contiguous()
                dists = torch.cdist(sub, cb_t)
                labels = dists.argmin(dim=-1)
                for g in range(opq_groups):
                    for k in labels[g].cpu().numpy():
                        opq_usage_counts[g, k] += 1
            del X, Y, flat, sub, dists, labels
        del R_t, cb_t
        torch.cuda.empty_cache()
        ppl_opq = np.exp(-(opq_usage_counts / opq_usage_counts.sum(-1, keepdims=True)
                          * np.log(opq_usage_counts / opq_usage_counts.sum(-1, keepdims=True)
                                   + 1e-30)).sum(-1)).mean()
        print(f"    OPQ empirical ppl={ppl_opq:.1f} (for prior init)")

    # Standard OPQ decode for test
    xhat_std = pq_encode_decode_features(
        features_test, codebooks_std, args.embedding_dim, args.norm_mode,
        device, R=R_std,
    )

    wrapper.backbone.to(device)
    if wrapper.head is not None:
        wrapper.head.to(device)
    torch.cuda.empty_cache()

    acc_std = evaluate_accuracy(
        xhat_std, basenames_test, gt_test,
        wrapper, layer_idx, device,
    )
    results['std_opq_acc'] = float(acc_std)
    print(f"  * Standard OPQ Acc = {acc_std:.4f}")
    del xhat_std

    # Standard OPQ ΔL_ref
    tail_blocks_ref = list(wrapper.backbone.blocks[layer_idx + 1:])
    norm_ref = wrapper.backbone.norm
    if is_clip:
        tail = CLIPFrozenTail(tail_blocks_ref, norm_ref, device=device)
    else:
        tail = FrozenTail(tail_blocks_ref, norm_ref, device=device)

    std_delta_l = evaluate_delta_l_ref(
        features_test, tail, args.norm_mode, device,
        codebooks=codebooks_std, R=R_std,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
    )
    results['std_opq_delta_l'] = float(std_delta_l)
    print(f"  * Standard OPQ ΔL_ref = {std_delta_l:.1f}")

    # ================================================================
    #   (B) Codec training (or load from checkpoint)
    # ================================================================
    pq_groups = num_groups

    if getattr(args, 'eval_only', False) and args.ckpt_path:
        print(f"\n{'=' * 60}")
        print(f"  [Eval-Only] Loading codec from: {args.ckpt_path}")
        print(f"{'=' * 60}")
        codec = load_codec(args.ckpt_path, device=device)
        train_time = 0.0
        history = []
        tail.to('cpu')
        torch.cuda.empty_cache()
    else:
        bt_str = f"bt={bt_dim}" if bt_dim > 0 else "no transform"
        print(f"\n{'=' * 60}")
        tau_str = f", τ={args.tau_start}→{args.tau_end}" if args.tau_start > 0 else ""
        print(f"  [Codec] K={args.K}, λ={args.lmbda}, {bt_str}, "
              f"epochs={args.epochs}{tau_str}")
        print(f"{'=' * 60}")

        for i, blk in enumerate(wrapper.backbone.blocks):
            if i <= layer_idx:
                blk.cpu()
        if wrapper.head is not None:
            wrapper.head.cpu()
        if args.mse_loss:
            tail.to('cpu')
        torch.cuda.empty_cache()

        transform = None
        if bt_dim > 0 and bt_dim == D:
            transform = OrthogonalTransform(D)
        elif bt_dim > 0:
            transform = FeatureTransform(D, bt_dim)

        can_warmstart = (args.warm_start_opq and bt_dim == D)
        if args.init_rotation == 'opq':
            if can_warmstart:
                R_ws = R_std.copy()
                C_ws = [c.copy() for c in codebooks_std]
                if np.linalg.det(R_ws) < 0:
                    R_ws[:, -1] *= -1
                    C_ws[-1][:, -1] *= -1
                    print(f"    det(R_opq)<0: flipped last col to SO(D)")
            else:
                R_ws = None
                C_ws = None
        else:
            if transform is not None and R_alt is not None and bt_dim == D:
                transform.init_from_opq(R_alt)
                print(f"    Transform set to {args.init_rotation} rotation")
            R_ws = None
            C_ws = None
        if bt_dim > 0 and bt_dim != D:
            print(f"    Bottleneck D'={bt_dim} < D={D}: "
                  f"k-means init (OPQ warm-start N/A)")
            opq_usage_counts = None

        t0 = time.time()
        codec, history = train_soft_pq(
            features_train=features_train_sub,
            tail=tail,
            G=pq_groups,
            K=args.K,
            d=args.embedding_dim,
            norm_mode=args.norm_mode,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            val_features=val_features,
            verbose=True,
            transform=transform,
            R_init=R_ws,
            codebooks_init=C_ws,
            kmeans_max_samples=args.kmeans_max_samples,
            use_mse_loss=args.mse_loss,
            lmbda=args.lmbda,
            prior_init_counts=opq_usage_counts,
            grad_clip=args.grad_clip,
            freeze_transform=args.freeze_transform,
            freeze_codebooks=args.freeze_codebooks,
            prior_floor=args.prior_floor,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
            tau_schedule=args.tau_schedule,
        )
        train_time = time.time() - t0
        print(f"  Codec training: {train_time:.1f}s")

        codec_dir = os.path.join(ORFC_ROOT, 'checkpoints', args.backbone)
        os.makedirs(codec_dir, exist_ok=True)
        bt_tag = f"bt{bt_dim}" if bt_dim > 0 else "noBt"
        ws_tag = "ws" if args.warm_start_opq else "km"
        mse_tag = "_mse" if args.mse_loss else ""
        rate_tag = f"_lmbda{args.lmbda}" if args.lmbda > 0 else ""
        fz_tag_ck = ""
        if args.freeze_transform:
            fz_tag_ck += "_fzR"
        if args.freeze_codebooks:
            fz_tag_ck += "_fzC"
        _rot_short = args.init_rotation.replace('_', '')
        rot_tag_ck = f"_rot{_rot_short}" if args.init_rotation != 'opq' else ""
        pfloor_tag_ck = f"_pf{args.prior_floor}" if args.prior_floor > 0 else ""
        tau_tag = f"_tau{args.tau_start}" if args.tau_start > 0 else ""
        tau_end_tag = f"_te{args.tau_end}" if args.tau_end != 0.005 and args.tau_start > 0 else ""
        tau_sched_tag = f"_ts{args.tau_schedule[:3]}" if args.tau_schedule != "exponential" else ""
        ckpt_name = (f"{args.layer}_K{args.K}_emb{args.embedding_dim}"
                     f"_{bt_tag}_{ws_tag}{mse_tag}{rate_tag}"
                     f"{fz_tag_ck}{rot_tag_ck}{pfloor_tag_ck}{tau_tag}{tau_end_tag}{tau_sched_tag}"
                     f"_lr{args.lr}_ep{args.epochs}_n{args.max_train_images}_s{args.seed}")
        ckpt_path = os.path.join(codec_dir, f"{ckpt_name}.pt")
        save_codec(codec, ckpt_path)
        print(f"  Codec saved: {ckpt_path}")

        tail.to('cpu')
        torch.cuda.empty_cache()

    # Codec decode for test
    xhat_spq = soft_pq_encode_decode(
        features_test, codec, args.norm_mode, device,
    )

    # Classification
    wrapper.backbone.to(device)
    if wrapper.head is not None:
        wrapper.head.to(device)
    torch.cuda.empty_cache()

    acc_spq = evaluate_accuracy(
        xhat_spq, basenames_test, gt_test,
        wrapper, layer_idx, device,
    )
    results['soft_pq_acc'] = float(acc_spq)
    delta_acc = acc_spq - acc_std
    print(f"  * Codec Acc = {acc_spq:.4f} (d={delta_acc:+.4f})")
    del xhat_spq

    # Codec ΔL_ref
    tail_blocks_ref2 = list(wrapper.backbone.blocks[layer_idx + 1:])
    norm_ref2 = wrapper.backbone.norm
    if is_clip:
        tail2 = CLIPFrozenTail(tail_blocks_ref2, norm_ref2, device=device)
    else:
        tail2 = FrozenTail(tail_blocks_ref2, norm_ref2, device=device)

    spq_delta_l = evaluate_delta_l_ref(
        features_test, tail2, args.norm_mode, device,
        codec=codec,
        batch_size=args.batch_size,
    )
    results['soft_pq_delta_l'] = float(spq_delta_l)
    delta_dl = spq_delta_l - std_delta_l
    print(f"  * Codec ΔL_ref = {spq_delta_l:.1f} (d={delta_dl:+.1f})")

    tail2.to('cpu')
    torch.cuda.empty_cache()

    # Build train histogram PMF for fair entropy-coding baseline
    print(f"  Computing train histogram PMF...")
    train_labels = _codec_labels(
        features_train_sub, codec, args.norm_mode, device,
        batch_size=args.batch_size,
    )
    train_pmf = _histogram_pmf(train_labels, pq_groups, args.K)

    # Rate evaluation (on test features, using train-fitted PMFs)
    rate_info = evaluate_rate(
        features_test, codec, args.norm_mode, device,
        batch_size=args.batch_size,
        train_pmf=train_pmf,
    )
    results['rate_info'] = rate_info
    xent_str = f"xent={rate_info['xent_rate_bpt']:.2f}"
    if 'xent_train_bpt' in rate_info:
        xent_str += f"  xent_train={rate_info['xent_train_bpt']:.2f}"
    rans_str = ""
    if 'rans_bpt' in rate_info:
        rans_str = f"  rANS={rate_info['rans_bpt']:.2f}"
    if 'rans_train_bpt' in rate_info:
        rans_str += f"  rANS_train={rate_info['rans_train_bpt']:.2f}"
    print(f"  * Rate: {xent_str}  "
          f"H_emp={rate_info['empirical_entropy_bpt']:.2f}  "
          f"max={rate_info['max_rate_bpt']:.0f}{rans_str}  bits/token")

    # ================================================================
    #   (C) Segmentation evaluation (optional)
    # ================================================================
    if args.eval_seg:
        if is_clip:
            print(f"\n  [Segmentation] Skipped (no segmentation head for CLIP)")
        else:
            seg_feat_dir = Path(args.seg_feat_root) / args.backbone / args.layer

            print(f"\n{'=' * 60}")
            print(f"  [Segmentation] VOC2012 mIoU evaluation")
            print(f"  seg features: {seg_feat_dir}")
            print(f"{'=' * 60}")

            wrapper.backbone.cpu()
            if wrapper.head is not None:
                wrapper.head.cpu()
            torch.cuda.empty_cache()

            opq_seg = OPQSegmentationEvaluator(
                R=R_std, codebooks=codebooks_std,
                embedding_dim=args.embedding_dim, norm_mode=args.norm_mode,
                layer_idx=layer_idx,
                voc_root=args.voc_root,
                weights_root=wrapper.weights_root, device=device,
                feat_dim=D, model_name=args.backbone)
            seg_std = opq_seg.evaluate(
                seg_feat_dir=str(seg_feat_dir),
                image_list=args.seg_image_list, verbose=True)
            results['std_opq_miou'] = float(seg_std['miou'])
            print(f"  * Standard OPQ mIoU = {seg_std['miou']:.4f}")
            del opq_seg
            torch.cuda.empty_cache()

            codec_seg = CodecSegmentationEvaluator(
                codec=codec,
                norm_mode=args.norm_mode, layer_idx=layer_idx,
                voc_root=args.voc_root,
                weights_root=wrapper.weights_root, device=device,
                feat_dim=D, model_name=args.backbone)
            seg_spq = codec_seg.evaluate(
                seg_feat_dir=str(seg_feat_dir),
                image_list=args.seg_image_list, verbose=True)
            results['soft_pq_miou'] = float(seg_spq['miou'])
            results['delta_miou'] = float(seg_spq['miou'] - seg_std['miou'])
            print(f"  * Codec       mIoU = {seg_spq['miou']:.4f}")
            print(f"  * Δ(mIoU) = {results['delta_miou']:+.4f}")
            del codec_seg
            torch.cuda.empty_cache()

            # --- VOC segmentation rate ---
            print(f"  Computing VOC seg rate...")
            seg_features_flat = []
            with open(args.seg_image_list) as fimg:
                seg_names = [l.strip() for l in fimg if l.strip()]
            for sname in seg_names:
                fp = os.path.join(str(seg_feat_dir), f"{sname}.npy")
                if os.path.exists(fp):
                    darr = np.load(fp)
                    for si in range(darr.shape[0]):
                        seg_features_flat.append(darr[si])
            if seg_features_flat:
                codec.to(device)
                seg_rate_info = evaluate_rate(
                    seg_features_flat, codec, args.norm_mode, device,
                    batch_size=1, train_pmf=train_pmf,
                )
                results['seg_rate_info'] = seg_rate_info
                seg_rans = seg_rate_info.get('rans_bpt')
                seg_rans_tr = seg_rate_info.get('rans_train_bpt')
                D_feat = seg_features_flat[0].shape[-1]
                seg_rans_s = f"rANS={seg_rans:.2f}" if seg_rans else "rANS=N/A"
                seg_rans_tr_s = f"rANS_train={seg_rans_tr:.2f}" if seg_rans_tr else ""
                seg_bpfp_s = f"BPFP={seg_rans / D_feat:.4f}" if seg_rans else ""
                print(f"  * VOC Rate: {seg_rans_s}  {seg_rans_tr_s}  "
                      f"{seg_bpfp_s}  bits/token")
                del seg_features_flat
            else:
                print(f"  * VOC Rate: no features found")

    # ================================================================
    #   Summary & save
    # ================================================================
    opq_bpt = opq_groups * math.log2(args.K)
    codec_bpt = pq_groups * math.log2(args.K)
    print(f"\n{'=' * 60}")
    print(f"  Summary: {args.layer} K={args.K} emb={args.embedding_dim}"
          f"  bt={bt_dim if bt_dim > 0 else 'none'}")
    print(f"  bits/token: OPQ={opq_bpt:.0f} (G={opq_groups})"
          f"  codec={codec_bpt:.0f} (G={pq_groups})")
    print(f"  Standard OPQ  Acc={acc_std:.4f}  ΔL_ref={std_delta_l:.1f}")
    print(f"  Codec         Acc={acc_spq:.4f}  ΔL_ref={spq_delta_l:.1f}")
    print(f"  Δ(Acc)={delta_acc:+.4f}  Δ(ΔL_ref)={delta_dl:+.1f}")
    if 'std_opq_miou' in results:
        print(f"  Standard OPQ mIoU={results['std_opq_miou']:.4f}")
        print(f"  Codec       mIoU={results['soft_pq_miou']:.4f}")
        print(f"  Δ(mIoU)={results['delta_miou']:+.4f}")
    if 'rate_info' in results:
        ri = results['rate_info']
        xent_s = f"xent={ri['xent_rate_bpt']:.2f}"
        if 'xent_train_bpt' in ri:
            xent_s += f"  xent_train={ri['xent_train_bpt']:.2f}"
        rans_s = ""
        if 'rans_bpt' in ri:
            rans_s = f"  rANS={ri['rans_bpt']:.2f}"
        print(f"  Rate: {xent_s}  H_emp={ri['empirical_entropy_bpt']:.2f}  "
              f"max={ri['max_rate_bpt']:.0f}{rans_s} bits/token")
    print(f"  Train time: std={std_time:.1f}s  codec={train_time:.1f}s")
    print(f"{'=' * 60}")

    results['delta_acc'] = float(delta_acc)
    results['delta_dl_ref'] = float(delta_dl)
    results['train_time_std'] = float(std_time)
    results['train_time_spq'] = float(train_time)
    results['history'] = history

    out_dir = os.path.join(ORFC_ROOT, 'results', 'soft_pq', args.backbone)
    os.makedirs(out_dir, exist_ok=True)
    bt_tag = f"bt{bt_dim}" if bt_dim > 0 else "noBt"
    ws_tag = "ws" if args.warm_start_opq else "km"
    mse_tag = "_mse" if args.mse_loss else ""
    rate_tag = f"_lmbda{args.lmbda}" if args.lmbda > 0 else ""
    fz_tag = ""
    if args.freeze_transform:
        fz_tag += "_fzR"
    if args.freeze_codebooks:
        fz_tag += "_fzC"
    rot_tag = f"_rot{args.init_rotation.replace('_', '')}" if args.init_rotation != 'opq' else ""
    pfloor_tag = f"_pf{args.prior_floor}" if args.prior_floor > 0 else ""
    tau_tag = f"_tau{args.tau_start}" if args.tau_start > 0 else ""
    tau_end_tag = f"_te{args.tau_end}" if args.tau_end != 0.005 and args.tau_start > 0 else ""
    tau_sched_tag = f"_ts{args.tau_schedule[:3]}" if args.tau_schedule != "exponential" else ""
    tag = (f"{args.layer}_K{args.K}_emb{args.embedding_dim}"
           f"_{bt_tag}_{ws_tag}{mse_tag}{rate_tag}"
           f"{fz_tag}{rot_tag}"
           f"{pfloor_tag}{tau_tag}{tau_end_tag}{tau_sched_tag}"
           f"_lr{args.lr}_ep{args.epochs}_n{args.max_train_images}_s{args.seed}")
    out_path = os.path.join(out_dir, f'{tag}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")

    return results


# ================================================================
#                    CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Soft-PQ (ΔL_ref) experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--layer", type=str, default="blk20")
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--norm_mode", type=str, default="per_image")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--bottleneck_dim", type=int, default=0,
                        help="Transform bottleneck dim (0=no transform, D=same-dim OrthogonalTransform)")
    parser.add_argument("--init_rotation", type=str, default="opq",
                        choices=["opq", "random_orth", "pca", "identity"],
                        help="Fixed rotation for transform init (opq=use OPQ R)")
    parser.add_argument("--warm_start_opq", action="store_true",
                        help="Warm-start transform+codebooks from OPQ solution")
    parser.add_argument("--freeze_transform", action="store_true",
                        help="Freeze transform params (train codebooks only)")
    parser.add_argument("--freeze_codebooks", action="store_true",
                        help="Freeze codebook params (train transform only)")
    parser.add_argument("--mse_loss", action="store_true",
                        help="Use MSE in normalised space instead of ΔL_ref")
    parser.add_argument("--lmbda", type=float, default=0.0,
                        help="R-D Lagrange multiplier (FCVQ: J=R+lmbda*D, 0=no rate)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max gradient norm for clipping (0=disabled)")
    parser.add_argument("--prior_floor", type=float, default=0.0,
                        help="Uniform-prior mixing weight to prevent -log2(p) explosion")
    parser.add_argument("--tau_start", type=float, default=0.5,
                        help="Initial temperature for soft PQ (0=hard PQ only)")
    parser.add_argument("--tau_end", type=float, default=0.005,
                        help="Final temperature (annealed exponentially)")
    parser.add_argument("--tau_schedule", type=str, default="exponential",
                        choices=["exponential", "linear"],
                        help="Temperature annealing schedule")

    parser.add_argument("--max_train_images", type=int, default=500)
    parser.add_argument("--kmeans_max_samples", type=int, default=2_000_000,
                        help="Max flat vectors for OPQ / k-means init")
    parser.add_argument("--n_val", type=int, default=200,
                        help="Validation set size (held-out from train pool)")
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features"))
    parser.add_argument("--train_subset", type=str, default="train")
    parser.add_argument("--backbone", type=str, default="dinov2_vitl14",
                        help="Backbone name (dinov2_vitl14, dinov2_vitg14, or clip_vitl14)")
    parser.add_argument("--classnames", type=str,
                        default=os.path.join(PROJECT_ROOT, "utils", "classnames.txt"),
                        help="Classnames file for CLIP zero-shot (wnid + name per line)")
    parser.add_argument("--gt_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "utils",
                                             "imagenet_selected_label500.txt"))
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, load codec from --ckpt_path and evaluate")
    parser.add_argument("--ckpt_path", type=str, default="",
                        help="Checkpoint path for --eval_only mode")

    parser.add_argument("--eval_seg", action="store_true",
                        help="Evaluate segmentation mIoU (VOC2012)")
    parser.add_argument("--seg_feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features", "voc2012_100"))
    parser.add_argument("--voc_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "VOCdevkit", "VOC2012"))
    parser.add_argument("--seg_image_list", type=str,
                        default=os.path.join(PROJECT_ROOT, "utils", "voc2012_val_100.txt"))

    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
