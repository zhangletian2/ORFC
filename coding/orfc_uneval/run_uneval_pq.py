#!/usr/bin/env python
"""
Non-uniform bit-allocation PQ with frozen OPQ rotation.

Pipeline:
  Stage 1  Learn OPQ rotation (uniform K).  Run cls_ablation on the OPQ
           codec to obtain per-group importance scores ΔL_cls[g].
  Stage 2  Constrained DP bit allocation: distribute a fixed bit budget
           (= G * log2(K_ref), matching the uniform-K BPFP) across groups
           weighted by importance.
  Stage 3  Build FeatureCodec(VAQSoftPQ, OrthogonalTransform).  Per-group
           k-means init in the OPQ-rotated space.  Fine-tune codebooks
           only (frozen rotation, frozen allocation) with ΔL_ref.
  Stage 4  Evaluate: accuracy, ΔL_ref, variable-K rate.

Usage:
    python run_uneval_pq.py --layer blk20 --K 64 --embedding_dim 32 --epochs 100

    # Quick smoke test
    python run_uneval_pq.py --layer blk20 --K 16 --epochs 5 --max_train_images 50
"""

import os, sys, argparse, json, math, time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# ---------- path setup (same convention as orfc scripts) ----------
UNEVAL_ROOT = os.path.dirname(os.path.abspath(__file__))
CODING_ROOT = os.path.normpath(os.path.join(UNEVAL_ROOT, ".."))
ORFC_ROOT = os.path.join(CODING_ROOT, "orfc")
VAQ_ROOT = os.path.join(CODING_ROOT, "vaq")
PROJECT_ROOT = os.path.normpath(os.path.join(CODING_ROOT, ".."))

for _p in (UNEVAL_ROOT, ORFC_ROOT, VAQ_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_multilayer_calibrator import (
    set_seed, preload_features, load_gt, evaluate_accuracy,
)
from opq import (
    batch_normalize_gpu, batch_inv_normalize_gpu,
    batched_assign, learn_opq_rotation, batched_kmeans,
    learn_pca_rotation,
)
from backbone.wrapper import Dinov2Wrapper, SegmentationEvaluator
from soft_pq import (
    SoftPQ, OrthogonalTransform, FeatureCodec,
    FrozenTail,
)
from compute_cls_sensitivity import (
    learn_opq, build_codec_opq, cls_ablation,
    collect_normalised, stats,
)
from vaq_soft import VAQSoftPQ

from allocate_bits import allocate_bits_importance
from train_uneval import train_uneval_pq

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
import logging
from mmcv.utils import get_logger
get_logger('mmcv').setLevel(logging.WARNING)


# ================================================================
#       rANS / rate helpers (reuse from run_soft_pq)
# ================================================================
try:
    from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
    from compressai import ans as _ans
    _HAS_ANS = True
except (ImportError, ModuleNotFoundError):
    _HAS_ANS = False


def soft_pq_encode_decode_varK(features, codec, norm_mode, device,
                               chunk_images=None):
    """Encode/decode supporting VAQSoftPQ (variable K_per_group)."""
    codec.eval()
    N = len(features)

    if chunk_images is None:
        tokens_per_img = features[0].shape[0]
        G = codec.pq.G
        K_max = max(codec.pq.K_per_group)
        max_tokens = max(tokens_per_img, int(3e9 / (G * K_max * 4)))
        chunk_images = max(1, min(200, max_tokens // tokens_per_img))

    all_xhat = []
    with torch.no_grad():
        for start in range(0, N, chunk_images):
            end = min(start + chunk_images, N)
            X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
            B = X.shape[0]
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            Y_hat, _ = codec(Y)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            for i in range(B):
                all_xhat.append(X_hat[i].cpu().numpy())
            del X, Y, Mu, Std, Y_hat, X_hat
        torch.cuda.empty_cache()
    return all_xhat


def _codec_labels(features, codec, norm_mode, device, batch_size=32):
    codec.eval()
    pq = codec.pq
    all_labels = []
    with torch.no_grad():
        for s in range(0, len(features), batch_size):
            e = min(s + batch_size, len(features))
            X = torch.from_numpy(
                np.stack(features[s:e])
            ).float().to(device)
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
            _ = codec(Y)
            all_labels.append(pq._last_labels.cpu())
            del X, Y
    return torch.cat(all_labels, dim=1).numpy()  # [G, N_total]


def _histogram_pmf(labels, G, K_per_group, smoothing=1.0):
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
    if not _HAS_ANS:
        return None
    if isinstance(K_per_group, int):
        K_per_group = [K_per_group] * G
    encoder = _ans.RansEncoder()
    N = labels_np.shape[1]
    cdfs, cdf_sizes = [], []
    for g in range(G):
        p = torch.from_numpy(pmf_list[g]).float()
        overflow = (1.0 - p.sum()).clamp_min(0)
        p = torch.cat([p, overflow.unsqueeze(0)])
        cdf = _pmf_to_quantized_cdf(p.tolist(), precision)
        cdfs.append(cdf)
        cdf_sizes.append(K_per_group[g] + 2)
    symbols, cdf_indices = [], []
    for n in range(N):
        for g in range(G):
            symbols.append(int(labels_np[g, n]))
            cdf_indices.append(g)
    byte_string = encoder.encode_with_indexes(
        symbols, cdf_indices, cdfs, cdf_sizes, [0] * G)
    return len(byte_string) * 8 / N


# ================================================================
#       evaluate_rate_varK  (adapted for variable K_per_group)
# ================================================================

def evaluate_rate_varK(features, codec, norm_mode, device,
                       batch_size=32, train_pmf=None):
    """Rate evaluation supporting variable K_per_group."""
    pq = codec.pq
    G = pq.G
    K_per_group = list(pq.K_per_group)

    test_labels = _codec_labels(features, codec, norm_mode, device,
                                batch_size=batch_size)
    N = test_labels.shape[1]

    # Primary PMF
    if hasattr(pq, 'get_prior_pmf') and pq.use_rate:
        primary_pmf = pq.get_prior_pmf()
    elif train_pmf is not None:
        primary_pmf = train_pmf
    else:
        primary_pmf = [np.full(k, 1.0 / k) for k in K_per_group]

    if isinstance(primary_pmf, np.ndarray) and primary_pmf.ndim == 2:
        primary_pmf = [primary_pmf[g, :K_per_group[g]] for g in range(G)]

    # Cross-entropy with primary PMF
    xent_primary = 0.0
    for g in range(G):
        log2_p = np.log2(np.asarray(primary_pmf[g]) + 1e-30)
        xent_primary += -log2_p[test_labels[g]].sum()
    xent_primary_bpt = xent_primary / N

    # Cross-entropy with train histogram PMF
    xent_train_bpt = None
    if train_pmf is not None:
        xent_train = 0.0
        for g in range(G):
            log2_t = np.log2(np.asarray(train_pmf[g]) + 1e-30)
            xent_train += -log2_t[test_labels[g]].sum()
        xent_train_bpt = xent_train / N

    # Empirical entropy
    test_pmf = _histogram_pmf(test_labels, G, K_per_group, smoothing=0)
    empirical_entropy = 0.0
    for g in range(G):
        pg = test_pmf[g]
        pg = pg[pg > 0]
        empirical_entropy += -np.sum(pg * np.log2(pg))

    # rANS
    rans_bpt = _rans_encode_bpt(test_labels, primary_pmf, G, K_per_group)
    rans_train_bpt = None
    if train_pmf is not None:
        rans_train_bpt = _rans_encode_bpt(test_labels, train_pmf, G, K_per_group)

    max_rate = sum(math.log2(k) for k in K_per_group)
    result = {
        'xent_rate_bpt': float(xent_primary_bpt),
        'empirical_entropy_bpt': float(empirical_entropy),
        'max_rate_bpt': float(max_rate),
        'K_per_group': K_per_group,
    }
    if xent_train_bpt is not None:
        result['xent_train_bpt'] = float(xent_train_bpt)
    if rans_bpt is not None:
        result['rans_bpt'] = float(rans_bpt)
    if rans_train_bpt is not None:
        result['rans_train_bpt'] = float(rans_train_bpt)
    return result


# ================================================================
#       ΔL_ref evaluation  (from run_soft_pq, codec path)
# ================================================================

def evaluate_delta_l_ref(features, tail, norm_mode, device, codec,
                         batch_size=4):
    N = len(features)
    total_loss = 0.0
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            X = torch.from_numpy(np.stack(features[s:e])).float().to(device)
            B = X.shape[0]
            Y_teacher = tail.forward_nograd(X)
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            Y_hat, _ = codec(Y)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            Y_student = tail.forward_nograd(X_hat)
            loss = ((Y_teacher - Y_student) ** 2).sum().item() / B
            total_loss += loss * B
            del X, X_hat, Y_teacher, Y_student, Y, Mu, Std, Y_hat
            torch.cuda.empty_cache()
    return total_loss / N


# ================================================================
#       Segmentation evaluators
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
#       Per-group k-means in rotated space
# ================================================================

def _kmeans_single_group(data_g, k, device, max_iter=100, seed=42):
    """K-means for a single PQ group.  data_g: [N, d] numpy."""
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
        empty = counts == 0
        new_c = sums / counts.clamp_min(1.0)[:, None]
        if empty.any():
            ei = empty.nonzero(as_tuple=False).flatten()
            repl = rng.choice(n, int(ei.numel()), replace=n < int(ei.numel()))
            new_c[ei] = torch.from_numpy(data_g[repl]).float().to(device)
        shift = (new_c - centroids).norm(dim=1).max().item()
        centroids = new_c
        if shift < 1e-4:
            break
    return centroids.cpu().numpy().astype(np.float32)


def init_codebooks_varK(features_train, R_np, G, d, K_per_group,
                        norm_mode, device, max_vectors=2_000_000,
                        seed=42):
    """Collect rotated vectors, then per-group k-means with variable K."""
    flat = collect_normalised(features_train, norm_mode, device,
                              max_vecs=max_vectors)
    R_t = torch.from_numpy(R_np).float().to(device)
    Z = (flat.to(device) @ R_t).cpu().numpy()
    del flat, R_t
    torch.cuda.empty_cache()

    centroids = []
    for g in range(G):
        start, end = g * d, (g + 1) * d
        data_g = Z[:, start:end]
        k_g = K_per_group[g]
        c_g = _kmeans_single_group(data_g, k_g, device,
                                   max_iter=100, seed=seed + g)
        centroids.append(c_g)
    del Z
    torch.cuda.empty_cache()
    return centroids


# ================================================================
#       PCA rotation (consecutive group assignment, no interleave)
# ================================================================

def learn_pca_consecutive(features_train, D, G, d, K, norm_mode, device,
                          seed=42):
    """PCA rotation with consecutive eigenvector assignment + uniform k-means.

    Unlike ``learn_pca_rotation`` in opq.py (which interleaves eigenvectors
    to *balance* group variances), this assigns consecutive eigenvectors to
    each group so that groups are ordered by *decreasing* variance — matching
    the VAQ paper's strategy.

    Returns (R, codebooks, usage) with the same interface as ``learn_opq``.
    """
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
        R = eigvecs[:, idx].contiguous()          # [D, D]

    R_np = R.cpu().numpy().astype(np.float32)

    Z = X @ R                                     # [N, D]
    sub_3d = Z.reshape(N, G, d).permute(1, 0, 2).contiguous()  # [G, N, d]
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
#       Save / load codec  (variable-K aware)
# ================================================================

def save_uneval_codec(codec, path, extra_meta=None):
    pq = codec.pq
    meta = {
        'K_per_group': list(pq.K_per_group),
        'G': pq.G,
        'd': pq.d,
        'lmbda': pq.lmbda,
        'prior_floor': pq.prior_floor,
        'has_transform': codec.transform is not None,
        'transform_type': (type(codec.transform).__name__
                           if codec.transform else None),
    }
    if codec.transform is not None and hasattr(codec.transform, 'D'):
        meta['D'] = codec.transform.D
    meta['state_dict'] = codec.state_dict()
    if extra_meta:
        meta.update(extra_meta)
    torch.save(meta, path)


def load_uneval_codec(path, device='cuda'):
    meta = torch.load(path, map_location='cpu', weights_only=False)
    K_per_group = meta['K_per_group']
    d = meta['d']
    pq = VAQSoftPQ(
        K_per_group=K_per_group, d=d,
        lmbda=meta.get('lmbda', 0.0),
        prior_floor=meta.get('prior_floor', 0.0),
    )
    transform = None
    if meta.get('has_transform'):
        ttype = meta.get('transform_type')
        if ttype == 'OrthogonalTransform':
            transform = OrthogonalTransform(meta['D'])
    codec = FeatureCodec(pq, transform)
    codec.load_state_dict(meta['state_dict'])
    return codec.to(device).eval()


# ================================================================
#       Main experiment
# ================================================================

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    layer_idx = int(args.layer[-2:])

    print(f"\n{'#' * 70}")
    print(f"# Non-uniform Bit-Allocation PQ (frozen {args.rotation.upper()} rotation)")
    print(f"# layer={args.layer}, K_ref={args.K}, emb={args.embedding_dim}")
    print(f"# rotation={args.rotation}, monotonic={args.monotonic}")
    print(f"# epochs={args.epochs}, lr={args.lr}, lmbda={args.lmbda}")
    print(f"# alloc: min_bits={args.min_bits}, max_bits={args.max_bits}, "
          f"objective={args.alloc_objective}")
    print(f"# {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'#' * 70}")

    # ---- Load features ----
    train_dir = Path(args.feat_root) / args.train_subset / args.backbone / args.layer
    test_dir = Path(args.feat_root) / args.test_subset / args.backbone / args.layer
    train_files = sorted(train_dir.glob("*.npy"))
    test_files = sorted(test_dir.glob("*.npy"))

    print(f"\nData: train={len(train_files)}, test={len(test_files)}")
    if not train_files:
        raise FileNotFoundError(f"No train features in {train_dir}")
    if not test_files:
        raise FileNotFoundError(f"No test features in {test_dir}")

    features_train, _ = preload_features(train_files, num_workers=8)
    features_test, basenames_test = preload_features(test_files, num_workers=8)
    gt_test = load_gt(args.gt_path)

    D = features_train[0].shape[1]
    G = D // args.embedding_dim
    d = args.embedding_dim
    K_ref = args.K
    bit_budget = int(G * math.log2(K_ref))

    print(f"  D={D}, G={G}, d={d}, K_ref={K_ref}")
    print(f"  bit_budget={bit_budget} (= {G} * log2({K_ref}))")

    if args.max_train_images > 0 and len(features_train) > args.max_train_images:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(features_train), args.max_train_images, replace=False)
        features_train = [features_train[i] for i in idx]
        print(f"  Training subset: {len(features_train)} images")

    # Validation split
    n_val = min(args.n_val, len(features_train))
    rng_val = np.random.RandomState(args.seed + 1)
    val_idx = rng_val.choice(len(features_train), n_val, replace=False)
    val_features = [features_train[i] for i in val_idx]

    # Diag split for cls_ablation
    rng_diag = np.random.RandomState(args.seed)
    perm = rng_diag.permutation(len(features_train))
    n_diag = min(args.n_diag, len(features_train) // 2)
    diag_idx = perm[:n_diag]
    train_idx = perm[n_diag:]
    features_diag = [features_train[i] for i in diag_idx]
    features_tr = [features_train[i] for i in train_idx]

    results = {'config': vars(args)}

    # ---- Load backbone ----
    print(f"\nLoading DINOv2 ({args.backbone}) ...")
    wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone, device=device)

    # ================================================================
    #   Stage 1: Rotation + importance estimation
    # ================================================================
    rot_label = args.rotation.upper()
    print(f"\n{'=' * 60}")
    print(f"  Stage 1: {rot_label} rotation + cls_ablation importance")
    print(f"{'=' * 60}")

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

    if args.rotation == 'pca':
        print(f"  Learning PCA-consecutive (K={K_ref}, G={G}, d={d}) ...")
        t0 = time.time()
        R_opq, codebooks_opq, opq_usage = learn_pca_consecutive(
            features_tr, D, G, d, K_ref, args.norm_mode, device, args.seed)
        print(f"  PCA done ({time.time() - t0:.1f}s)")
    else:
        print(f"  Learning OPQ (K={K_ref}, G={G}, d={d}) ...")
        t0 = time.time()
        R_opq, codebooks_opq, opq_usage = learn_opq(
            features_tr, D, G, d, K_ref, args.norm_mode, device, args.seed)
        print(f"  OPQ done ({time.time() - t0:.1f}s)")

    print(f"  Building {rot_label} codec for cls_ablation ...")
    codec_opq = build_codec_opq(
        R_opq, codebooks_opq, G, K_ref, d, D, device, opq_usage)
    t0 = time.time()
    importance = cls_ablation(
        codec_opq, tail, head, features_diag, args.norm_mode, device,
        batch_size=args.ablation_batch_size)
    print(f"  cls_ablation done ({time.time() - t0:.1f}s)")
    print(f"  importance: {stats(importance)}")
    results['importance'] = stats(importance)

    del codec_opq
    head.cpu()
    torch.cuda.empty_cache()

    # ================================================================
    #   Stage 2: Constrained bit allocation
    # ================================================================
    mono_tag = " (monotonic gaps)" if args.monotonic else ""
    print(f"\n{'=' * 60}")
    print(f"  Stage 2: Importance-weighted bit allocation{mono_tag}")
    print(f"  bit_budget={bit_budget}, min_bits={args.min_bits}, "
          f"max_bits={args.max_bits}")
    print(f"{'=' * 60}")

    bits_alloc = allocate_bits_importance(
        importance, bit_budget,
        min_bits=args.min_bits, max_bits=args.max_bits,
        d=d, objective=args.alloc_objective,
        monotonic=args.monotonic,
    )
    K_per_group = [1 << b for b in bits_alloc]

    print(f"  bits_alloc = {bits_alloc}")
    print(f"  K_per_group = {K_per_group}")
    print(f"  sum(bits) = {sum(bits_alloc)}  "
          f"(budget={bit_budget}, uniform={int(math.log2(K_ref))} each)")
    print(f"  max_rate = {sum(math.log2(k) for k in K_per_group):.1f} bpt")
    results['bits_alloc'] = bits_alloc
    results['K_per_group'] = K_per_group

    # ================================================================
    #   Stage 3: Build codec + train codebooks
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  Stage 3: Build non-uniform codec & train codebooks (ΔL_ref)")
    print(f"{'=' * 60}")

    if not args.eval_only:
        # 3a. Frozen OPQ rotation
        transform = OrthogonalTransform(D)
        R_ws = R_opq.copy()
        if np.linalg.det(R_ws) < 0:
            R_ws[:, -1] *= -1
        transform.init_from_opq(R_ws)

        # 3b. Per-group k-means init in rotated space
        print(f"  Per-group k-means init (variable K) ...")
        t0 = time.time()
        centroids = init_codebooks_varK(
            features_tr, R_ws, G, d, K_per_group,
            args.norm_mode, device,
            max_vectors=args.kmeans_max_samples, seed=args.seed)
        print(f"  k-means done ({time.time() - t0:.1f}s)")

        # 3c. Build codec
        pq = VAQSoftPQ(
            K_per_group=K_per_group, d=d,
            lmbda=args.lmbda, prior_floor=args.prior_floor,
        ).to(device)
        pq.init_from_centroids(centroids)

        # Optionally init prior from OPQ usage counts (needs re-compute
        # for variable K — use uniform for now)
        codec = FeatureCodec(pq, transform=transform).to(device)

        # 3d. Train (frozen rotation, only codebooks)
        if not args.mse_loss:
            tail.to(device)
        torch.cuda.empty_cache()

        t0 = time.time()
        history, snapshots = train_uneval_pq(
            codec=codec,
            features_train=features_tr,
            norm_mode=args.norm_mode,
            batch_normalize_gpu=batch_normalize_gpu,
            batch_inv_normalize_gpu=batch_inv_normalize_gpu,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            val_features=val_features,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
            tau_schedule=args.tau_schedule,
            grad_clip=args.grad_clip,
            freeze_transform=True,
            freeze_codebooks=False,
            verbose=True,
            tail=None if args.mse_loss else tail,
        )
        train_time = time.time() - t0
        print(f"  Training done ({train_time:.1f}s)")
        results['train_time'] = float(train_time)
        results['history'] = history

        # Save checkpoint
        ckpt_dir = os.path.join(UNEVAL_ROOT, 'checkpoints', args.backbone)
        os.makedirs(ckpt_dir, exist_ok=True)
        rot_tag = f"_{args.rotation}" if args.rotation != "opq" else ""
        mono_tag = "_mono" if args.monotonic else ""
        rate_tag = f"_lmbda{args.lmbda}" if args.lmbda > 0 else ""
        tau_tag = f"_tau{args.tau_start}" if args.tau_start > 0 else ""
        loss_tag = "_mse" if args.mse_loss else ""
        ckpt_name = (f"{args.layer}_Kref{K_ref}_emb{d}"
                     f"_minb{args.min_bits}_maxb{args.max_bits}"
                     f"_{args.alloc_objective}{rot_tag}{mono_tag}"
                     f"{rate_tag}{tau_tag}{loss_tag}"
                     f"_lr{args.lr}_ep{args.epochs}"
                     f"_n{args.max_train_images}_s{args.seed}")
        ckpt_path = os.path.join(ckpt_dir, f"{ckpt_name}.pt")
        save_uneval_codec(codec, ckpt_path, extra_meta={
            'bits_alloc': bits_alloc,
            'importance': importance.tolist(),
        })
        print(f"  Saved: {ckpt_path}")

        tail.to('cpu')
        torch.cuda.empty_cache()
    else:
        # eval-only: load codec
        print(f"  Loading codec from {args.ckpt_path}")
        codec = load_uneval_codec(args.ckpt_path, device=device)

    # ================================================================
    #   Stage 4: Evaluate
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  Stage 4: Evaluation")
    print(f"{'=' * 60}")

    codec.eval()

    # --- OPQ baseline (uniform K) for comparison ---
    print(f"  [OPQ baseline] encode/decode ...")
    from run_soft_pq import pq_encode_decode_features, evaluate_delta_l_ref as eval_dl_std
    xhat_opq = pq_encode_decode_features(
        features_test, codebooks_opq, d, args.norm_mode, device,
        R=R_opq)

    wrapper.backbone.to(device)
    if wrapper.head is not None:
        wrapper.head.to(device)
    torch.cuda.empty_cache()

    acc_opq = evaluate_accuracy(
        xhat_opq, basenames_test, gt_test, wrapper, layer_idx, device)
    results['opq_acc'] = float(acc_opq)
    print(f"  OPQ Acc = {acc_opq:.4f}")
    del xhat_opq

    # --- Non-uniform codec ---
    print(f"  [Non-uniform codec] encode/decode ...")
    xhat_nu = soft_pq_encode_decode_varK(
        features_test, codec, args.norm_mode, device)
    acc_nu = evaluate_accuracy(
        xhat_nu, basenames_test, gt_test, wrapper, layer_idx, device)
    results['uneval_acc'] = float(acc_nu)
    delta_acc = acc_nu - acc_opq
    print(f"  Non-uniform Acc = {acc_nu:.4f} (Δ={delta_acc:+.4f})")
    del xhat_nu

    # --- ΔL_ref ---
    tail_blocks2 = list(wrapper.backbone.blocks[layer_idx + 1:])
    tail2 = FrozenTail(tail_blocks2, wrapper.backbone.norm, device=device)

    opq_dl = eval_dl_std(
        features_test, tail2, args.norm_mode, device,
        codebooks=codebooks_opq, R=R_opq, embedding_dim=d,
        batch_size=args.batch_size)
    results['opq_delta_l'] = float(opq_dl)

    nu_dl = evaluate_delta_l_ref(
        features_test, tail2, args.norm_mode, device, codec=codec,
        batch_size=args.batch_size)
    results['uneval_delta_l'] = float(nu_dl)
    delta_dl = nu_dl - opq_dl
    print(f"  OPQ ΔL_ref       = {opq_dl:.1f}")
    print(f"  Non-uniform ΔL_ref = {nu_dl:.1f} (Δ={delta_dl:+.1f})")

    tail2.to('cpu')
    torch.cuda.empty_cache()

    # --- Rate ---
    print(f"  Computing rate ...")
    train_labels = _codec_labels(
        features_tr, codec, args.norm_mode, device,
        batch_size=args.batch_size)
    train_pmf = _histogram_pmf(train_labels, G, K_per_group)

    rate_info = evaluate_rate_varK(
        features_test, codec, args.norm_mode, device,
        batch_size=args.batch_size, train_pmf=train_pmf)
    results['rate_info'] = rate_info

    xent_str = f"xent={rate_info['xent_rate_bpt']:.2f}"
    if 'xent_train_bpt' in rate_info:
        xent_str += f"  xent_train={rate_info['xent_train_bpt']:.2f}"
    rans_str = ""
    if 'rans_bpt' in rate_info:
        rans_str = f"  rANS={rate_info['rans_bpt']:.2f}"
    if 'rans_train_bpt' in rate_info:
        rans_str += f"  rANS_train={rate_info['rans_train_bpt']:.2f}"
    print(f"  Rate: {xent_str}  "
          f"H_emp={rate_info['empirical_entropy_bpt']:.2f}  "
          f"max={rate_info['max_rate_bpt']:.0f}{rans_str}  bpt")

    # --- Segmentation evaluation (optional) ---
    if args.eval_seg:
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
            R=R_opq, codebooks=codebooks_opq,
            embedding_dim=d, norm_mode=args.norm_mode,
            layer_idx=layer_idx,
            voc_root=args.voc_root,
            weights_root=wrapper.weights_root, device=device,
            feat_dim=D, model_name=args.backbone)
        seg_opq = opq_seg.evaluate(
            seg_feat_dir=str(seg_feat_dir),
            image_list=args.seg_image_list, verbose=True)
        results['opq_miou'] = float(seg_opq['miou'])
        print(f"  * OPQ       mIoU = {seg_opq['miou']:.4f}")
        del opq_seg
        torch.cuda.empty_cache()

        codec_seg = CodecSegmentationEvaluator(
            codec=codec,
            norm_mode=args.norm_mode, layer_idx=layer_idx,
            voc_root=args.voc_root,
            weights_root=wrapper.weights_root, device=device,
            feat_dim=D, model_name=args.backbone)
        seg_nu = codec_seg.evaluate(
            seg_feat_dir=str(seg_feat_dir),
            image_list=args.seg_image_list, verbose=True)
        results['uneval_miou'] = float(seg_nu['miou'])
        results['delta_miou'] = float(seg_nu['miou'] - seg_opq['miou'])
        print(f"  * Non-uniform mIoU = {seg_nu['miou']:.4f}")
        print(f"  * Δ(mIoU) = {results['delta_miou']:+.4f}")
        del codec_seg
        torch.cuda.empty_cache()

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
            seg_rate_info = evaluate_rate_varK(
                seg_features_flat, codec, args.norm_mode, device,
                batch_size=1, train_pmf=train_pmf)
            results['seg_rate_info'] = seg_rate_info
            seg_rans = seg_rate_info.get('rans_bpt')
            seg_rans_tr = seg_rate_info.get('rans_train_bpt')
            seg_rans_s = f"rANS={seg_rans:.2f}" if seg_rans else "rANS=N/A"
            seg_rans_tr_s = f"rANS_train={seg_rans_tr:.2f}" if seg_rans_tr else ""
            seg_bpfp_s = f"BPFP={seg_rans / D:.4f}" if seg_rans else ""
            print(f"  * VOC Rate: {seg_rans_s}  {seg_rans_tr_s}  "
                  f"{seg_bpfp_s}  bits/token")
            del seg_features_flat
        else:
            print(f"  * VOC Rate: no features found")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"  Summary: {args.layer} K_ref={K_ref} emb={d}")
    print(f"  bits_alloc  = {bits_alloc}")
    print(f"  K_per_group = {K_per_group}")
    print(f"  OPQ baseline   Acc={acc_opq:.4f}  ΔL_ref={opq_dl:.1f}")
    print(f"  Non-uniform    Acc={acc_nu:.4f}  ΔL_ref={nu_dl:.1f}")
    print(f"  Δ(Acc)={delta_acc:+.4f}  Δ(ΔL_ref)={delta_dl:+.1f}")
    if 'opq_miou' in results:
        print(f"  OPQ       mIoU={results['opq_miou']:.4f}")
        print(f"  Non-uniform mIoU={results['uneval_miou']:.4f}")
        print(f"  Δ(mIoU)={results['delta_miou']:+.4f}")
    print(f"{'=' * 60}")

    results['delta_acc'] = float(delta_acc)
    results['delta_dl_ref'] = float(delta_dl)

    # ---- Save results ----
    out_dir = os.path.join(UNEVAL_ROOT, 'results', 'uneval_pq', args.backbone)
    os.makedirs(out_dir, exist_ok=True)
    rot_tag = f"_{args.rotation}" if args.rotation != "opq" else ""
    mono_tag = "_mono" if args.monotonic else ""
    rate_tag = f"_lmbda{args.lmbda}" if args.lmbda > 0 else ""
    tau_tag = f"_tau{args.tau_start}" if args.tau_start > 0 else ""
    loss_tag = "_mse" if args.mse_loss else ""
    tag = (f"{args.layer}_Kref{K_ref}_emb{d}"
           f"_minb{args.min_bits}_maxb{args.max_bits}"
           f"_{args.alloc_objective}{rot_tag}{mono_tag}"
           f"{rate_tag}{tau_tag}{loss_tag}"
           f"_lr{args.lr}_ep{args.epochs}"
           f"_n{args.max_train_images}_s{args.seed}")
    if args.result_suffix:
        tag = f"{tag}_{args.result_suffix}"
    out_path = os.path.join(out_dir, f'{tag}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")
    return results


# ================================================================
#       CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Non-uniform bit-allocation PQ (frozen OPQ rotation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--layer", type=str, default="blk20")
    parser.add_argument("--K", type=int, default=64,
                        help="Reference uniform K (defines total bit budget)")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--norm_mode", type=str, default="per_image")

    # Rotation & allocation
    parser.add_argument("--rotation", type=str, default="opq",
                        choices=["opq", "pca"],
                        help="Rotation type: opq (balanced groups) or "
                             "pca (consecutive, variance-ordered groups)")
    parser.add_argument("--monotonic", action="store_true",
                        help="VAQ-style gaps constraint: sort groups by "
                             "importance and enforce smooth bit descent")
    parser.add_argument("--min_bits", type=int, default=1)
    parser.add_argument("--max_bits", type=int, default=10)
    parser.add_argument("--alloc_objective", type=str, default="rd",
                        choices=["rd", "linear"])

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lmbda", type=float, default=0.0,
                        help="R-D Lagrange multiplier (0 = no rate term)")
    parser.add_argument("--prior_floor", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--mse_loss", action="store_true",
                        help="Use MSE instead of ΔL_ref")
    parser.add_argument("--tau_start", type=float, default=0.5)
    parser.add_argument("--tau_end", type=float, default=0.005)
    parser.add_argument("--tau_schedule", type=str, default="exponential",
                        choices=["exponential", "linear"])

    # Data
    parser.add_argument("--max_train_images", type=int, default=5000)
    parser.add_argument("--kmeans_max_samples", type=int, default=2_000_000)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--n_diag", type=int, default=200,
                        help="Images reserved for cls_ablation diagnostics")
    parser.add_argument("--ablation_batch_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features"))
    parser.add_argument("--train_subset", type=str, default="train")
    parser.add_argument("--test_subset", type=str, default="test")
    parser.add_argument("--backbone", type=str, default="dinov2_vitl14")
    parser.add_argument("--gt_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "utils",
                                             "imagenet_selected_label500.txt"))
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--result_suffix", type=str, default="")

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
    run_experiment(args)


if __name__ == '__main__':
    main()
