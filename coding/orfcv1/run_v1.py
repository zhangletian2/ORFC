#!/usr/bin/env python
"""
ORFC-v1 experiment runner.

Implements the full pipeline from TODO.md Sections 1-8 + review §11-12:
  1. Deterministic data split + seed management
  2. OPQ baseline (shared artifact, §11.8)
  3. Codec training with elastic loss
     Phase A: train U only  |  Phase B: joint  |  Phase C: alternating
  4. Evaluation: D0, accuracy, rate, per-group diagnostics
  5. Held-out elasticity evaluation on val and test (§11.4)
  6. Structured result JSON (§11.8)

Usage:
    python run_v1.py --layer blk20 --K 8 --beta 0.03 --alpha 0.1

    # Beta=0 regression (pure D0):
    python run_v1.py --layer blk20 --K 8 --beta 0

    # Phase A: fix codebooks, train U only:
    python run_v1.py --layer blk20 --K 8 --beta 0.03 --freeze_codebooks

    # Phase B: joint training:
    python run_v1.py --layer blk20 --K 8 --beta 0.03 --step_mode joint

    # Phase C: alternating:
    python run_v1.py --layer blk20 --K 8 --beta 0.03 --step_mode alternating
"""

import os, sys, argparse, json, math, time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

V1_ROOT = os.path.dirname(os.path.abspath(__file__))
ORFC_ROOT = os.path.normpath(os.path.join(V1_ROOT, '..', 'orfc'))
PROJECT_ROOT = os.path.normpath(os.path.join(V1_ROOT, '..', '..', '..'))
CODE_VERSION = 'orfcv1-delivery-20260726'

if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)
if V1_ROOT not in sys.path:
    sys.path.insert(0, V1_ROOT)

from run_multilayer_calibrator import (
    set_seed, preload_features, load_gt, evaluate_accuracy,
)
from opq import (
    batch_normalize_gpu, batch_inv_normalize_gpu, batched_assign,
    learn_opq_rotation,
)
from backbone.wrapper import Dinov2Wrapper
from soft_pq import (
    SoftPQ, OrthogonalTransform, FeatureTransform,
    FrozenTail, soft_pq_encode_decode,
)

from data_utils import (
    make_split, save_manifest, make_seeds,
    build_result_skeleton, git_status_porcelain,
)
from codec_v1 import (
    FeatureCodecV1, reconstruction_audit,
    save_codec_v1, load_codec_v1,
)
from train_v1 import train_v1, evaluate_heldout_elasticity

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
import logging
try:
    from mmcv.utils import get_logger
    logger = get_logger('mmcv')
except ImportError:
    logger = logging.getLogger('mmcv')
logger.setLevel(logging.WARNING)

# ================================================================
#  Standard OPQ encode/decode (from run_soft_pq.py)
# ================================================================

def pq_encode_decode_features(features, codebooks, embedding_dim, norm_mode,
                              device, R=None, chunk_images=500):
    num_groups = len(codebooks)
    N_img = len(features)
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
            Y_hat = Y_hat.reshape(B, -1, C)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        for i in range(B):
            all_xhat.append(X_hat[i].cpu().numpy())
        del X, Y, Mu, Std, Z, z_3d, z_hat_3d, flat_hat, Y_hat, X_hat
        torch.cuda.empty_cache()
    return all_xhat


# ================================================================
#  ΔL_ref evaluation (no-grad)
# ================================================================

def evaluate_delta_l_ref(features, tail, norm_mode, device,
                         codec=None,
                         codebooks=None, R=None, embedding_dim=None,
                         batch_size=4):
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
#  Rate evaluation
# ================================================================

try:
    from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
    from compressai import ans as _ans
    _HAS_ANS = True
except (ImportError, ModuleNotFoundError):
    _HAS_ANS = False


def _codec_labels(features, codec, norm_mode, device, batch_size=32):
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
    return torch.cat(all_labels, dim=1).numpy()


def _histogram_pmf(labels, G, K, smoothing=1.0):
    pmfs = []
    for g in range(G):
        counts = np.zeros(K, dtype=np.float64)
        np.add.at(counts, labels[g], 1)
        counts += smoothing
        pmfs.append(counts / counts.sum())
    return pmfs


def evaluate_rate(features, codec, norm_mode, device, batch_size=32,
                  train_pmf=None):
    pq = codec.pq
    G, K = pq.G, pq.K
    test_labels = _codec_labels(features, codec, norm_mode, device,
                                batch_size=batch_size)
    N = test_labels.shape[1]

    test_pmf = _histogram_pmf(test_labels, G, K, smoothing=0)
    empirical_entropy = 0.0
    for g in range(G):
        pg = test_pmf[g]
        pg = pg[pg > 0]
        empirical_entropy += -np.sum(pg * np.log2(pg))

    result = {
        'empirical_entropy_bpt': float(empirical_entropy),
        'max_rate_bpt': float(G * math.log2(K)),
    }

    if train_pmf is not None:
        xent_train = 0.0
        for g in range(G):
            log2_t = np.log2(train_pmf[g] + 1e-30)
            xent_train += -log2_t[test_labels[g]].sum()
        result['xent_train_bpt'] = float(xent_train / N)

        if _HAS_ANS:
            encoder = _ans.RansEncoder()
            cdfs, cdf_sizes = [], []
            for g in range(G):
                p = torch.from_numpy(train_pmf[g]).float()
                overflow = (1.0 - p.sum()).clamp_min(0)
                p = torch.cat([p, overflow.unsqueeze(0)])
                cdf = _pmf_to_quantized_cdf(p.tolist(), 16)
                cdfs.append(cdf)
                cdf_sizes.append(K + 2)
            symbols, cdf_indices = [], []
            for n in range(N):
                for g in range(G):
                    symbols.append(int(test_labels[g, n]))
                    cdf_indices.append(g)
            byte_string = encoder.encode_with_indexes(
                symbols, cdf_indices, cdfs, cdf_sizes, [0] * G)
            result['rans_train_bpt'] = float(len(byte_string) * 8 / N)

    return result


# ================================================================
#  V1 codec encode/decode for evaluation
# ================================================================

def codec_encode_decode(features, codec, norm_mode, device,
                        chunk_images=200):
    codec.eval()
    N = len(features)
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


# ================================================================
#  OPQ artifact management (§11.8)
# ================================================================

def _opq_artifact_path(args, seeds):
    """Deterministic path for shared OPQ initialization artifact (§13.1)."""
    art_dir = os.path.join(V1_ROOT, 'artifacts', args.backbone)
    os.makedirs(art_dir, exist_ok=True)
    name = (f"opq_{args.layer}_K{args.K}_emb{args.embedding_dim}"
            f"_{args.norm_mode}_n{args.n_train}"
            f"_ss{seeds['split']}_is{seeds['init']}"
            f"_oi{args.opq_iter}_ki{args.kmeans_iter}"
            f"_km{args.kmeans_max_samples}"
            f"_{CODE_VERSION}.npz")
    return os.path.join(art_dir, name)


def save_opq_artifact(path, R, codebooks, hist, meta):
    """Atomic save with metadata for reproducibility checks (§13.1)."""
    import tempfile
    dir_name = os.path.dirname(path) or '.'
    fd, tmp_base = tempfile.mkstemp(dir=dir_name, suffix='_tmp')
    os.close(fd)
    os.remove(tmp_base)
    tmp_npz = tmp_base  # np.savez adds .npz automatically
    try:
        save_dict = {
            'R': R,
            'meta_json': json.dumps(meta),
        }
        for g, cb in enumerate(codebooks):
            save_dict[f'cb_{g}'] = cb
        np.savez(tmp_npz, **save_dict)
        actual_tmp = tmp_npz + '.npz' if not tmp_npz.endswith('.npz') else tmp_npz
        os.replace(actual_tmp, path)
    except BaseException:
        for candidate in [tmp_npz, tmp_npz + '.npz']:
            if os.path.exists(candidate):
                os.remove(candidate)
        raise


def load_opq_artifact(path, G, expected_meta=None):
    """Load OPQ artifact and strictly verify its metadata contract."""
    with np.load(path, allow_pickle=True) as data:
        R = data['R'].copy()
        codebooks = [data[f'cb_{g}'].copy() for g in range(G)]
        if 'meta_json' not in data:
            raise ValueError(f"OPQ artifact has no metadata: {path}")
        meta = json.loads(str(data['meta_json']))
    if expected_meta is not None:
        missing = sorted(set(expected_meta) - set(meta))
        if missing:
            raise ValueError(
                f"OPQ artifact missing metadata fields: {missing}")
        for key, expected in expected_meta.items():
            if meta[key] != expected:
                raise ValueError(
                    f"OPQ artifact mismatch: {key}={meta[key]!r} "
                    f"vs expected {expected!r}")
    return R, codebooks, meta


def _opq_lock_path(art_path):
    return art_path + '.lock'


def _acquire_opq_lock(lock_path, timeout=600):
    """Simple file lock to prevent concurrent OPQ creation (§13.1)."""
    import time as _time
    t0 = _time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            if _time.time() - t0 > timeout:
                raise TimeoutError(f"OPQ lock timeout ({timeout}s): {lock_path}")
            _time.sleep(2)


def _release_opq_lock(lock_path):
    try:
        os.remove(lock_path)
    except OSError:
        pass


def check_cayley_init(transform, R_opq, verbose=True):
    """Verify OPQ→Cayley initialization quality (§13.1d).

    Returns dict with:
        rel_error   : ||U_Cayley − R_OPQ||_F / ||R_OPQ||_F
        cond_R_plus_I : cond(R_OPQ + I)
        orth_error  : ||U'U − I||_F
    """
    import torch
    R_t = torch.from_numpy(R_opq).float()
    U = transform.get_rotation().detach().cpu()

    rel_err = (U - R_t).norm().item() / R_t.norm().item()
    cond_val = float(np.linalg.cond(R_opq + np.eye(R_opq.shape[0])))
    orth_err = transform.orth_error()

    result = {
        'rel_error': rel_err,
        'cond_R_plus_I': cond_val,
        'orth_error': orth_err,
    }
    if verbose:
        print(f"  Cayley init check:")
        print(f"    ||U_Cayley−R_OPQ||/||R_OPQ|| = {rel_err:.2e}")
        print(f"    cond(R_OPQ+I) = {cond_val:.2e}")
        print(f"    ||U'U−I||_F = {orth_err:.2e}")
        if rel_err > 0.01:
            print(f"    WARNING: Cayley init relative error > 1%")
        if cond_val > 1e6:
            print(f"    WARNING: cond(R+I) very large, Cayley may be unstable")
    return result


# ================================================================
#  Shared feature / teacher cache infrastructure
# ================================================================

def _cache_dir(args):
    d = os.path.join(V1_ROOT, 'artifacts', args.backbone, 'cache')
    os.makedirs(d, exist_ok=True)
    return d


def _features_cache_path(args, seeds, split_name, n_images):
    return os.path.join(
        _cache_dir(args),
        f"features_{split_name}_{args.layer}_n{n_images}"
        f"_ss{seeds['split']}.npy")


def _teacher_cache_path(args, seeds, split_name, n_images):
    return os.path.join(
        _cache_dir(args),
        f"teacher_{split_name}_{args.layer}_n{n_images}"
        f"_ss{seeds['split']}.npy")


def _save_array_atomic(path, arr):
    """Save numpy array atomically (temp + rename).

    np.save appends '.npy' if the filename doesn't already end with it,
    so the temp file must use '.npy' suffix to avoid writing to a
    different path than we rename.
    """
    import tempfile
    d = os.path.dirname(path) or '.'
    fd, tmp = tempfile.mkstemp(dir=d, suffix='.npy')
    os.close(fd)
    try:
        np.save(tmp, arr)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _load_or_create_array(path, create_fn, lock_timeout=600, mmap=True):
    """Load cached array, or create it exactly once across processes.

    Uses the same file-lock pattern as OPQ artifacts (§13.1).
    Returns a numpy array (mmap'd if mmap=True and file existed or was created).
    """
    if os.path.isfile(path):
        return np.load(path, mmap_mode='r' if mmap else None)

    lock = path + '.lock'
    _acquire_opq_lock(lock, timeout=lock_timeout)
    try:
        if os.path.isfile(path):
            return np.load(path, mmap_mode='r' if mmap else None)
        arr = create_fn()
        _save_array_atomic(path, arr)
        if mmap:
            return np.load(path, mmap_mode='r')
        return arr
    finally:
        _release_opq_lock(lock)


def _compute_teacher_cache(features_array, tail, batch_size, device):
    """Compute teacher outputs for all images."""
    N = features_array.shape[0]
    cache = np.empty_like(features_array)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X = torch.from_numpy(
                features_array[start:end]).float().to(device)
            cache[start:end] = tail.forward_nograd(X).cpu().numpy()
            del X
    torch.cuda.empty_cache()
    return cache


# ================================================================
#  Main experiment
# ================================================================

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    layer_idx = int(args.layer[-2:])
    seeds = make_seeds(args.seed)
    set_seed(seeds['base'])

    D = 1024  # ViT-L/14
    bt_dim = args.bottleneck_dim
    Dp = bt_dim if bt_dim > 0 else D
    num_groups = Dp // args.embedding_dim
    bits_per_token = num_groups * math.log2(args.K)

    # Determine phase tag
    if args.freeze_codebooks and not args.freeze_transform:
        phase_tag = "A"
        phase_desc = "U only"
    elif args.step_mode == 'alternating':
        phase_tag = "C"
        phase_desc = "alternating"
    else:
        phase_tag = "B"
        phase_desc = "joint"

    print(f"\n{'#' * 70}")
    print(f"# ORFC-v1 Elastic Experiment")
    print(f"# layer={args.layer}, K={args.K}, G={num_groups}, d={args.embedding_dim}")
    print(f"# β={args.beta}, α={args.alpha}, τ_el={args.elastic_tau}")
    print(f"# bits/token={bits_per_token:.0f}")
    print(f"# Phase: {phase_tag} ({phase_desc})")
    print(f"# step_mode: {args.step_mode}")
    print(f"# seeds: {seeds}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    # ---- Load features ----
    train_dir = Path(args.feat_root) / args.train_subset / args.backbone / args.layer
    test_dir = Path(args.feat_root) / args.test_subset / args.backbone / args.layer
    train_files = sorted(train_dir.glob("*.npy"))
    test_files = sorted(test_dir.glob("*.npy"))
    if args.n_test > 0:
        test_files = test_files[:args.n_test]

    if not train_files:
        raise FileNotFoundError(f"No train features in {train_dir}")
    if not test_files:
        raise FileNotFoundError(f"No test features in {test_dir}")

    print(f"\nData: train_pool={len(train_files)}, test={len(test_files)}")

    # ---- Deterministic split (Section 1.3) ----
    split, manifest = make_split(
        train_files, test_files,
        n_train=args.n_train, n_val=args.n_val,
        split_seed=seeds['split'],
    )

    out_dir = os.path.join(
        V1_ROOT, 'results', args.backbone, args.run_id)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, f'split_manifest_s{args.seed}.json')
    save_manifest(manifest, manifest_path)
    print(f"  Split: train={len(split['train'])}, "
          f"val={len(split['val'])}, test={len(split['test'])}")
    print(f"  Manifest: {manifest_path}")

    # ---- Load split features (shared cache for stacked arrays) ----
    gt_test = None if args.skip_test_eval else load_gt(args.gt_path)

    feat_train_path = _features_cache_path(args, seeds, 'train', args.n_train)
    feat_val_path = _features_cache_path(args, seeds, 'val', args.n_val)

    if os.path.isfile(feat_train_path) and os.path.isfile(feat_val_path):
        print(f"  Loading cached stacked features (mmap)...")
        features_train_array = np.load(feat_train_path, mmap_mode='r')
        features_val_array = np.load(feat_val_path, mmap_mode='r')
        if args.skip_test_eval:
            features_test, basenames_test = None, []
        else:
            features_test, basenames_test = preload_features(
                split['test'], num_workers=4)
    else:
        features_train, _ = preload_features(split['train'], num_workers=4)
        features_val, _ = preload_features(split['val'], num_workers=4)
        if args.skip_test_eval:
            features_test, basenames_test = None, []
        else:
            features_test, basenames_test = preload_features(
                split['test'], num_workers=4)

        features_train_array = np.stack(features_train)
        features_val_array = np.stack(features_val)
        del features_train, features_val

        print(f"  Saving stacked features to cache...")
        _save_array_atomic(feat_train_path, features_train_array)
        _save_array_atomic(feat_val_path, features_val_array)
        features_train_array = np.load(feat_train_path, mmap_mode='r')
        features_val_array = np.load(feat_val_path, mmap_mode='r')

    T_tokens = features_train_array.shape[1]
    D = features_train_array.shape[2]
    print(f"  D={D}, T={T_tokens}")

    # ---- Build result skeleton ----
    config = vars(args)
    results = build_result_skeleton(config, seeds, manifest)
    results['run_id'] = args.run_id
    results['code_version'] = CODE_VERSION
    results['source_files'] = [
        'run_v1.py', 'train_v1.py', 'codec_v1.py', 'elastic.py',
        'data_utils.py', 'run_exp_v1.sh', 'select_phasec.py',
        'select_final.py', 'run_smoke_v1.sh',
        'benchmark_probe_chunks_v1.sh', 'test_v1_contracts.py',
        '../orfc/soft_pq.py', '../orfc/opq.py',
    ]

    # Record tracked and untracked changes without file checksums.
    git_porcelain = git_status_porcelain(V1_ROOT)
    if git_porcelain is not None:
        results['git_status_porcelain'] = git_porcelain

    # ---- Load backbone ----
    print(f"\nLoading DINOv2 ({args.backbone})...")
    wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone,
                            device=device)
    n_blocks = len(wrapper.backbone.blocks)
    tail_blocks = list(wrapper.backbone.blocks[layer_idx + 1:])
    norm_layer = wrapper.backbone.norm
    n_tail = len(tail_blocks)
    print(f"  Tail: {n_tail} blocks + norm")

    # ================================================================
    #  (A) Standard OPQ baseline (shared artifact, §11.8)
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  [Standard OPQ] baseline")
    print(f"{'=' * 60}")

    opq_groups = D // args.embedding_dim
    opq_art_path = _opq_artifact_path(args, seeds)

    opq_expected_meta = {
        'code_version': CODE_VERSION,
        'layer': args.layer, 'K': args.K,
        'embedding_dim': args.embedding_dim,
        'norm_mode': args.norm_mode,
        'n_train': args.n_train,
        'split_seed': seeds['split'],
        'init_seed': seeds['init'],
        'opq_iter': args.opq_iter,
        'kmeans_iter': args.kmeans_iter,
        'kmeans_max_samples': args.kmeans_max_samples,
        'train_feature_ids': manifest['train_basenames'],
    }

    if os.path.isfile(opq_art_path):
        print(f"  Loading shared OPQ artifact: {opq_art_path}")
        R_std, codebooks_std, opq_meta = load_opq_artifact(
            opq_art_path, opq_groups, expected_meta=opq_expected_meta)
        std_time = 0.0
    else:
        lock_path = _opq_lock_path(opq_art_path)
        print(f"  Acquiring OPQ lock: {lock_path}")
        _acquire_opq_lock(lock_path)
        try:
            if os.path.isfile(opq_art_path):
                print(f"  OPQ created by another process, loading...")
                R_std, codebooks_std, opq_meta = load_opq_artifact(
                    opq_art_path, opq_groups, expected_meta=opq_expected_meta)
                std_time = 0.0
            else:
                wrapper.backbone.cpu()
                if wrapper.head is not None:
                    wrapper.head.cpu()
                torch.cuda.empty_cache()

                t0 = time.time()
                all_vectors = []
                for start in range(0, len(features_train_array), 200):
                    end = min(start + 200, len(features_train_array))
                    X = torch.from_numpy(
                        features_train_array[start:end]).float().to(device)
                    with torch.no_grad():
                        Y, _, _ = batch_normalize_gpu(X, mode=args.norm_mode)
                    all_vectors.append(Y.reshape(-1, D).cpu().numpy())
                    del X, Y
                full_vectors = np.concatenate(all_vectors, axis=0)
                del all_vectors

                max_flat = args.kmeans_max_samples // opq_groups
                if full_vectors.shape[0] > max_flat:
                    rng2 = np.random.RandomState(seeds['init'])
                    idx2 = rng2.choice(full_vectors.shape[0], max_flat,
                                       replace=False)
                    full_vectors = full_vectors[idx2]

                R_std, codebooks_std, hist_std = learn_opq_rotation(
                    full_vectors, opq_groups, args.embedding_dim, args.K,
                    max_iter_opq=args.opq_iter,
                    max_iter_kmeans=args.kmeans_iter,
                    device=device, verbose=False,
                )
                del full_vectors
                torch.cuda.empty_cache()
                std_time = time.time() - t0
                print(f"  OPQ done: MSE={hist_std[-1][0]:.8f} ({std_time:.1f}s)")

                save_opq_artifact(opq_art_path, R_std, codebooks_std,
                                  hist_std, opq_expected_meta)
                print(f"  Saved OPQ artifact: {opq_art_path}")

                wrapper.backbone.to(device)
                if wrapper.head is not None:
                    wrapper.head.to(device)
                torch.cuda.empty_cache()
        finally:
            _release_opq_lock(lock_path)

    tail = FrozenTail(tail_blocks, norm_layer, device=device)
    if args.skip_test_eval:
        acc_std = None
        std_delta_l = evaluate_delta_l_ref(
            features_val_array, tail, args.norm_mode, device,
            codebooks=codebooks_std, R=R_std,
            embedding_dim=args.embedding_dim,
            batch_size=args.batch_size,
        )
        results['std_opq_val_delta_l'] = float(std_delta_l)
        print(f"  * OPQ validation D0 = {std_delta_l:.1f}")
    else:
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
        print(f"  * OPQ Acc = {acc_std:.4f}")
        del xhat_std
        std_delta_l = evaluate_delta_l_ref(
            features_test, tail, args.norm_mode, device,
            codebooks=codebooks_std, R=R_std,
            embedding_dim=args.embedding_dim,
            batch_size=args.batch_size,
        )
        results['std_opq_delta_l'] = float(std_delta_l)
        print(f"  * OPQ D0 = {std_delta_l:.1f}")

    # ---- Shared teacher cache (computed once, reused by all processes) ----
    tc_train_path = _teacher_cache_path(args, seeds, 'train', args.n_train)
    tc_val_path = _teacher_cache_path(args, seeds, 'val', args.n_val)

    def _make_train_tc():
        print(f"  Computing shared teacher cache (train, {args.n_train} images)...")
        return _compute_teacher_cache(
            features_train_array, tail, args.batch_size, device)

    def _make_val_tc():
        print(f"  Computing shared teacher cache (val, {args.n_val} images)...")
        return _compute_teacher_cache(
            features_val_array, tail, args.batch_size, device)

    t_tc = time.time()
    teacher_cache_train = _load_or_create_array(
        tc_train_path, _make_train_tc, mmap=True)
    teacher_cache_val = _load_or_create_array(
        tc_val_path, _make_val_tc, mmap=True)
    print(f"  Teacher caches ready: "
          f"train={teacher_cache_train.nbytes/1e9:.1f}GB "
          f"val={teacher_cache_val.nbytes/1e9:.1f}GB "
          f"({time.time()-t_tc:.1f}s)")

    # ================================================================
    #  (B) V1 Codec training
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  [V1 Codec] Phase {phase_tag}  β={args.beta}, α={args.alpha}")
    print(f"{'=' * 60}")

    for i, blk in enumerate(wrapper.backbone.blocks):
        if i <= layer_idx:
            blk.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()

    # Build transform
    transform = OrthogonalTransform(D) if bt_dim == D else None
    if bt_dim > 0 and bt_dim != D:
        transform = FeatureTransform(D, bt_dim)

    # Warm-start from OPQ
    R_ws = R_std.copy()
    C_ws = [c.copy() for c in codebooks_std]
    if np.linalg.det(R_ws) < 0:
        R_ws[:, -1] *= -1
        C_ws[-1][:, -1] *= -1
        print(f"  det(R_opq)<0: flipped to SO(D)")

    # OPQ→Cayley initialization check (§13.1d)
    if transform is not None and hasattr(transform, 'init_from_opq'):
        transform_check = OrthogonalTransform(D).to(device)
        transform_check.init_from_opq(R_ws)
        cayley_check = check_cayley_init(transform_check, R_ws, verbose=True)
        results['cayley_init_check'] = cayley_check
        if cayley_check['rel_error'] > 0.1:
            raise RuntimeError(
                f"Cayley init failed: rel_error={cayley_check['rel_error']:.4f} > 0.1")
        del transform_check

    t0 = time.time()
    codec, history = train_v1(
        features_array=features_train_array,
        T_tokens=T_tokens,
        tail=tail,
        G=num_groups, K=args.K, d=args.embedding_dim,
        norm_mode=args.norm_mode,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        seeds=seeds,
        val_array=features_val_array,
        verbose=True,
        transform=transform,
        R_init=R_ws,
        codebooks_init=C_ws,
        kmeans_max_samples=args.kmeans_max_samples,
        beta=args.beta,
        alpha=args.alpha,
        elastic_tau=args.elastic_tau,
        n_groups_per_batch=args.n_groups_per_batch,
        compute_diagnostics=True,
        n_interaction_pairs=args.n_interaction_pairs,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        tau_schedule=args.tau_schedule,
        freeze_transform=args.freeze_transform,
        freeze_codebooks=args.freeze_codebooks,
        lmbda=0.0,
        grad_clip=args.grad_clip,
        step_mode=args.step_mode,
        alt_u_steps=args.alt_u_steps,
        alt_c_steps=args.alt_c_steps,
        val_elasticity_interval=args.val_elasticity_interval,
        probe_group_chunk=args.probe_group_chunk,
        heldout_probe_group_chunk=args.heldout_probe_group_chunk,
        teacher_cache_precomputed=teacher_cache_train,
        val_teacher_cache_precomputed=teacher_cache_val,
    )
    train_time = time.time() - t0
    print(f"  Training: {train_time:.1f}s")

    results['history'] = history

    # Save checkpoint (§13.1: fully disambiguated filename + atomic write)
    ckpt_dir = os.path.join(
        V1_ROOT, 'checkpoints', args.backbone, args.run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    fz_tag = ""
    if args.freeze_transform:
        fz_tag += "_fzR"
    if args.freeze_codebooks:
        fz_tag += "_fzC"
    step_tag = f"_{args.step_mode}"
    if args.step_mode == 'alternating':
        step_tag += f"_u{args.alt_u_steps}c{args.alt_c_steps}"
    ckpt_name = (f"{args.layer}_K{args.K}_emb{args.embedding_dim}"
                 f"_b{args.beta}_a{args.alpha}"
                 f"_lr{args.lr}_tau{args.elastic_tau}"
                 f"_gpb{args.n_groups_per_batch}"
                 f"{fz_tag}{step_tag}"
                 f"_ep{args.epochs}_s{args.seed}")
    if args.result_suffix:
        ckpt_name += f"_{args.result_suffix}"
    ckpt_path = os.path.join(ckpt_dir, f"{ckpt_name}.pt")
    save_codec_v1(codec, ckpt_path)
    print(f"  Saved: {ckpt_path}")

    # ---- Reconstruction audit (Section 2) ----
    print(f"\n  Reconstruction audit...")
    audit_features = (
        features_val_array[:50] if args.skip_test_eval
        else features_test[:50])
    audit_info = reconstruction_audit(
        audit_features, codec, args.norm_mode, device,
        batch_size=args.batch_size, return_details=True,
    )
    results['reconstruction_audit'] = audit_info
    print(
        f"  Audit: quant={audit_info['quantisation_relative_error']:.2e} "
        f"combined={audit_info['combined_relative_error']:.2e} "
        f"roundtrip={audit_info['roundtrip_relative_magnitude']:.2e} "
        f"{'PASS' if audit_info['passed'] else 'FAIL'}")
    if not audit_info['passed']:
        raise RuntimeError(
            f"Reconstruction audit failed: {audit_info}")

    # ---- Orthogonality check (Section 1.7) ----
    if transform and hasattr(transform, 'orth_error'):
        orth_err = transform.orth_error()
        results['final_orth_error'] = float(orth_err)
        print(f"  ||U'U-I||_F = {orth_err:.2e}")

    # ---- Final held-out validation elasticity (§11.4) ----
    val_el_final = None
    if history and 'val_elasticity' in history[-1]:
        val_el_final = history[-1]['val_elasticity']
    elif args.force_val_heldout:
        print(f"\n  Computing forced validation held-out elasticity...")
        val_el_final = evaluate_heldout_elasticity(
            features_val_array, teacher_cache_val, codec, tail,
            num_groups, args.alpha, args.norm_mode,
            args.batch_size, device,
            max_images=(args.heldout_max_images or None),
            probe_group_chunk=args.heldout_probe_group_chunk)
    if val_el_final is not None:
        results.setdefault('heldout_metrics', {})['val'] = val_el_final

    if args.skip_test_eval:
        results['train_time_std'] = float(std_time)
        results['train_time_v1'] = float(train_time)
        results['phase'] = phase_tag
        tag = (f"{args.layer}_K{args.K}_b{args.beta}_a{args.alpha}"
               f"_lr{args.lr}_tau{args.elastic_tau}"
               f"_gpb{args.n_groups_per_batch}"
               f"{fz_tag}{step_tag}"
               f"_ep{args.epochs}_s{args.seed}")
        if args.result_suffix:
            tag += f"_{args.result_suffix}"
        result_path = os.path.join(out_dir, f'{tag}.json')
        import tempfile
        fd, tmp_result = tempfile.mkstemp(
            dir=out_dir, suffix='.json.tmp')
        os.close(fd)
        try:
            with open(tmp_result, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            os.replace(tmp_result, result_path)
        except BaseException:
            if os.path.exists(tmp_result):
                os.remove(tmp_result)
            raise
        print(f"\nValidation-only results: {result_path}")
        return results

    # ---- Held-out elasticity on test set (§11.4) ----
    if args.beta > 0:
        print(f"\n  Computing test-set held-out elasticity...")
        n_test = len(features_test)
        tc_test_path = _teacher_cache_path(
            args, seeds, 'test', n_test)
        feat_test_path = _features_cache_path(
            args, seeds, 'test', n_test)

        test_array = _load_or_create_array(
            feat_test_path,
            lambda: np.stack(features_test), mmap=True)

        test_teacher_cache = _load_or_create_array(
            tc_test_path,
            lambda: _compute_teacher_cache(
                test_array, tail, args.batch_size, device),
            mmap=True)

        test_el = evaluate_heldout_elasticity(
            test_array, test_teacher_cache, codec, tail,
            num_groups, args.alpha, args.norm_mode,
            args.batch_size, device,
            probe_group_chunk=args.heldout_probe_group_chunk)
        results.setdefault('heldout_metrics', {})['test'] = test_el
        print(f"  Test ε_g: mean={test_el['eps_g']['global_mean']:.4f} "
              f"CV={test_el['eps_g']['global_cv']:.4f} "
              f"span={test_el['eps_g']['global_span']:.4f}")
        del test_array, test_teacher_cache

    tail.to('cpu')
    torch.cuda.empty_cache()

    # ================================================================
    #  (C) Evaluation on test set
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  [Evaluation] test set")
    print(f"{'=' * 60}")

    xhat_v1 = codec_encode_decode(
        features_test, codec, args.norm_mode, device)

    wrapper.backbone.to(device)
    if wrapper.head is not None:
        wrapper.head.to(device)
    torch.cuda.empty_cache()

    acc_v1 = evaluate_accuracy(
        xhat_v1, basenames_test, gt_test,
        wrapper, layer_idx, device,
    )
    results['v1_acc'] = float(acc_v1)
    delta_acc = acc_v1 - acc_std
    print(f"  * V1 Acc = {acc_v1:.4f} (Δ={delta_acc:+.4f})")
    del xhat_v1

    # V1 ΔL_ref
    tail_blocks2 = list(wrapper.backbone.blocks[layer_idx + 1:])
    norm2 = wrapper.backbone.norm
    tail2 = FrozenTail(tail_blocks2, norm2, device=device)

    v1_delta_l = evaluate_delta_l_ref(
        features_test, tail2, args.norm_mode, device,
        codec=codec, batch_size=args.batch_size,
    )
    results['v1_delta_l'] = float(v1_delta_l)
    delta_dl = v1_delta_l - std_delta_l
    print(f"  * V1 D0 = {v1_delta_l:.1f} (Δ={delta_dl:+.1f})")

    tail2.to('cpu')
    torch.cuda.empty_cache()

    # Rate
    print(f"  Computing rate metrics...")
    train_labels = _codec_labels(
        [features_train_array[i] for i in range(features_train_array.shape[0])],
        codec, args.norm_mode, device, batch_size=args.batch_size)
    train_pmf = _histogram_pmf(train_labels, num_groups, args.K)
    rate_info = evaluate_rate(
        features_test, codec, args.norm_mode, device,
        batch_size=args.batch_size, train_pmf=train_pmf)
    results['rate_info'] = rate_info
    print(f"  * H_emp={rate_info['empirical_entropy_bpt']:.2f} "
          f"max={rate_info['max_rate_bpt']:.0f} bits/token")
    if 'rans_train_bpt' in rate_info:
        print(f"  * rANS_train={rate_info['rans_train_bpt']:.2f} bits/token")

    # ================================================================
    #  Summary
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  Summary: {args.layer} K={args.K} β={args.beta} Phase {phase_tag}")
    print(f"  OPQ:  Acc={acc_std:.4f}  D0={std_delta_l:.1f}")
    print(f"  V1:   Acc={acc_v1:.4f}  D0={v1_delta_l:.1f}")
    print(f"  Δ(Acc)={delta_acc:+.4f}  Δ(D0)={delta_dl:+.1f}")
    print(f"  bits/token: {bits_per_token:.0f}")
    print(f"{'=' * 60}")

    results['delta_acc'] = float(delta_acc)
    results['delta_dl'] = float(delta_dl)
    results['train_time_std'] = float(std_time)
    results['train_time_v1'] = float(train_time)
    results['phase'] = phase_tag

    # Save results (§13.1: fully disambiguated filename + atomic write)
    tag = (f"{args.layer}_K{args.K}_b{args.beta}_a{args.alpha}"
           f"_lr{args.lr}_tau{args.elastic_tau}"
           f"_gpb{args.n_groups_per_batch}"
           f"{fz_tag}{step_tag}"
           f"_ep{args.epochs}_s{args.seed}")
    if args.result_suffix:
        tag += f"_{args.result_suffix}"
    result_path = os.path.join(out_dir, f'{tag}.json')
    import tempfile
    fd, tmp_result = tempfile.mkstemp(
        dir=out_dir, suffix='.json.tmp')
    os.close(fd)
    try:
        with open(tmp_result, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        os.replace(tmp_result, result_path)
    except BaseException:
        if os.path.exists(tmp_result):
            os.remove(tmp_result)
        raise
    print(f"\nResults: {result_path}")

    return results


# ================================================================
#  CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ORFC-v1 elastic experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Architecture
    parser.add_argument("--layer", type=str, default="blk20")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--bottleneck_dim", type=int, default=1024)
    parser.add_argument("--norm_mode", type=str, default="per_image")

    # V1 elastic
    parser.add_argument("--beta", type=float, default=0.0,
                        help="Elastic loss weight (0=D0 only)")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Perturbation scale for elasticity probe")
    parser.add_argument("--elastic_tau", type=float, default=1.0,
                        help="Temperature for smooth max/min in elastic loss")
    parser.add_argument("--n_groups_per_batch", type=int, default=4,
                        help="Groups sampled per batch for elasticity")
    parser.add_argument("--n_interaction_pairs", type=int, default=4,
                        help="Cross-group pairs sampled per batch")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Temperature
    parser.add_argument("--tau_start", type=float, default=0.5)
    parser.add_argument("--tau_end", type=float, default=0.005)
    parser.add_argument("--tau_schedule", type=str, default="exponential")

    # Freezing (Phase A/B)
    parser.add_argument("--freeze_transform", action="store_true")
    parser.add_argument("--freeze_codebooks", action="store_true")

    # Phase C: alternating (§12)
    parser.add_argument("--step_mode", type=str, default="joint",
                        choices=["joint", "alternating"],
                        help="'joint' = update all; 'alternating' = U/C steps")
    parser.add_argument("--alt_u_steps", type=int, default=1,
                        help="U-step count per alternating cycle")
    parser.add_argument("--alt_c_steps", type=int, default=1,
                        help="C-step count per alternating cycle")

    # Held-out evaluation (§11.4)
    parser.add_argument("--val_elasticity_interval", type=int, default=20,
                        help="Epochs between validation elasticity evaluation")
    parser.add_argument("--probe_group_chunk", type=int, default=4,
                        help="Training groups batched in one tail probe")
    parser.add_argument("--heldout_probe_group_chunk", type=int, default=4,
                        help="Held-out groups batched in one no-grad tail probe")
    parser.add_argument("--force_val_heldout", action="store_true",
                        help="Evaluate validation elasticity even when beta=0")
    parser.add_argument("--heldout_max_images", type=int, default=0,
                        help="Limit forced validation held-out; 0 uses all")

    # Data
    parser.add_argument("--n_train", type=int, default=4500,
                        help="Training images (from 5000 pool)")
    parser.add_argument("--n_val", type=int, default=500,
                        help="Validation images (held-out from pool)")
    parser.add_argument("--kmeans_max_samples", type=int, default=2_000_000)
    parser.add_argument("--opq_iter", type=int, default=20)
    parser.add_argument("--kmeans_iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_test", type=int, default=0,
                        help="Limit test images; 0 uses the full test set")
    parser.add_argument("--skip_test_eval", action="store_true",
                        help="Do not load or evaluate test features")

    parser.add_argument("--feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features"))
    parser.add_argument("--train_subset", type=str, default="train")
    parser.add_argument("--test_subset", type=str, default="test")
    parser.add_argument("--backbone", type=str, default="dinov2_vitl14")
    parser.add_argument("--gt_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "utils",
                                             "imagenet_selected_label500.txt"))

    parser.add_argument("--result_suffix", type=str, default="")
    parser.add_argument("--run_id", type=str, default="manual",
                        help="Isolated result/checkpoint namespace")

    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
