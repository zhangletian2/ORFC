#!/usr/bin/env python
"""
深度估计评测接口 — Soft-PQ & OPQ

纯评测工具 —— 所有路径、参数、checkpoint 配置由 CLI 指定。
支持两种方法:
  - soft_pq: 加载预训练 Soft-PQ codec checkpoint 评测
  - opq:     在线训练标准 OPQ (旋转+k-means)，支持缓存

Usage:
    python eval_depth_soft_pq.py \
        --feat_root  /path/to/features/dinov2_vitl14 \
        --data_root  /path/to/NYU_Test80 \
        --split_file /path/to/nyu_test_80.txt \
        --ckpt_dir   /path/to/checkpoints/dinov2_vitl14 \
        --weights_root /path/to/pretrained \
        --config_json  /path/to/depth_eval_configs.json \
        --layers blk05 blk10 \
        --methods soft_pq opq \
        --opq_cache_dir ./opq_cache \
        --model vitl14 --dim 1024 --norm_mode per_image
"""

import os, sys, json, math, time, argparse
import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent


def _setup_paths(project_root):
    """Register project sub-packages onto sys.path."""
    root = Path(project_root)
    for sub in ("tools", os.path.join("backbone", "dinov2")):
        p = str(root / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


os.environ['USE_XFORMERS'] = '0'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="xFormers")
warnings.filterwarnings("ignore", message="TypedStorage")

import logging
logging.getLogger('mmcv').setLevel(logging.WARNING)


# ══════════════════════════════════════════════════════════════════════
#  rANS BPFP — 通用 (给定 labels [G, N])
# ══════════════════════════════════════════════════════════════════════

def _histogram_pmf(labels, G, K, smoothing=1.0):
    """Build G PMFs from [G, N] labels with Laplace smoothing."""
    pmfs = []
    for g in range(G):
        counts = np.zeros(K, dtype=np.float64)
        np.add.at(counts, labels[g], 1)
        counts += smoothing
        pmfs.append(counts / counts.sum())
    return pmfs


def _rans_encode_bpt(labels_np, pmf_list, G, K, precision=16):
    """rANS encode, return bits per token."""
    try:
        from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
        from compressai import ans as _ans
    except (ImportError, ModuleNotFoundError):
        return None

    encoder = _ans.RansEncoder()
    N = labels_np.shape[1]
    cdfs, cdf_sizes = [], []
    for g in range(G):
        p = torch.from_numpy(pmf_list[g]).float()
        overflow = (1.0 - p.sum()).clamp_min(0)
        p = torch.cat([p, overflow.unsqueeze(0)])
        cdfs.append(_pmf_to_quantized_cdf(p.tolist(), precision))
        cdf_sizes.append(K + 2)
    symbols, cdf_indices = [], []
    for n in range(N):
        for g in range(G):
            symbols.append(int(labels_np[g, n]))
            cdf_indices.append(g)
    byte_string = encoder.encode_with_indexes(
        symbols, cdf_indices, cdfs, cdf_sizes, [0] * G,
    )
    return len(byte_string) * 8 / N


def _compute_bpfp_from_labels(labels_np, G, K, dim):
    """Given labels [G, N], compute rANS BPFP."""
    pmf_list = _histogram_pmf(labels_np, G, K)
    rans_bpt = _rans_encode_bpt(labels_np, pmf_list, G, K)
    if rans_bpt is None:
        return G * math.log2(K) / dim
    return rans_bpt / dim


# ══════════════════════════════════════════════════════════════════════
#  Soft-PQ: 编解码 + BPFP
# ══════════════════════════════════════════════════════════════════════

def _softpq_labels(features, codec, norm_mode, device, batch_size=4):
    """Run Soft-PQ codec, return hard labels [G, N_total]."""
    from opq import batch_normalize_gpu

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


def compute_softpq_bpfp(features, codec, norm_mode, dim, device):
    """Compute rANS-coded BPFP for Soft-PQ."""
    pq = codec.pq
    G, K = pq.G, pq.K
    labels = _softpq_labels(features, codec, norm_mode, device)
    return _compute_bpfp_from_labels(labels, G, K, dim)


# ══════════════════════════════════════════════════════════════════════
#  OPQ: 训练 / 缓存 / 编解码 / BPFP
# ══════════════════════════════════════════════════════════════════════

def _opq_cache_path(cache_dir, layer, emb_dim, K):
    """Construct cache file path for a trained OPQ model."""
    return Path(cache_dir) / f"{layer}_opq_e{emb_dim}_K{K}.npz"


def load_opq_train_features(train_feat_root, layer, max_images=5000, seed=42):
    """Load training features from dedicated train set directory."""
    feat_dir = Path(train_feat_root) / layer
    files = sorted(feat_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No train features in {feat_dir}")
    if max_images > 0 and len(files) > max_images:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(files), max_images, replace=False)
        files = [files[i] for i in idx]
    features = [np.load(str(f)) for f in files]
    print(f"    [train] loaded {len(features)} features from {feat_dir.name}/ "
          f"shape={features[0].shape}")
    return features


def train_or_load_opq(train_feat_root, layer, emb_dim, K, dim, norm_mode,
                      device, cache_dir=None,
                      max_iter_opq=20, max_iter_kmeans=100,
                      max_train=5000, kmeans_max_samples=2_000_000,
                      seed=42):
    """Train OPQ on dedicated training features or load from cache."""
    from opq import batch_normalize_gpu, learn_opq_rotation

    if cache_dir:
        cache_path = _opq_cache_path(cache_dir, layer, emb_dim, K)
        if cache_path.exists():
            data = np.load(str(cache_path), allow_pickle=True)
            R = data['R']
            codebooks = list(data['codebooks'])
            print(f"    [cache] loaded {cache_path.name}")
            return R, codebooks

    train_features = load_opq_train_features(train_feat_root, layer,
                                             max_images=max_train, seed=seed)

    num_groups = dim // emb_dim
    flat_vectors = []
    with torch.no_grad():
        for start in range(0, len(train_features), 200):
            end = min(start + 200, len(train_features))
            X = torch.from_numpy(
                np.stack(train_features[start:end])
            ).float().to(device)
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
            flat_vectors.append(Y.reshape(-1, dim).cpu().numpy())
            del X, Y
    del train_features
    vectors = np.concatenate(flat_vectors, axis=0)

    max_flat = kmeans_max_samples // num_groups
    if vectors.shape[0] > max_flat:
        rng = np.random.RandomState(seed)
        idx = rng.choice(vectors.shape[0], max_flat, replace=False)
        vectors = vectors[idx]

    print(f"    训练 OPQ: {vectors.shape[0]} vectors, G={num_groups}, "
          f"e={emb_dim}, K={K}")
    R, codebooks, _ = learn_opq_rotation(
        vectors, num_groups, emb_dim, K,
        max_iter_opq=max_iter_opq,
        max_iter_kmeans=max_iter_kmeans,
        device=device, verbose=False,
    )

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = _opq_cache_path(cache_dir, layer, emb_dim, K)
        np.savez(str(cache_path), R=R,
                 codebooks=np.array(codebooks, dtype=object))
        print(f"    [cache] saved {cache_path.name}")

    return R, codebooks


def opq_encode_decode(features, R, codebooks, emb_dim, norm_mode,
                      dim, device, chunk_images=32):
    """OPQ encode/decode a list of features."""
    from opq import batch_normalize_gpu, batch_inv_normalize_gpu, batched_assign

    num_groups = len(codebooks)
    cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
    R_t = torch.from_numpy(R).float().to(device)
    all_xhat = []
    for start in range(0, len(features), chunk_images):
        end = min(start + chunk_images, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        B, T, C = X.shape
        with torch.no_grad():
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            flat = Y.reshape(-1, C)
            Z = flat @ R_t
            z_3d = Z.reshape(-1, num_groups, emb_dim) \
                    .permute(1, 0, 2).contiguous()
            z_hat_3d, _ = batched_assign(z_3d, cb_t, device=device)
            flat_hat = z_hat_3d.permute(1, 0, 2).reshape(-1, C)
            Y_hat = flat_hat @ R_t.T
            Y_hat = Y_hat.reshape(B, T, C)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        for i in range(B):
            all_xhat.append(X_hat[i].cpu().numpy())
        del X, Y, Mu, Std, Z, z_3d, z_hat_3d, flat_hat, Y_hat, X_hat
        torch.cuda.empty_cache()
    return all_xhat


def compute_opq_bpfp(features, R, codebooks, emb_dim, norm_mode,
                     dim, device, batch_size=8):
    """Compute rANS BPFP for OPQ."""
    from opq import batch_normalize_gpu, batched_assign

    num_groups = len(codebooks)
    K = codebooks[0].shape[0]
    cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
    R_t = torch.from_numpy(R).float().to(device)

    all_labels = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            X = torch.from_numpy(
                np.stack(features[start:end])).float().to(device)
            B, T, C = X.shape
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
            flat = Y.reshape(-1, C)
            Z = flat @ R_t
            z_3d = Z.reshape(-1, num_groups, emb_dim) \
                    .permute(1, 0, 2).contiguous()
            _, labels = batched_assign(z_3d, cb_t, device=device)
            all_labels.append(labels.cpu())
            del X, Y, Z, z_3d, labels
    labels_np = torch.cat(all_labels, dim=1).numpy()
    return _compute_bpfp_from_labels(labels_np, num_groups, K, dim)


# ══════════════════════════════════════════════════════════════════════
#  Feature I/O
# ══════════════════════════════════════════════════════════════════════

def load_depth_features(feat_root, layer, samples):
    """加载深度特征，squeeze batch 维 → list of (T, D)。"""
    feat_dir = Path(feat_root) / layer
    features = []
    for rgb_rel, _, _ in samples:
        name = rgb_rel.replace('/', '_').rsplit('.', 1)[0]
        feat = np.load(str(feat_dir / f"{name}.npy"))
        if feat.ndim == 3:
            feat = feat.squeeze(0)
        features.append(feat)
    return features


# ══════════════════════════════════════════════════════════════════════
#  Depth evaluation
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_rmse(backbone, head, features, layer_idx, sample_meta, device):
    """Replay 特征 → 深度图 → 计算 mean RMSE。"""
    from dinov2_depth_pipeline import decode_depth, load_depth_gt, compute_depth_metrics

    rmse_list = []
    for i, feat in enumerate(features):
        ft = feat if feat.ndim == 3 else feat[np.newaxis, ...]
        ft = torch.from_numpy(ft).float().to(device)
        ori_shape, pad_shape, gt_path = sample_meta[i]
        pred = decode_depth(backbone, head, ft, layer_idx,
                            pad_shape, ori_shape, device)
        gt = load_depth_gt(gt_path)
        rmse_list.append(compute_depth_metrics(pred, gt)['rmse'])
    return float(np.mean(rmse_list))


# ══════════════════════════════════════════════════════════════════════
#  Config loading
# ══════════════════════════════════════════════════════════════════════

def load_configs(config_json_path):
    """Load checkpoint configs from JSON file."""
    with open(config_json_path, 'r') as f:
        raw = json.load(f)
    configs = {}
    for layer, entries in raw.items():
        configs[layer] = [
            (e["tag"], e.get("ckpt", ""), e["emb_dim"], e["K"])
            for e in entries
        ]
    return configs


def derive_opq_configs(layer_configs):
    """Extract unique (emb_dim, K) pairs from Soft-PQ configs for OPQ."""
    seen = set()
    opq_cfgs = []
    for tag, _, emb_dim, K in layer_configs:
        key = (emb_dim, K)
        if key not in seen:
            seen.add(key)
            opq_cfgs.append((f"K{K}_e{emb_dim}", emb_dim, K))
    opq_cfgs.sort(key=lambda x: (x[1], x[2]))
    return opq_cfgs


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Depth estimation evaluation: Soft-PQ & OPQ")

    # Paths
    p.add_argument("--feat_root", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--split_file", type=str, required=True)
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--weights_root", type=str, required=True)
    p.add_argument("--config_json", type=str, required=True)

    # Evaluation scope
    p.add_argument("--layers", nargs="+", required=True)
    p.add_argument("--methods", nargs="+", default=["soft_pq"],
                   choices=["soft_pq", "opq"],
                   help="Methods to evaluate (default: soft_pq)")

    # Model
    p.add_argument("--model", type=str, default="vitl14")
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--norm_mode", type=str, default="per_image")

    # OPQ specific
    p.add_argument("--opq_train_feat_root", type=str, default=None,
                   help="Root dir for OPQ training features (e.g. features/train/dinov2_vitl14)")
    p.add_argument("--opq_cache_dir", type=str, default=None,
                   help="Cache dir for trained OPQ models (skip re-training)")
    p.add_argument("--opq_iter", type=int, default=20,
                   help="OPQ outer iterations (default: 20)")
    p.add_argument("--opq_kmeans_iter", type=int, default=100,
                   help="k-means iterations per OPQ iter (default: 100)")
    p.add_argument("--opq_max_train", type=int, default=5000,
                   help="Max training images for OPQ (default: 5000)")

    # Misc
    p.add_argument("--project_root", type=str,
                   default=str(SCRIPT_DIR.parents[1]))
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    _setup_paths(args.project_root)
    from soft_pq import load_codec, soft_pq_encode_decode
    from dinov2_depth_pipeline import (
        preprocess_image, parse_split_file,
        _load_backbone, _load_depth_head,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    configs = load_configs(args.config_json)
    run_softpq = "soft_pq" in args.methods
    run_opq = "opq" in args.methods

    if run_opq and not args.opq_train_feat_root:
        raise ValueError("--opq_train_feat_root is required when evaluating OPQ")

    print("=" * 70)
    print("  Depth Eval — Soft-PQ & OPQ")
    print(f"  methods:      {args.methods}")
    print(f"  feat_root:    {args.feat_root}")
    print(f"  data_root:    {args.data_root}")
    print(f"  ckpt_dir:     {args.ckpt_dir}")
    print(f"  config_json:  {args.config_json}")
    print(f"  layers:       {args.layers}")
    print(f"  model:        {args.model}, dim={args.dim}, norm={args.norm_mode}")
    if run_opq:
        print(f"  opq_train:    {args.opq_train_feat_root}")
        print(f"  opq_cache:    {args.opq_cache_dir or '(disabled)'}")
        print(f"  opq_iter:     {args.opq_iter}, kmeans_iter={args.opq_kmeans_iter}")
        print(f"  opq_max_train:{args.opq_max_train}")
    print("=" * 70)

    samples = parse_split_file(args.split_file)
    print(f"\n  样本数: {len(samples)}")

    print("  预处理图像尺寸...")
    data_root = Path(args.data_root)
    sample_meta = []
    for rgb_rel, depth_rel, _ in samples:
        _, ori, pad = preprocess_image(str(data_root / rgb_rel))
        sample_meta.append((ori, pad, str(data_root / depth_rel)))

    print("\n[1/2] 加载模型...")
    model_args = SimpleNamespace(
        model=args.model, weights_root=args.weights_root, device=device)
    backbone, _ = _load_backbone(model_args)
    head = _load_depth_head(model_args)

    results = []  # (layer, method, tag, bpfp, rmse)
    anchors = {}

    print(f"\n[2/2] 评测中...\n")

    for layer in args.layers:
        if layer not in configs:
            print(f"  [SKIP] {layer}: no configs found in {args.config_json}")
            continue

        layer_idx = int(layer[-2:])
        layer_configs = configs[layer]
        n_tail = 23 - layer_idx

        print(f"{'═' * 70}")
        print(f"  {layer} (layer_idx={layer_idx}, tail={n_tail} blocks)")
        print(f"{'═' * 70}")

        feats = load_depth_features(args.feat_root, layer, samples)
        print(f"  特征: {len(feats)} × {feats[0].shape}")

        t0 = time.time()
        anchor = eval_rmse(backbone, head, feats, layer_idx,
                           sample_meta, device)
        anchors[layer] = anchor
        print(f"  Anchor RMSE = {anchor:.4f}  ({time.time()-t0:.1f}s)")

        # ── OPQ ──
        if run_opq:
            opq_cfgs = derive_opq_configs(layer_configs)
            print(f"\n  ─── OPQ ({len(opq_cfgs)} configs) ───")
            for tag, emb_dim, K_val in opq_cfgs:
                t0 = time.time()
                R, codebooks = train_or_load_opq(
                    args.opq_train_feat_root, layer, emb_dim, K_val,
                    args.dim, args.norm_mode, device,
                    cache_dir=args.opq_cache_dir,
                    max_iter_opq=args.opq_iter,
                    max_iter_kmeans=args.opq_kmeans_iter,
                    max_train=args.opq_max_train,
                )
                recs = opq_encode_decode(
                    feats, R, codebooks, emb_dim, args.norm_mode,
                    args.dim, device)
                rmse = eval_rmse(backbone, head, recs, layer_idx,
                                 sample_meta, device)
                bpfp = compute_opq_bpfp(
                    feats, R, codebooks, emb_dim, args.norm_mode,
                    args.dim, device)
                delta = rmse - anchor
                elapsed = time.time() - t0

                results.append((layer, "OPQ", tag, bpfp, rmse))
                print(f"  {tag:<16s}  BPFP={bpfp:.4f}  "
                      f"RMSE={rmse:.4f}  ΔRMSE={delta:+.4f}  ({elapsed:.1f}s)")

                del R, codebooks, recs
                torch.cuda.empty_cache()

        # ── Soft-PQ ──
        if run_softpq:
            print(f"\n  ─── Soft-PQ ({len(layer_configs)} configs) ───")
            for tag, ckpt_name, emb, K_val in layer_configs:
                ckpt_path = str(Path(args.ckpt_dir) / ckpt_name)
                if not os.path.isfile(ckpt_path):
                    print(f"  [SKIP] {tag}: checkpoint not found")
                    continue

                t0 = time.time()
                codec = load_codec(ckpt_path, device=device)
                recs = soft_pq_encode_decode(
                    feats, codec, args.norm_mode, device)
                rmse = eval_rmse(backbone, head, recs, layer_idx,
                                 sample_meta, device)
                bpfp = compute_softpq_bpfp(
                    feats, codec, args.norm_mode, args.dim, device)
                delta = rmse - anchor
                elapsed = time.time() - t0

                results.append((layer, "SoftPQ", tag, bpfp, rmse))
                print(f"  {tag:<16s}  BPFP={bpfp:.4f}  "
                      f"RMSE={rmse:.4f}  ΔRMSE={delta:+.4f}  ({elapsed:.1f}s)")

                del codec, recs
                torch.cuda.empty_cache()

        del feats
        torch.cuda.empty_cache()
        print()

    # ─── 汇总表 ───
    print("=" * 70)
    print("  汇总表")
    print("=" * 70)
    header = f"{'Layer':<8} {'Method':<8} {'Config':<16} {'BPFP':>8} {'RMSE':>8} {'ΔRMSE':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for layer in args.layers:
        if layer not in anchors:
            continue
        print(f"{layer:<8} {'—':<8} {'Anchor':<16} {'—':>8} "
              f"{anchors[layer]:>8.4f} {'—':>8}")
        for l, method, tag, bpfp, rmse in results:
            if l == layer:
                delta = rmse - anchors[l]
                print(f"{l:<8} {method:<8} {tag:<16} {bpfp:>8.4f} "
                      f"{rmse:>8.4f} {delta:>+8.4f}")
        print()

    total = time.time() - t_start
    print(f"总耗时: {total:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
