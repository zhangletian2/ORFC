#!/usr/bin/env python
"""Spatial codec + 4-channel residual guidance.

Stage 1 (no ORFC, no PQ):

    Z = E_θ(X)                         Conv2d(D,C,k,s=2)  k=2 or 3
    X_0 = U_θ(Z)                       ConvTranspose2d(C,D,k,s=2)  k=2 or 3
    G = (X - X_0) W_g                  c=4
    hat_X = X_0 + F_φ(X_0, G)

Ablations (``--residual_ablation``):
    main    hat_X = U(E(X))
    recon0  hat_X = X_0 + F_φ(X_0, 0)
    full    hat_X = X_0 + F_φ(X_0, G)

    CUDA_VISIBLE_DEVICES=2 python -u run_bilinear_residual.py \\
        --stage residual --residual_mode both --layer blk20 \\
        --spatial_down conv2 --spatial_up conv2 --residual_decoder conv \\
        --no-residual_orfc --no-residual_quantize \\
        --residual_epochs 30 --n_val 0

``E_θ``, ``U_θ``, ``W_g`` and ``F_φ`` are trained jointly (cosine + grad clip).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
_ORFC_DIR = os.path.normpath(os.path.join(HERE, "..", "orfc"))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if _ORFC_DIR not in sys.path:
    sys.path.insert(0, _ORFC_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from backbone.wrapper import Dinov2Wrapper  # noqa: E402
from opq import (  # noqa: E402
    batch_inv_normalize_gpu,
    batch_normalize_gpu,
    batched_assign,
    learn_opq_rotation,
)
from run_multilayer_calibrator import (  # noqa: E402
    evaluate_accuracy,
    load_gt,
    preload_features,
    set_seed,
)
from soft_pq import (  # noqa: E402
    FeatureCodec,
    FrozenTail,
    OrthogonalTransform,
    SoftPQ,
    compute_perplexity,
    load_codec,
    save_codec,
)

from bilinear_residual import (  # noqa: E402
    BITS_PER_CHANNEL,
    RESIDUAL_ABLATIONS,
    BilinearSpatialCodec,
    ResidualLinearCodec,
    apply_residual,
    fit_residual_basis,
    freeze_module,
    load_residual_codec,
    load_residual_init,
    load_spatial_weights,
    overflow_mask,
    reconstruct_base,
    save_residual_codec,
    save_residual_init,
    spatial_roundtrip,
)
from run_soft_pq import _histogram_pmf, _rans_encode_bpt  # noqa: E402
from soft_pq import FrozenTail  # noqa: E402
from backbone.dinov3_tail import build_dinov3_tail, Dinov3FrozenTail  # noqa: E402

# -- inlined from run_haar_orfc_finetune / run_haar_orfc_train / eval_haar_orfc --

NORM_BITS_PER_IMAGE = 32.0  # one mu+sigma group, fp16 each
GRID_BITS_PER_IMAGE = 16.0  # patch grid (H, W) as two uint8


def norm_bits_for(norm_mode, n_prefix):
    """Side-info bits for the transmitted mu/sigma: one group is 32 bits.

    split_per_reg_cls_patch gives each register token [1, n_prefix) its own
    group and shares one between CLS and the patches -> n_prefix groups, not the
    one group the old hardcoded 32.0 assumed.  The pooled split_reg_cls_patch
    sends one register group plus the shared CLS/patch group -> 2.
    per_token_ln transmits nothing.
    """
    groups = {"split_per_reg_cls_patch": int(n_prefix),
              "split_reg_cls_patch": 2, "split_cls_patch": 2,
              "per_token_ln": 0}.get(norm_mode, 1)
    return NORM_BITS_PER_IMAGE * groups


def _cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def split_train_val(features, max_train_images=5000, n_val=200, seed=42):
    n_all = len(features)
    rng = np.random.RandomState(seed)
    if max_train_images > 0 and n_all > max_train_images:
        pool = rng.choice(n_all, max_train_images, replace=False)
    else:
        pool = np.arange(n_all, dtype=np.int64)
    n_val = min(int(n_val), len(pool))
    rng_val = np.random.RandomState(seed + 1)
    val_pos = rng_val.choice(len(pool), n_val, replace=False) if n_val > 0 else np.array([], dtype=np.int64)
    val_mask = np.zeros(len(pool), dtype=bool)
    if n_val > 0:
        val_mask[val_pos] = True
    train_idx = np.asarray(pool[~val_mask], dtype=np.int64)
    val_idx = np.asarray(pool[val_mask], dtype=np.int64)
    train_feat = [features[int(i)] for i in train_idx]
    val_feat = [features[int(i)] for i in val_idx]
    return train_feat, val_feat, train_idx, val_idx


def subsample_vectors(vectors, opq_groups, kmeans_max_samples, seed):
    max_flat = kmeans_max_samples // max(opq_groups, 1)
    if vectors.shape[0] > max_flat:
        rng = np.random.RandomState(seed)
        idx = rng.choice(vectors.shape[0], max_flat, replace=False)
        vectors = vectors[idx]
    return vectors


def _build_dinov3_tail(model, layer_idx, token_hw, device):
    """Build RoPE-aware frozen tail directly from a timm Eva model."""
    from backbone.dinov3_tail import Dinov3FrozenTail
    from soft_pq import FrozenTail as _FT
    n_blocks = len(model.blocks)
    if layer_idx >= n_blocks - 1:
        return _FT([], model.norm, device=str(device))
    h_p, w_p = token_hw
    dummy = torch.zeros(1, 3, h_p * model.patch_embed.proj.kernel_size[0],
                        w_p * model.patch_embed.proj.kernel_size[0], device=device)
    with torch.no_grad():
        x_emb = model.patch_embed(dummy)
        _, rot_pos_embed = model._pos_embed(x_emb)
    return Dinov3FrozenTail(model, layer_idx, rot_pos_embed, None, device=str(device))


def _rebuild_train_tail(wrapper, layer_idx, device):
    if getattr(wrapper, "_is_dinov3", False):
        for i, blk in enumerate(wrapper.backbone.blocks):
            if i <= layer_idx:
                blk.cpu()
        torch.cuda.empty_cache()
        return _build_dinov3_tail(wrapper.backbone, layer_idx, wrapper._token_hw, device)
    for i, blk in enumerate(wrapper.backbone.blocks):
        if i <= layer_idx:
            blk.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()
    return FrozenTail(
        list(wrapper.backbone.blocks[layer_idx + 1:]),
        wrapper.backbone.norm,
        device=device,
    )


def _rate_from_labels(test_labels, train_labels, G, K):
    import math as _math
    N = test_labels.shape[1]
    train_pmf = _histogram_pmf(train_labels, G, K)
    primary_pmf = train_pmf
    xent_train = 0.0
    for g in range(G):
        xent_train += -np.log2(train_pmf[g] + 1e-30)[test_labels[g]].sum()
    test_pmf = _histogram_pmf(test_labels, G, K, smoothing=0)
    emp = 0.0
    for g in range(G):
        pg = test_pmf[g]
        pg = pg[pg > 0]
        emp += -np.sum(pg * np.log2(pg))
    rans_train = _rans_encode_bpt(test_labels, train_pmf, G, K)
    max_rate = G * _math.log2(K)
    out = {
        "xent_train_bpt": float(xent_train / N),
        "empirical_entropy_bpt": float(emp),
        "max_rate_bpt": float(max_rate),
        "n_coded_tokens_total": int(N),
    }
    if rans_train is not None:
        out["rans_train_bpt"] = float(rans_train)
    return out, primary_pmf


def collect_orfc_labels(token_seqs, orfc, device, batch_size):
    orfc.eval()
    rows = []
    n = token_seqs.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        seq = torch.from_numpy(token_seqs[start:end]).float().to(device)
        _ = orfc(seq)
        rows.append(orfc.pq._last_labels.cpu())
        del seq
    return torch.cat(rows, dim=1).numpy()

warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

RESIDUAL_MODES = ("fixed", "recon", "both")
GUIDANCE_BITS_PER_PATCH = 4 * BITS_PER_CHANNEL  # c=4 default; overwritten from args


class FeatureTeacherDataset(Dataset):
    def __init__(self, features, teacher):
        self.features = features
        self.teacher = teacher

    def __len__(self):
        return int(self.features.shape[0])

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).float()
        t = torch.from_numpy(self.teacher[idx]).float()
        return x, t


def main_stem(args):
    bt_tag = f"bt{args.bottleneck_dim}" if args.bottleneck_dim > 0 else "noBt"
    ws_tag = "ws" if args.warm_start_opq else "km"
    tau_tag = f"_tau{args.tau_start}" if args.tau_start > 0 else ""
    tau_end_tag = (
        f"_te{args.tau_end}"
        if args.tau_start > 0 and args.tau_end != 0.005
        else ""
    )
    tau_sched_tag = (
        f"_ts{args.tau_schedule[:3]}"
        if args.tau_schedule != "exponential"
        else ""
    )
    nval_tag = f"_nval{args.n_val}" if args.n_val > 0 else ""
    return (
        f"{args.layer}_bili65_K{args.K}_emb{args.embedding_dim}"
        f"_{bt_tag}_{ws_tag}_lmbda{args.lmbda}{tau_tag}{tau_end_tag}"
        f"{tau_sched_tag}_bv_lr{args.lr}_ep{args.epochs}"
        f"_n{args.max_train_images}{nval_tag}_s{args.seed}"
    )


def main_ckpt_path(args):
    if args.main_ckpt:
        return Path(args.main_ckpt)
    return Path(args.ckpt_dir) / args.backbone / f"{main_stem(args)}.pt"


def init_stem(args):
    return f"{args.layer}_bili65_K{args.K}_c{args.c}_residual_init_s{args.seed}"


def spatial_init_stem(args):
    tag = spatial_codec_tag(args)
    return (
        f"{args.layer}_bili65{tag}_K{args.K}_c{args.c}"
        f"_spatial_residual_init_s{args.seed}"
    )


def use_orfc(args):
    return bool(getattr(args, "residual_orfc", True))


def init_path(args):
    if args.init_ckpt:
        return Path(args.init_ckpt)
    stem = spatial_init_stem(args) if not use_orfc(args) else init_stem(args)
    return Path(args.result_dir) / args.backbone / f"{stem}.pt"


def residual_decoder(args):
    return getattr(args, "residual_decoder", "linear")


def residual_ablation(args):
    return getattr(args, "residual_ablation", "full")


def residual_ablation_tag(args):
    ab = residual_ablation(args)
    return "" if ab == "full" else f"_ablation_{ab}"


def spatial_down(args):
    return getattr(args, "spatial_down", "conv2")


def spatial_up(args):
    return getattr(args, "spatial_up", "conv2")


def spatial_codec_tag(args):
    etag = "" if spatial_down(args) == "conv2" else f"_{spatial_down(args)}"
    utag = "" if spatial_up(args) == "conv2" else f"_u{spatial_up(args)}"
    lc = getattr(args, "latent_channels", 0)
    ctag = f"_C{lc}" if lc and lc > 0 else ""
    ortho = "_ortho" if getattr(args, "ortho_init", False) else ""
    clstag = "_clsid" if getattr(args, "cls_mode", "learned") == "identity" else ""
    return etag + utag + ctag + ortho + clstag


def joint_spatial_train(args, orfc):
    """Stage-1: learn ``E_θ`` / ``U_θ`` together with residual (no ORFC)."""
    return orfc is None



def _extra_tag(args):
    return ""

def residual_stem(args, mode):
    tag = f"_bv" if mode != "fixed" else ""
    qtag = "" if getattr(args, "residual_quantize", True) else "_noq"
    ptag = "" if getattr(args, "residual_orfc", True) else "_pre"
    dtag = "" if residual_decoder(args) == "linear" else f"_{residual_decoder(args)}"
    stag = spatial_codec_tag(args)
    atag = residual_ablation_tag(args)
    wd = getattr(args, "weight_decay", 0.0)
    wdtag = f"_wd{wd}" if wd > 0 else ""
    gc = getattr(args, "grad_clip", 1.0)
    cliptag = f"_clip{gc}" if gc != 1.0 else ""
    return (
        f"{args.layer}_bili65{stag}_K{args.K}_c{args.c}_{mode}"
        f"_wa{dtag}{ptag}{qtag}{tag}{atag}"
        f"_lr{args.residual_lr}_ep{args.residual_epochs}"
        f"{wdtag}{cliptag}"
        f"_n{args.max_train_images}"
        f"{_extra_tag(args)}"
        f"_nval{args.n_val}_s{args.seed}"
    )


def residual_ckpt_path(args, mode):
    return Path(args.result_dir) / args.backbone / f"{residual_stem(args, mode)}.pt"


def summary_path(args):
    return (
        Path(args.result_dir) / args.backbone
        / f"{args.layer}_bili65{spatial_codec_tag(args)}"
        f"_K{args.K}_c{args.c}_wa"
        f"{'' if residual_decoder(args) == 'linear' else '_' + residual_decoder(args)}"
        f"{'' if getattr(args, 'residual_orfc', True) else '_pre'}"
        f"{'' if getattr(args, 'residual_quantize', True) else '_noq'}"
        f"{residual_ablation_tag(args)}_summary.json"
    )


def split_npz_path(args):
    return (
        Path(args.result_dir) / "splits"
        / f"{args.layer}_n{args.max_train_images}_nval{args.n_val}_s{args.seed}.npz"
    )


def save_split(path, train_idx, val_idx, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
        max_train_images=np.int64(args.max_train_images),
        n_val=np.int64(args.n_val),
        seed=np.int64(args.seed),
    )


def make_split(features_train, args):
    train_feat, val_feat, train_idx, val_idx = split_train_val(
        features_train, args.max_train_images, args.n_val, args.seed)
    save_split(split_npz_path(args), train_idx, val_idx, args)
    print(f"  split: pool={args.max_train_images}  "
          f"train={len(train_feat)}  val={len(val_feat)}  "
          f"saved {split_npz_path(args).name}", flush=True)
    return train_feat, val_feat, train_idx, val_idx


def load_features(args):
    train_dir = Path(args.feat_root) / args.train_subset / args.backbone / args.layer
    test_dir = Path(args.feat_root) / args.test_subset / args.backbone / args.layer
    train_files = sorted(train_dir.glob("*.npy"))
    test_files = sorted(test_dir.glob("*.npy"))
    if not train_files or not test_files:
        raise FileNotFoundError(f"missing features in {train_dir} or {test_dir}")
    features_train, _ = preload_features(train_files, num_workers=4)
    features_test, basenames_test = preload_features(test_files, num_workers=4)
    return features_train, features_test, basenames_test


def _as_array(features):
    if isinstance(features, np.ndarray) and features.ndim == 3:
        return features
    return np.stack(features)


def cache_teacher(features, tail, device, batch_size, verbose=True):
    feat = _as_array(features)
    cache = np.empty_like(feat)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, feat.shape[0], batch_size):
            end = min(start + batch_size, feat.shape[0])
            X = torch.from_numpy(feat[start:end]).float().to(device)
            cache[start:end] = tail.forward_nograd(X).cpu().numpy()
            del X
    torch.cuda.empty_cache()
    if verbose:
        print(f"  Teacher cache: {cache.nbytes / 1e9:.1f} GB CPU "
              f"({time.time() - t0:.1f}s)", flush=True)
    return cache


def collect_bilinear_opq_vectors(features, spatial, norm_mode, device, batch_size):
    chunks = []
    n_prefix = spatial.n_prefix
    spatial.eval()
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
            seq, _ = spatial.encode(Y)
        chunks.append(seq.reshape(-1, seq.shape[-1]).cpu().numpy())
        del X, Y, seq
    return np.concatenate(chunks, axis=0)


def opq_usage_on_bilinear(features, spatial, R, codebooks, embedding_dim,
                          norm_mode, device, batch_size):
    R_t = torch.from_numpy(R).float().to(device)
    cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
    G = len(codebooks)
    K = codebooks[0].shape[0]
    counts = np.zeros((G, K), dtype=np.float64)
    n_prefix = spatial.n_prefix
    spatial.eval()
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
            seq, _ = spatial.encode(Y)
            flat = seq.reshape(-1, seq.shape[-1]) @ R_t
            sub = flat.reshape(-1, G, embedding_dim).permute(1, 0, 2).contiguous()
            labels = torch.cdist(sub, cb_t).argmin(dim=-1)
            for gi in range(G):
                np.add.at(counts[gi], labels[gi].cpu().numpy(), 1)
        del X, Y, seq, flat, sub, labels
    del R_t, cb_t
    torch.cuda.empty_cache()
    return counts


def bilinear_encode_all(features, spatial, norm_mode, device, batch_size):
    spatial.eval()
    chunks = []
    n_prefix = spatial.n_prefix
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
            seq, _ = spatial.encode(Y)
        chunks.append(seq.cpu().numpy())
        del X, Y, seq
    return np.concatenate(chunks, axis=0)


def pack_rate(bpt_dict, n_coded, n_patch, c, norm_bits, include_guidance=True,
              n_orig=257, feat_dim=1024, grid_bits=GRID_BITS_PER_IMAGE):
    rans = bpt_dict.get("rans_train_bpt")
    xent = bpt_dict.get("xent_train_bpt")
    pq_bpt = rans if rans is not None else xent
    pq_bits = float(pq_bpt) * n_coded
    guidance_bits = float(n_patch * c * BITS_PER_CHANNEL) if include_guidance else 0.0
    max_pq = float(bpt_dict.get("max_rate_bpt", 0.0)) * n_coded
    total = pq_bits + guidance_bits + norm_bits + grid_bits
    return {
        "n_coded_tokens": int(n_coded),
        "n_patch": int(n_patch),
        "c": int(c),
        "pq_bpt": float(pq_bpt),
        "pq_bits_per_image": pq_bits,
        "pq_max_bits_per_image": float(max_pq),
        "guidance_bits_per_image": guidance_bits,
        "norm_bits_per_image": float(norm_bits),
        "grid_bits_per_image": float(grid_bits),
        "bits_per_image": total,
        "bpfp": total / (n_orig * feat_dim),
        **{k: v for k, v in bpt_dict.items() if k not in ("G", "K")},
    }


def _tau_at_epoch(epoch, epochs, tau_start, tau_end, tau_schedule):
    if tau_start <= 0:
        return 0.0
    if tau_schedule == "constant" or tau_start == tau_end or epochs <= 1:
        return float(tau_start)
    progress = epoch / (epochs - 1)
    if tau_schedule == "linear":
        return float(tau_start + (tau_end - tau_start) * progress)
    return float(tau_start * (tau_end / tau_start) ** progress)


def train_bilinear_orfc(
    features_train, spatial, tail, G, K, d,
    norm_mode="per_image",
    epochs=100, lr=3e-4, batch_size=32, device="cuda", seed=42,
    val_features=None, verbose=True,
    transform=None, R_init=None, codebooks_init=None,
    kmeans_max_samples=2_000_000, lmbda=0.5, prior_init_counts=None,
    grad_clip=1.0, tau_start=2.0, tau_end=2.0, tau_schedule="constant",
    n_prefix=1,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    features_array = _as_array(features_train)
    N_img = features_array.shape[0]
    D = features_array.shape[2]
    freeze_module(spatial)

    pq = SoftPQ(G, K, d, lmbda=lmbda).to(device)
    if transform is not None:
        transform = transform.to(device)
    codec = FeatureCodec(pq, transform).to(device)
    use_soft = tau_start > 0

    if R_init is not None and codebooks_init is not None and transform is not None:
        if verbose:
            print("  Warm-start from OPQ on bilinear-65 tokens", flush=True)
        transform.init_from_opq(R_init)
        pq.init_codebooks(codebooks_init)
        if pq.use_rate and prior_init_counts is not None:
            pq.init_prior_from_freq(prior_init_counts)
            if verbose:
                print("  log_prior init from OPQ empirical frequency", flush=True)
    elif codebooks_init is not None:
        pq.init_codebooks(codebooks_init)
        if pq.use_rate and prior_init_counts is not None:
            pq.init_prior_from_freq(prior_init_counts)
    else:
        if verbose:
            print(f"  K-means init on bilinear tokens ({N_img} images)...", flush=True)
        all_Z = []
        for start in range(0, N_img, 200):
            end = min(start + 200, N_img)
            X = torch.from_numpy(features_array[start:end]).float().to(device)
            with torch.no_grad():
                Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
                seq, _ = spatial.encode(Y)
                flat = seq.reshape(-1, D)
                Z = transform.encode(flat) if transform is not None else flat
            all_Z.append(Z.cpu())
            del X, Y, seq, flat, Z
        Z_flat = torch.cat(all_Z, dim=0)
        if Z_flat.shape[0] > kmeans_max_samples:
            idx = np.random.choice(Z_flat.shape[0], kmeans_max_samples, replace=False)
            Z_flat = Z_flat[idx]
        pq.init_from_kmeans(Z_flat, device=device)
        del all_Z, Z_flat
        torch.cuda.empty_cache()

    trainable = [p for p in codec.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)
    if verbose:
        n_tr = sum(p.numel() for p in trainable)
        print(f"  Trainable params: {n_tr:,}  (bilinear + ViT frozen)", flush=True)
        if transform is not None and hasattr(transform, "orth_error"):
            print(f"  Orthogonal transform: ||R'R-I||={transform.orth_error():.2e}",
                  flush=True)
        if use_soft:
            print(f"  Soft PQ: τ={tau_start:.2f} ({tau_schedule})", flush=True)
        print("  Checkpoint: restore ORFC at best val hard ΔL_ref", flush=True)

    if verbose:
        print(f"  Pre-computing teacher outputs ({N_img} images)...", flush=True)
    teacher_cache = cache_teacher(
        features_array, tail, device, batch_size, verbose=verbose)

    val_array = val_teacher = None
    if val_features is not None and len(val_features) > 0:
        val_array = _as_array(val_features)
        val_teacher = cache_teacher(
            val_array, tail, device, batch_size, verbose=False)

    loader = DataLoader(
        FeatureTeacherDataset(features_array, teacher_cache),
        batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )
    history = []
    Tm_rate = None
    best_val = best_epoch = None
    best_codec = None

    for epoch in range(epochs):
        t_epoch = time.time()
        pq.temperature = _tau_at_epoch(
            epoch, epochs, tau_start, tau_end, tau_schedule)

        total_distortion = 0.0
        total_rate = 0.0
        usage_acc = torch.zeros(G, K, device=device)
        codec.train()
        spatial.eval()

        for X, Y_teacher in loader:
            X = X.to(device, non_blocking=True)
            Y_teacher = Y_teacher.to(device, non_blocking=True)
            B = X.shape[0]
            with torch.no_grad():
                Y, Mu, Std = batch_normalize_gpu(
                    X, mode=norm_mode, n_prefix=n_prefix)
                seq, aux = spatial.encode(Y)
            Tm = seq.shape[1]
            if Tm_rate is None:
                Tm_rate = int(Tm)
            seq_hat, usage = codec(seq)
            Y_hat = spatial.decode(seq_hat, aux)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            distortion = ((Y_teacher - tail(X_hat)) ** 2).sum() / B
            if codec.use_rate:
                loss = codec._last_rate * Tm + lmbda * distortion
            else:
                loss = distortion
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(codec.parameters(), grad_clip)
            optimizer.step()
            total_distortion += distortion.item() * B
            if codec.use_rate:
                total_rate += codec._last_rate.item() * B
            usage_acc += usage.detach()
            del X, Y, Mu, Std, seq, seq_hat, Y_hat, X_hat
            del Y_teacher, loss, distortion

        scheduler.step()
        avg_distortion = total_distortion / N_img
        avg_rate = total_rate / N_img if codec.use_rate else 0.0
        perplexity = compute_perplexity(usage_acc)
        dead_entries = int((usage_acc == 0).sum().item())

        val_loss = None
        if val_array is not None:
            val_sum = 0.0
            n_val = val_array.shape[0]
            codec.eval()
            with torch.no_grad():
                for vs in range(0, n_val, batch_size):
                    ve = min(vs + batch_size, n_val)
                    X_v = torch.from_numpy(val_array[vs:ve]).float().to(device)
                    Bv = X_v.shape[0]
                    Y_v, Mu_v, Std_v = batch_normalize_gpu(
                        X_v, mode=norm_mode, n_prefix=n_prefix)
                    seq_v, aux_v = spatial.encode(Y_v)
                    seqh_v, _ = codec(seq_v)
                    Yh_v = spatial.decode(seqh_v, aux_v)
                    Yt_v = torch.from_numpy(val_teacher[vs:ve]).float().to(device)
                    Xh_v = batch_inv_normalize_gpu(Yh_v, Mu_v, Std_v)
                    Xo_v = tail.forward_nograd(Xh_v)
                    val_sum += ((Yt_v - Xo_v) ** 2).sum().item()
                    del X_v, Y_v, Mu_v, Std_v, seq_v, seqh_v, Yh_v
                    del Yt_v, Xh_v, Xo_v
            val_loss = val_sum / n_val
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_codec = _cpu_state(codec)
                if verbose:
                    print(f"  * val-best  ep={epoch}  val_ΔL={val_loss:.1f}",
                          flush=True)

        Tm_tokens = Tm_rate if Tm_rate is not None else 65
        epoch_info = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "loss_distortion": avg_distortion,
            "perplexity": perplexity,
            "val_loss": val_loss,
            "rate_bits": avg_rate,
            "rate_per_image": avg_rate * Tm_tokens if codec.use_rate else 0.0,
            "n_coded": Tm_tokens,
            "dead_entries": dead_entries,
            "temperature": pq.temperature,
            "time": time.time() - t_epoch,
            "is_best_val": bool(best_epoch == epoch),
        }
        if transform is not None and hasattr(transform, "orth_error"):
            epoch_info["orth_error"] = transform.orth_error()
        history.append(epoch_info)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            val_str = f"  val={val_loss:.1f}" if val_loss is not None else ""
            rate_str = ""
            if codec.use_rate:
                rate_str = (f"  R={avg_rate:.2f}b/t"
                            f"  λD={lmbda * avg_distortion:.1f}"
                            f"  dead={dead_entries}")
            tau_str = f"  τ={pq.temperature:.4f}" if use_soft else ""
            print(f"  ep {epoch:3d}/{epochs}  "
                  f"D={avg_distortion:.1f}  ppl={perplexity:.1f}"
                  f"{rate_str}{tau_str}{val_str}  ({time.time() - t_epoch:.1f}s)",
                  flush=True)

    best_info = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "restored": False,
    }
    if best_codec is not None:
        codec.load_state_dict(best_codec)
        codec.to(device)
        best_info["restored"] = True
        if verbose:
            print(f"  restored ORFC val-best  ep={best_epoch}  "
                  f"val_ΔL={best_val:.1f}", flush=True)
    return codec, history, best_info


@torch.no_grad()
def eval_cascade(name, features, spatial, orfc, residual, tail,
                 norm_mode, device, batch_size, n_prefix=None,
                 basenames=None, gt=None, wrapper=None, layer_idx=None,
                 quantize=True, ablation="full"):
    # Derive from the codec rather than defaulting to 1: a hardcoded 1 makes
    # the split_*reg* modes silently fall back to global normalization for
    # DINOv3 (n_prefix=5), so eval would not match training.
    if n_prefix is None:
        n_prefix = int(getattr(spatial, "n_prefix", 1))
    spatial.eval()
    if orfc is not None:
        orfc.eval()
    if residual is not None:
        residual.eval()
    n = len(features)
    if n == 0:
        return {
            "name": name,
            "delta_l": None,
            "mse_patch": None,
            "mse_prefix": None,
            "n_image": 0,
            "time_s": 0.0,
        }
    delta_l = 0.0
    mse = 0.0
    n_elem = 0
    mse_pre = 0.0
    n_pre = 0
    ov_sum = 0.0
    ov_n = 0
    ov_ch = None
    all_xhat = []
    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
        Y0, _, _, _, _ = reconstruct_base(Y, spatial, orfc)
        Y_hat, G, _, _ = apply_residual(
            Y, Y0, residual, n_prefix=n_prefix, quantize=quantize,
            ablation=ablation)
        if G is not None and quantize and residual is not None and ablation == "full":
            ov = overflow_mask(G, residual.quant.scale)
            ov_sum += float(ov.float().sum().item())
            ov_n += int(ov.numel())
            ch = ov.reshape(-1, ov.shape[-1]).float().sum(0)
            ov_ch = ch if ov_ch is None else ov_ch + ch
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        teacher = tail.forward_nograd(X)
        student = tail.forward_nograd(X_hat)
        delta_l += ((teacher - student) ** 2).sum().item()
        diff = Y[:, n_prefix:] - Y_hat[:, n_prefix:]
        mse += (diff ** 2).sum().item()
        n_elem += diff.numel()
        # Prefix (CLS + reg) tokens are excluded from mse_patch but DO feed the
        # tail, so track them separately to attribute delta_l.
        if n_prefix > 0:
            dpre = Y[:, :n_prefix] - Y_hat[:, :n_prefix]
            mse_pre += (dpre ** 2).sum().item()
            n_pre += dpre.numel()
        if basenames is not None:
            all_xhat.extend(X_hat.cpu().numpy())
        del X, Y, Mu, Std, Y0, Y_hat, X_hat, teacher, student, G
    row = {
        "name": name,
        "delta_l": float(delta_l / n),
        "mse_patch": float(mse / max(n_elem, 1)),
        "mse_prefix": float(mse_pre / max(n_pre, 1)),
        "t_s": time.time() - t0,
        "n": int(n),
        "ablation": ablation,
    }
    if ov_n > 0:
        row["overflow_frac"] = float(ov_sum / ov_n)
        n_img_coeff = ov_n / residual.c
        row["overflow_frac_per_channel"] = [
            float(x) / n_img_coeff for x in ov_ch.cpu().tolist()]
    if (basenames is not None and wrapper is not None and gt is not None
            and hasattr(wrapper, "forward_from_tokens")):
        acc = evaluate_accuracy(
            all_xhat, basenames, gt, wrapper, layer_idx, device)
        row["acc"] = float(acc)
    acc_str = f"  Acc={row['acc']:.4f}" if "acc" in row else ""
    ov_str = (f"  ov={row['overflow_frac']:.4f}" if "overflow_frac" in row else "")
    print(f"  {name:28s} ΔL={row['delta_l']:.1f}  "
          f"MSE_p={row['mse_patch']:.6f}  "
          f"MSE_pre={row['mse_prefix']:.6f}{acc_str}{ov_str}", flush=True)
    return row


def train_residual(
    features_train, spatial, orfc, residual, tail,
    epochs, lr, batch_size, device, seed, mode,
    val_features=None, norm_mode="per_image", n_prefix=1, verbose=True,
    quantize=True, train_spatial=False, grad_clip=1.0, ablation="full",
    weight_decay=0.0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if orfc is not None:
        freeze_module(orfc)
    if train_spatial:
        spatial.train()
        for p in spatial.parameters():
            p.requires_grad_(True)
    else:
        freeze_module(spatial)
    residual = residual.to(device)
    train_mode_res = (
        "fixed" if ablation == "main" else
        "recon" if ablation == "recon0" else mode)
    residual.set_mode(train_mode_res)

    features_array = _as_array(features_train)
    N_img = features_array.shape[0]
    if verbose:
        print(f"  Pre-computing teacher outputs ({N_img} images)...", flush=True)
    teacher_cache = cache_teacher(
        features_array, tail, device, batch_size, verbose=verbose)

    val_array = val_teacher = None
    if val_features is not None and len(val_features) > 0:
        val_array = _as_array(val_features)
        val_teacher = cache_teacher(
            val_array, tail, device, batch_size, verbose=False)

    def _reconstruct(Y):
        if train_spatial and orfc is None:
            return spatial_roundtrip(Y, spatial, None)
        return reconstruct_base(Y, spatial, orfc)

    def _lref_split(feat, teacher, train_mode):
        residual.train(train_mode)
        residual.set_mode(train_mode_res)
        if train_spatial:
            spatial.train(train_mode)
        total = 0.0
        ov_sum = 0.0
        ov_n = 0
        n = feat.shape[0]
        with torch.no_grad():
            for vs in range(0, n, batch_size):
                ve = min(vs + batch_size, n)
                X = torch.from_numpy(feat[vs:ve]).float().to(device)
                Y, Mu, Std = batch_normalize_gpu(
                    X, mode=norm_mode, n_prefix=n_prefix)
                Y0, _, _, _, _ = _reconstruct(Y)
                Y_hat, G, _, _ = apply_residual(
                    Y, Y0, residual, n_prefix=n_prefix, quantize=quantize,
                    ablation=ablation)
                Xh = batch_inv_normalize_gpu(Y_hat, Mu, Std)
                Yt = torch.from_numpy(teacher[vs:ve]).float().to(device)
                Xo = tail.forward_nograd(Xh)
                total += ((Yt - Xo) ** 2).sum().item()
                if G is not None:
                    ov = overflow_mask(G, residual.quant.scale)
                    ov_sum += float(ov.float().sum().item())
                    ov_n += int(ov.numel())
                del X, Y, Mu, Std, Y0, G, Y_hat, Xh, Yt, Xo
        return total / n, (ov_sum / max(ov_n, 1))

    init_train, init_ov = _lref_split(features_array, teacher_cache, False)
    init_val = init_val_ov = None
    if val_array is not None:
        init_val, init_val_ov = _lref_split(val_array, val_teacher, False)
    if verbose:
        val_str = f"  val={init_val:.1f}" if init_val is not None else ""
        print(f"  epoch0 (fixed-init)  train_ΔL={init_train:.1f}  "
              f"ov={init_ov:.4f}{val_str}", flush=True)

    history = [{
        "epoch": -1,
        "loss_distortion": init_train,
        "val_loss": init_val,
        "overflow_frac": init_ov,
        "val_overflow_frac": init_val_ov,
        "lr": 0.0,
        "time": 0.0,
        "is_best_val": True,
    }]
    best_info = {
        "best_epoch": -1,
        "best_val_loss": init_val,
        "restored": False,
        "init_train_lref": init_train,
        "init_val_lref": init_val,
        "init_overflow": init_ov,
    }

    if mode == "fixed" or epochs <= 0:
        return residual, history, best_info

    trainable = [p for p in residual.parameters() if p.requires_grad]
    if train_spatial:
        trainable = trainable + [p for p in spatial.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(f"mode={mode} has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=lr * 0.01)
    if verbose:
        n_tr = sum(p.numel() for p in trainable)
        n_sp = sum(p.numel() for p in spatial.parameters()) if train_spatial else 0
        decoder = getattr(residual, "decoder", "linear")
        names = []
        if train_spatial:
            if getattr(spatial, "analysis", None) is not None:
                names.append("E_θ")
            if getattr(spatial, "synthesis", None) is not None:
                names.append("U_θ")
        if train_mode_res == "both":
            names.extend(
                ["W_g", "F_φ"] if decoder == "conv" else ["W_g", "W_r", "W_a"])
        elif train_mode_res == "recon":
            names.extend(["F_φ"] if decoder == "conv" else ["W_r", "W_a"])
        print(f"  Trainable: {n_tr:,}  ({', '.join(names)})  "
              f"decoder={decoder}  ablation={ablation}  "
              f"down={getattr(spatial, 'down', 'bilinear')}  "
              f"up={getattr(spatial, 'up', 'bilinear')}  "
              f"spatial={n_sp:,}  lr={lr:g}  cosine  clip={grad_clip:g}  "
              f"ep={epochs}  Q_G={'on' if quantize else 'off'}", flush=True)

    loader = DataLoader(
        FeatureTeacherDataset(features_array, teacher_cache),
        batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )
    best_val = init_val
    best_epoch = -1
    best_state = {
        "residual": _cpu_state(residual),
        "spatial": _cpu_state(spatial) if train_spatial else None,
    }

    for epoch in range(epochs):
        t_epoch = time.time()
        residual.train()
        residual.set_mode(train_mode_res)
        if train_spatial:
            spatial.train()
            for p in spatial.parameters():
                p.requires_grad_(True)
        total = 0.0
        ov_sum = 0.0
        ov_n = 0
        for X, Y_teacher in loader:
            X = X.to(device, non_blocking=True)
            Y_teacher = Y_teacher.to(device, non_blocking=True)
            B = X.shape[0]
            with torch.no_grad():
                Y, Mu, Std = batch_normalize_gpu(
                    X, mode=norm_mode, n_prefix=n_prefix)
            Y0, _, _, _, _ = _reconstruct(Y)
            Y_hat, G, _, _ = apply_residual(
                Y, Y0, residual, n_prefix=n_prefix, quantize=quantize,
                ablation=ablation)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            distortion = ((Y_teacher - tail(X_hat)) ** 2).sum() / B
            optimizer.zero_grad()
            distortion.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            total += distortion.item() * B
            if G is not None:
                ov = overflow_mask(G.detach(), residual.quant.scale)
                ov_sum += float(ov.float().sum().item())
                ov_n += int(ov.numel())
            del X, Y, Mu, Std, Y0, G, Y_hat, X_hat, Y_teacher, distortion

        scheduler.step()
        train_lref = total / N_img
        train_ov = ov_sum / max(ov_n, 1)
        val_loss = val_ov = None
        if val_array is not None:
            val_loss, val_ov = _lref_split(val_array, val_teacher, False)
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {
                    "residual": _cpu_state(residual),
                    "spatial": _cpu_state(spatial) if train_spatial else None,
                }
                if verbose:
                    print(f"  * val-best  ep={epoch}  val_ΔL={val_loss:.1f}",
                          flush=True)
        else:
            best_epoch = epoch
            best_state = {
                "residual": _cpu_state(residual),
                "spatial": _cpu_state(spatial) if train_spatial else None,
            }
        epoch_info = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "loss_distortion": train_lref,
            "val_loss": val_loss,
            "overflow_frac": train_ov,
            "val_overflow_frac": val_ov,
            "time": time.time() - t_epoch,
            "is_best_val": bool(best_epoch == epoch),
        }
        history.append(epoch_info)
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            val_str = f"  val={val_loss:.1f}" if val_loss is not None else ""
            print(f"  ep {epoch:3d}/{epochs}  D={train_lref:.1f}  "
                  f"ov={train_ov:.4f}{val_str}  ({time.time() - t_epoch:.1f}s)",
                  flush=True)

    residual.load_state_dict(best_state["residual"])
    residual.to(device)
    if train_spatial and best_state["spatial"] is not None:
        spatial.load_state_dict(best_state["spatial"])
        spatial.to(device)
    best_info.update({
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "restored": True,
    })
    if verbose:
        if val_array is not None:
            print(f"  restored residual val-best  ep={best_epoch}  "
                  f"val_ΔL={best_val:.1f}", flush=True)
        else:
            print(f"  kept last epoch  ep={best_epoch}  "
                  f"(no val split, linear4 protocol)", flush=True)
    return residual, history, best_info


def init_orfc_from_opq(args, D, G, spatial, train_feat, device):
    print(f"\n{'=' * 60}", flush=True)
    print("  [OPQ] on bilinear-65 tokens (train hold-out split)", flush=True)
    print(f"{'=' * 60}", flush=True)
    t0 = time.time()
    vectors = collect_bilinear_opq_vectors(
        train_feat, spatial, args.norm_mode, device, batch_size=200)
    vectors = subsample_vectors(
        vectors, G, args.kmeans_max_samples, args.seed)
    R_std, codebooks_std, hist_std = learn_opq_rotation(
        vectors, G, args.embedding_dim, args.K,
        max_iter_opq=20, max_iter_kmeans=100,
        device=device, verbose=False)
    del vectors
    torch.cuda.empty_cache()
    print(f"    OPQ done: MSE={hist_std[-1][0]:.8f} ({time.time() - t0:.1f}s)",
          flush=True)

    opq_usage_counts = None
    if args.lmbda > 0:
        opq_usage_counts = opq_usage_on_bilinear(
            train_feat, spatial, R_std, codebooks_std, args.embedding_dim,
            args.norm_mode, device, 200)
        ppl_opq = np.exp(
            -(opq_usage_counts / opq_usage_counts.sum(-1, keepdims=True)
              * np.log(opq_usage_counts / opq_usage_counts.sum(-1, keepdims=True)
                       + 1e-30)).sum(-1)
        ).mean()
        print(f"    OPQ empirical ppl={ppl_opq:.1f} (for prior init)", flush=True)

    C_dim = getattr(spatial, "C", D)
    transform = OrthogonalTransform(C_dim)
    R_ws, C_ws = R_std, codebooks_std
    if transform is not None:
        R_ws = R_std.copy()
        C_ws = [c.copy() for c in codebooks_std]
        if np.linalg.det(R_ws) < 0:
            R_ws[:, -1] *= -1
            C_ws[-1][:, -1] *= -1
            print("    det(R_opq)<0: flipped last col to SO(D)", flush=True)
    return transform, R_ws, C_ws, opq_usage_counts, R_std, codebooks_std


def build_spatial(D, args, C=None, cls_mode=None):
    if C is None:
        C = getattr(args, "latent_channels", None)
        if C is not None and C <= 0:
            C = None
    if cls_mode is None:
        cls_mode = getattr(args, "cls_mode", "learned")
    spatial = BilinearSpatialCodec(
        D, n_prefix=getattr(args, "n_prefix", 1), scale=args.scale, grid=None,
        down=spatial_down(args), up=spatial_up(args), C=C,
        cls_mode=cls_mode)
    if getattr(args, "ortho_init", False) and spatial.C != spatial.D:
        spatial.init_orthogonal_projection(seed=getattr(args, "seed", 42))
    return spatial


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", default="all",
                   choices=["main", "init", "residual", "eval", "all"])
    p.add_argument("--residual_mode", default="all",
                   choices=["fixed", "recon", "both", "all"])
    p.add_argument("--layer", default="blk05")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--embedding_dim", type=int, default=32)
    p.add_argument("--bottleneck_dim", type=int, default=1024)
    p.add_argument("--c", type=int, default=4)
    p.add_argument("--scale", type=int, default=2)

    p.add_argument("--latent_channels", type=int, default=0,
                   help="Latent channel width C for spatial codec. "
                        "0 means same as D (no reduction).")
    p.add_argument("--cls_mode", default="learned",
                   choices=["learned", "identity"],
                   help="CLS prefix handling: learned Linear, or identity "
                        "slice/pad.  Ignored when C == D.")
    p.add_argument("--ortho_init", action="store_true",
                   help="Paired orthogonal projection init for E/U (C<D only)")
    p.add_argument("--norm_mode", default="per_image")
    p.add_argument("--n_prefix", type=int, default=1,
                   help="CLS+register prefix length (DINOv3: 5)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lmbda", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="AdamW decoupled weight decay (0 = no decay)")
    p.add_argument("--tau_start", type=float, default=2.0)
    p.add_argument("--tau_end", type=float, default=2.0)
    p.add_argument("--tau_schedule", default="constant",
                   choices=["exponential", "linear", "constant"])
    p.add_argument("--residual_epochs", type=int, default=100)
    p.add_argument("--residual_lr", type=float, default=3e-4)
    p.add_argument("--warm_start_opq", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--max_train_images", type=int, default=5000)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--kmeans_max_samples", type=int, default=2_000_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", default="dinov2_vitl14")
    p.add_argument("--feat_root", default=os.path.join(PROJECT_ROOT, "features"))
    p.add_argument("--train_subset", default="train")
    p.add_argument("--test_subset", default="test")

    p.add_argument("--gt_path", default=os.path.join(
        PROJECT_ROOT, "utils", "imagenet_selected_label500.txt"))
    p.add_argument("--ckpt_dir", default=os.path.join(HERE, "checkpoints"))
    p.add_argument("--result_dir", default=os.path.join(
        HERE, "results", "bilinear_residual"))
    p.add_argument("--main_ckpt", default="")
    p.add_argument("--init_ckpt", default="")
    p.add_argument("--skip_existing", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--residual_quantize", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="If false, residual forward skips Q_G")
    p.add_argument("--residual_orfc", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="If false, stage-1 spatial pretrain: X0 is bilinear "
                        "only (no ORFC / Q_ORFC)")
    p.add_argument("--residual_decoder", default="linear",
                   choices=["linear", "conv"],
                   help="linear: W_r/W_a; conv: 1x1-GELU-DW3x3-1x1 F_θ(X0,G)")
    p.add_argument("--spatial_down", default="conv2",
                   choices=["conv2", "conv3"],
                   help="Main-path E: Conv2d k=2 s=2 or Conv2d k=3 s=2 p=1")
    p.add_argument("--spatial_up", default="conv2",
                   choices=["conv2", "conv3"],
                   help="Main-path U: ConvTranspose2d k=2 s=2 or k=3 s=2 p=1 op=1")
    p.add_argument("--residual_ablation", default="full",
                   choices=list(RESIDUAL_ABLATIONS),
                   help="main: hat=U(E(X)); recon0: hat=X0+F(X0,0); "
                        "full: hat=X0+F(X0,G)")
    p.add_argument("--spatial_ckpt", default="",
                   help="Path to residual ckpt with pretrained spatial weights. "
                        "Loads spatial_state_dict and freezes spatial for ORFC-only training.")
    p.add_argument("--skip_test_acc", action="store_true")
    return p.parse_args()


def resolve_modes(args):
    if args.residual_mode == "all":
        return list(RESIDUAL_MODES)
    return [args.residual_mode]


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  saved {path}", flush=True)


def rate_for(features_train, features_test, spatial, orfc, args, device,
             include_guidance):
    n_prefix = int(getattr(spatial, "n_prefix", 1))
    n_patch = int(features_test[0].shape[0]) - n_prefix
    n_coded = spatial.coded_tokens(int(features_test[0].shape[0]))
    nbits = norm_bits_for(args.norm_mode, n_prefix)
    if orfc is None:
        return pack_rate(
            {"rans_train_bpt": 0.0, "max_rate_bpt": 0.0},
            n_coded=n_coded, n_patch=n_patch, c=args.c,
            include_guidance=include_guidance, norm_bits=nbits)
    train_seq = bilinear_encode_all(
        features_train, spatial, args.norm_mode, device, args.batch_size)
    test_seq = bilinear_encode_all(
        features_test, spatial, args.norm_mode, device, args.batch_size)
    train_labels = collect_orfc_labels(
        train_seq, orfc, device, args.batch_size)
    test_labels = collect_orfc_labels(
        test_seq, orfc, device, args.batch_size)
    bpt, _ = _rate_from_labels(test_labels, train_labels, orfc.pq.G, args.K)
    return pack_rate(
        bpt, n_coded=int(test_seq.shape[1]), n_patch=n_patch, c=args.c,
        include_guidance=include_guidance, norm_bits=nbits)


def run_main_stage(args, device, train_feat, val_feat, features_test,
                   basenames_test, gt_test, spatial, D, G, wrapper, tail,
                   layer_idx):
    ckpt = main_ckpt_path(args)
    if args.skip_existing and ckpt.is_file() and args.stage in ("all", "eval"):
        print(f"  skip existing main ckpt: {ckpt}", flush=True)
        codec = load_codec(str(ckpt), device=device)
        freeze_module(codec)
        return codec, {"skipped": True, "ckpt": str(ckpt)}

    print(f"\n{'#' * 70}", flush=True)
    print("# Bilinear-65 + trainable ORFC", flush=True)
    print(f"# layer={args.layer}  K={args.K}  e={args.embedding_dim}  "
          f"G={G}  λ={args.lmbda}  τ={args.tau_start} ({args.tau_schedule})",
          flush=True)
    print(f"# lr={args.lr}  ep={args.epochs}  batch={args.batch_size}",
          flush=True)
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'#' * 70}", flush=True)

    wrapper.backbone.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()
    transform, R_ws, C_ws, opq_usage, R_std, codebooks_std = init_orfc_from_opq(
        args, D, G, spatial, train_feat, device)

    wrapper.backbone.to(device)
    if wrapper.head is not None:
        wrapper.head.to(device)
    tail = _rebuild_train_tail(wrapper, layer_idx, device)

    t0 = time.time()
    codec, history, best_info = train_bilinear_orfc(
        features_train=train_feat,
        spatial=spatial,
        tail=tail,
        G=G, K=args.K, d=args.embedding_dim,
        norm_mode=args.norm_mode,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        val_features=val_feat,
        verbose=True,
        transform=transform,
        R_init=R_ws,
        codebooks_init=C_ws,
        kmeans_max_samples=args.kmeans_max_samples,
        lmbda=args.lmbda,
        prior_init_counts=opq_usage,
        grad_clip=args.grad_clip,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        tau_schedule=args.tau_schedule,
        n_prefix=spatial.n_prefix,
    )
    train_time = time.time() - t0
    print(f"  Codec training: {train_time:.1f}s", flush=True)

    ckpt.parent.mkdir(parents=True, exist_ok=True)
    save_codec(codec, str(ckpt))
    extra = torch.load(str(ckpt), map_location="cpu")
    extra.update({
        "spatial": "bilinear",
        "n_coded": int(spatial.coded_tokens(train_feat[0].shape[0])),
        "layer": args.layer,
        "scale": args.scale,
        "best_epoch": best_info.get("best_epoch"),
        "best_val_loss": best_info.get("best_val_loss"),
    })
    torch.save(extra, str(ckpt))
    print(f"  Codec saved: {ckpt}", flush=True)

    tail.to("cpu")
    torch.cuda.empty_cache()
    wrapper.backbone.to(device)
    if wrapper.head is not None:
        wrapper.head.to(device)
    tail = _rebuild_train_tail(wrapper, layer_idx, device)

    freeze_module(codec)
    results = {
        "config": vars(args),
        "ckpt": str(ckpt),
        "train_time_s": float(train_time),
        "best": best_info,
        "history": history,
        "n_train": len(train_feat),
        "n_val": len(val_feat),
    }
    results["val"] = eval_cascade(
        "bili+ORFC val", val_feat, spatial, codec, None, tail,
        args.norm_mode, device, args.batch_size)
    if not args.skip_test_acc:
        results["test"] = eval_cascade(
            "bili+ORFC test", features_test, spatial, codec, None, tail,
            args.norm_mode, device, args.batch_size,
            basenames=basenames_test, gt=gt_test, wrapper=wrapper,
            layer_idx=layer_idx)
    results["rate"] = rate_for(
        train_feat, features_test, spatial, codec, args, device,
        include_guidance=False)
    print(f"  pq_bpt={results['rate']['pq_bpt']:.3f}  "
          f"bits/img={results['rate']['bits_per_image']:.1f}  "
          f"maxPQ={results['rate']['pq_max_bits_per_image']:.0f}", flush=True)
    dump_json(Path(args.result_dir) / args.backbone / f"{main_stem(args)}.json",
              results)
    return codec, results


def run_init_stage(args, device, train_feat, spatial, orfc):
    path = init_path(args)
    if args.skip_existing and path.is_file() and args.stage != "init":
        print(f"  skip existing residual init: {path}", flush=True)
        B0, scale, meta = load_residual_init(path, device="cpu")
        return B0, scale, meta

    print(f"\n{'=' * 60}", flush=True)
    print(f"  [B0] uncentered Gram, c={args.c}  "
          f"{'ORFC residual' if orfc is not None else 'spatial-only residual (no PQ)'}",
          flush=True)
    print(f"{'=' * 60}", flush=True)
    t0 = time.time()
    if orfc is not None:
        freeze_module(orfc)
    B0, scale, stats = fit_residual_basis(
        train_feat, spatial, orfc, args.norm_mode, device,
        batch_size=args.batch_size, c=args.c, n_prefix=spatial.n_prefix)
    print(f"  n_patch={stats['n_patch']}  residual MSE={stats['mse_residual']:.6f}",
          flush=True)
    print(f"  evals={['%.4g' % x for x in stats['eigenvalues']]}  "
          f"explained={stats['explained_topc']:.4f}", flush=True)
    print(f"  scale={['%.5f' % x for x in stats['scale']]}", flush=True)
    ov = stats["overflow"]
    print(f"  G0 overflow |G|>2s  frac={ov['frac']:.4f}  "
          f"per_ch={[round(x, 4) for x in ov['frac_per_channel']]}", flush=True)
    print(f"  B0 fit {time.time() - t0:.1f}s", flush=True)

    extra = {
        "main_ckpt": str(main_ckpt_path(args)) if orfc is not None else "",
        "layer": args.layer,
        "c": args.c,
        "K": args.K,
        "scale_spatial": args.scale,
        "residual_orfc": orfc is not None,
        "spatial_down": spatial_down(args),
        "spatial_up": spatial_up(args),
        "C": int(spatial.C),
    }
    save_residual_init(path, B0, scale, stats, extra=extra)
    dump_json(path.with_suffix(".json"), {
        "init": str(path), **extra, "stats": stats,
    })
    print(f"  saved init {path}", flush=True)
    return B0, scale, {"stats": stats, **extra}


def run_residual_stage(args, device, train_feat, val_feat, features_test,
                       basenames_test, gt_test, spatial, orfc, B0, scale,
                       wrapper, tail, layer_idx, modes):
    if orfc is not None:
        freeze_module(orfc)
    train_spatial = joint_spatial_train(args, orfc)
    if not train_spatial:
        freeze_module(spatial)
    summary = {
        "modes": {},
        "main_ckpt": str(main_ckpt_path(args)) if orfc is not None else "",
        "init_ckpt": str(init_path(args)),
        "residual_orfc": orfc is not None,
        "residual_quantize": bool(args.residual_quantize),
        "residual_decoder": residual_decoder(args),
        "residual_ablation": residual_ablation(args),
        "spatial_down": spatial_down(args),
        "spatial_up": spatial_up(args),
    }
    D = int(B0.shape[0])
    c = int(B0.shape[1])
    ablation = residual_ablation(args)
    use_g = bool(args.residual_quantize) and ablation == "full"

    def _eval_split(tag, feat, **kwargs):
        if feat is None or len(feat) == 0:
            return None
        return eval_cascade(
            tag, feat, spatial, orfc, residual, tail,
            args.norm_mode, device, args.batch_size,
            quantize=args.residual_quantize, ablation=ablation, **kwargs)

    for mode in modes:
        print(f"\n{'=' * 60}", flush=True)
        print(f"  [residual] mode={mode}  c={c}  "
              f"ep={0 if mode == 'fixed' else args.residual_epochs}  "
              f"ORFC={'on' if orfc is not None else 'off'}  "
              f"down={spatial_down(args)}  up={spatial_up(args)}  "
              f"decoder={getattr(args, 'residual_decoder', 'linear')}  "
              f"ablation={ablation}  "
              f"Q_G={'on' if args.residual_quantize else 'off'}", flush=True)
        print(f"{'=' * 60}", flush=True)
        ckpt = residual_ckpt_path(args, mode)
        if args.skip_existing and ckpt.is_file() and mode != "fixed":
            print(f"  skip existing residual {mode}: {ckpt.name}", flush=True)
            residual, meta = load_residual_codec(str(ckpt), device=device)
            load_spatial_weights(spatial, meta)
            freeze_module(residual)
            freeze_module(spatial)
            row = {"mode": mode, "ckpt": str(ckpt), "skipped": True,
                   "quantize": bool(args.residual_quantize)}
            row["train"] = _eval_split(f"{mode} train", train_feat)
            row["val"] = _eval_split(f"{mode} val", val_feat)
            if not args.skip_test_acc:
                row["test"] = _eval_split(
                    f"{mode} test", features_test,
                    basenames=basenames_test, gt=gt_test, wrapper=wrapper,
                    layer_idx=layer_idx)
            row["rate"] = rate_for(
                train_feat, features_test, spatial, orfc, args, device,
                include_guidance=use_g)
            summary["modes"][mode] = row
            continue

        residual = ResidualLinearCodec(
            D, c, B0=B0, scale=scale,
            decoder=residual_decoder(args)).to(device)
        residual.quant.set_scale(scale)
        residual.set_mode(mode)
        epochs = 0 if mode == "fixed" else args.residual_epochs
        t0 = time.time()
        residual, history, best_info = train_residual(
            train_feat, spatial, orfc, residual, tail,
            epochs=epochs, lr=args.residual_lr, batch_size=args.batch_size,
            device=device, seed=args.seed, mode=mode,
            val_features=val_feat, norm_mode=args.norm_mode,
            n_prefix=spatial.n_prefix, verbose=True,
            quantize=args.residual_quantize,
            train_spatial=train_spatial, grad_clip=args.grad_clip,
            weight_decay=args.weight_decay, ablation=ablation)
        train_time = time.time() - t0
        save_residual_codec(residual, ckpt, extra={
            "mode": mode,
            "main_ckpt": str(main_ckpt_path(args)) if orfc is not None else "",
            "init_ckpt": str(init_path(args)),
            "layer": args.layer,
            "best": best_info,
            "residual_quantize": bool(args.residual_quantize),
            "residual_orfc": orfc is not None,
            "residual_decoder": residual_decoder(args),
            "residual_ablation": ablation,
            "spatial_down": spatial_down(args),
            "spatial_up": spatial_up(args),
            "spatial_state_dict": spatial.state_dict(),
            "D": int(spatial.D),
            "C": int(spatial.C),
            "cls_mode": getattr(spatial, "cls_mode", "learned"),
            "ortho_init": bool(getattr(args, "ortho_init", False)),
            # Needed by stage-2: under cls_mode="none" no parameter shape
            # depends on n_prefix, so a mismatch is otherwise undetectable.
            "n_prefix": int(spatial.n_prefix),
            "norm_mode": str(args.norm_mode),
        })
        print(f"  saved {ckpt}", flush=True)

        freeze_module(residual)
        freeze_module(spatial)
        row = {
            "mode": mode,
            "ckpt": str(ckpt),
            "quantize": bool(args.residual_quantize),
            "residual_ablation": ablation,
            "train_time_s": float(train_time),
            "best": best_info,
            "history": history,
        }
        row["train"] = _eval_split(f"{mode} train", train_feat)
        row["val"] = _eval_split(f"{mode} val", val_feat)
        # --skip_test_acc suppresses only the classification-accuracy probe;
        # test distortion (ΔL / MSE) is always measured so DINOv2 and DINOv3
        # report the same columns.
        row["test"] = _eval_split(
            f"{mode} test", features_test,
            basenames=None if args.skip_test_acc else basenames_test,
            gt=None if args.skip_test_acc else gt_test,
            wrapper=None if args.skip_test_acc else wrapper,
            layer_idx=layer_idx)
        if not args.residual_quantize and orfc is not None:
            if val_feat:
                row["val_qg"] = eval_cascade(
                    f"{mode} val QG", val_feat, spatial, orfc, residual, tail,
                    args.norm_mode, device, args.batch_size, quantize=True,
                    ablation=ablation)
            if True:
                row["test_qg"] = eval_cascade(
                    f"{mode} test QG", features_test, spatial, orfc, residual,
                    tail, args.norm_mode, device, args.batch_size,
                    quantize=True, ablation=ablation)
        row["rate"] = rate_for(
            train_feat, features_test, spatial, orfc, args, device,
            include_guidance=use_g)
        dump_json(
            Path(args.result_dir) / args.backbone / f"{residual_stem(args, mode)}.json",
            row)
        summary["modes"][mode] = row

    dump_json(summary_path(args), summary)
    return summary

def main():
    args = parse_args()
    if not use_orfc(args) and args.residual_quantize:
        args.residual_quantize = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    layer_idx = int(args.layer[-2:])

    print(f"\n{'#' * 70}", flush=True)
    print("# Spatial codec + residual guidance", flush=True)
    print(f"# stage={args.stage}  layer={args.layer}  K={args.K}  c={args.c}  "
          f"ORFC={'on' if use_orfc(args) else 'off'}  "
          f"down={spatial_down(args)}  up={spatial_up(args)}  "
          f"decoder={residual_decoder(args)}  "
          f"ablation={residual_ablation(args)}  "
          f"Q_G={'on' if args.residual_quantize else 'off'}",
          flush=True)
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'#' * 70}", flush=True)

    features_train, features_test, basenames_test = load_features(args)
    gt_test = load_gt(args.gt_path)
    train_feat, val_feat, train_idx, val_idx = make_split(features_train, args)

    D = int(train_feat[0].shape[1])
    T = int(train_feat[0].shape[0])
    bt_dim = args.bottleneck_dim if args.bottleneck_dim > 0 else D
    G = bt_dim // args.embedding_dim
    spatial = build_spatial(D, args).to(device)
    if args.spatial_ckpt:
        _sp_meta = torch.load(args.spatial_ckpt, map_location="cpu")
        _sp_sd = _sp_meta.get("spatial_state_dict", {})
        if _sp_sd:
            spatial.load_state_dict(_sp_sd, strict=False)
            print(f"  Loaded spatial weights from {args.spatial_ckpt}", flush=True)
            print(f"    keys: {list(_sp_sd.keys())}", flush=True)
        else:
            raise FileNotFoundError(
                f"No spatial_state_dict in {args.spatial_ckpt}")
        del _sp_meta, _sp_sd
    freeze_module(spatial)
    n_coded = spatial.coded_tokens(T)
    C_latent = spatial.C
    print(f"  D={D} C={C_latent} T={T}  G={G} d={args.embedding_dim} K={args.K}  "
          f"coded={n_coded}  train={len(train_feat)} val={len(val_feat)} "
          f"test={len(features_test)}", flush=True)

    print(f"\nLoading {args.backbone}...", flush=True)
    if args.backbone.startswith("dinov3"):
        import timm
        from types import SimpleNamespace
        _ckpt = "/data4/workspace/zlt/cache/torch/hub/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        _model = timm.create_model("vit_large_patch16_dinov3",
                                    pretrained=False, img_size=224,
                                    dynamic_img_size=True)
        _sd = torch.load(_ckpt, map_location="cpu", weights_only=True)
        _sd.pop("mask_token", None)
        # The checkpoint uses facebook naming (ls1.gamma / ls2.gamma /
        # storage_tokens); timm expects gamma_1 / gamma_2 / reg_token.  Without
        # this filter, load_state_dict(strict=False) silently leaves all 48
        # LayerScale gammas at their 1e-5 init, which collapses every residual
        # update in the tail.  Matches cofai's Dinov3TimmBackbone.
        from timm.models.eva import checkpoint_filter_fn as _dinov3_filter
        _sd = _dinov3_filter(_sd, _model)
        _res = _model.load_state_dict(_sd, strict=False)
        if _res.missing_keys or _res.unexpected_keys:
            raise RuntimeError(
                f"DINOv3 checkpoint load mismatch: "
                f"missing={_res.missing_keys[:8]} "
                f"unexpected={_res.unexpected_keys[:8]}")
        _model.eval().to(device)
        _token_hw = (14, 14)
        tail = _build_dinov3_tail(_model, layer_idx, _token_hw, device)
        n_tail = len(_model.blocks) - layer_idx - 1
        wrapper = SimpleNamespace(backbone=_model, head=None,
                                  _is_dinov3=True, _token_hw=_token_hw)
        print(f"  DINOv3 Tail: {n_tail} blocks (RoPE-aware)", flush=True)
    else:
        wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone, device=device)
        tail = FrozenTail(
            list(wrapper.backbone.blocks[layer_idx + 1:]),
            wrapper.backbone.norm, device=device)
        print(f"  Tail: {len(wrapper.backbone.blocks) - layer_idx - 1} blocks",
              flush=True)

    codec = None
    run_main = args.stage == "main" or (args.stage == "all" and use_orfc(args))
    if run_main:
        codec, _main_results = run_main_stage(
            args, device, train_feat, val_feat, features_test, basenames_test,
            gt_test, spatial, D, G, wrapper, tail, layer_idx)
        tail = _rebuild_train_tail(wrapper, layer_idx, device)

    if args.stage in ("init", "residual", "eval", "all"):
        if use_orfc(args):
            if codec is None:
                ckpt = main_ckpt_path(args)
                if not ckpt.is_file():
                    raise FileNotFoundError(
                        f"main ORFC checkpoint missing: {ckpt}")
                print(f"  load main ORFC {ckpt}", flush=True)
                codec = load_codec(str(ckpt), device=device)
                freeze_module(codec)
        else:
            codec = None
            print("  stage-1: U(E(X)), no ORFC / no PQ  "
                  f"down={spatial_down(args)}  up={spatial_up(args)}", flush=True)

    B0 = scale = None
    if args.stage in ("init", "residual", "eval", "all"):
        init_file = init_path(args)
        fit_init = (args.stage in ("init", "all") or not init_file.is_file())
        if fit_init:
            B0, scale, _init_meta = run_init_stage(
                args, device, train_feat, spatial, codec)
        else:
            B0, scale, _init_meta = load_residual_init(init_file, device="cpu")
            print(f"  load residual init {init_file}", flush=True)

    if args.stage in ("residual", "eval", "all"):
        if args.stage == "eval":
            modes = resolve_modes(args)
            summary = {"modes": {}}
            for mode in modes:
                residual, meta = load_residual_codec(
                    residual_ckpt_path(args, mode), device=device)
                load_spatial_weights(spatial, meta)
                freeze_module(residual)
                freeze_module(spatial)
                row = {"mode": mode, "ckpt": str(residual_ckpt_path(args, mode))}
                row["val"] = eval_cascade(
                    f"{mode} val", val_feat, spatial, codec, residual, tail,
                    args.norm_mode, device, args.batch_size,
                    quantize=args.residual_quantize,
                    ablation=residual_ablation(args))
                # --skip_test_acc drops only the accuracy probe; test
                # distortion is always measured (see run_residual_stage).
                row["test"] = eval_cascade(
                    f"{mode} test", features_test, spatial, codec, residual,
                    tail, args.norm_mode, device, args.batch_size,
                    basenames=None if args.skip_test_acc else basenames_test,
                    gt=None if args.skip_test_acc else gt_test,
                    wrapper=None if args.skip_test_acc else wrapper,
                    layer_idx=layer_idx, quantize=args.residual_quantize,
                    ablation=residual_ablation(args))
                summary["modes"][mode] = row
            dump_json(summary_path(args), summary)
        else:
            run_residual_stage(
                args, device, train_feat, val_feat, features_test,
                basenames_test, gt_test, spatial, codec, B0, scale,
                wrapper, tail, layer_idx, resolve_modes(args))

    print("done", flush=True)


if __name__ == "__main__":
    main()
