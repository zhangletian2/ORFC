#!/usr/bin/env python
"""
Per-group sensitivity ablation (restore-one-group).

Extended from compute_cls_sensitivity.py:
  - L_cls ablation (classification cross-entropy)
  - L_ref ablation (frozen-tail feature distortion, any backbone)
  - Gini coefficient of sensitivity distribution

Supports: dinov2_vitl14, dinov2_vitg14, clip_vitl14.
CLIP uses zero-shot classification via text embeddings.

Usage:
    python compute_sensitivity.py --gpu 0 --layer blk20 --backbone dinov2_vitl14 \
        --codec_path checkpoints/dinov2_vitl14/blk20_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt

    python compute_sensitivity.py --gpu 0 --layer blk20 --backbone clip_vitl14 \
        --codec_path /data4/workspace/zlt/featcodec/coding/vq/v3.4/checkpoints/clip_vitl14/blk20_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt
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
from soft_pq import (
    SoftPQ, OrthogonalTransform, FeatureCodec,
    FrozenTail, CLIPFrozenTail, train_soft_pq, load_codec,
)

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
import logging
from mmcv.utils import get_logger
get_logger('mmcv').setLevel(logging.WARNING)


# ── helpers ──────────────────────────────────────────────────────────

def gini(arr):
    """Gini coefficient: 0 = perfectly equal, 1 = maximally unequal."""
    a = np.abs(np.sort(arr))
    n = len(a)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * a) / (n * np.sum(a) + 1e-30))
                 - (n + 1) / n)


def stats(arr):
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'cv': float(np.std(arr) / (np.mean(arr) + 1e-10)),
        'gini': gini(arr),
        'values': arr.tolist(),
    }


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


# ── ablation functions ───────────────────────────────────────────────

def lref_ablation(codec, tail, features, norm_mode, device, batch_size=8):
    """Restore-one-group ablation for L_ref (feature distortion).

    ΔL_ref[g] = L_ref(all quant) - L_ref(group g restored)
    where L_ref = ||tail(X_hat) - tail(X)||^2.
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

            ref_out = tail.forward_nograd(X)

            Z = codec.transform.encode(flat) if codec.transform else flat
            Z_hat, _ = pq._quantise(Z)

            Y_hat_base = (codec.transform.decode(Z_hat)
                          if codec.transform else Z_hat)
            X_hat_base = batch_inv_normalize_gpu(
                Y_hat_base.reshape(B, T, C), Mu, Std)
            out_base = tail.forward_nograd(X_hat_base)
            L_base = ((out_base - ref_out) ** 2).sum()

            Z_g = Z.reshape(-1, G, d)
            Z_hat_g = Z_hat.reshape(-1, G, d)

            for g in range(G):
                oracle = Z_hat_g.clone()
                oracle[:, g, :] = Z_g[:, g, :]
                Y_hat_g = (codec.transform.decode(oracle.reshape(-1, D))
                           if codec.transform else oracle.reshape(-1, D))
                X_hat_g = batch_inv_normalize_gpu(
                    Y_hat_g.reshape(B, T, C), Mu, Std)
                out_g = tail.forward_nograd(X_hat_g)
                L_g = ((out_g - ref_out) ** 2).sum()
                delta_L[g] += (L_base - L_g).item()

            n += 1
            del X, Y, Mu, Std, flat, Z, Z_hat, ref_out
            torch.cuda.empty_cache()

    return delta_L / max(n, 1)


def cls_ablation(codec, tail, head, features, norm_mode,
                 device, batch_size=8):
    """Restore-one-group ablation for L_cls (DINOv2 linear head)."""
    pq = codec.pq
    G, d = pq.G, pq.d
    D = G * d
    delta_L = np.zeros(G)
    n = 0
    codec.eval()

    def _logits(X_hat):
        x = tail.forward_nograd(X_hat)
        cls_tok = x[:, 0]
        lin_in = torch.cat([cls_tok, x[:, 1:].mean(1)], dim=1)
        return head(lin_in)

    with torch.no_grad():
        for s in range(0, len(features), batch_size):
            e = min(s + batch_size, len(features))
            X = torch.from_numpy(np.stack(features[s:e])).float().to(device)
            B, T, C = X.shape
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            flat = Y.reshape(B * T, C)

            labels = _logits(X).argmax(dim=1)

            Z = codec.transform.encode(flat) if codec.transform else flat
            Z_hat, _ = pq._quantise(Z)

            Y_hat_base = (codec.transform.decode(Z_hat)
                          if codec.transform else Z_hat)
            X_hat_base = batch_inv_normalize_gpu(
                Y_hat_base.reshape(B, T, C), Mu, Std)
            L_base = F.cross_entropy(_logits(X_hat_base), labels,
                                     reduction='sum')

            Z_g = Z.reshape(-1, G, d)
            Z_hat_g = Z_hat.reshape(-1, G, d)

            for g in range(G):
                oracle = Z_hat_g.clone()
                oracle[:, g, :] = Z_g[:, g, :]
                Y_hat_g = (codec.transform.decode(oracle.reshape(-1, D))
                           if codec.transform else oracle.reshape(-1, D))
                X_hat_g = batch_inv_normalize_gpu(
                    Y_hat_g.reshape(B, T, C), Mu, Std)
                L_g = F.cross_entropy(_logits(X_hat_g), labels,
                                      reduction='sum')
                delta_L[g] += (L_base - L_g).item()

            n += 1
            del X, Y, Mu, Std, flat, Z, Z_hat
            torch.cuda.empty_cache()

    return delta_L / max(n, 1)


def cls_ablation_clip(codec, tail, proj, text_emb, logit_scale,
                      features, norm_mode, device, batch_size=8):
    """Restore-one-group ablation for L_cls (CLIP zero-shot)."""
    pq = codec.pq
    G, d = pq.G, pq.d
    D = G * d
    delta_L = np.zeros(G)
    n = 0
    codec.eval()

    def _logits(X_hat):
        x = tail.forward_nograd(X_hat)
        cls = x[:, 0]
        if isinstance(proj, torch.Tensor):
            img_feat = cls @ proj
        else:
            img_feat = proj(cls)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return logit_scale * img_feat @ text_emb.t()

    with torch.no_grad():
        for s in range(0, len(features), batch_size):
            e = min(s + batch_size, len(features))
            X = torch.from_numpy(np.stack(features[s:e])).float().to(device)
            B, T, C = X.shape
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            flat = Y.reshape(B * T, C)

            labels = _logits(X).argmax(dim=1)

            Z = codec.transform.encode(flat) if codec.transform else flat
            Z_hat, _ = pq._quantise(Z)

            Y_hat_base = (codec.transform.decode(Z_hat)
                          if codec.transform else Z_hat)
            X_hat_base = batch_inv_normalize_gpu(
                Y_hat_base.reshape(B, T, C), Mu, Std)
            L_base = F.cross_entropy(_logits(X_hat_base), labels,
                                     reduction='sum')

            Z_g = Z.reshape(-1, G, d)
            Z_hat_g = Z_hat.reshape(-1, G, d)

            for g in range(G):
                oracle = Z_hat_g.clone()
                oracle[:, g, :] = Z_g[:, g, :]
                Y_hat_g = (codec.transform.decode(oracle.reshape(-1, D))
                           if codec.transform else oracle.reshape(-1, D))
                X_hat_g = batch_inv_normalize_gpu(
                    Y_hat_g.reshape(B, T, C), Mu, Std)
                L_g = F.cross_entropy(_logits(X_hat_g), labels,
                                      reduction='sum')
                delta_L[g] += (L_base - L_g).item()

            n += 1
            del X, Y, Mu, Std, flat, Z, Z_hat
            torch.cuda.empty_cache()

    return delta_L / max(n, 1)


# ── main ─────────────────────────────────────────────────────────────

_EMBED_DIMS = {
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
    'clip_vitl14': 1024,
}


def main():
    parser = argparse.ArgumentParser(
        description="Per-group sensitivity ablation (cls + lref)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--layer", type=str, default="blk20")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--norm_mode", type=str, default="per_image")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_diag", type=int, default=200)
    parser.add_argument("--max_train_images", type=int, default=5000)
    parser.add_argument("--codec_path", type=str, default="",
                        help="Trained codec checkpoint")
    parser.add_argument("--feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features"))
    parser.add_argument("--backbone", type=str, default="dinov2_vitl14")
    parser.add_argument("--classnames", type=str,
                        default=os.path.join(PROJECT_ROOT, "utils",
                                             "classnames.txt"),
                        help="Classnames file for CLIP zero-shot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_cls", action="store_true",
                        help="Skip L_cls ablation")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    set_seed(args.seed)

    layer_idx = int(args.layer[-2:])
    is_clip = args.backbone.startswith("clip")

    if is_clip:
        from backbone.wrapper import ClipWrapper as BackboneWrapper
    else:
        from backbone.wrapper import Dinov2Wrapper as BackboneWrapper

    D = _EMBED_DIMS.get(args.backbone, 1024)

    G = D // args.embedding_dim
    d = args.embedding_dim
    K = args.K
    norm_mode = args.norm_mode

    print(f"\n{'#' * 70}")
    print(f"# Sensitivity ablation (restore-one-group)")
    print(f"# backbone={args.backbone}, layer={args.layer}")
    print(f"# K={K}, G={G}, d={d}, D={D}")
    print(f"# n_diag={args.n_diag}, max_train={args.max_train_images}")
    print(f"# {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'#' * 70}")

    feat_dir = Path(args.feat_root) / "train" / args.backbone / args.layer
    feat_files = sorted(feat_dir.glob("*.npy"))
    print(f"\nLoading features: {len(feat_files)} files from {feat_dir}")
    features_all, _ = preload_features(feat_files, num_workers=8)

    if args.max_train_images > 0 and len(features_all) > args.max_train_images:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(features_all), args.max_train_images, replace=False)
        features_all = [features_all[i] for i in idx]

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(features_all))
    diag_idx = perm[:args.n_diag]
    train_idx = perm[args.n_diag:]
    features_diag = [features_all[i] for i in diag_idx]
    features_tr = [features_all[i] for i in train_idx]
    del features_all
    print(f"Split: diagnostic={len(features_diag)}, training={len(features_tr)}")

    print(f"Loading {args.backbone} ...")
    if is_clip:
        wrapper = BackboneWrapper(args.classnames, device=device)
    elif args.backbone == 'dinov2_vitg14':
        wrapper = BackboneWrapper(head_layers=1, model_name='dinov2_vitg14',
                                  device=device)
    else:
        wrapper = BackboneWrapper(head_layers=1, device=device)

    # CLIP zero-shot classification components (keep on GPU)
    clip_proj = clip_text_emb = clip_logit_scale = None
    if is_clip:
        clip_proj = wrapper._proj
        if isinstance(clip_proj, torch.Tensor):
            clip_proj = clip_proj.to(device)
        clip_text_emb = wrapper.text_emb.to(device)
        clip_logit_scale = wrapper.model.logit_scale.exp().to(device)

    wrapper.backbone.cpu()
    if hasattr(wrapper, 'head') and wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()

    tail_blocks = list(wrapper.backbone.blocks[layer_idx + 1:])
    if is_clip:
        tail = CLIPFrozenTail(tail_blocks, wrapper.backbone.norm,
                              device=device)
    else:
        tail = FrozenTail(tail_blocks, wrapper.backbone.norm, device=device)

    head = None
    run_cls = not args.skip_cls
    if run_cls and not is_clip:
        head = wrapper.head.to(device)
        head.eval()
        for p in head.parameters():
            p.requires_grad_(False)

    print("Learning OPQ ...")
    R_opq, codebooks_opq, opq_usage = learn_opq(
        features_tr, D, G, d, K, norm_mode, device, args.seed)

    all_results = {}

    conditions = [
        ('identity', 'Identity', lambda: build_codec_identity(
            features_tr, G, K, d, norm_mode, device)),
        ('opq', 'OPQ', lambda: build_codec_opq(
            R_opq, codebooks_opq, G, K, d, D, device, opq_usage)),
    ]

    if args.codec_path and os.path.exists(args.codec_path):
        conditions.append(
            ('trained', 'Trained (ckpt)', lambda: load_codec(
                args.codec_path, device=device)))

    for key, label, build_fn in conditions:
        print(f"\n{'=' * 60}")
        print(f"  [{label}]")
        t0 = time.time()
        codec = build_fn()
        codec.eval()

        res = {'condition': label}

        abl_lref = lref_ablation(codec, tail, features_diag, norm_mode,
                                 device, batch_size=args.batch_size)
        res['ablation_lref'] = stats(abl_lref)
        print(f"  L_ref: CV={res['ablation_lref']['cv']:.4f}  "
              f"Gini={res['ablation_lref']['gini']:.4f}")

        if run_cls:
            if is_clip:
                abl_cls = cls_ablation_clip(
                    codec, tail, clip_proj, clip_text_emb,
                    clip_logit_scale, features_diag,
                    norm_mode, device, batch_size=args.batch_size)
            else:
                abl_cls = cls_ablation(codec, tail, head, features_diag,
                                       norm_mode, device,
                                       batch_size=args.batch_size)
            res['ablation_cls'] = stats(abl_cls)
            print(f"  L_cls: CV={res['ablation_cls']['cv']:.4f}  "
                  f"Gini={res['ablation_cls']['gini']:.4f}")

        all_results[key] = res
        print(f"  ({time.time() - t0:.1f}s)")
        del codec
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"  {'Condition':<25} {'CV_lref':>8} {'Gini_lref':>10}"
          + (' {:>8} {:>10}'.format('CV_cls', 'Gini_cls') if run_cls else ''))
    for key, res in all_results.items():
        lr = res['ablation_lref']
        line = f"  {res['condition']:<25} {lr['cv']:>8.4f} {lr['gini']:>10.4f}"
        if 'ablation_cls' in res:
            cl = res['ablation_cls']
            line += f" {cl['cv']:>8.4f} {cl['gini']:>10.4f}"
        print(line)

    out_dir = os.path.join(ORFC_ROOT, 'results', 'sensitivity')
    os.makedirs(out_dir, exist_ok=True)
    tag = (f"sens_{args.backbone}_{args.layer}_K{K}_emb{d}"
           f"_ndiag{args.n_diag}")
    out_path = os.path.join(out_dir, f'{tag}.json')
    output = {
        'config': {
            'backbone': args.backbone, 'layer': args.layer,
            'K': K, 'embedding_dim': d, 'G': G, 'D': D,
            'n_diag': args.n_diag, 'n_train': len(features_tr),
            'codec_path': args.codec_path or None,
            'norm_mode': norm_mode, 'seed': args.seed,
        },
        'conditions': all_results,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")
    return out_path


if __name__ == '__main__':
    main()
