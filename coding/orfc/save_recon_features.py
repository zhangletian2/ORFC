#!/usr/bin/env python
"""
Save reconstructed features for OPQ and Ours (DOPQ) methods.

Outputs .npy files with the same shape as the original features:
  [num_slides, 1+N, D]

Usage:
    conda activate featcodec2
    python save_recon_features.py --layer blk10 --image 2009_005148 --gpu 0
"""

import os, sys, glob, argparse
import numpy as np
import torch
from pathlib import Path

ORFC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

import warnings
warnings.filterwarnings("ignore")

from run_multilayer_calibrator import set_seed, preload_features
from opq import (
    batch_normalize_gpu, batch_inv_normalize_gpu,
    batched_assign, learn_opq_rotation,
)
from soft_pq import load_codec


def per_image_norm(x, eps=1e-5):
    mu = x.mean()
    var = ((x - mu) ** 2).mean()
    std = np.sqrt(var + eps)
    return ((x - mu) / std).astype(np.float32), \
           np.array([[mu]], dtype=np.float32), \
           np.array([[std]], dtype=np.float32)


@torch.no_grad()
def reconstruct_opq(features, R_t, cb_t, num_groups, embedding_dim, feat_dim, device):
    """Reconstruct all slides with OPQ."""
    out = np.zeros_like(features, dtype=np.float32)
    for s in range(features.shape[0]):
        tokens = features[s]
        y, mu, std = per_image_norm(tokens)
        Y = torch.from_numpy(y).float().to(device).unsqueeze(0)
        flat = Y.reshape(-1, feat_dim)
        Z = flat @ R_t
        z_3d = Z.reshape(-1, num_groups, embedding_dim).permute(1, 0, 2).contiguous()
        z_hat_3d, _ = batched_assign(z_3d, cb_t, device=device)
        flat_hat = z_hat_3d.permute(1, 0, 2).reshape(-1, feat_dim)
        Y_hat = (flat_hat @ R_t.T).reshape(1, -1, feat_dim)
        Mu = torch.from_numpy(mu).float().to(device).unsqueeze(0)
        Std = torch.from_numpy(std).float().to(device).unsqueeze(0)
        X_hat = (Y_hat * Std + Mu).squeeze(0).cpu().numpy()
        out[s] = X_hat
    return out


@torch.no_grad()
def reconstruct_codec(features, codec, device):
    """Reconstruct all slides with our codec."""
    out = np.zeros_like(features, dtype=np.float32)
    for s in range(features.shape[0]):
        X = torch.from_numpy(features[s].astype(np.float32)).unsqueeze(0).to(device)
        Y, Mu, Std = batch_normalize_gpu(X, mode='per_image')
        Y_hat, _ = codec(Y)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        out[s] = X_hat.squeeze(0).cpu().numpy()
    return out


def train_opq(feat_dir, embedding_dim, K, device, max_train=5000, seed=42):
    feat_files = sorted(Path(feat_dir).glob("*.npy"))
    print(f"  Loading {len(feat_files)} training features ...")
    features, _ = preload_features(feat_files, num_workers=8)

    if max_train > 0 and len(features) > max_train:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(features), max_train, replace=False)
        features = [features[i] for i in idx]

    D = features[0].shape[1]
    G = D // embedding_dim

    all_vecs = []
    for s in range(0, len(features), 200):
        e = min(s + 200, len(features))
        X = torch.from_numpy(np.stack(features[s:e])).float().to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode='per_image')
        all_vecs.append(Y.reshape(-1, D).cpu().numpy())
        del X, Y
    flat = np.concatenate(all_vecs, axis=0)
    del all_vecs

    max_flat = 2_000_000 // G
    if flat.shape[0] > max_flat:
        flat = flat[np.random.RandomState(seed).choice(flat.shape[0], max_flat, replace=False)]

    print(f"  Training OPQ: G={G}, K={K}, d={embedding_dim}, vectors={flat.shape[0]}")
    R, codebooks, hist = learn_opq_rotation(
        flat, G, embedding_dim, K,
        max_iter_opq=20, max_iter_kmeans=100,
        device=device, verbose=False)
    print(f"  OPQ trained. MSE={hist[-1][0]:.8f}")
    del flat
    torch.cuda.empty_cache()
    return R, codebooks


def main():
    parser = argparse.ArgumentParser(
        description="Save reconstructed features for OPQ and Ours",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--image", type=str, default="2009_005148")
    parser.add_argument("--layer", type=str, default="blk10")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--backbone", type=str, default="dinov2_vitl14")
    parser.add_argument("--feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features"))
    parser.add_argument("--codec_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    set_seed(args.seed)

    D = 1024
    G = D // args.embedding_dim

    # Derive paths
    orig_feat_dir = os.path.join(args.feat_root, "voc2012_100", args.backbone, args.layer)
    train_feat_dir = os.path.join(args.feat_root, "train", args.backbone, args.layer)

    if args.codec_path is None:
        ckpt_dir = os.path.join(ORFC_ROOT, "checkpoints", args.backbone)
        pattern = f"{args.layer}_K{args.K}_emb{args.embedding_dim}_*_tau*_*.pt"
        candidates = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint matching {pattern} in {ckpt_dir}")
        args.codec_path = candidates[-1]

    if args.out_dir is None:
        args.out_dir = os.path.join(ORFC_ROOT, "features_recon",
                                    f"{args.backbone}_{args.layer}_K{args.K}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load original features
    orig_path = os.path.join(orig_feat_dir, f"{args.image}.npy")
    orig = np.load(orig_path)
    print(f"Original features: {orig_path}")
    print(f"  shape={orig.shape}, dtype={orig.dtype}")

    # Save original for reference
    orig_out = os.path.join(args.out_dir, f"{args.image}_original.npy")
    np.save(orig_out, orig.astype(np.float32))
    print(f"  Saved: {orig_out}")

    # Train OPQ
    print(f"\nTraining OPQ (K={args.K}, d={args.embedding_dim}) ...")
    R, codebooks = train_opq(train_feat_dir, args.embedding_dim, args.K,
                             device, seed=args.seed)
    R_t = torch.from_numpy(R).float().to(device)
    cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)

    # Reconstruct with OPQ
    print(f"\nReconstructing with OPQ ...")
    opq_recon = reconstruct_opq(orig, R_t, cb_t, G, args.embedding_dim, D, device)
    opq_out = os.path.join(args.out_dir, f"{args.image}_opq.npy")
    np.save(opq_out, opq_recon)
    mse_opq = np.mean((orig.astype(np.float32) - opq_recon) ** 2)
    print(f"  MSE={mse_opq:.6f}  shape={opq_recon.shape}")
    print(f"  Saved: {opq_out}")

    # Load codec and reconstruct
    print(f"\nLoading codec: {args.codec_path}")
    codec = load_codec(args.codec_path, device=device)
    codec.eval()
    print(f"  G={codec.pq.G}, K={codec.pq.K}, d={codec.pq.d}")

    print(f"Reconstructing with Ours ...")
    ours_recon = reconstruct_codec(orig, codec, device)
    ours_out = os.path.join(args.out_dir, f"{args.image}_ours.npy")
    np.save(ours_out, ours_recon)
    mse_ours = np.mean((orig.astype(np.float32) - ours_recon) ** 2)
    print(f"  MSE={mse_ours:.6f}  shape={ours_recon.shape}")
    print(f"  Saved: {ours_out}")

    print(f"\nDone. Output directory: {args.out_dir}")
    print(f"  {args.image}_original.npy")
    print(f"  {args.image}_opq.npy      MSE={mse_opq:.6f}")
    print(f"  {args.image}_ours.npy     MSE={mse_ours:.6f}")


if __name__ == '__main__':
    main()
