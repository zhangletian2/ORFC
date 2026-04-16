#!/usr/bin/env python
"""
Compute feature-space MSE for DOPQ points in correlation_blk20.json
that are missing the 'soft_pq_mse' field.

Loads the checkpoint for each point, runs encode-decode on test features,
and writes MSE back into the JSON.

Usage:
    python eval_mse_correlation.py --gpu 0
"""

import argparse, json, os, sys, time
import numpy as np
import torch

ORFC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

from soft_pq import load_codec, batch_normalize_gpu, batch_inv_normalize_gpu


def compute_codec_mse(features, codec, norm_mode, device, batch_size=32):
    codec.eval()
    total_mse = 0.0
    N = len(features)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
            B = X.shape[0]
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            Y_hat, _ = codec(Y)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            mse = ((X - X_hat) ** 2).sum().item() / B
            total_mse += mse * B
            del X, Y, Mu, Std, Y_hat, X_hat
    return total_mse / N


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--json_path", default=os.path.join(
        ORFC_ROOT, "results", "analysis_intro_v2", "correlation_blk20.json"))
    parser.add_argument("--ckpt_dir", default=os.path.join(
        ORFC_ROOT, "checkpoints", "dinov2_vitl14"))
    parser.add_argument("--feat_root",
                        default=os.path.join(PROJECT_ROOT, "features"))
    parser.add_argument("--backbone", default="dinov2_vitl14")
    parser.add_argument("--layer", default="blk20")
    parser.add_argument("--norm_mode", default="per_image")
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    with open(args.json_path) as f:
        data = json.load(f)

    dopq_indices = []
    for i, p in enumerate(data['points']):
        if p.get('method') != 'dopq':
            continue
        if p.get('soft_pq_mse') is not None:
            continue
        src = p.get('source', '')
        ckpt_name = src.replace('.json', '.pt')
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
        if not os.path.isfile(ckpt_path):
            print(f"  SKIP (no ckpt): {ckpt_name}")
            continue
        dopq_indices.append((i, ckpt_path, ckpt_name))

    print(f"DOPQ points to evaluate: {len(dopq_indices)}")
    if not dopq_indices:
        print("Nothing to do.")
        return

    feat_dir = os.path.join(args.feat_root, "test", args.backbone, args.layer)
    print(f"Loading test features from {feat_dir} ...")
    all_files = sorted(f for f in os.listdir(feat_dir) if f.endswith('.npy'))
    rng = np.random.RandomState(args.seed)
    n_test = min(args.n_test, len(all_files))
    perm = rng.permutation(len(all_files))
    test_idx = sorted(perm[:n_test])
    features_test = [np.load(os.path.join(feat_dir, all_files[i]))
                     for i in test_idx]
    print(f"  Loaded {n_test} test features, shape={features_test[0].shape}")

    updated = 0
    for idx, ckpt_path, ckpt_name in dopq_indices:
        p = data['points'][idx]
        print(f"\n{'=' * 60}")
        print(f"  [{updated+1}/{len(dopq_indices)}] {ckpt_name}")
        print(f"  K={p['K']}, emb={p['emb']}, lmbda={p['lmbda']}, "
              f"fzR={p.get('freeze_transform', False)}")

        t0 = time.time()
        codec = load_codec(ckpt_path, device=device)
        mse = compute_codec_mse(
            features_test, codec, args.norm_mode, device,
            batch_size=args.batch_size)
        elapsed = time.time() - t0

        data['points'][idx]['soft_pq_mse'] = float(mse)
        updated += 1
        print(f"  MSE = {mse:.1f}  ({elapsed:.1f}s)")

        del codec
        torch.cuda.empty_cache()

    with open(args.json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n{'=' * 60}")
    print(f"Updated {updated} points in {args.json_path}")


if __name__ == "__main__":
    main()
