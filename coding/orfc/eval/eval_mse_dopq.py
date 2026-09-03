#!/usr/bin/env python
"""
Evaluate feature-space MSE for trained DOPQ codecs.

For each checkpoint, loads the codec, runs encode-decode on test features,
and computes MSE = ||X - X_hat||².  Appends 'soft_pq_mse' to the
corresponding result JSON.

Usage:
    python eval_mse_dopq.py --layer blk20 --ckpt_dir checkpoints/dinov2_vitl14
"""

import argparse, json, os, sys, time, glob
import numpy as np
import torch

ORFC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

from soft_pq import (
    FeatureCodec, load_codec, soft_pq_encode_decode,
    batch_normalize_gpu, batch_inv_normalize_gpu,
)


def compute_codec_mse(features, codec, norm_mode, device, batch_size=32):
    """Feature-space MSE: mean over images of sum-of-squared-error per image."""
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
    parser.add_argument("--layer", default="blk20")
    parser.add_argument("--ckpt_dir", default="checkpoints/dinov2_vitl14")
    parser.add_argument("--result_dir", default="results/soft_pq/dinov2_vitl14")
    parser.add_argument("--feat_root", default=os.path.join(PROJECT_ROOT, "features"))
    parser.add_argument("--backbone", default="dinov2_vitl14")
    parser.add_argument("--norm_mode", default="per_image")
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pattern", default=None,
                        help="Glob pattern to filter checkpoints (e.g. '*K16_emb32*')")
    args = parser.parse_args()

    ckpt_dir = os.path.join(ORFC_ROOT, args.ckpt_dir)
    result_dir = os.path.join(ORFC_ROOT, args.result_dir)
    device = torch.device("cuda")

    pattern = args.pattern or f"{args.layer}_*.pt"
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
    print(f"Found {len(ckpt_files)} checkpoints matching '{pattern}'")

    if not ckpt_files:
        print("No checkpoints found. Exiting.")
        return

    # Load test features once
    feat_dir = os.path.join(args.feat_root, "test",
                            args.backbone, args.layer)
    print(f"Loading test features from {feat_dir} ...")
    all_files = sorted([f for f in os.listdir(feat_dir) if f.endswith('.npy')])
    rng = np.random.RandomState(args.seed)
    n_total = len(all_files)
    n_test = min(args.n_test, n_total)
    perm = rng.permutation(n_total)
    test_idx = sorted(perm[:n_test])
    features_test = [np.load(os.path.join(feat_dir, all_files[i]))
                     for i in test_idx]
    print(f"  Loaded {n_test} test features, shape={features_test[0].shape}")

    for ckpt_path in ckpt_files:
        name = os.path.splitext(os.path.basename(ckpt_path))[0]
        json_path = os.path.join(result_dir, f"{name}.json")

        print(f"\n{'=' * 60}")
        print(f"  {name}")

        if not os.path.exists(json_path):
            print(f"  WARNING: No result JSON found at {json_path}, skipping")
            continue

        with open(json_path) as f:
            result = json.load(f)

        if 'soft_pq_mse' in result:
            print(f"  Already has MSE = {result['soft_pq_mse']:.1f}, skipping")
            continue

        t0 = time.time()
        codec = load_codec(ckpt_path, device=device)
        print(f"  Loaded codec ({time.time()-t0:.1f}s)")

        t1 = time.time()
        mse = compute_codec_mse(
            features_test, codec, args.norm_mode, device,
            batch_size=args.batch_size)
        print(f"  MSE = {mse:.1f} ({time.time()-t1:.1f}s)")

        result['soft_pq_mse'] = float(mse)
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {json_path}")

        del codec
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
