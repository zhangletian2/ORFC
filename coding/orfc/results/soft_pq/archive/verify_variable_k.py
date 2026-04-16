#!/usr/bin/env python3
"""Small-scale verification: does a better allocate_K improve distortion?

This script does NOT modify soft_pq.py.  It monkey-patches allocate_K
at runtime and calls the standard run_soft_pq pipeline with short epochs.

Usage:
    # Conservative allocation (min_ratio=4.0)
    CUDA_VISIBLE_DEVICES=0 python verify_variable_k.py --min_ratio 4.0

    # Moderate allocation (min_ratio=2.0)
    CUDA_VISIBLE_DEVICES=1 python verify_variable_k.py --min_ratio 2.0

    # Uniform baseline (no variable K, for comparison)
    CUDA_VISIBLE_DEVICES=2 python verify_variable_k.py --uniform_baseline
"""
import sys, os, math, argparse
import numpy as np

ORFC_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

import soft_pq
import torch


def allocate_K_fixed(s_g, G, base_K=16, K_choices=(4, 8, 16, 32, 64),
                     min_ratio=2.0):
    """Fixed allocate_K: only swap bits when sensitivity ratio >= min_ratio."""
    K_choices = sorted(K_choices)
    log2K_choices = [int(math.log2(k)) for k in K_choices]
    base_log2 = int(math.log2(base_K))
    total_bits = G * base_log2

    if isinstance(s_g, torch.Tensor):
        s_g = s_g.cpu().numpy()
    s_g = np.asarray(s_g, dtype=np.float64)

    alloc = [base_log2] * G

    while True:
        best_ratio = min_ratio
        best_pair = None

        for g_donor in range(G):
            if alloc[g_donor] <= log2K_choices[0]:
                continue
            for g_recv in range(G):
                if g_recv == g_donor:
                    continue
                if alloc[g_recv] >= log2K_choices[-1]:
                    continue
                ratio = s_g[g_recv] / (s_g[g_donor] + 1e-30)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pair = (g_donor, g_recv)

        if best_pair is not None:
            alloc[best_pair[0]] -= 1
            alloc[best_pair[1]] += 1
        else:
            break

    assert sum(alloc) == total_bits
    return [2 ** b for b in alloc]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_ratio", type=float, default=4.0,
                        help="Marginal gain threshold for bit swap")
    parser.add_argument("--uniform_baseline", action="store_true",
                        help="Run uniform K=16 baseline (no variable K)")
    parser.add_argument("--lmbda", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    # Monkey-patch allocate_K with fixed version
    _mr = args.min_ratio

    def patched_allocate_K(s_g, G, base_K=16, K_choices=(4, 8, 16, 32, 64)):
        return allocate_K_fixed(s_g, G, base_K, K_choices, min_ratio=_mr)

    soft_pq.allocate_K = patched_allocate_K
    print(f"[verify] Patched allocate_K with min_ratio={_mr}")

    # Build CLI args for run_soft_pq
    cli = [
        "run_soft_pq.py",
        "--layer", "blk20",
        "--K", "16",
        "--embedding_dim", "32",
        "--lmbda", str(args.lmbda),
        "--norm_mode", "per_image",
        "--bottleneck_dim", "1024",
        "--warm_start_opq",
        "--max_train_images", "5000",
        "--epochs", str(args.epochs),
        "--lr", "1e-4",
        "--batch_size", "32",
        "--n_val", "200",
        "--seed", "42",
        "--eval_seg",
        "--seg_feat_root", os.path.join(PROJECT_ROOT, "features", "voc2012_100"),
        "--voc_root", os.path.join(PROJECT_ROOT, "data", "VOCdevkit", "VOC2012"),
        "--seg_image_list", os.path.join(PROJECT_ROOT, "utils", "voc2012_val_100.txt"),
    ]

    if not args.uniform_baseline:
        cli += ["--variable_K", "--sensitivity_warmup", "10"]

    sys.argv = cli
    import run_soft_pq
    run_soft_pq.main()


if __name__ == "__main__":
    main()
