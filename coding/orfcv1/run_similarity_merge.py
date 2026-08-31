#!/usr/bin/env python
"""Token merge + position residual decoder (classification).

No R / PQ / entropy.  Global greedy matching of ``r`` disjoint pairs,
``y_m = (x_i+x_j)/2``, decoder ``x̂ = y_m + r_ψ(y_m, Δp)``.  Train ``r_ψ``
with ΔL_ref on the 5k training split.

Matching score (``--match_mode``):
  cosine     : cosine similarity of patch tokens (default, previous runs)
  lref_proxy : Hutchinson estimate of ||J δ||² for the mean-merge
               perturbation through the frozen ViT tail (L_ref proxy)

Reports:
  copy   : decoder off (both slots = y_m), zero-train baseline
  trained: after ΔL_ref optimisation of r_ψ

Usage (blk05, L_ref-proxy matching, GPU 2):
    CUDA_VISIBLE_DEVICES=2 python -u run_similarity_merge.py --layer blk05 \
        --match_mode lref_proxy --save_codec
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_ORFC_DIR = os.path.normpath(os.path.join(HERE, '..', 'orfc'))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
if _ORFC_DIR not in sys.path:
    sys.path.insert(0, _ORFC_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from run_multilayer_calibrator import (  # noqa: E402
    set_seed, preload_features, load_gt, evaluate_accuracy,
)
from backbone.wrapper import Dinov2Wrapper  # noqa: E402
from soft_pq import FrozenTail  # noqa: E402
from opq import batch_normalize_gpu, batch_inv_normalize_gpu  # noqa: E402
from run_soft_pq import evaluate_delta_l_ref  # noqa: E402

from similarity_merge import (  # noqa: E402
    SimilarityMergeCodec, train_similarity_merge_codec, save_merge_codec,
    precompute_lref_pairs, precompute_exact_lref_pairs,
    cosine_match_pairs_numpy, matching_jaccard,
)

import warnings  # noqa: E402
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")


@torch.no_grad()
def encode_decode_with_stats(features, codec, norm_mode, device, n_prefix,
                             batch_size=32, pair_left=None, pair_right=None):
    mse_sum = 0.0
    energy_sum = 0.0
    n_elem = 0
    cos_acc = 0.0
    dist_acc = 0.0
    n_batch = 0
    all_xhat = []
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
        pairs = None
        if pair_left is not None:
            pairs = (
                torch.from_numpy(pair_left[start:end]).long().to(device),
                torch.from_numpy(pair_right[start:end]).long().to(device),
            )
        Y_hat, _ = codec(Y, X=X, pairs=pairs)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        p = n_prefix
        diff = Y[:, p:, :] - Y_hat[:, p:, :]
        mse_sum += (diff ** 2).sum().item()
        energy_sum += (Y[:, p:, :] ** 2).sum().item()
        n_elem += diff.numel()
        diag = codec.match_diagnostics(Y, X=X, pairs=pairs)
        cos_acc += diag['mean_matched_cos']
        dist_acc += diag['mean_grid_l2']
        n_batch += 1
        for i in range(X_hat.shape[0]):
            all_xhat.append(X_hat[i].cpu().numpy())
        del X, Y, Mu, Std, Y_hat, X_hat, diff
    stats = {
        'mse_patch': float(mse_sum / max(n_elem, 1)),
        'rel_energy_err': float(mse_sum / max(energy_sum, 1e-12)),
        'mean_matched_cos': float(cos_acc / max(n_batch, 1)),
        'mean_grid_l2': float(dist_acc / max(n_batch, 1)),
    }
    return all_xhat, stats


@torch.no_grad()
def evaluate_delta_l_ref_pairs(features, tail, codec, norm_mode, device,
                               n_prefix, batch_size=4, pair_left=None,
                               pair_right=None):
    """ΔL_ref with optional precomputed pairs (needed for lref_proxy)."""
    total = 0.0
    n = len(features)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        B = X.shape[0]
        Y_teacher = tail.forward_nograd(X)
        Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
        pairs = None
        if pair_left is not None:
            pairs = (
                torch.from_numpy(pair_left[start:end]).long().to(device),
                torch.from_numpy(pair_right[start:end]).long().to(device),
            )
        Y_hat, _ = codec(Y, X=X, pairs=pairs)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        total += ((Y_teacher - tail.forward_nograd(X_hat)) ** 2).sum().item()
        del X, Y, Mu, Std, Y_hat, X_hat, Y_teacher
    return total / max(n, 1)


def _eval(name, codec, features, basenames, gt, wrapper, layer_idx, device,
          norm_mode, n_prefix, batch_size, T_full, pair_left=None,
          pair_right=None):
    codec = codec.to(device).eval()
    wrapper.backbone.to(device)
    if wrapper.head is not None:
        wrapper.head.to(device)
    torch.cuda.empty_cache()

    t0 = time.time()
    xhat, stats = encode_decode_with_stats(
        features, codec, norm_mode, device, n_prefix, batch_size,
        pair_left=pair_left, pair_right=pair_right)
    acc = evaluate_accuracy(xhat, basenames, gt, wrapper, layer_idx, device)
    del xhat
    t_acc = time.time() - t0

    tail = FrozenTail(list(wrapper.backbone.blocks[layer_idx + 1:]),
                      wrapper.backbone.norm, device=device)
    if getattr(codec, 'match_mode', 'cosine') in ('lref', 'lref_proxy'):
        codec.bind_tail(tail)
    t1 = time.time()
    if pair_left is not None or getattr(codec, 'match_mode', 'cosine') != 'cosine':
        dl = evaluate_delta_l_ref_pairs(
            features, tail, codec, norm_mode, device, n_prefix,
            batch_size=batch_size, pair_left=pair_left, pair_right=pair_right)
    else:
        dl = evaluate_delta_l_ref(features, tail, norm_mode, device,
                                  codec=codec, batch_size=batch_size)
    t_dl = time.time() - t1
    tail.to('cpu')
    torch.cuda.empty_cache()

    row = {
        'name': name,
        'acc': float(acc),
        'delta_l': float(dl),
        'coded_tokens': codec.coded_tokens(T_full),
        **stats,
        't_acc_s': t_acc,
        't_dl_s': t_dl,
    }
    print(f"  Acc={row['acc']:.4f}  ΔL_ref={row['delta_l']:.1f}  "
          f"MSE_p={row['mse_patch']:.6f}  relE={row['rel_energy_err']:.4f}  "
          f"cos={row['mean_matched_cos']:.4f}  "
          f"gridL2={row['mean_grid_l2']:.2f}  "
          f"coded={row['coded_tokens']}/{T_full}")
    return row


def _rebuild_train_tail(wrapper, layer_idx, device):
    for i, blk in enumerate(wrapper.backbone.blocks):
        if i <= layer_idx:
            blk.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()
    return FrozenTail(list(wrapper.backbone.blocks[layer_idx + 1:]),
                      wrapper.backbone.norm, device=device)


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--layer", type=str, default="blk05")
    p.add_argument("--r", type=int, default=128,
                   help="number of merged pairs (ignored if --keep_patch>0)")
    p.add_argument("--keep_patch", type=int, default=0,
                   help="coded patch tokens = unmatched + y_m; sets r=T_patch-keep_patch")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n_prefix", type=int, default=1)
    p.add_argument("--grid", type=int, default=0)
    p.add_argument("--norm_mode", type=str, default="per_image",
                   choices=["per_image", "split_cls_patch"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--match_mode", type=str, default="cosine",
                   choices=["cosine", "lref", "lref_proxy"])
    p.add_argument("--n_probe", type=int, default=1,
                   help="L_ref-proxy probes only")
    p.add_argument("--n_cand", type=int, default=8,
                   help="exact L_ref: candidate pairs scored per round")
    p.add_argument("--commit_k", type=int, default=1,
                   help="exact L_ref: pairs committed per round")
    p.add_argument("--match_batch", type=int, default=4,
                   help="image batch for L_ref pair precompute")
    p.add_argument("--max_train_images", type=int, default=5000)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--save_codec", action="store_true")
    p.add_argument("--feat_root", type=str,
                   default=os.path.join(PROJECT_ROOT, "features"))
    p.add_argument("--train_subset", type=str, default="train")
    p.add_argument("--test_subset", type=str, default="test")
    p.add_argument("--backbone", type=str, default="dinov2_vitl14")
    p.add_argument("--gt_path", type=str,
                   default=os.path.join(PROJECT_ROOT, "utils",
                                        "imagenet_selected_label500.txt"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    layer_idx = int(args.layer[-2:])
    grid = args.grid if args.grid > 0 else None

    print(f"\n{'#' * 70}")
    print(f"# Token merge + r_ψ  (no R / PQ / entropy)")
    print(f"# {args.layer}  match={args.match_mode}  n_cand={args.n_cand}  "
          f"commit_k={args.commit_k}  r={args.r}  hidden={args.hidden}")
    print(f"# ep={args.epochs} lr={args.lr} seed={args.seed}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    train_dir = Path(args.feat_root) / args.train_subset / args.backbone / args.layer
    test_dir = Path(args.feat_root) / args.test_subset / args.backbone / args.layer
    train_files = sorted(train_dir.glob("*.npy"))
    test_files = sorted(test_dir.glob("*.npy"))
    if not train_files or not test_files:
        raise FileNotFoundError(f"features missing: {train_dir} / {test_dir}")
    features_train, _ = preload_features(train_files, num_workers=4)
    features_test, basenames_test = preload_features(test_files, num_workers=4)
    gt_test = load_gt(args.gt_path)

    T_full, D = features_train[0].shape
    T_patch = T_full - args.n_prefix
    if args.keep_patch > 0:
        if not (1 <= args.keep_patch <= T_patch):
            raise ValueError(f"keep_patch={args.keep_patch} not in [1, {T_patch}]")
        args.r = T_patch - args.keep_patch
        print(f"  keep_patch={args.keep_patch} → r={args.r}")
    if 2 * args.r > T_patch:
        raise ValueError(f"r={args.r} needs 2r <= T_patch={T_patch}")
    n_unmatched = T_patch - 2 * args.r
    n_coded_patch = n_unmatched + args.r
    print(f"\nData: train={len(features_train)} test={len(features_test)}  "
          f"D={D} T_full={T_full} T_patch={T_patch}")
    print(f"  merge r={args.r} pairs; unmatched={n_unmatched} identity; "
          f"coded patch={n_coded_patch}  coded total={T_full - args.r}")

    if 0 < args.max_train_images < len(features_train):
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(features_train), args.max_train_images,
                         replace=False)
        features_train_sub = [features_train[i] for i in idx]
    else:
        features_train_sub = features_train
    n_val = min(args.n_val, len(features_train))
    rng_val = np.random.RandomState(args.seed + 1)
    val_idx = rng_val.choice(len(features_train), n_val, replace=False)
    val_features = [features_train[i] for i in val_idx]

    results = {'config': vars(args), 'D': D, 'T_full': T_full,
               'T_patch': T_patch, 'n_unmatched': n_unmatched,
               'n_coded_patch': n_coded_patch,
               'n_coded_total': T_full - args.r}

    print(f"\nLoading DINOv2 ({args.backbone})...")
    wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone,
                            device=device)

    test_left = test_right = None
    train_left = train_right = None
    val_left = val_right = None
    if args.match_mode in ('lref', 'lref_proxy'):
        print(f"\nPrecomputing {args.match_mode} pairs...")
        tail_match = FrozenTail(
            list(wrapper.backbone.blocks[layer_idx + 1:]),
            wrapper.backbone.norm, device=device)
        print("  test...")
        if args.match_mode == 'lref':
            test_left, test_right = precompute_exact_lref_pairs(
                features_test, tail_match, args.r, args.n_prefix, device,
                batch_size=args.match_batch, n_cand=args.n_cand,
                commit_k=args.commit_k, grid=grid, verbose=True)
            if args.epochs > 0:
                print("  train...")
                train_left, train_right = precompute_exact_lref_pairs(
                    features_train_sub, tail_match, args.r, args.n_prefix,
                    device, batch_size=args.match_batch, n_cand=args.n_cand,
                    commit_k=args.commit_k, grid=grid, verbose=True)
                print("  val...")
                val_left, val_right = precompute_exact_lref_pairs(
                    val_features, tail_match, args.r, args.n_prefix, device,
                    batch_size=args.match_batch, n_cand=args.n_cand,
                    commit_k=args.commit_k, grid=grid, verbose=True)
        else:
            test_left, test_right = precompute_lref_pairs(
                features_test, tail_match, args.r, args.n_prefix, device,
                batch_size=args.match_batch, n_probe=args.n_probe,
                seed=args.seed, verbose=True)
            print("  train...")
            train_left, train_right = precompute_lref_pairs(
                features_train_sub, tail_match, args.r, args.n_prefix, device,
                batch_size=args.match_batch, n_probe=args.n_probe,
                seed=args.seed, verbose=True)
            print("  val...")
            val_left, val_right = precompute_lref_pairs(
                val_features, tail_match, args.r, args.n_prefix, device,
                batch_size=args.match_batch, n_probe=args.n_probe,
                seed=args.seed + 7, verbose=True)
        print("  cosine pairs on test (overlap diagnostic)...")
        cos_left, cos_right = cosine_match_pairs_numpy(
            features_test, args.r, args.n_prefix, device)
        jac = matching_jaccard(test_left, test_right, cos_left, cos_right)
        results['match_overlap_vs_cosine_jaccard'] = jac
        print(f"  Jaccard vs cosine matching (test): {jac:.4f}")
        tail_match.to('cpu')
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}\n  [copy] decoder off, x̂ = y_m  "
          f"match={args.match_mode}\n{'=' * 60}")
    copy_codec = SimilarityMergeCodec(
        D, n_prefix=args.n_prefix, r=args.r, grid=grid,
        hidden=args.hidden, use_decoder=False,
        match_mode=args.match_mode, n_probe=args.n_probe,
        n_cand=args.n_cand, commit_k=args.commit_k).to(device)
    results['copy'] = _eval(
        'copy', copy_codec, features_test, basenames_test, gt_test,
        wrapper, layer_idx, device, args.norm_mode, args.n_prefix,
        args.batch_size, T_full, pair_left=test_left, pair_right=test_right)
    del copy_codec
    torch.cuda.empty_cache()

    if args.epochs > 0:
        print(f"\n{'=' * 60}\n  [train] r_ψ with ΔL_ref, {args.epochs} ep, "
              f"n={len(features_train_sub)}  match={args.match_mode}\n{'=' * 60}")
        tail_train = _rebuild_train_tail(wrapper, layer_idx, device)
        t0 = time.time()
        codec, hist = train_similarity_merge_codec(
            features_train=features_train_sub, tail=tail_train, D=D,
            n_prefix=args.n_prefix, r=args.r, grid=grid, hidden=args.hidden,
            norm_mode=args.norm_mode, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, device=device, seed=args.seed,
            val_features=val_features, verbose=True, grad_clip=args.grad_clip,
            match_mode=args.match_mode, n_probe=args.n_probe,
            n_cand=args.n_cand, commit_k=args.commit_k,
            pair_left=train_left, pair_right=train_right,
            val_pair_left=val_left, val_pair_right=val_right)
        print(f"  training: {time.time() - t0:.1f}s")
        results['history'] = hist

        print(f"\n{'=' * 60}\n  [trained] r_ψ after ΔL_ref\n{'=' * 60}")
        results['trained'] = _eval(
            'trained', codec, features_test, basenames_test, gt_test,
            wrapper, layer_idx, device, args.norm_mode, args.n_prefix,
            args.batch_size, T_full, pair_left=test_left, pair_right=test_right)
    else:
        codec = None
        results['history'] = []
        results['trained'] = None

    out_dir = Path(HERE) / 'results' / 'similarity_merge' / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"{args.layer}_r{args.r}_h{args.hidden}"
           f"_lr{args.lr}_ep{args.epochs}_s{args.seed}")
    if args.match_mode != 'cosine':
        tag += f"_{args.match_mode}_c{args.n_cand}k{args.commit_k}"
    if args.save_codec and codec is not None:
        save_merge_codec(codec, out_dir / f"{tag}.pt",
                         meta_extra={'layer': args.layer,
                                     'match_mode': args.match_mode})
    if test_left is not None:
        payload = {'test_left': test_left, 'test_right': test_right}
        if train_left is not None:
            payload['train_left'] = train_left
            payload['train_right'] = train_right
        np.savez_compressed(out_dir / f"{tag}_pairs.npz", **payload)
    out_path = out_dir / f"{tag}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")
    print(f"  copy     Acc={results['copy']['acc']:.4f}  "
          f"ΔL={results['copy']['delta_l']:.1f}")
    if results.get('trained'):
        print(f"  trained  Acc={results['trained']['acc']:.4f}  "
              f"ΔL={results['trained']['delta_l']:.1f}")
    if 'match_overlap_vs_cosine_jaccard' in results:
        print(f"  Jaccard vs cosine: "
              f"{results['match_overlap_vs_cosine_jaccard']:.4f}")
    return results


if __name__ == '__main__':
    main()
