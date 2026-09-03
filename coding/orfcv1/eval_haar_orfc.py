#!/usr/bin/env python
"""Cascade eval: trained Haar-joint spatial codec + frozen ORFC weights.

Y(257) --Haar encode--> seq(65) --ORFC PQ--> seq_hat --Haar decode--> Y_hat

Rate per image:
    PQ bits (rANS / xent on 65 tokens)
  + grouping bits  log2(256!)   [ordered partition = permutation]
  + per-image μ/σ  32 bits
BPFP uses the original feature count 257×1024 so it is comparable to ORFC-only.

Usage:
    CUDA_VISIBLE_DEVICES=2 python -u eval_haar_orfc.py --layer blk05
    CUDA_VISIBLE_DEVICES=4 python -u eval_haar_orfc.py --layer blk20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_ORFC_DIR = os.path.normpath(os.path.join(HERE, "..", "orfc"))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if _ORFC_DIR not in sys.path:
    sys.path.insert(0, _ORFC_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from backbone.wrapper import Dinov2Wrapper  # noqa: E402
from opq import batch_inv_normalize_gpu, batch_normalize_gpu  # noqa: E402
from run_multilayer_calibrator import (  # noqa: E402
    evaluate_accuracy, load_gt, preload_features, set_seed,
)
from run_soft_pq import _histogram_pmf, _rans_encode_bpt  # noqa: E402
from soft_pq import FrozenTail, load_codec  # noqa: E402

from run_global_residual import (  # noqa: E402
    build_groups, load_global_detail,
)


def log2_factorial(n):
    return sum(math.log2(i) for i in range(2, n + 1))


# Ordered 4-tuples covering 256 patches ↔ permutation of 256 indices.
GROUPING_BITS_PERM = log2_factorial(256)
GROUPING_BITS_UINT8 = 256 * 8.0
NORM_BITS_PER_IMAGE = 32.0  # per_image μ+σ, 16+16
ORIG_TOKENS = 257
FEAT_DIM = 1024


def default_haar_ckpt(layer, backbone="dinov2_vitl14"):
    return (Path(HERE) / "results" / "global_residual" / backbone
            / f"{layer}_global_haar_joint_D_lr0.0003_ep10_s42.pt")


def default_orfc_ckpt(layer, K, ckpt_dir):
    # blk05 K64 used lr=5e-4; everything else in this sweep is 3e-4.
    lr = "0.0005" if (layer == "blk05" and int(K) == 64) else "0.0003"
    name = (f"{layer}_K{K}_emb32_bt1024_ws_lmbda0.5_tau0.5_"
            f"lr{lr}_ep100_n5000_s42.pt")
    return Path(ckpt_dir) / name


def _rate_from_labels(test_labels, train_labels, G, K):
    """Same metrics as run_soft_pq.evaluate_rate, given already-collected labels."""
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
    max_rate = G * math.log2(K)
    out = {
        "xent_train_bpt": float(xent_train / N),
        "empirical_entropy_bpt": float(emp),
        "max_rate_bpt": float(max_rate),
        "n_coded_tokens_total": int(N),
    }
    if rans_train is not None:
        out["rans_train_bpt"] = float(rans_train)
    return out, primary_pmf


def pack_image_rate(bpt_dict, n_coded, grouping_bits, n_orig=ORIG_TOKENS,
                    feat_dim=FEAT_DIM):
    """Convert bits/coded-token → bits/image and BPFP over original features."""
    rans = bpt_dict.get("rans_train_bpt")
    xent = bpt_dict.get("xent_train_bpt")
    pq_bpt = rans if rans is not None else xent
    pq_bits = float(pq_bpt) * n_coded
    total = pq_bits + grouping_bits + NORM_BITS_PER_IMAGE
    return {
        "n_coded_tokens": int(n_coded),
        "pq_bpt": float(pq_bpt),
        "pq_bits_per_image": pq_bits,
        "grouping_bits_perm": float(grouping_bits),
        "grouping_bits_uint8": GROUPING_BITS_UINT8,
        "norm_bits_per_image": NORM_BITS_PER_IMAGE,
        "bits_per_image": total,
        "bpfp": total / (n_orig * feat_dim),
        **bpt_dict,
    }


@torch.no_grad()
def collect_orfc_labels(token_seqs, orfc, device, batch_size):
    """token_seqs: already-normalized [N, Tm, D] numpy. Returns [G, N*Tm]."""
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


@torch.no_grad()
def haar_encode_all(features, groups, haar, norm_mode, device, batch_size):
    """Return normalized compressed sequences [N, Tm, D] float32 numpy."""
    haar.eval()
    chunks = []
    n = len(features)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        g = torch.from_numpy(groups[start:end].astype(np.int64)).to(device)
        Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=haar.n_prefix)
        seq, _ = haar.encode(Y, g)
        chunks.append(seq.cpu().numpy())
        del X, Y, g, seq
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def eval_cascade(name, features, basenames, groups, haar, orfc, tail, wrapper,
                 layer_idx, gt, norm_mode, device, batch_size):
    """Haar encode → optional ORFC → Haar decode.  Acc / ΔL_ref / MSE."""
    haar.eval()
    if orfc is not None:
        orfc.eval()
    all_xhat = []
    delta_l = 0.0
    mse = 0.0
    n_elem = 0
    t0 = time.time()
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        g = torch.from_numpy(groups[start:end].astype(np.int64)).to(device)
        Y, Mu, Std = batch_normalize_gpu(
            X, mode=norm_mode, n_prefix=haar.n_prefix)
        seq, aux = haar.encode(Y, g)
        if orfc is not None:
            seq, _ = orfc(seq)
        Y_hat = haar.decode(seq, aux)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        teacher = tail.forward_nograd(X)
        student = tail.forward_nograd(X_hat)
        delta_l += ((teacher - student) ** 2).sum().item()
        diff = Y[:, haar.n_prefix:] - Y_hat[:, haar.n_prefix:]
        mse += (diff ** 2).sum().item()
        n_elem += diff.numel()
        all_xhat.extend(X_hat.cpu().numpy())
        del X, Y, Mu, Std, seq, Y_hat, X_hat, teacher, student, g
    acc = evaluate_accuracy(all_xhat, basenames, gt, wrapper, layer_idx, device)
    row = {
        "name": name,
        "acc": float(acc),
        "delta_l": float(delta_l / len(features)),
        "mse_patch": float(mse / n_elem),
        "t_s": time.time() - t0,
    }
    print(f"  {name:28s} Acc={row['acc']:.4f}  ΔL={row['delta_l']:.1f}  "
          f"MSE_p={row['mse_patch']:.6f}")
    return row


@torch.no_grad()
def eval_orfc_only(name, features, basenames, orfc, tail, wrapper, layer_idx,
                   gt, norm_mode, device, batch_size):
    orfc.eval()
    all_xhat = []
    delta_l = 0.0
    mse = 0.0
    n_elem = 0
    t0 = time.time()
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=1)
        Y_hat, _ = orfc(Y)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        teacher = tail.forward_nograd(X)
        student = tail.forward_nograd(X_hat)
        delta_l += ((teacher - student) ** 2).sum().item()
        diff = Y[:, 1:] - Y_hat[:, 1:]
        mse += (diff ** 2).sum().item()
        n_elem += diff.numel()
        all_xhat.extend(X_hat.cpu().numpy())
        del X, Y, Mu, Std, Y_hat, X_hat, teacher, student
    acc = evaluate_accuracy(all_xhat, basenames, gt, wrapper, layer_idx, device)
    row = {
        "name": name,
        "acc": float(acc),
        "delta_l": float(delta_l / len(features)),
        "mse_patch": float(mse / n_elem),
        "t_s": time.time() - t0,
    }
    print(f"  {name:28s} Acc={row['acc']:.4f}  ΔL={row['delta_l']:.1f}  "
          f"MSE_p={row['mse_patch']:.6f}")
    return row


def collect_plain_orfc_labels(features, orfc, norm_mode, device, batch_size):
    orfc.eval()
    rows = []
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=1)
        _ = orfc(Y)
        rows.append(orfc.pq._last_labels.cpu())
        del X, Y
    return torch.cat(rows, dim=1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="blk05")
    ap.add_argument("--haar_ckpt", default=None)
    ap.add_argument("--orfc_dir", default=os.path.join(
        _ORFC_DIR, "checkpoints", "dinov2_vitl14"))
    ap.add_argument("--K", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--skip_orfc_only", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_train_images", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--norm_mode", default="per_image")
    ap.add_argument("--backbone", default="dinov2_vitl14")
    ap.add_argument("--feat_root", default=os.path.join(PROJECT_ROOT, "features"))
    ap.add_argument("--gt_path", default=os.path.join(
        PROJECT_ROOT, "utils", "imagenet_selected_label500.txt"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    layer_idx = int(args.layer[-2:])
    haar_path = Path(args.haar_ckpt) if args.haar_ckpt else default_haar_ckpt(
        args.layer, args.backbone)
    if not haar_path.is_file():
        raise FileNotFoundError(haar_path)

    print(f"\n{'#' * 70}")
    print(f"# Haar + frozen ORFC  {args.layer}  K={args.K}")
    print(f"# haar={haar_path.name}")
    print(f"# grouping = log2(256!) = {GROUPING_BITS_PERM:.2f} bits/img")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    train_files = sorted(
        (Path(args.feat_root) / "train" / args.backbone / args.layer).glob("*.npy"))
    test_files = sorted(
        (Path(args.feat_root) / "test" / args.backbone / args.layer).glob("*.npy"))
    train_features, _ = preload_features(train_files, num_workers=4)
    test_features, test_names = preload_features(test_files, num_workers=4)
    if 0 < args.max_train_images < len(train_features):
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(train_features), args.max_train_images, replace=False)
        train_features = [train_features[i] for i in idx]

    T_full, D = train_features[0].shape
    n_patch = T_full - 1
    print(f"Data: train={len(train_features)} test={len(test_features)}  "
          f"T={T_full} D={D}")

    print("Building groups (train+test)...")
    t_g = time.time()
    train_groups = build_groups(
        train_features, args.norm_mode, device, args.batch_size)
    test_groups = build_groups(
        test_features, args.norm_mode, device, args.batch_size)
    print(f"  groups: {time.time() - t_g:.1f}s  "
          f"shape train={train_groups.shape} test={test_groups.shape}")
    # permutation check
    g0 = test_groups[0].astype(np.int64).reshape(-1)
    assert g0.min() == 0 and g0.max() == n_patch - 1
    assert len(np.unique(g0)) == n_patch

    haar, haar_meta = load_global_detail(str(haar_path), device=str(device))
    if not haar.joint:
        raise ValueError("this eval requires joint Haar (low+detail in 64 tokens)")
    Tm = haar.coded_tokens(T_full)
    print(f"Loaded Haar  joint={haar.joint}  coded={Tm}/{T_full}")

    print(f"Loading {args.backbone}...")
    wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone, device=device)
    tail = FrozenTail(list(wrapper.backbone.blocks[layer_idx + 1:]),
                      wrapper.backbone.norm, device=device)
    gt = load_gt(args.gt_path)

    print("Haar-encoding train/test sequences...")
    t_e = time.time()
    train_seq = haar_encode_all(
        train_features, train_groups, haar, args.norm_mode, device, args.batch_size)
    test_seq = haar_encode_all(
        test_features, test_groups, haar, args.norm_mode, device, args.batch_size)
    print(f"  encode: {time.time() - t_e:.1f}s  seq={tuple(test_seq.shape)}")

    results = {
        "config": vars(args),
        "haar_ckpt": str(haar_path),
        "T_full": T_full, "Tm": Tm, "D": D,
        "grouping_bits_perm": GROUPING_BITS_PERM,
        "grouping_bits_uint8": GROUPING_BITS_UINT8,
        "rows": [],
    }

    print(f"\n{'=' * 60}\n  [haar-only] unquantized 65 tokens\n{'=' * 60}")
    row = eval_cascade(
        "haar_only", test_features, test_names, test_groups, haar, None,
        tail, wrapper, layer_idx, gt, args.norm_mode, device, args.batch_size)
    row["rate"] = {
        "n_coded_tokens": Tm,
        "pq_bpt": None,
        "pq_bits_per_image": None,
        "grouping_bits_perm": GROUPING_BITS_PERM,
        "norm_bits_per_image": NORM_BITS_PER_IMAGE,
        "bits_per_image": None,
        "bpfp": None,
        "note": "payload is unquantized float; only grouping+norm are finite",
    }
    results["rows"].append(row)

    for K in args.K:
        ckpt = default_orfc_ckpt(args.layer, K, args.orfc_dir)
        if not ckpt.is_file():
            print(f"  SKIP missing {ckpt.name}")
            continue
        print(f"\n{'=' * 60}\n  [haar+ORFC] {ckpt.name}\n{'=' * 60}")
        orfc = load_codec(str(ckpt), device=str(device))
        orfc.eval()
        G, K_cb = orfc.pq.G, orfc.pq.K

        row = eval_cascade(
            f"haar_orfc_K{K}", test_features, test_names, test_groups, haar,
            orfc, tail, wrapper, layer_idx, gt, args.norm_mode, device,
            args.batch_size)

        print("  collecting PQ labels (train PMF + test rANS)...")
        train_lab = collect_orfc_labels(
            train_seq, orfc, device, args.batch_size)
        test_lab = collect_orfc_labels(
            test_seq, orfc, device, args.batch_size)
        bpt, _ = _rate_from_labels(test_lab, train_lab, G, K_cb)
        row["rate"] = pack_image_rate(bpt, Tm, GROUPING_BITS_PERM)
        row["orfc_ckpt"] = ckpt.name
        row["orfc_K"] = int(K)
        r = row["rate"]
        print(f"  rate  pq={r['pq_bpt']:.2f} b/tok × {Tm} = "
              f"{r['pq_bits_per_image']:.1f}  "
              f"+group {r['grouping_bits_perm']:.1f}  +norm {r['norm_bits_per_image']:.0f}  "
              f"→ {r['bits_per_image']:.1f} bits/img  BPFP={r['bpfp']:.5f}")
        results["rows"].append(row)

        if not args.skip_orfc_only:
            print(f"\n  [ORFC-only 257 tok] {ckpt.name}")
            row_o = eval_orfc_only(
                f"orfc_only_K{K}", test_features, test_names, orfc, tail,
                wrapper, layer_idx, gt, args.norm_mode, device, args.batch_size)
            print("  collecting ORFC-only labels...")
            train_lab_o = collect_plain_orfc_labels(
                train_features, orfc, args.norm_mode, device, args.batch_size)
            test_lab_o = collect_plain_orfc_labels(
                test_features, orfc, args.norm_mode, device, args.batch_size)
            bpt_o, _ = _rate_from_labels(test_lab_o, train_lab_o, G, K_cb)
            row_o["rate"] = pack_image_rate(
                bpt_o, T_full, grouping_bits=0.0)
            row_o["orfc_ckpt"] = ckpt.name
            row_o["orfc_K"] = int(K)
            r = row_o["rate"]
            print(f"  rate  pq={r['pq_bpt']:.2f} b/tok × {T_full} = "
                  f"{r['pq_bits_per_image']:.1f}  +norm {r['norm_bits_per_image']:.0f}  "
                  f"→ {r['bits_per_image']:.1f} bits/img  BPFP={r['bpfp']:.5f}")
            results["rows"].append(row_o)

        del orfc
        torch.cuda.empty_cache()

    out_dir = Path(HERE) / "results" / "global_residual" / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    ktag = "-".join(str(k) for k in args.K)
    out_path = out_dir / f"{args.layer}_haar_orfc_K{ktag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {out_path}")
    print(f"\n{'name':32s} {'Acc':>7s} {'ΔL':>12s} {'bits/img':>12s} {'BPFP':>9s}")
    for row in results["rows"]:
        rate = row.get("rate") or {}
        bpi = rate.get("bits_per_image")
        bpfp = rate.get("bpfp")
        bpi_s = f"{bpi:.1f}" if bpi is not None else "n/a"
        bpfp_s = f"{bpfp:.5f}" if bpfp is not None else "n/a"
        print(f"  {row['name']:30s} {row['acc']:7.4f} {row['delta_l']:12.1f} "
              f"{bpi_s:>12s} {bpfp_s:>9s}")
    return results


if __name__ == "__main__":
    main()
