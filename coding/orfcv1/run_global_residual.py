#!/usr/bin/env python
"""Validate global token grouping with an unquantized residual side stream."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


HERE = os.path.dirname(os.path.abspath(__file__))
_ORFC_DIR = os.path.normpath(os.path.join(HERE, "..", "orfc"))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if _ORFC_DIR not in sys.path:
    sys.path.insert(0, _ORFC_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from backbone.wrapper import Dinov2Wrapper  # noqa: E402
from eval_greedy_4x import greedy_match_pairs  # noqa: E402
from opq import batch_inv_normalize_gpu, batch_normalize_gpu  # noqa: E402
from run_multilayer_calibrator import (  # noqa: E402
    evaluate_accuracy,
    load_gt,
    preload_features,
    set_seed,
)
from soft_pq import FrozenTail  # noqa: E402


@torch.no_grad()
def global_groups(Y, n_prefix=1):
    """Two global matchings produce disjoint groups ``[B, T/4, 4]``."""
    patch = Y[:, n_prefix:]
    B, T, D = patch.shape
    left1, right1 = greedy_match_pairs(patch, T // 2)
    xi = torch.gather(patch, 1, left1.unsqueeze(-1).expand(-1, -1, D))
    xj = torch.gather(patch, 1, right1.unsqueeze(-1).expand(-1, -1, D))
    pair_mean = 0.5 * (xi + xj)
    left2, right2 = greedy_match_pairs(pair_mean, T // 4)

    def select(index, pair_index):
        return torch.gather(index, 1, pair_index)

    return torch.stack([
        select(left1, left2), select(right1, left2),
        select(left1, right2), select(right1, right2),
    ], dim=2)


def grouped_tokens(patch, groups):
    B, M, Q = groups.shape
    D = patch.shape[-1]
    index = groups.reshape(B, M * Q).unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(patch, 1, index).view(B, M, Q, D)


def haar_analysis(patch, groups):
    a, b, c, d = grouped_tokens(patch, groups).unbind(dim=2)
    low = 0.5 * (a + b + c + d)
    root2 = 2.0 ** 0.5
    detail = torch.stack([
        (a - b) / root2,
        (c - d) / root2,
        0.5 * (a + b - c - d),
    ], dim=2)
    return low, detail


def haar_synthesis(low, detail, groups, n_patch):
    d1, d2, d3 = detail.unbind(dim=2)
    root2 = 2.0 ** 0.5
    pair1 = 0.5 * (low + d3)
    pair2 = 0.5 * (low - d3)
    restored = torch.stack([
        pair1 + d1 / root2,
        pair1 - d1 / root2,
        pair2 + d2 / root2,
        pair2 - d2 / root2,
    ], dim=2)
    B, M, Q, D = restored.shape
    patch = restored.new_zeros(B, n_patch, D)
    index = groups.reshape(B, M * Q).unsqueeze(-1).expand(-1, -1, D)
    patch.scatter_(1, index, restored.reshape(B, M * Q, D))
    return patch


class GlobalDetailCodec(nn.Module):
    """Encode either three details or low+details into one D token."""

    def __init__(self, D, joint=False, n_prefix=1):
        super().__init__()
        self.D = int(D)
        self.joint = bool(joint)
        self.n_prefix = int(n_prefix)
        n_parts = 4 if self.joint else 3
        self.analysis = nn.Linear(n_parts * D, D, bias=False)
        self.synthesis = nn.Linear(D, n_parts * D, bias=False)
        with torch.no_grad():
            self.analysis.weight.zero_()
            self.synthesis.weight.zero_()
            eye = torch.eye(D)
            block = 0 if self.joint else 2
            self.analysis.weight[:, block * D:(block + 1) * D] = eye
            self.synthesis.weight[block * D:(block + 1) * D, :] = eye

    def coded_tokens(self, T_full=None):
        if T_full is None:
            return None
        n_patch = int(T_full) - self.n_prefix
        extra = 0 if self.joint else n_patch // 4
        return self.n_prefix + n_patch // 4 + extra

    def encode(self, Y, groups):
        """``Y [B, 1+T, D]`` → compressed ``seq [B, 1+T/4, D]`` plus aux.

        Joint: 64 tokens carry mixed low+detail.  Non-joint: 64 residual
        tokens; unquantized low is kept in ``aux`` (not in the bitstream).
        """
        p = self.n_prefix
        prefix, patch = Y[:, :p], Y[:, p:]
        low, detail = haar_analysis(patch, groups)
        parts = torch.cat([low.unsqueeze(2), detail], dim=2) if self.joint else detail
        token = self.analysis(parts.flatten(2))
        seq = torch.cat([prefix, token], dim=1)
        aux = {"groups": groups, "n_patch": patch.shape[1]}
        if not self.joint:
            aux["low"] = low
        return seq, aux

    def decode(self, seq, aux):
        """Inverse of ``encode``.  ``seq`` may be PQ-quantized."""
        p = self.n_prefix
        prefix, token = seq[:, :p], seq[:, p:]
        groups = aux["groups"]
        n_patch = aux["n_patch"]
        n_parts = 4 if self.joint else 3
        decoded = self.synthesis(token).view(
            token.shape[0], token.shape[1], n_parts, self.D)
        if self.joint:
            low_hat, detail_hat = decoded[:, :, 0], decoded[:, :, 1:]
        else:
            low_hat, detail_hat = aux["low"], decoded
        patch_hat = haar_synthesis(low_hat, detail_hat, groups, n_patch)
        return torch.cat([prefix, patch_hat], dim=1)

    def forward(self, Y, groups):
        seq, aux = self.encode(Y, groups)
        return self.decode(seq, aux), seq[:, self.n_prefix:]


def save_global_detail(codec, path, extra=None):
    meta = {
        "format": "global_detail_v1",
        "D": codec.D,
        "joint": bool(codec.joint),
        "n_prefix": int(codec.n_prefix),
        "state_dict": codec.state_dict(),
    }
    if extra:
        meta.update(extra)
    torch.save(meta, path)


def load_global_detail(path, device="cuda"):
    meta = torch.load(path, map_location="cpu")
    D = int(meta["D"])
    sd = meta["state_dict"]
    w = sd["analysis.weight"]
    joint = bool(meta["joint"]) if "joint" in meta else (w.shape[1] == 4 * D)
    n_prefix = int(meta.get("n_prefix", 1))
    codec = GlobalDetailCodec(D, joint=joint, n_prefix=n_prefix)
    codec.load_state_dict(sd)
    return codec.to(device).eval(), meta


class HaarORFCWrapper(nn.Module):
    """Drop-in ``forward(Y) -> Y_hat``: Haar encode → ORFC → Haar decode.

    Groups are computed from ``Y`` unless passed in.  The decoder needs
    them, so they are side information (counted separately at eval).
    """

    def __init__(self, haar, orfc=None, freeze_haar=True, freeze_orfc=True):
        super().__init__()
        self.haar = haar
        self.orfc = orfc
        if freeze_haar:
            haar.eval()
            for p in haar.parameters():
                p.requires_grad_(False)
        if orfc is not None and freeze_orfc:
            orfc.eval()
            for p in orfc.parameters():
                p.requires_grad_(False)

    def coded_tokens(self, T_full=None):
        return self.haar.coded_tokens(T_full)

    @property
    def pq(self):
        return None if self.orfc is None else self.orfc.pq

    @property
    def use_rate(self):
        return False if self.orfc is None else bool(self.orfc.use_rate)

    @property
    def _last_rate(self):
        return None if self.orfc is None else self.orfc._last_rate

    def forward(self, Y, groups=None):
        if groups is None:
            groups = global_groups(Y, n_prefix=self.haar.n_prefix)
        seq, aux = self.haar.encode(Y, groups)
        usage = None
        if self.orfc is not None:
            seq, usage = self.orfc(seq)
        Y_hat = self.haar.decode(seq, aux)
        aux = dict(aux)
        aux["usage"] = usage
        return Y_hat, aux



@torch.no_grad()
def build_groups(features, norm_mode, device, batch_size):
    rows = []
    for start in range(0, len(features), batch_size):
        X = torch.from_numpy(np.stack(features[start:start + batch_size])).float().to(device)
        Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=1)
        rows.append(global_groups(Y).cpu())
    return torch.cat(rows).numpy().astype(np.int16)


@torch.no_grad()
def evaluate(name, features, basenames, groups, mode, codec, tail, wrapper,
             layer_idx, gt, norm_mode, device, batch_size):
    all_xhat = []
    delta_l = 0.0
    mse = 0.0
    n_elem = 0
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        group = torch.from_numpy(groups[start:end].astype(np.int64)).to(device)
        Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=1)
        if mode == "learned":
            Y_hat, _ = codec(Y, group)
        else:
            low, detail = haar_analysis(Y[:, 1:], group)
            if mode == "low":
                detail = torch.zeros_like(detail)
            patch_hat = haar_synthesis(low, detail, group, Y.shape[1] - 1)
            Y_hat = torch.cat([Y[:, :1], patch_hat], dim=1)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        teacher = tail.forward_nograd(X)
        delta_l += ((teacher - tail.forward_nograd(X_hat)) ** 2).sum().item()
        diff = Y[:, 1:] - Y_hat[:, 1:]
        mse += (diff ** 2).sum().item()
        n_elem += diff.numel()
        all_xhat.extend(X_hat.cpu().numpy())
    acc = evaluate_accuracy(all_xhat, basenames, gt, wrapper, layer_idx, device)
    row = {
        "name": name,
        "acc": float(acc),
        "delta_l": float(delta_l / len(features)),
        "mse_patch": float(mse / n_elem),
    }
    print(f"  {name:18s} Acc={acc:.4f}  delta_L={row['delta_l']:.1f}  "
          f"MSE_p={row['mse_patch']:.6f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="blk05")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_train_images", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--norm_mode", default="per_image")
    ap.add_argument("--joint", action="store_true")
    ap.add_argument("--backbone", default="dinov2_vitl14")
    ap.add_argument("--feat_root", default=os.path.join(PROJECT_ROOT, "features"))
    ap.add_argument("--gt_path", default=os.path.join(
        PROJECT_ROOT, "utils", "imagenet_selected_label500.txt"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    layer_idx = int(args.layer[-2:])
    train_files = sorted((Path(args.feat_root) / "train" / args.backbone / args.layer).glob("*.npy"))
    test_files = sorted((Path(args.feat_root) / "test" / args.backbone / args.layer).glob("*.npy"))
    train_features, _ = preload_features(train_files, num_workers=4)
    test_features, test_names = preload_features(test_files, num_workers=4)
    if args.max_train_images < len(train_features):
        rng = np.random.RandomState(args.seed)
        chosen = rng.choice(len(train_features), args.max_train_images, replace=False)
        train_features = [train_features[i] for i in chosen]

    print(f"\nGlobal relation: train={len(train_features)} test={len(test_features)}")
    train_groups = build_groups(train_features, args.norm_mode, device, args.batch_size)
    test_groups = build_groups(test_features, args.norm_mode, device, args.batch_size)
    D = train_features[0].shape[-1]

    X0 = torch.from_numpy(np.stack(test_features[:2])).float().to(device)
    Y0, _, _ = batch_normalize_gpu(X0, mode=args.norm_mode, n_prefix=1)
    G0 = torch.from_numpy(test_groups[:2].astype(np.int64)).to(device)
    low0, detail0 = haar_analysis(Y0[:, 1:], G0)
    exact0 = haar_synthesis(low0, detail0, G0, Y0.shape[1] - 1)
    exact_error = (exact0 - Y0[:, 1:]).abs().max().item()
    print(f"Haar full-detail max error: {exact_error:.3e}")

    print(f"Loading {args.backbone}...")
    wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone, device=device)
    tail = FrozenTail(list(wrapper.backbone.blocks[layer_idx + 1:]),
                      wrapper.backbone.norm, device=device)
    gt = load_gt(args.gt_path)
    codec = GlobalDetailCodec(D, joint=args.joint).to(device)
    kind = "joint" if args.joint else "residual"
    n_coded = 65 if args.joint else 129
    print(f"layer={args.layer}  kind={kind}  coded={n_coded}  "
          f"ep={args.epochs} lr={args.lr}  joint={args.joint}")

    results = {"config": vars(args), "main_patch_tokens": 64,
               "residual_tokens": 0 if args.joint else 64,
               "total_tokens": 65 if args.joint else 129,
               "full_detail_max_abs": exact_error}
    results["low_only"] = evaluate(
        "global-low", test_features, test_names, test_groups, "low", codec,
        tail, wrapper, layer_idx, gt, args.norm_mode, device, args.batch_size)
    results[f"{kind}_init"] = evaluate(
        f"{kind}-init", test_features, test_names, test_groups, "learned", codec,
        tail, wrapper, layer_idx, gt, args.norm_mode, device, args.batch_size)

    features_array = np.stack(train_features)
    teacher_cache = np.empty_like(features_array)
    with torch.no_grad():
        for start in range(0, len(features_array), args.batch_size):
            end = min(start + args.batch_size, len(features_array))
            X = torch.from_numpy(features_array[start:end]).float().to(device)
            teacher_cache[start:end] = tail.forward_nograd(X).cpu().numpy()

    optimizer = torch.optim.Adam(codec.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    history = []
    for epoch in range(args.epochs):
        started = time.time()
        order = np.random.permutation(len(features_array))
        total = 0.0
        codec.train()
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            X = torch.from_numpy(features_array[index]).float().to(device)
            group = torch.from_numpy(train_groups[index].astype(np.int64)).to(device)
            Y, Mu, Std = batch_normalize_gpu(X, mode=args.norm_mode, n_prefix=1)
            Y_hat, _ = codec(Y, group)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            teacher = torch.from_numpy(teacher_cache[index]).float().to(device)
            loss = ((teacher - tail(X_hat)) ** 2).sum() / X.shape[0]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * X.shape[0]
        scheduler.step()
        avg = total / len(features_array)
        history.append({"epoch": epoch, "loss": avg})
        print(f"  ep {epoch:3d}/{args.epochs} loss={avg:.1f} "
              f"({time.time() - started:.1f}s)")

    codec.eval()
    results["history"] = history
    results[f"{kind}_trained"] = evaluate(
        f"{kind}-trained", test_features, test_names, test_groups, "learned", codec,
        tail, wrapper, layer_idx, gt, args.norm_mode, device, args.batch_size)

    out_dir = Path(HERE) / "results" / "global_residual" / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.layer}_global_haar_{kind}_D_lr{args.lr}_ep{args.epochs}_s{args.seed}"
    save_global_detail(
        codec, out_dir / f"{stem}.pt",
        extra={"layer": args.layer, "joint": bool(args.joint)})
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_dir / (stem + '.json')}")


if __name__ == "__main__":
    main()
