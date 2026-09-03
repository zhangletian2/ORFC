#!/usr/bin/env python
"""Untrained greedy cosine merge at the same coded length as 2×2 reassembly.

Spatial reassembly: 16×16 → 8×8 patch tokens (256 → 64, 4×).  Disjoint
pairwise greedy can only halve per round, so 4× is two perfect-matching
rounds: 256 → 128 → 64, copy-mean unmerge.  Also reports one-round
max pairwise (256 → 128) for context.

Usage:
    CUDA_VISIBLE_DEVICES=3 python -u eval_greedy_4x.py --layers blk05 blk20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
_ORFC_DIR = os.path.normpath(os.path.join(HERE, "..", "orfc"))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if _ORFC_DIR not in sys.path:
    sys.path.insert(0, _ORFC_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from run_multilayer_calibrator import (  # noqa: E402
    set_seed, preload_features, load_gt,
)
from backbone.wrapper import Dinov2Wrapper  # noqa: E402
from run_spatial_reassembly import _eval  # noqa: E402

import warnings  # noqa: E402
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

_NEG = -1.0e4


@torch.no_grad()
def greedy_match_pairs(tokens, r):
    """Global cosine greedy matching.  ``tokens [B, T, D]`` → ``left, right [B, r]``."""
    B, T, _D = tokens.shape
    if r > T // 2:
        raise ValueError(f"r={r} > T/2={T // 2}")
    x = F.normalize(tokens, dim=-1)
    sim = x @ x.transpose(-1, -2)
    triu = torch.triu(torch.ones(T, T, device=tokens.device, dtype=torch.bool),
                      diagonal=1)
    work = sim.masked_fill(~triu, _NEG)
    used = torch.zeros(B, T, dtype=torch.bool, device=tokens.device)
    left = torch.empty(B, r, dtype=torch.long, device=tokens.device)
    right = torch.empty(B, r, dtype=torch.long, device=tokens.device)
    arange = torch.arange(B, device=tokens.device)
    for k in range(r):
        work.masked_fill_(used.unsqueeze(2) | used.unsqueeze(1), _NEG)
        idx = work.reshape(B, -1).argmax(dim=-1)
        i = torch.div(idx, T, rounding_mode="floor")
        j = idx - i * T
        left[:, k] = i
        right[:, k] = j
        used[arange, i] = True
        used[arange, j] = True
    return left, right


def _pair_mean(patch, left, right):
    D = patch.shape[-1]
    xi = torch.gather(patch, 1, left.unsqueeze(-1).expand(-1, -1, D))
    xj = torch.gather(patch, 1, right.unsqueeze(-1).expand(-1, -1, D))
    return 0.5 * (xi + xj)


def _scatter_copy(ym, left, right, n):
    B, r, D = ym.shape
    out = ym.new_zeros(B, n, D)
    out.scatter_(1, left.unsqueeze(-1).expand(-1, -1, D), ym)
    out.scatter_(1, right.unsqueeze(-1).expand(-1, -1, D), ym)
    return out


class GreedyCopyMeanCodec(nn.Module):
    """``n_rounds`` successive perfect matchings (each halves patch count)."""

    def __init__(self, D, n_prefix=1, n_rounds=2):
        super().__init__()
        self.D = int(D)
        self.n_prefix = int(n_prefix)
        self.n_rounds = int(n_rounds)

    def coded_tokens(self, T_full=None):
        if T_full is None:
            return None
        n_patch = int(T_full) - self.n_prefix
        return self.n_prefix + n_patch // (2 ** self.n_rounds)

    def forward(self, Y, **_kwargs):
        p = self.n_prefix
        patch = Y[:, p:, :]
        levels = []
        cur = patch
        for _ in range(self.n_rounds):
            T = cur.shape[1]
            r = T // 2
            left, right = greedy_match_pairs(cur, r)
            ym = _pair_mean(cur, left, right)
            levels.append((left, right, T))
            cur = ym
        hat = cur
        for left, right, T in reversed(levels):
            hat = _scatter_copy(hat, left, right, T)
        if p == 0:
            return hat, None
        return torch.cat([Y[:, :p, :], hat], dim=1), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", default=["blk05", "blk20"])
    ap.add_argument("--n_prefix", type=int, default=1)
    ap.add_argument("--norm_mode", type=str, default="per_image")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feat_root", type=str,
                    default=os.path.join(PROJECT_ROOT, "features"))
    ap.add_argument("--test_subset", type=str, default="test")
    ap.add_argument("--backbone", type=str, default="dinov2_vitl14")
    ap.add_argument("--gt_path", type=str,
                    default=os.path.join(PROJECT_ROOT, "utils",
                                         "imagenet_selected_label500.txt"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    gt = load_gt(args.gt_path)

    print(f"\n{'#' * 70}")
    print("# Greedy copy-mean vs 2×2 spatial reassembly (same coded length)")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    print(f"\nLoading DINOv2 ({args.backbone})...")
    wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone,
                            device=device)

    all_out = {"config": vars(args), "rows": []}
    for layer in args.layers:
        layer_idx = int(layer[-2:])
        test_dir = Path(args.feat_root) / args.test_subset / args.backbone / layer
        files = sorted(test_dir.glob("*.npy"))
        features, basenames = preload_features(files, num_workers=4)
        T_full, D = features[0].shape
        n_patch = T_full - args.n_prefix
        print(f"\n{'=' * 60}\n  {layer}  test={len(features)}  "
              f"T={T_full} D={D}\n{'=' * 60}")

        for n_rounds, tag in ((1, "greedy_2x_r128"), (2, "greedy_4x_r128+64")):
            codec = GreedyCopyMeanCodec(
                D, n_prefix=args.n_prefix, n_rounds=n_rounds).to(device)
            Tm = codec.coded_tokens(T_full)
            print(f"\n  [{tag}]  coded={Tm}/{T_full}  "
                  f"patch {n_patch} -> {n_patch // (2 ** n_rounds)}")
            row = _eval(
                f"{layer}/{tag}", codec, features, basenames, gt, wrapper,
                layer_idx, device, args.norm_mode, args.n_prefix,
                args.batch_size, T_full)
            row.update({"layer": layer, "n_rounds": n_rounds,
                        "keep_frac": 1.0 / (2 ** n_rounds)})
            all_out["rows"].append(row)
            del codec
            torch.cuda.empty_cache()

    out_dir = Path(HERE) / "results" / "spatial_reassembly" / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "greedy_vs_reassembly_4x.json"
    with open(out_path, "w") as f:
        json.dump(all_out, f, indent=2, default=str)
    print(f"\nSaved {out_path}")
    for row in all_out["rows"]:
        print(f"  {row['name']:28s}  Acc={row['acc']:.4f}  "
              f"ΔL={row['delta_l']:.1f}  coded={row['coded_tokens']}")


if __name__ == "__main__":
    main()
