#!/usr/bin/env python
"""Ablate integer vs generic sampling on a trained s=2 reassembly codec.

Same weights; only the forward (pool / window / kernel layout) changes.
``integer`` should reproduce the trained json.  ``generic`` is the
arbitrary-ratio path evaluated at H'=H/2.

Usage:
    CUDA_VISIBLE_DEVICES=3 python -u eval_generic_s2.py --layers blk05 blk20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

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
from spatial_reassembly import (  # noqa: E402
    load_spatial_reassembly, FORWARD_MODES,
)

import warnings  # noqa: E402
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

CKPT_DIR = Path(HERE) / "results" / "spatial_reassembly" / "dinov2_vitl14"
MODES = ("integer", "down_pool", "down_win", "up_kern", "up_win", "generic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", default=["blk05", "blk20"])
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--n_prefix", type=int, default=1)
    ap.add_argument("--norm_mode", type=str, default="per_image")
    ap.add_argument("--batch_size", type=int, default=32)
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
    print("# Integer vs generic forward on trained s=2 weights")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    print(f"\nLoading DINOv2 ({args.backbone})...")
    wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone,
                            device=device)

    out = {"config": vars(args), "modes": {m: FORWARD_MODES[m] for m in args.modes
                                           if m in FORWARD_MODES},
           "rows": []}

    for layer in args.layers:
        ckpt = CKPT_DIR / f"{layer}_s2_k5_lr0.0003_ep50_s42.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        layer_idx = int(layer[-2:])
        test_dir = (Path(args.feat_root) / args.test_subset / args.backbone / layer)
        files = sorted(test_dir.glob("*.npy"))
        features, basenames = preload_features(files, num_workers=4)
        T_full = features[0].shape[0]
        print(f"\n{'=' * 60}\n  {layer}  {ckpt.name}  test={len(features)}\n"
              f"{'=' * 60}")

        codec, meta = load_spatial_reassembly(str(ckpt), device=device)
        print(f"  loaded D={meta['D']} scale={meta['scale']} k={meta['k']}")

        for mode in args.modes:
            codec.set_forward_mode(mode)
            print(f"\n  [{mode}]  {FORWARD_MODES[mode]}")
            row = _eval(
                f"{layer}/{mode}", codec, features, basenames, gt, wrapper,
                layer_idx, device, args.norm_mode, args.n_prefix,
                args.batch_size, T_full)
            row.update({"layer": layer, "mode": mode, **FORWARD_MODES[mode]})
            out["rows"].append(row)
            torch.cuda.empty_cache()

        del codec
        torch.cuda.empty_cache()

    out_path = CKPT_DIR / "generic_vs_integer_s2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {out_path}")
    print(f"\n{'mode':12s} {'layer':6s} {'Acc':8s} {'ΔL_ref':12s}  vs integer")
    by = {}
    for row in out["rows"]:
        by[(row["layer"], row["mode"])] = row
    for layer in args.layers:
        base = by.get((layer, "integer"))
        for mode in args.modes:
            row = by.get((layer, mode))
            if row is None:
                continue
            dacc = row["acc"] - base["acc"] if base else 0.0
            ddl = row["delta_l"] - base["delta_l"] if base else 0.0
            print(f"  {mode:12s} {layer:6s} {row['acc']:.4f}  "
                  f"{row['delta_l']:.1f}  dAcc={dacc:+.4f} dL={ddl:+.1f}")


if __name__ == "__main__":
    main()
