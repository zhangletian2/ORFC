#!/usr/bin/env python
"""Paired per-image tail-MSE comparison of non-uniform allocations.

All arms are scored on the *same* ``features_test`` with the *same* frozen
tail (identical data + teacher + pipeline; only the codec/allocation differs),
so image-level noise cancels in the paired delta.  This is the honest
apples-to-apples test of "is allocation A better than allocation B".

Usage:
    python paired_eval_nonuniform.py \
        --ckpt t00=results/.../blk20_R64_t00_...s42.pt \
        --ckpt t04=results/.../blk20_R64_t04_...s42.pt \
        --ckpt t10=results/.../blk20_R64_t10_...s42.pt \
        --ref t10
"""

import os, sys, argparse, json
import numpy as np
import torch
from pathlib import Path

ORFC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))

from run_multilayer_calibrator import preload_features
from opq import batch_normalize_gpu, batch_inv_normalize_gpu
from backbone.wrapper import Dinov2Wrapper
from soft_pq import FrozenTail
from run_soft_pq_nonuniform import load_nonuniform_codec

import warnings
warnings.filterwarnings("ignore")


@torch.no_grad()
def per_image_tail_mse(features, codec, tail, norm_mode, device, batch_size=32):
    """Per-image tail MSE ``[N]`` (sum over token*dim, one value per image)."""
    vals = []
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        B = X.shape[0]
        Y_teacher = tail.forward_nograd(X)
        Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
        Y_hat, _ = codec(Y)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        Y_student = tail.forward_nograd(X_hat)
        per = ((Y_teacher - Y_student) ** 2).reshape(B, -1).sum(-1)
        vals.append(per.cpu().numpy())
        del X, X_hat, Y_teacher, Y_student
        torch.cuda.empty_cache()
    return np.concatenate(vals)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", action="append", default=[], required=True,
                   help="repeatable NAME=PATH to a saved non-uniform codec .pt")
    p.add_argument("--ref", default=None,
                   help="reference arm name for paired delta (default: best mean)")
    p.add_argument("--layer", type=str, default="blk20")
    p.add_argument("--norm_mode", type=str, default="per_image")
    p.add_argument("--backbone", type=str, default="dinov2_vitl14")
    p.add_argument("--feat_root", type=str,
                   default=os.path.join(PROJECT_ROOT, "features"))
    p.add_argument("--test_subset", type=str, default="test")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_idx = int(args.layer[-2:])

    arms = []
    for spec in args.ckpt:
        name, _, path = spec.partition("=")
        if not name or not path:
            raise SystemExit(f"--ckpt {spec!r} must look like NAME=PATH")
        arms.append((name, path))

    test_dir = Path(args.feat_root) / args.test_subset / args.backbone / args.layer
    test_files = sorted(test_dir.glob("*.npy"))
    if not test_files:
        raise FileNotFoundError(f"no test features under {test_dir}")
    features_test, _ = preload_features(test_files, num_workers=4)
    print(f"# test images: {len(features_test)}  (dir={test_dir})")

    wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone, device=device)
    tail = FrozenTail(list(wrapper.backbone.blocks[layer_idx + 1:]),
                      wrapper.backbone.norm, device=device)

    per_image = {}
    summary = {}
    for name, path in arms:
        codec, meta = load_nonuniform_codec(path, device=device)
        vals = per_image_tail_mse(features_test, codec, tail,
                                  args.norm_mode, device, args.batch_size)
        per_image[name] = vals
        summary[name] = {
            "path": path,
            "t": meta.get("t"),
            "nominal_bpt": meta.get("nominal_bpt"),
            "K_counts": {
                "K8": meta["K_per_group"].count(8),
                "K4": meta["K_per_group"].count(4),
                "K2": meta["K_per_group"].count(2),
            },
            "tail_mse_mean": float(vals.mean()),
            "tail_mse_sem": float(vals.std(ddof=1) / np.sqrt(vals.size)),
            "n": int(vals.size),
        }
        c = summary[name]["K_counts"]
        print(f"  {name}: t={meta.get('t')} "
              f"K8/K4/K2={c['K8']}/{c['K4']}/{c['K2']}  "
              f"TailMSE={vals.mean():.1f} ± {summary[name]['tail_mse_sem']:.1f}")
        del codec
        torch.cuda.empty_cache()

    # pick reference: explicit, else lowest mean
    ref = args.ref or min(summary, key=lambda k: summary[k]["tail_mse_mean"])
    if ref not in per_image:
        raise SystemExit(f"--ref {ref} not among arms {list(per_image)}")
    ref_vals = per_image[ref]
    print(f"\n# paired vs reference '{ref}' "
          f"(mean {summary[ref]['tail_mse_mean']:.1f})")
    print(f"{'arm':>6} {'mean':>12} {'d(arm-ref)':>12} {'sem':>9} "
          f"{'t':>8} {'%':>7}")
    for name in per_image:
        d = per_image[name] - ref_vals
        delta = float(d.mean())
        sem = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0
        tstat = delta / sem if sem > 0 else float("nan")
        pct = 100.0 * delta / summary[ref]["tail_mse_mean"]
        summary[name]["paired_vs_ref"] = {
            "ref": ref, "delta": delta, "sem": sem,
            "t_stat": tstat, "percent": pct,
        }
        print(f"{name:>6} {summary[name]['tail_mse_mean']:>12.1f} "
              f"{delta:>+12.1f} {sem:>9.1f} {tstat:>8.2f} {pct:>+7.2f}")

    report = {"reference": ref, "arms": summary}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
