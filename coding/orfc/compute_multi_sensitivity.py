#!/usr/bin/env python
"""
Compute per-group L_ref gradient sensitivity for multiple DOPQ
configurations (varying K, fixed emb_dim=32 / G=32).

Saves results to JSON, then generates the plot.

Usage:
    python compute_multi_sensitivity.py --gpu 0
"""

import os, sys, json, argparse, time
import numpy as np
import torch
from pathlib import Path

ORFC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

from run_multilayer_calibrator import set_seed, preload_features, load_gt
from opq import batch_normalize_gpu, batch_inv_normalize_gpu
from backbone.wrapper import Dinov2Wrapper
from soft_pq import SoftPQ, FrozenTail, FeatureCodec, load_codec

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
import logging
from mmcv.utils import get_logger
get_logger('mmcv').setLevel(logging.WARNING)


CONFIGS = [
    ("K=4",   "blk20_K4_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"),
    ("K=8",   "blk20_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"),
    ("K=16",  "blk20_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"),
    ("K=32",  "blk20_K32_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"),
    ("K=256", "blk20_K256_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"),
]

LAYER = "blk20"
LAYER_IDX = 20
NORM_MODE = "per_image"
N_DIAG = 200
SEED = 42
FEAT_ROOT = os.path.join(PROJECT_ROOT, "features")
BACKBONE = "dinov2_vitl14"


def gradient_sensitivity(codec, tail, features, device, batch_size=8):
    """Compute ||dL_ref / dC_g|| per group."""
    pq = codec.pq
    G = pq.G
    grad_sum = torch.zeros(G, device=device)
    n = 0

    codec.train()
    old_tau = pq.temperature
    pq.temperature = 0.1

    for s in range(0, len(features), batch_size):
        e = min(s + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[s:e])).float().to(device)
        B, T, C = X.shape
        with torch.no_grad():
            Y, Mu, Std = batch_normalize_gpu(X, mode=NORM_MODE)
        Y_hat, _ = codec(Y)

        with torch.no_grad():
            Y_teacher = tail.forward_nograd(X)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        F_hat = tail(X_hat)
        loss = ((Y_teacher - F_hat) ** 2).sum() / B

        grad_C = torch.autograd.grad(loss, pq.codebooks, retain_graph=False)[0]
        grad_sum += grad_C.reshape(G, -1).norm(dim=1).detach()
        n += 1
        del X, Y, Mu, Std, Y_hat, loss, grad_C, X_hat, F_hat, Y_teacher
        torch.cuda.empty_cache()

    codec.eval()
    pq.temperature = old_tau
    return (grad_sum / max(n, 1)).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    set_seed(SEED)

    # ── Load features ──
    feat_dir = Path(FEAT_ROOT) / "train" / BACKBONE / LAYER
    feat_files = sorted(feat_dir.glob("*.npy"))
    print(f"Loading train features: {len(feat_files)} files")
    features_all, _ = preload_features(feat_files, num_workers=8)

    rng = np.random.RandomState(SEED)
    diag_idx = rng.choice(len(features_all), N_DIAG, replace=False)
    features_diag = [features_all[i] for i in diag_idx]
    del features_all
    print(f"Using {N_DIAG} diagnostic images")

    # ── Load DINOv2 & build Frozen-Tail ──
    print("Loading DINOv2 ViT-L/14 ...")
    wrapper = Dinov2Wrapper(head_layers=1, device=device)
    wrapper.backbone.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()
    tail_blocks = list(wrapper.backbone.blocks[LAYER_IDX + 1:])
    tail = FrozenTail(tail_blocks, wrapper.backbone.norm, device=device)
    print("Frozen-Tail ready")

    # ── Compute sensitivity for each config ──
    ckpt_dir = os.path.join(ORFC_ROOT, "checkpoints", BACKBONE)
    results = {}

    for label, ckpt_name in CONFIGS:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {label}: checkpoint not found")
            continue
        print(f"\n{'='*50}")
        print(f"  Computing sensitivity for {label} ...")
        t0 = time.time()
        codec = load_codec(ckpt_path, device=device)
        sens = gradient_sensitivity(codec, tail, features_diag, device)
        elapsed = time.time() - t0
        cv = float(np.std(sens) / np.mean(sens))
        print(f"  {label}: CV={cv:.4f}, mean={np.mean(sens):.2f}, "
              f"max/min={np.max(sens):.2f}/{np.min(sens):.2f}, "
              f"time={elapsed:.1f}s")
        results[label] = {
            "values": sens.tolist(),
            "cv": cv,
            "mean": float(np.mean(sens)),
            "ckpt": ckpt_name,
        }
        del codec
        torch.cuda.empty_cache()

    # ── Save JSON ──
    out_dir = os.path.join(ORFC_ROOT, "results", "analysis_intro_v2")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "multi_sensitivity_blk20_emb32.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Plot ──
    plot_from_json(out_path)


def plot_from_json(json_path):
    """Generate the figure from saved JSON."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "font.family": "sans-serif",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    })

    with open(json_path) as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(4.5, 2.4))

    markers = ["o", "s", "^", "D", "v", "P"]
    for i, (label, data) in enumerate(results.items()):
        vals = np.array(data["values"])
        xs = np.arange(1, len(vals) + 1)
        ax.plot(xs, vals, f"-{markers[i % len(markers)]}", ms=3, lw=1.2,
                alpha=0.85, label=label)

    ax.set_xlabel("Group index")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(0.5, len(vals) + 0.5)
    ax.set_xticks(np.arange(1, len(vals) + 1, 4))
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(7)

    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.15),
              ncol=len(results), framealpha=0.9, borderpad=0.3,
              handlelength=1.5, columnspacing=1.0)

    fig.tight_layout()

    fig_dir = os.path.join(ORFC_ROOT, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(fig_dir, f"fig_multi_sensitivity.{ext}"),
                    dpi=300, bbox_inches="tight")
    print(f"Saved: {fig_dir}/fig_multi_sensitivity.[pdf|png]")


if __name__ == "__main__":
    main()
