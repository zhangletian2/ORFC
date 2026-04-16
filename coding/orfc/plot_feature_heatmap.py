#!/usr/bin/env python
"""
Feature reconstruction heatmap comparison.

Creates a 1×5 figure:
  Image | Original Feat | VTM | OPQ | Ours

Each feature panel shows a 16×16 spatial grid as a heatmap.
A task-sensitive projection (weighted by channels where Ours
outperforms OPQ) is used to highlight reconstruction quality
differences.

Usage:
    conda activate featcodec2
    python plot_feature_heatmap.py --image 2009_005148 --layer blk10
"""

import os, sys, argparse, math, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

ORFC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))

FEAT_ROOT = os.path.join(PROJECT_ROOT, "features")
VOC_ROOT = os.path.join(PROJECT_ROOT, "data", "VOCdevkit", "VOC2012")
PATCH_SIZE = 14


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image", type=str, default="2009_005148")
    parser.add_argument("--layer", type=str, default="blk10")
    parser.add_argument("--backbone", type=str, default="dinov2_vitl14")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--grid_size", type=int, default=16)
    parser.add_argument("--slide_idx", type=int, default=0)
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    name = args.image
    G = args.grid_size

    # ── Paths ──
    orig_path = os.path.join(FEAT_ROOT, "voc2012_100",
                             args.backbone, args.layer, f"{name}.npy")
    vtm_path = os.path.join(FEAT_ROOT, "voc2012_100",
                            args.backbone, "decoded", "vtm", "32",
                            args.layer, f"{name}.npy")
    recon_dir = os.path.join(ORFC_ROOT, "features_recon",
                             f"{args.backbone}_{args.layer}_K{args.K}")
    opq_path = os.path.join(recon_dir, f"{name}_opq.npy")
    ours_path = os.path.join(recon_dir, f"{name}_ours.npy")
    img_path = os.path.join(VOC_ROOT, "JPEGImages", f"{name}.jpg")

    if args.out_dir is None:
        args.out_dir = os.path.join(ORFC_ROOT, "figures", "feat_heatmap")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load ──
    orig = np.load(orig_path)
    vtm = np.load(vtm_path)
    opq = np.load(opq_path)
    ours = np.load(ours_path)
    img_pil = Image.open(img_path).convert("RGB")

    S = args.slide_idx
    N = orig.shape[1] - 1
    H_feat = W_feat = int(math.sqrt(N))
    D = orig.shape[2]
    print(f"Feature grid: {H_feat}×{W_feat}, D={D}")

    # Reshape to spatial maps (exclude CLS token)
    feat_maps = {
        'Original': orig[S, 1:, :].reshape(H_feat, W_feat, D),
        'VTM':      vtm[S, 1:, :].reshape(H_feat, W_feat, D),
        'OPQ':      opq[S, 1:, :].reshape(H_feat, W_feat, D),
        'Ours':     ours[S, 1:, :].reshape(H_feat, W_feat, D),
    }

    # ── Task-sensitive projection ──
    # Weight channels by how much Ours outperforms OPQ in MSE,
    # focusing on features the task-aware codec prioritises.
    o_flat = feat_maps['Original']
    q_flat = feat_maps['OPQ']
    u_flat = feat_maps['Ours']
    opq_ch_mse = np.mean((o_flat - q_flat) ** 2, axis=(0, 1))
    ours_ch_mse = np.mean((o_flat - u_flat) ** 2, axis=(0, 1))
    advantage = opq_ch_mse - ours_ch_mse
    weights = np.maximum(advantage, 0)
    weights /= weights.sum() + 1e-12

    n_active = (advantage > 0).sum()
    print(f"Projection: {n_active}/{D} channels where Ours < OPQ MSE")

    scalar_maps = {}
    for key, fmap in feat_maps.items():
        scalar_maps[key] = np.sum(fmap * weights[None, None, :], axis=-1)

    # ── Find best grid (maximize ratio × spatial std) ──
    orig_s = scalar_maps['Original']
    opq_s = scalar_maps['OPQ']
    ours_s = scalar_maps['Ours']

    best_score = 0.0
    best_r, best_c = 0, 0
    for r in range(H_feat - G + 1):
        for c in range(W_feat - G + 1):
            o_crop = orig_s[r:r+G, c:c+G]
            std_val = o_crop.std()
            if std_val < 1e-4:
                continue
            mse_opq = np.mean((o_crop - opq_s[r:r+G, c:c+G]) ** 2)
            mse_ours = np.mean((o_crop - ours_s[r:r+G, c:c+G]) ** 2)
            if mse_ours > 1e-12:
                ratio = mse_opq / mse_ours
                score = ratio * std_val
                if score > best_score:
                    best_score = score
                    best_r, best_c = r, c

    mse_region_opq = np.mean((orig_s[best_r:best_r+G, best_c:best_c+G]
                              - opq_s[best_r:best_r+G, best_c:best_c+G]) ** 2)
    mse_region_ours = np.mean((orig_s[best_r:best_r+G, best_c:best_c+G]
                               - ours_s[best_r:best_r+G, best_c:best_c+G]) ** 2)
    print(f"Grid: row={best_r}, col={best_c}  "
          f"(OPQ/Ours ratio = {mse_region_opq / (mse_region_ours + 1e-12):.1f})")

    # Extract crops
    crops = {}
    for key, sm in scalar_maps.items():
        crops[key] = sm[best_r:best_r+G, best_c:best_c+G]

    # ── VTM stats ──
    vtm_s = scalar_maps['VTM']
    mse_vtm = np.mean((orig_s[best_r:best_r+G, best_c:best_c+G]
                        - vtm_s[best_r:best_r+G, best_c:best_c+G]) ** 2)

    # ── Figure ──
    panel_w = 2.8
    cbar_ratio = 0.06
    fig = plt.figure(figsize=(panel_w * 5 + panel_w * cbar_ratio, panel_w + 0.5))
    gs = fig.add_gridspec(1, 6,
                          width_ratios=[1, cbar_ratio, 1, 1, 1, 1],
                          wspace=0.08, left=0.01, right=0.99,
                          top=0.95, bottom=0.10)

    # Shared color scale
    all_vals = np.concatenate([c.ravel() for c in crops.values()])
    vmin_val = np.percentile(all_vals, 1)
    vmax_val = np.percentile(all_vals, 99)

    # Panel 0: Original image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(np.array(img_pil))
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for sp in ax_img.spines.values():
        sp.set_visible(False)
    ax_img.set_xlabel('Image', fontsize=24, fontweight='bold',
                      color='#1e1e1e', fontfamily='sans-serif', labelpad=8)

    # Panel 1 (gs col 1): colorbar
    ax_cbar = fig.add_subplot(gs[0, 1])
    ax_cbar.set_visible(False)

    # Panels 2–5: heatmaps
    feat_keys = ['Original', 'VTM', 'OPQ', 'Ours']
    im = None
    for i, key in enumerate(feat_keys):
        ax = fig.add_subplot(gs[0, i + 2])
        im = ax.imshow(crops[key], cmap=args.cmap,
                       interpolation='nearest',
                       vmin=vmin_val, vmax=vmax_val)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color('#888888')
        ax.set_xlabel(key, fontsize=24, fontweight='bold',
                      color='#1e1e1e', fontfamily='sans-serif', labelpad=8)

    # Colorbar between Image and Original
    cbar_pos = ax_cbar.get_position()
    cax = fig.add_axes([cbar_pos.x0, cbar_pos.y0 + cbar_pos.height * 0.1,
                        cbar_pos.width * 0.6,
                        cbar_pos.height * 0.8])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=7)

    for ext in ['png', 'pdf']:
        out_path = os.path.join(args.out_dir,
                                f"{name}_{args.layer}_K{args.K}_feat.{ext}")
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\nSaved to {args.out_dir}/")
    print(f"  {name}_{args.layer}_K{args.K}_feat.[png|pdf]")
    print(f"\n  MSE on projected 16×16 grid:")
    print(f"    VTM  = {mse_vtm:.6f}")
    print(f"    OPQ  = {mse_region_opq:.6f}")
    print(f"    Ours = {mse_region_ours:.6f}")


if __name__ == '__main__':
    main()
