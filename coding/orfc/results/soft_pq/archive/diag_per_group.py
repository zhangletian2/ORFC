#!/usr/bin/env python
"""
Diagnostic: per-group MSE and ΔL_ref contribution after OPQ init.

Measures:
  1. Per-group MSE_g = E[||z_g - c_g[k_g]||²]
  2. Per-group ΔL_ref sensitivity s_g = E[||∂D/∂ẑ_g||²]
  3. Per-group ΔL_ref contribution ≈ s_g * MSE_g  (first-order approx)
  4. Leave-one-group-out ΔL_ref  (exact per-group contribution)

Prints ratio (max/min) and CV for each metric.
"""

import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F_fn

ORFC_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

from run_multilayer_calibrator import preload_features
from opq import (batch_normalize_gpu, batch_inv_normalize_gpu,
                 learn_opq_rotation)
from backbone.wrapper import Dinov2Wrapper
from soft_pq import FrozenTail

import warnings
warnings.filterwarnings("ignore")
import logging
from mmcv.utils import get_logger
get_logger('mmseg', log_level=logging.WARNING)


def main():
    device = 'cuda'
    layer = 'blk20'
    layer_idx = 20
    K = 16
    emb_dim = 32
    norm_mode = 'per_image'
    max_images = 500
    seed = 42

    from pathlib import Path
    feat_root = Path(os.path.join(PROJECT_ROOT, "features"))
    train_dir = feat_root / "train" / "dinov2_vitl14" / layer
    train_files = sorted(train_dir.glob("*.npy"))

    print(f"Loading features from {train_dir}...")
    features_train, _ = preload_features(train_files, num_workers=16)

    rng = np.random.RandomState(seed)
    if len(features_train) > max_images:
        idx = rng.choice(len(features_train), max_images, replace=False)
        features_train = [features_train[i] for i in idx]

    D = features_train[0].shape[1]
    T = features_train[0].shape[0]
    G = D // emb_dim
    d = emb_dim
    print(f"D={D}, T={T}, G={G}, d={d}, K={K}")
    print(f"N_images={len(features_train)}")

    # --- OPQ ---
    print("\nRunning OPQ...")
    all_vectors = []
    for start in range(0, len(features_train), 200):
        end = min(start + 200, len(features_train))
        X = torch.from_numpy(
            np.stack(features_train[start:end])).float().to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
        all_vectors.append(Y.reshape(-1, D).cpu().numpy())
        del X, Y
    full_vectors = np.concatenate(all_vectors)
    del all_vectors

    R, codebooks, hist = learn_opq_rotation(
        full_vectors, G, d, K,
        max_iter_opq=20, max_iter_kmeans=100,
        device=device, verbose=False)
    print(f"OPQ done: final MSE={hist[-1][0]:.8f}")

    # --- Per-group MSE ---
    print("\n=== Per-group MSE (after OPQ) ===")
    R_t = torch.from_numpy(R).float().to(device)
    cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)  # [G, K, d]

    mse_per_group = np.zeros(G, dtype=np.float64)
    n_tokens = 0

    for start in range(0, len(features_train), 100):
        end = min(start + 100, len(features_train))
        X = torch.from_numpy(
            np.stack(features_train[start:end])).float().to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
            flat = Y.reshape(-1, D) @ R_t                        # [NT, D]
            sub = flat.reshape(-1, G, d).permute(1, 0, 2)        # [G, NT, d]
            dists = torch.cdist(sub, cb_t).pow(2)                # [G, NT, K]
            min_dists = dists.min(dim=-1).values                  # [G, NT]
            mse_per_group += min_dists.sum(dim=1).cpu().numpy()
            n_tokens += flat.shape[0]
        del X, Y, flat, sub, dists, min_dists

    mse_per_group /= n_tokens
    mse_ratio = mse_per_group.max() / mse_per_group.min()
    mse_cv = mse_per_group.std() / mse_per_group.mean()
    print(f"  MSE per group: mean={mse_per_group.mean():.6f}")
    print(f"  range: [{mse_per_group.min():.6f}, {mse_per_group.max():.6f}]")
    print(f"  ratio (max/min): {mse_ratio:.3f}x")
    print(f"  CV: {mse_cv:.4f}")

    # --- Load DINOv2 for ΔL_ref ---
    print("\nLoading DINOv2 for ΔL_ref measurement...")
    dino = Dinov2Wrapper(head_layers=1, device=device)
    n_blocks = len(dino.backbone.blocks)
    tail_blocks = list(dino.backbone.blocks[layer_idx + 1:])
    norm_layer = dino.backbone.norm

    dino.backbone.cpu()
    if dino.head is not None:
        dino.head.cpu()
    torch.cuda.empty_cache()

    tail = FrozenTail(tail_blocks, norm_layer, device=device)

    # --- Per-group sensitivity s_g = E[||∂D/∂ẑ_g||²] ---
    print("\n=== Per-group ΔL_ref sensitivity (gradient energy) ===")
    sg = torch.zeros(G, device=device, dtype=torch.float64)
    sg_count = 0

    for start in range(0, len(features_train), 4):
        end = min(start + 4, len(features_train))
        X = torch.from_numpy(
            np.stack(features_train[start:end])).float().to(device)
        B = X.shape[0]
        with torch.no_grad():
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            Y_teacher = tail.forward_nograd(X)
            flat = Y.reshape(-1, D) @ R_t
            sub = flat.reshape(-1, G, d).permute(1, 0, 2)
            dists = torch.cdist(sub, cb_t).pow(2)
            labels = dists.argmin(dim=-1)                         # [G, NT]

        flat_q = flat.detach().clone()
        flat_q.requires_grad_(True)
        sub_q = flat_q.reshape(-1, G, d).permute(1, 0, 2)
        selected = torch.gather(
            cb_t.unsqueeze(1).expand(-1, sub_q.shape[1], -1, -1), 2,
            labels.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, d),
        ).squeeze(2)
        z_hat = selected.permute(1, 0, 2).reshape(-1, D)
        z_hat_perm = z_hat.detach().clone()
        z_hat_perm.requires_grad_(True)

        y_hat = z_hat_perm @ R_t.t()
        Y_hat_img = y_hat.reshape(B, T, D)
        X_hat = batch_inv_normalize_gpu(Y_hat_img, Mu, Std)
        X_hat_out = tail(X_hat)
        loss = ((Y_teacher - X_hat_out) ** 2).sum() / B

        loss.backward()

        grad = z_hat_perm.grad.detach()                           # [NT, D]
        grad_g = grad.reshape(-1, G, d)
        sg += (grad_g.double() ** 2).sum(dim=(0, 2))
        sg_count += grad.shape[0]

        del X, Y, Mu, Std, Y_teacher, flat, sub, dists, labels
        del flat_q, sub_q, selected, z_hat, z_hat_perm, y_hat
        del Y_hat_img, X_hat, X_hat_out, loss, grad, grad_g

    sg_np = (sg / sg_count).cpu().numpy()
    sg_ratio = sg_np.max() / sg_np.min()
    sg_cv = sg_np.std() / sg_np.mean()
    print(f"  s_g (grad energy): mean={sg_np.mean():.6e}")
    print(f"  range: [{sg_np.min():.6e}, {sg_np.max():.6e}]")
    print(f"  ratio (max/min): {sg_ratio:.3f}x")
    print(f"  CV: {sg_cv:.4f}")

    # --- Per-group ΔL_ref contribution ≈ s_g * MSE_g ---
    contrib = sg_np * mse_per_group
    contrib_ratio = contrib.max() / contrib.min()
    contrib_cv = contrib.std() / contrib.mean()
    print(f"\n=== Per-group ΔL_ref contribution (s_g × MSE_g) ===")
    print(f"  mean={contrib.mean():.6e}")
    print(f"  range: [{contrib.min():.6e}, {contrib.max():.6e}]")
    print(f"  ratio (max/min): {contrib_ratio:.3f}x")
    print(f"  CV: {contrib_cv:.4f}")

    # --- Leave-one-group-out exact ΔL_ref ---
    print("\n=== Leave-one-group-out ΔL_ref (exact) ===")
    dl_full = 0.0
    dl_logo = np.zeros(G, dtype=np.float64)
    n_logo = 0

    for start in range(0, min(len(features_train), 100), 4):
        end = min(start + 4, min(len(features_train), 100))
        X = torch.from_numpy(
            np.stack(features_train[start:end])).float().to(device)
        B = X.shape[0]
        with torch.no_grad():
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            Y_teacher = tail.forward_nograd(X)
            flat = Y.reshape(-1, D) @ R_t
            sub = flat.reshape(-1, G, d).permute(1, 0, 2)
            dists = torch.cdist(sub, cb_t).pow(2)
            labels = dists.argmin(dim=-1)

            selected = torch.gather(
                cb_t.unsqueeze(1).expand(-1, sub.shape[1], -1, -1), 2,
                labels.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, d),
            ).squeeze(2)                                         # [G, NT, d]

            z_hat_full = selected.permute(1, 0, 2).reshape(-1, D)
            y_hat_full = z_hat_full @ R_t.t()
            X_hat_full = batch_inv_normalize_gpu(
                y_hat_full.reshape(B, T, D), Mu, Std)
            X_out_full = tail.forward_nograd(X_hat_full)
            dl_full_batch = ((Y_teacher - X_out_full) ** 2).sum().item() / B
            dl_full += dl_full_batch * B

            for g in range(G):
                z_hat_g = flat.clone()
                sub_g_sel = sub.clone()
                sub_g_sel[g] = sub[g]       # keep original (no quantization) for group g
                z_hat_noquant = sub.clone()
                z_hat_noquant[g] = sub[g]   # unquantized for group g
                for gg in range(G):
                    if gg != g:
                        z_hat_noquant[gg] = selected[gg]

                z_logo = z_hat_noquant.permute(1, 0, 2).reshape(-1, D)
                y_logo = z_logo @ R_t.t()
                X_hat_logo = batch_inv_normalize_gpu(
                    y_logo.reshape(B, T, D), Mu, Std)
                X_out_logo = tail.forward_nograd(X_hat_logo)
                dl_logo_batch = ((Y_teacher - X_out_logo) ** 2).sum().item() / B
                dl_logo[g] += dl_logo_batch * B

            n_logo += B
        del X, Y, Mu, Std, Y_teacher, flat, sub, dists, labels, selected

    dl_full /= n_logo
    dl_logo /= n_logo
    dl_diff = dl_full - dl_logo  # positive = group g contributes to ΔL_ref
    dl_diff_ratio = dl_diff.max() / max(dl_diff.min(), 1e-30)
    dl_diff_cv = dl_diff.std() / max(dl_diff.mean(), 1e-30)

    print(f"  Full ΔL_ref (all quantized): {dl_full:.1f}")
    print(f"  ΔL_ref reduction when skipping group g:")
    print(f"    mean reduction: {dl_diff.mean():.1f}")
    print(f"    range: [{dl_diff.min():.1f}, {dl_diff.max():.1f}]")
    print(f"    ratio (max/min): {dl_diff_ratio:.3f}x")
    print(f"    CV: {dl_diff_cv:.4f}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: Per-group heterogeneity after OPQ")
    print(f"{'=' * 60}")
    print(f"  MSE per group:           ratio={mse_ratio:.3f}x  CV={mse_cv:.4f}")
    print(f"  ΔL_ref sensitivity:      ratio={sg_ratio:.3f}x  CV={sg_cv:.4f}")
    print(f"  s_g × MSE_g:             ratio={contrib_ratio:.3f}x  CV={contrib_cv:.4f}")
    print(f"  Leave-one-out ΔL_ref:    ratio={dl_diff_ratio:.3f}x  CV={dl_diff_cv:.4f}")
    print()

    # Theoretical gain from optimal bit allocation
    print(f"=== Theoretical gain from optimal bit allocation ===")
    for label, values in [("MSE", mse_per_group),
                          ("s_g", sg_np),
                          ("s_g×MSE", contrib),
                          ("LOGO ΔL_ref", dl_diff)]:
        values = np.maximum(values, 1e-30)
        am = values.mean()
        gm = np.exp(np.log(values).mean())
        gain = am / gm
        gain_pct = (gain - 1) * 100
        print(f"  {label:20s}: AM/GM = {gain:.6f}  "
              f"→ {gain_pct:.2f}% potential distortion reduction")
    print(f"\n  (AM/GM gives the distortion ratio: D_uniform / D_optimal)")
    print(f"  (Under RD theory for independent Gaussian sub-sources)")

    # Print per-group detail
    print(f"\n=== Per-group detail ===")
    print(f"  {'g':>3s} {'MSE_g':>10s} {'s_g':>12s} {'s_g*MSE':>12s} {'LOGO':>10s}")
    for g in range(G):
        print(f"  {g:3d} {mse_per_group[g]:10.6f} {sg_np[g]:12.6e} "
              f"{contrib[g]:12.6e} {dl_diff[g]:10.1f}")


if __name__ == "__main__":
    main()
