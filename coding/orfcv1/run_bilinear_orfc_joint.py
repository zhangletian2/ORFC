#!/usr/bin/env python
"""Joint conv2 spatial + ORFC from OPQ, aligned with linear4+ORFC.

Load stage-1 conv2 (no PQ), then OPQ on E(X) and jointly train with ORFC.

    main     hat X = U(ORFC(E(X)))
    recon0   hat X = U(ORFC(E(X))) + F_φ(X0, 0)

Default ``--residual_ablation recon0`` keeps existing ckpt names
(``conv2_recon0_jointopq``).  ``main`` writes ``conv2_main_jointopq``.

Protocol matches ``launch_linear4_orfc_joint.py``:
λ=0.5, ORFC lr=3e-4, spatial lr=3e-5, τ=2.0 fixed, 100 epoch, val-best.

    CUDA_VISIBLE_DEVICES=2 python -u run_bilinear_orfc_joint.py \\
        --layer blk05 --K 4 --epochs 100 --n_val 200
    CUDA_VISIBLE_DEVICES=2 python -u run_bilinear_orfc_joint.py \\
        --layer blk05 --K 4 --residual_ablation main
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
_ORFC_DIR = os.path.normpath(os.path.join(HERE, "..", "orfc"))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
for p in (_ORFC_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from backbone.wrapper import Dinov2Wrapper  # noqa: E402
from opq import batch_inv_normalize_gpu, batch_normalize_gpu  # noqa: E402
from run_multilayer_calibrator import load_gt, set_seed  # noqa: E402
from soft_pq import (  # noqa: E402
    FeatureCodec, FrozenTail, OrthogonalTransform, SoftPQ,
    compute_perplexity, save_codec,
)

from bilinear_residual import (  # noqa: E402
    RESIDUAL_ABLATIONS, BilinearSpatialCodec, apply_residual, freeze_module,
    load_residual_codec, load_spatial_weights, save_residual_codec,
)
from run_bilinear_residual import (  # noqa: E402
    FeatureTeacherDataset, cache_teacher, eval_cascade, init_orfc_from_opq,
    load_features, make_split, rate_for, residual_ckpt_path,
)
def _cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _build_dinov3_tail(model, layer_idx, token_hw, device):
    """Build RoPE-aware frozen tail directly from a timm Eva model."""
    from backbone.dinov3_tail import Dinov3FrozenTail
    from soft_pq import FrozenTail as _FT
    n_blocks = len(model.blocks)
    if layer_idx >= n_blocks - 1:
        return _FT([], model.norm, device=str(device))
    h_p, w_p = token_hw
    dummy = torch.zeros(1, 3, h_p * model.patch_embed.proj.kernel_size[0],
                        w_p * model.patch_embed.proj.kernel_size[0], device=device)
    with torch.no_grad():
        x_emb = model.patch_embed(dummy)
        _, rot_pos_embed = model._pos_embed(x_emb)
    return Dinov3FrozenTail(model, layer_idx, rot_pos_embed, None, device=str(device))


def _full_tail(wrapper, layer_idx, device):
    """Tail over all blocks after layer_idx, RoPE-aware for DINOv3."""
    if getattr(wrapper, "_is_dinov3", False):
        return _build_dinov3_tail(
            wrapper.backbone, layer_idx, wrapper._token_hw, device)
    return FrozenTail(
        list(wrapper.backbone.blocks[layer_idx + 1:]),
        wrapper.backbone.norm, device=device)


def _rebuild_train_tail(wrapper, layer_idx, device):
    if getattr(wrapper, "_is_dinov3", False):
        for i, blk in enumerate(wrapper.backbone.blocks):
            if i <= layer_idx:
                blk.cpu()
        torch.cuda.empty_cache()
        return _build_dinov3_tail(
            wrapper.backbone, layer_idx, wrapper._token_hw, device)
    for i, blk in enumerate(wrapper.backbone.blocks):
        if i <= layer_idx:
            blk.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()
    return FrozenTail(
        list(wrapper.backbone.blocks[layer_idx + 1:]),
        wrapper.backbone.norm,
        device=device,
    )

warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

def joint_ablation(args_or_ab="recon0"):
    if isinstance(args_or_ab, str):
        ab = args_or_ab
    else:
        ab = getattr(args_or_ab, "residual_ablation", "recon0")
    ab = str(ab or "recon0")
    if ab not in RESIDUAL_ABLATIONS:
        raise ValueError(f"unknown residual ablation {ab!r}")
    return ab


def spatial_tag(args_or_ab="recon0"):
    ab = joint_ablation(args_or_ab)
    tag = f"conv2_{ab}"
    if isinstance(args_or_ab, SimpleNamespace):
        cm = getattr(args_or_ab, "cls_mode", "learned")
    elif not isinstance(args_or_ab, str):
        cm = getattr(args_or_ab, "cls_mode", "learned")
    else:
        cm = "learned"
    if cm == "conv2":
        tag += "_clsconv2"
    return tag


def default_spatial_ckpt(layer, backbone="dinov2_vitl14", result_dir=None,
                         ablation="recon0"):
    if result_dir is None:
        result_dir = os.path.join(HERE, "results", "bilinear_residual")
    ns = SimpleNamespace(
        layer=layer, K=4, c=4, backbone=backbone,
        residual_lr=3e-4, residual_epochs=30,
        max_train_images=5000, n_val=0, seed=42,
        residual_quantize=False, residual_orfc=False,
        residual_decoder="conv", residual_ablation=joint_ablation(ablation),
        spatial_down="conv2", spatial_up="conv2",
        latent_channels=0, cls_mode="learned",
        result_dir=result_dir,
    )
    return residual_ckpt_path(ns, "both")


def default_recon0_ckpt(layer, backbone="dinov2_vitl14", result_dir=None):
    return default_spatial_ckpt(
        layer, backbone=backbone, result_dir=result_dir, ablation="recon0")


def joint_stem(args):
    bt_tag = f"bt{args.bottleneck_dim}" if args.bottleneck_dim > 0 else "noBt"
    lc = getattr(args, "latent_channels", 0)
    ct_tag = f"_C{lc}" if lc and lc > 0 else ""
    slr = float(getattr(args, "spatial_lr_scale", 0.1))
    slr_tag = "" if abs(slr - 1.0) < 1e-12 else f"_hlr{slr:g}"
    return (
        f"{args.layer}_{spatial_tag(args)}_jointopq_K{args.K}"
        f"_emb{args.embedding_dim}"
        f"_{bt_tag}{ct_tag}_ws_lmbda{args.lmbda}_tau{args.tau_start}"
        f"_te{args.tau_end}_tscon{slr_tag}_bv_lr{args.lr}_ep{args.epochs}"
        f"_n{args.max_train_images}_nval{args.n_val}_s{args.seed}"
    )


def orfc_ckpt_path(args):
    return Path(args.ckpt_dir) / args.backbone / f"{joint_stem(args)}.pt"


def spatial_out_path(args):
    return Path(args.ckpt_dir) / args.backbone / f"{joint_stem(args)}_spatial.pt"


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  saved {path}", flush=True)


def build_spatial(D, args, C=None, cls_mode=None):
    if C is None:
        C = getattr(args, "latent_channels", None)
        if C is not None and C <= 0:
            C = None
    if cls_mode is None:
        cls_mode = getattr(args, "cls_mode", "learned")
    spatial = BilinearSpatialCodec(
        D, n_prefix=getattr(args, "n_prefix", 1), scale=args.scale, grid=None,
        down="conv2", up="conv2", C=C, cls_mode=cls_mode)
    if getattr(args, "ortho_init", False) and spatial.C != spatial.D:
        spatial.init_orthogonal_projection(seed=getattr(args, "seed", 42))
    return spatial


def load_stage1_spatial(args, D, device):
    ablation = joint_ablation(args)
    ckpt = Path(args.residual_ckpt) if args.residual_ckpt else default_spatial_ckpt(
        args.layer, args.backbone, args.residual_dir, ablation=ablation)
    if not ckpt.is_file():
        raise FileNotFoundError(f"{ablation} stage-1 ckpt missing: {ckpt}")
    residual, meta = load_residual_codec(str(ckpt), device=device)
    C_ckpt = meta.get("C", None)
    cls_ckpt = meta.get("cls_mode", "learned")
    # Under cls_mode="none" no parameter shape depends on n_prefix, so a
    # mismatch loads cleanly and misreads the register tokens as patches.
    # Checkpoints predating these meta keys carry None -> nothing to check.
    for key, want in (("n_prefix", args.n_prefix),
                      ("norm_mode", args.norm_mode)):
        got = meta.get(key)
        if got is not None and str(got) != str(want):
            raise ValueError(
                f"stage-1 was trained with {key}={got!r}, but --{key}={want!r}")
    spatial = build_spatial(int(meta.get("D", D)), args, C=C_ckpt,
                            cls_mode=cls_ckpt).to(device)
    load_spatial_weights(spatial, meta)
    residual.set_mode("fixed" if ablation == "main" else "recon")
    print(f"  load stage-1 {ablation} {ckpt.name}", flush=True)
    return spatial, residual, ckpt, meta


def load_recon0(args, D, device):
    return load_stage1_spatial(args, D, device)


def train_recon0_orfc_joint(
    features_train, spatial, residual, codec, tail,
    epochs, lr, spatial_lr_scale, batch_size, device, seed,
    val_features=None, lmbda=0.5, grad_clip=1.0,
    tau_start=2.0, tau_end=2.0, tau_schedule="constant",
    n_prefix=1, verbose=True, ablation="recon0", norm_mode="per_image",
):
    """E/U (and F_φ unless ``main``) stay in-graph with ORFC R/PQ/prior."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if isinstance(features_train, np.ndarray) and features_train.ndim == 3:
        features_array = features_train
    else:
        features_array = np.stack(features_train)
    N_img = features_array.shape[0]
    pq = codec.pq
    G, K = pq.G, pq.K
    pq.ste_mode = "softmax"
    use_soft = tau_start > 0
    tau_fixed = use_soft and (tau_schedule == "constant" or tau_start == tau_end)

    lr_orfc = float(lr)
    lr_spat = float(lr) * float(spatial_lr_scale)

    ablation = joint_ablation(ablation)
    train_fphi = ablation != "main"
    res_mode = "recon" if train_fphi else "fixed"

    spatial.train()
    for p in spatial.parameters():
        p.requires_grad_(True)
    residual.set_mode(res_mode)
    residual.train(train_fphi)
    codec.train()
    for p in codec.parameters():
        p.requires_grad_(True)

    spat_params = [p for p in spatial.parameters() if p.requires_grad]
    recon_params = (
        [p for p in residual.recon_parameters() if p.requires_grad]
        if train_fphi else [])
    orfc_params = [p for p in codec.parameters() if p.requires_grad]
    pre_params = spat_params + recon_params
    trainable = pre_params + orfc_params
    optimizer = torch.optim.Adam([
        {"params": pre_params, "lr": lr_spat},
        {"params": orfc_params, "lr": lr_orfc},
    ])
    t_max = max(epochs, 1)

    def _cosine_factor(step):
        t = max(int(step), 0)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * t / t_max))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_cosine_factor)
    if verbose:
        n_s = sum(p.numel() for p in spat_params)
        n_r = sum(p.numel() for p in recon_params)
        n_c = sum(p.numel() for p in orfc_params)
        if train_fphi:
            print(f"  Trainable: E/U {n_s:,} + F_φ {n_r:,} + ORFC {n_c:,} "
                  f"= {n_s + n_r + n_c:,}", flush=True)
            print(f"  LR: spatial+F_φ {lr_spat:g}  ORFC {lr_orfc:g}  "
                  f"(spatial_lr_scale={spatial_lr_scale:g})", flush=True)
        else:
            print(f"  Trainable: E/U {n_s:,} + ORFC {n_c:,} "
                  f"= {n_s + n_c:,}  (main: F_φ frozen/unused)", flush=True)
            print(f"  LR: spatial {lr_spat:g}  ORFC {lr_orfc:g}  "
                  f"(spatial_lr_scale={spatial_lr_scale:g})", flush=True)
        if use_soft:
            if tau_fixed:
                print(f"  Soft PQ: τ={tau_start:.2f} fixed ({tau_schedule})",
                      flush=True)
            else:
                print(f"  Soft PQ: τ {tau_start:.2f} → {tau_end:.4f} "
                      f"({tau_schedule})", flush=True)
        if train_fphi:
            print("  Checkpoint: restore E/U/F_φ+ORFC at best val ΔL_ref",
                  flush=True)
            print("  recon: hat X = U(ORFC(E(X))) + F_φ(X0, 0)", flush=True)
        else:
            print("  Checkpoint: restore E/U+ORFC at best val ΔL_ref",
                  flush=True)
            print("  recon: hat X = U(ORFC(E(X)))  (main, no F_φ)", flush=True)

    if verbose:
        print(f"  Pre-computing teacher outputs ({N_img} images)...",
              flush=True)
    teacher_cache = cache_teacher(
        features_array, tail, device, batch_size, verbose=verbose)

    val_array = val_teacher = None
    if val_features is not None and len(val_features) > 0:
        val_array = (val_features if isinstance(val_features, np.ndarray)
                     else np.stack(val_features))
        val_teacher = cache_teacher(
            val_array, tail, device, batch_size, verbose=False)

    loader = DataLoader(
        FeatureTeacherDataset(features_array, teacher_cache),
        batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )
    history = []
    best_val = best_epoch = None
    best_spatial = best_residual = best_codec = None

    def _reconstruct(Y, train_mode):
        spatial.train(train_mode)
        residual.train(bool(train_mode) and train_fphi)
        codec.train(train_mode)
        seq, aux = spatial.encode(Y)
        seq_hat, usage = codec(seq)
        Y0 = spatial.decode(seq_hat, aux)
        Y_hat, _, _, _ = apply_residual(
            Y, Y0, residual, n_prefix=n_prefix, quantize=False,
            ablation=ablation)
        return Y_hat, seq, usage

    for epoch in range(epochs):
        t_epoch = time.time()
        if tau_fixed:
            pq.temperature = tau_start
        elif use_soft and epochs > 1:
            progress = epoch / (epochs - 1)
            if tau_schedule == "linear":
                tau = tau_start + (tau_end - tau_start) * progress
            else:
                tau = tau_start * (tau_end / tau_start) ** progress
            pq.temperature = tau
        elif use_soft:
            pq.temperature = tau_start
        else:
            pq.temperature = 0.0

        total_distortion = 0.0
        total_rate = 0.0
        usage_acc = torch.zeros(G, K, device=device)
        spatial.train()
        residual.train(train_fphi)
        codec.train()
        residual.set_mode(res_mode)
        Tm_rate = int(spatial.coded_tokens(features_array.shape[1]))

        for X, Y_teacher in loader:
            X = X.to(device, non_blocking=True)
            Y_teacher = Y_teacher.to(device, non_blocking=True)
            B = X.shape[0]
            with torch.no_grad():
                Y, Mu, Std = batch_normalize_gpu(
                    X, mode=norm_mode, n_prefix=n_prefix)
            Y_hat, seq, usage = _reconstruct(Y, True)
            Tm_rate = int(seq.shape[1])
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            distortion = ((Y_teacher - tail(X_hat)) ** 2).sum() / B
            if codec.use_rate:
                loss = codec._last_rate * seq.shape[1] + lmbda * distortion
            else:
                loss = distortion
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            total_distortion += distortion.item() * B
            if codec.use_rate:
                total_rate += codec._last_rate.item() * B
            usage_acc += usage.detach()
            del X, Y, Mu, Std, seq, Y_hat, X_hat, Y_teacher, loss, distortion

        scheduler.step()
        avg_distortion = total_distortion / N_img
        avg_rate = total_rate / N_img if codec.use_rate else 0.0
        perplexity = compute_perplexity(usage_acc)
        dead_entries = int((usage_acc == 0).sum().item())

        val_loss = None
        if val_array is not None:
            val_sum = 0.0
            n_val = val_array.shape[0]
            spatial.eval()
            residual.eval()
            codec.eval()
            with torch.no_grad():
                for vs in range(0, n_val, batch_size):
                    ve = min(vs + batch_size, n_val)
                    X_v = torch.from_numpy(val_array[vs:ve]).float().to(device)
                    Y_v, Mu_v, Std_v = batch_normalize_gpu(
                        X_v, mode=norm_mode, n_prefix=n_prefix)
                    Yh_v, _, _ = _reconstruct(Y_v, False)
                    Yt_v = torch.from_numpy(val_teacher[vs:ve]).float().to(device)
                    Xh_v = batch_inv_normalize_gpu(Yh_v, Mu_v, Std_v)
                    Xo_v = tail.forward_nograd(Xh_v)
                    val_sum += ((Yt_v - Xo_v) ** 2).sum().item()
                    del X_v, Y_v, Mu_v, Std_v, Yh_v, Yt_v, Xh_v, Xo_v
            val_loss = val_sum / n_val
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_spatial = _cpu_state(spatial)
                best_residual = _cpu_state(residual)
                best_codec = _cpu_state(codec)
                if verbose:
                    print(f"  * val-best  ep={epoch}  val_ΔL={val_loss:.1f}",
                          flush=True)

        epoch_info = {
            "epoch": epoch,
            "lr": optimizer.param_groups[-1]["lr"],
            "lr_spatial": optimizer.param_groups[0]["lr"],
            "loss_distortion": avg_distortion,
            "perplexity": perplexity,
            "val_loss": val_loss,
            "rate_bits": avg_rate,
            "dead_entries": dead_entries,
            "temperature": pq.temperature,
            "n_coded": Tm_rate,
            "time": time.time() - t_epoch,
            "is_best_val": bool(best_epoch == epoch),
        }
        history.append(epoch_info)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            val_str = f"  val={val_loss:.1f}" if val_loss is not None else ""
            rate_str = ""
            if codec.use_rate:
                rate_str = (f"  R={avg_rate:.2f}b/t"
                            f"  λD={lmbda * avg_distortion:.1f}"
                            f"  dead={dead_entries}")
            tau_str = f"  τ={pq.temperature:.4f}" if use_soft else ""
            print(f"  ep {epoch:3d}/{epochs}  "
                  f"D={avg_distortion:.1f}  ppl={perplexity:.1f}"
                  f"{rate_str}{tau_str}{val_str}  ({time.time() - t_epoch:.1f}s)",
                  flush=True)

    best_info = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "restored": False,
        "spatial_lr_scale": float(spatial_lr_scale),
        "lr_spatial": lr_spat,
        "lr_orfc": lr_orfc,
    }
    if best_spatial is not None:
        spatial.load_state_dict(best_spatial)
        residual.load_state_dict(best_residual)
        codec.load_state_dict(best_codec)
        spatial.to(device)
        residual.to(device)
        codec.to(device)
        best_info["restored"] = True
        if verbose:
            print(f"  restored {'E/U/F_φ+ORFC' if train_fphi else 'E/U+ORFC'} "
                  f"val-best  ep={best_epoch}  val_ΔL={best_val:.1f}",
                  flush=True)
    return spatial, residual, codec, history, best_info


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--layer", default="blk05")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--embedding_dim", type=int, default=32)
    p.add_argument("--bottleneck_dim", type=int, default=1024)
    p.add_argument("--c", type=int, default=4)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--latent_channels", type=int, default=0,
                   help="Latent channel width C for spatial codec. "
                        "0 means same as D (no reduction).")
    p.add_argument("--norm_mode", default="per_image")
    p.add_argument("--n_prefix", type=int, default=1,
                   help="CLS+register prefix length (DINOv3: 5)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--spatial_lr_scale", type=float, default=0.1)
    p.add_argument("--cls_mode", default="learned",
                   choices=["learned", "identity", "conv2"],
                   help="CLS prefix handling for the stage-1 spatial codec. "
                        "Used to build the output stem; the checkpoint's own "
                        "cls_mode (loaded from --residual_ckpt) decides the "
                        "actual construction.")
    p.add_argument("--lmbda", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--tau_start", type=float, default=2.0)
    p.add_argument("--tau_end", type=float, default=2.0)
    p.add_argument("--tau_schedule", default="constant",
                   choices=["exponential", "linear", "constant"])
    p.add_argument("--max_train_images", type=int, default=5000)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--kmeans_max_samples", type=int, default=2_000_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", default="dinov2_vitl14")
    p.add_argument("--feat_root", default=os.path.join(PROJECT_ROOT, "features"))
    p.add_argument("--train_subset", default="train")
    p.add_argument("--test_subset", default="test")
    p.add_argument("--gt_path", default=os.path.join(
        PROJECT_ROOT, "utils", "imagenet_selected_label500.txt"))
    p.add_argument("--residual_ckpt", default="")
    p.add_argument("--residual_ablation", default="recon0",
                   choices=list(RESIDUAL_ABLATIONS),
                   help="main: hat=U(ORFC(E(X))); recon0: +F_φ(X0,0); "
                        "full: +F_φ(X0,G).  Changes ckpt tag (conv2_{ablation}).")
    p.add_argument("--residual_dir", default=os.path.join(
        HERE, "results", "bilinear_residual"))
    p.add_argument("--ckpt_dir", default=os.path.join(HERE, "checkpoints"))
    p.add_argument("--result_dir", default=os.path.join(
        HERE, "results", "bilinear_orfc_jointopq"))
    p.add_argument("--skip_baselines", action="store_true")
    p.add_argument("--skip_test_acc", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    layer_idx = int(args.layer[-2:])
    stem = joint_stem(args)

    ablation = joint_ablation(args)
    print(f"\n{'#' * 70}", flush=True)
    print(f"# conv2 {ablation} + ORFC joint  (linear4 protocol)", flush=True)
    print(f"# layer={args.layer}  K={args.K}  λ={args.lmbda}  "
          f"lr_orfc={args.lr:g}  lr_spat={args.lr * args.spatial_lr_scale:g}  "
          f"ep={args.epochs}  τ={args.tau_start} ({args.tau_schedule})",
          flush=True)
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'#' * 70}", flush=True)

    features_train, features_test, basenames_test = load_features(args)
    gt_test = load_gt(args.gt_path)
    train_feat, val_feat, train_idx, val_idx = make_split(features_train, args)
    D = int(train_feat[0].shape[1])
    T = int(train_feat[0].shape[0])
    spatial, residual, stage1_ckpt, _meta = load_stage1_spatial(args, D, device)
    C_latent = spatial.C
    G = C_latent // args.embedding_dim
    n_coded = spatial.coded_tokens(T)
    print(f"  D={D} C={C_latent} T={T}  G={G} d={args.embedding_dim} K={args.K}  "
          f"coded={n_coded}  train={len(train_feat)} val={len(val_feat)} "
          f"test={len(features_test)}", flush=True)

    print(f"\nLoading {args.backbone}...", flush=True)
    if args.backbone.startswith("dinov3"):
        import timm
        from types import SimpleNamespace
        _ckpt = "/data4/workspace/zlt/cache/torch/hub/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        _model = timm.create_model("vit_large_patch16_dinov3",
                                   pretrained=False, img_size=224,
                                   dynamic_img_size=True)
        _sd = torch.load(_ckpt, map_location="cpu", weights_only=True)
        _sd.pop("mask_token", None)
        # The checkpoint uses facebook naming (ls1.gamma / ls2.gamma /
        # storage_tokens); timm expects gamma_1 / gamma_2 / reg_token.  Without
        # this filter, load_state_dict(strict=False) silently leaves all 48
        # LayerScale gammas at their 1e-5 init, which collapses every residual
        # update in the tail.  Matches cofai's Dinov3TimmBackbone.
        from timm.models.eva import checkpoint_filter_fn as _dinov3_filter
        _sd = _dinov3_filter(_sd, _model)
        _res = _model.load_state_dict(_sd, strict=False)
        if _res.missing_keys or _res.unexpected_keys:
            raise RuntimeError(
                f"DINOv3 checkpoint load mismatch: "
                f"missing={_res.missing_keys[:8]} "
                f"unexpected={_res.unexpected_keys[:8]}")
        _model.eval().to(device)
        wrapper = SimpleNamespace(backbone=_model, head=None,
                                  _is_dinov3=True, _token_hw=(14, 14))
        print(f"  DINOv3 Tail: {len(_model.blocks) - layer_idx - 1} blocks "
              f"(RoPE-aware)", flush=True)
    else:
        wrapper = Dinov2Wrapper(
            head_layers=1, model_name=args.backbone, device=device)
    results = {
        "config": vars(args),
        "stage": "joint",
        "ablation": ablation,
        "stage1_ckpt": str(stage1_ckpt),
        "recon0_ckpt": str(stage1_ckpt),
        "n_coded": int(n_coded),
        "n_train": len(train_feat),
        "n_val": len(val_feat),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
    }

    wrapper.backbone.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()
    freeze_module(spatial)
    freeze_module(residual)
    transform, R_ws, C_ws, opq_usage, R_std, codebooks_std = init_orfc_from_opq(
        args, D, G, spatial, train_feat, device)
    pq = SoftPQ(G, args.K, args.embedding_dim, lmbda=args.lmbda).to(device)
    codec = FeatureCodec(pq, transform.to(device) if transform is not None else None).to(device)
    if transform is not None:
        transform.init_from_opq(R_ws)
        print(f"    ||R'R-I||={transform.orth_error():.2e}", flush=True)
    pq.init_codebooks(C_ws)
    if pq.use_rate and opq_usage is not None:
        pq.init_prior_from_freq(opq_usage)

    wrapper.backbone.to(device)
    if wrapper.head is not None:
        wrapper.head.to(device)
    tail = _full_tail(wrapper, layer_idx, device)

    if not args.skip_baselines:
        freeze_module(spatial)
        freeze_module(residual)
        freeze_module(codec)
        results["stage1_only"] = eval_cascade(
            f"{ablation}-only", features_test, spatial, None, residual, tail,
            args.norm_mode, device, args.batch_size, ablation=ablation,
            quantize=False, basenames=basenames_test, gt=gt_test,
            wrapper=wrapper, layer_idx=layer_idx)
        results["recon0_only"] = results["stage1_only"]

    print(f"\n{'=' * 60}", flush=True)
    print(f"  [Joint from OPQ] K={args.K}  λ={args.lmbda}  "
          f"lr_orfc={args.lr:g}  lr_spat={args.lr * args.spatial_lr_scale:g}  "
          f"ep={args.epochs}  ablation={ablation}", flush=True)
    print(f"{'=' * 60}", flush=True)
    tail = _rebuild_train_tail(wrapper, layer_idx, device)

    t0 = time.time()
    spatial, residual, codec, history, best_info = train_recon0_orfc_joint(
        train_feat, spatial, residual, codec, tail,
        epochs=args.epochs, lr=args.lr,
        spatial_lr_scale=args.spatial_lr_scale,
        batch_size=args.batch_size, device=device, seed=args.seed,
        val_features=val_feat if val_feat else None,
        lmbda=args.lmbda, grad_clip=args.grad_clip,
        tau_start=args.tau_start, tau_end=args.tau_end,
        tau_schedule=args.tau_schedule, n_prefix=spatial.n_prefix,
        ablation=ablation, norm_mode=args.norm_mode)
    train_time = time.time() - t0
    print(f"  Joint from OPQ: {train_time:.1f}s", flush=True)

    orfc_out = orfc_ckpt_path(args)
    spat_out = spatial_out_path(args)
    orfc_out.parent.mkdir(parents=True, exist_ok=True)
    save_codec(codec, str(orfc_out))
    extra = torch.load(str(orfc_out), map_location="cpu")
    extra.update({
        "spatial_ckpt": str(spat_out),
        "stage1_init": str(stage1_ckpt),
        "recon0_init": str(stage1_ckpt),
        "n_coded": int(n_coded),
        "D": int(D),
        "C": int(C_latent),
        "layer": args.layer,
        "joint": True,
        "joint_from_opq": True,
        "ablation": ablation,
        "residual_ablation": ablation,
        "spatial_down": "conv2",
        "spatial_up": "conv2",
        "cls_mode": getattr(spatial, "cls_mode", "learned"),
        "n_prefix": int(spatial.n_prefix),
        "norm_mode": str(args.norm_mode),
        "spatial_lr_scale": args.spatial_lr_scale,
        "best_epoch": best_info.get("best_epoch"),
        "best_val_loss": best_info.get("best_val_loss"),
        "val_best_restored": best_info.get("restored"),
    })
    torch.save(extra, str(orfc_out))
    save_residual_codec(residual, spat_out, extra={
        "mode": "both",
        "D": int(D),
        "C": int(C_latent),
        "layer": args.layer,
        "orfc_ckpt": str(orfc_out),
        "residual_ablation": ablation,
        "residual_orfc": True,
        "residual_quantize": False,
        "residual_decoder": "conv",
        "spatial_down": "conv2",
        "spatial_up": "conv2",
        "cls_mode": getattr(spatial, "cls_mode", "learned"),
        "n_prefix": int(spatial.n_prefix),
        "norm_mode": str(args.norm_mode),
        "spatial_state_dict": spatial.state_dict(),
        "best": best_info,
    })
    print(f"  saved ORFC {orfc_out.name}", flush=True)
    print(f"  saved spatial {spat_out.name}", flush=True)

    tail.to("cpu")
    torch.cuda.empty_cache()
    wrapper.backbone.to(device)
    if wrapper.head is not None:
        wrapper.head.to(device)
    tail = _full_tail(wrapper, layer_idx, device)
    freeze_module(spatial)
    freeze_module(residual)
    freeze_module(codec)

    results["ckpt"] = str(orfc_out)
    results["spatial_ckpt"] = str(spat_out)
    results["train_time_s"] = float(train_time)
    results["best"] = best_info
    results["history"] = history
    results["val"] = eval_cascade(
        "joint val", val_feat, spatial, codec, residual, tail,
        args.norm_mode, device, args.batch_size, ablation=ablation,
        quantize=False)
    # --skip_test_acc drops only the accuracy probe; test distortion (ΔL /
    # MSE) is always measured so DINOv2 and DINOv3 report the same columns.
    results["test"] = eval_cascade(
        "joint test", features_test, spatial, codec, residual, tail,
        args.norm_mode, device, args.batch_size, ablation=ablation,
        quantize=False,
        basenames=None if args.skip_test_acc else basenames_test,
        gt=None if args.skip_test_acc else gt_test,
        wrapper=None if args.skip_test_acc else wrapper,
        layer_idx=layer_idx)
    results["rate"] = rate_for(
        train_feat, features_test, spatial, codec, args, device,
        include_guidance=False)
    print(f"  pq_bpt={results['rate']['pq_bpt']:.3f}  "
          f"bits/img={results['rate']['bits_per_image']:.1f}  "
          f"maxPQ={results['rate']['pq_max_bits_per_image']:.0f}", flush=True)
    dump_json(Path(args.result_dir) / args.backbone / f"{stem}.json", results)
    print("done", flush=True)


if __name__ == "__main__":
    main()
