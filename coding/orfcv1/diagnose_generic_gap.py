#!/usr/bin/env python
"""Why generic trails integer on blk05 but not blk20.

Compares trained integer/generic ckpts: kernel stats, residual vs bilinear
prior, 2x2 phase diversity, encode/decode split, init gradients.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
_ORFC_DIR = os.path.normpath(os.path.join(HERE, "..", "orfc"))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if _ORFC_DIR not in sys.path:
    sys.path.insert(0, _ORFC_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from run_multilayer_calibrator import set_seed, preload_features  # noqa: E402
from backbone.wrapper import Dinov2Wrapper  # noqa: E402
from soft_pq import FrozenTail  # noqa: E402
from opq import batch_normalize_gpu, batch_inv_normalize_gpu  # noqa: E402
from spatial_reassembly import (  # noqa: E402
    SpatialReassemblyCodec, load_spatial_reassembly, reshape_map,
    bilinear_up_logit_map, bilinear_up_kernel, kernel_to_bias,
)

import warnings  # noqa: E402
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

CKPT_DIR = Path(HERE) / "results" / "spatial_reassembly" / "dinov2_vitl14"


def _entropy(p, dim=1):
    return -(p * p.clamp_min(1e-8).log()).sum(dim=dim)


def _kl(p, q, dim=1):
    return (p * (p.clamp_min(1e-8).log() - q.clamp_min(1e-8).log())).sum(dim=dim)


def bilinear_beta(H, W, Hm, Wm, k, device, dtype):
    logits = bilinear_up_logit_map(H, W, Hm, Wm, k, device).to(dtype)
    return torch.softmax(logits, dim=1)


def integer_bilinear_beta(k, scale, H, W, device, dtype):
    s, kk = scale, k * k
    bias = torch.zeros(s * s * kk, device=device, dtype=dtype)
    for dy in range(s):
        for dx in range(s):
            ker = bilinear_up_kernel(k, dy, dx, scale=s).to(device=device, dtype=dtype)
            b = kernel_to_bias(ker).reshape(-1)
            for c in range(kk):
                bias[c * s * s + dy * s + dx] = b[c]
    low = bias.view(1, s * s * kk, 1, 1).expand(1, s * s * kk, H // s, W // s)
    return torch.softmax(F.pixel_shuffle(low, s), dim=1)


def up_logit_parts(codec, y):
    """learned residual, prior (or None), full logits, phase-std of pred_up."""
    v = codec.route_up(y)
    g = codec.global_fc(v.mean(dim=(2, 3), keepdim=True))
    logits_low = codec.pred_up(v + g)
    s, kk = codec.scale, codec.k * codec.k
    Hm, Wm = y.shape[-2:]
    H, W = Hm * s, Wm * s
    phase = logits_low.view(logits_low.shape[0], kk, s * s, Hm, Wm)
    phase_std = phase.std(dim=2).mean().item()
    phase_rms = phase.pow(2).mean().sqrt().item()
    if codec.up_kern == "shuffle":
        full = F.pixel_shuffle(logits_low, s)
        return dict(learned=full, prior=None, full=full,
                    phase_std=phase_std, phase_rms=phase_rms)
    collapsed = phase.mean(dim=2)
    learned = F.interpolate(
        collapsed, size=(H, W), mode="bilinear", align_corners=False)
    prior = bilinear_up_logit_map(
        H, W, Hm, Wm, codec.k, device=y.device).to(dtype=learned.dtype)
    return dict(learned=learned, prior=prior, full=learned + prior,
                phase_std=phase_std, phase_rms=phase_rms,
                collapsed_rms=collapsed.pow(2).mean().sqrt().item())


def kernel_pack(alpha):
    pmax = alpha.max(dim=1).values
    return {
        "pmax_mean": float(pmax.mean()),
        "entropy_mean": float(_entropy(alpha).mean()),
        "eff_support": float((alpha > 0.05).float().sum(dim=1).mean()),
    }


def phase_cosine(beta, s=2):
    """Mean pairwise cosine of the s² phase kernels inside each coarse cell."""
    B, kk, H, W = beta.shape
    Hm, Wm = H // s, W // s
    ph = beta.view(B, kk, Hm, s, Wm, s).permute(0, 2, 4, 3, 5, 1)
    ph = ph.reshape(B, Hm, Wm, s * s, kk)
    ph = ph / ph.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # [B,Hm,Wm,s²,s²]
    sim = torch.matmul(ph, ph.transpose(-1, -2))
    eye = torch.eye(s * s, device=beta.device, dtype=beta.dtype)
    off = 1.0 - eye
    return float((sim * off).sum() / off.sum() / (B * Hm * Wm))


@torch.no_grad()
def stats_on_batch(codec, X, n_prefix=1, norm_mode="per_image"):
    Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
    p = n_prefix
    H = W = int(round((Y.shape[1] - p) ** 0.5))
    x = reshape_map(Y[:, p:, :], H, W)
    y, alpha = codec.encode_map(x)
    parts = up_logit_parts(codec, y)
    beta = torch.softmax(parts["full"], dim=1)
    xhat, _ = codec.decode_map(y)
    bili_y = F.interpolate(x, size=y.shape[-2:], mode="bilinear",
                           align_corners=False)
    bili_x = F.interpolate(bili_y, size=(H, W), mode="bilinear",
                           align_corners=False)
    Hm, Wm = y.shape[-2:]
    if codec.up_kern == "interp":
        ref_beta = bilinear_beta(H, W, Hm, Wm, codec.k, x.device, x.dtype)
    else:
        ref_beta = integer_bilinear_beta(
            codec.k, codec.scale, H, W, x.device, x.dtype)

    out = {
        "alpha": kernel_pack(alpha),
        "beta": kernel_pack(beta),
        "beta_kl_bili": float(_kl(beta, ref_beta.expand_as(beta)).mean()),
        "mse_to_input": float(((xhat - x) ** 2).mean()),
        "mse_to_bili": float(((xhat - bili_x) ** 2).mean()),
        "mse_y_to_bili": float(((y - bili_y) ** 2).mean()),
        "phase_std": parts["phase_std"],
        "phase_rms": parts["phase_rms"],
        "beta_phase_cos": phase_cosine(beta, codec.scale),
        "learned_logit_rms": float(parts["learned"].pow(2).mean().sqrt()),
        "full_logit_rms": float(parts["full"].pow(2).mean().sqrt()),
    }
    if parts["prior"] is not None:
        prior_rms = float(parts["prior"].pow(2).mean().sqrt())
        out["prior_logit_rms"] = prior_rms
        out["learned_over_prior"] = out["learned_logit_rms"] / max(prior_rms, 1e-8)
        out["collapsed_rms"] = parts.get("collapsed_rms")
        # residual-only vs prior-only reconstructions
        beta_prior = torch.softmax(parts["prior"].expand_as(parts["full"]), dim=1)
        beta_res = torch.softmax(parts["learned"], dim=1)
        win = codec._up_windows(y, H, W)
        x_prior = (win * beta_prior.unsqueeze(1)).sum(dim=2)
        x_res = (win * beta_res.unsqueeze(1)).sum(dim=2)
        out["mse_prior_only_to_bili"] = float(((x_prior - bili_x) ** 2).mean())
        out["mse_residual_only_to_input"] = float(((x_res - x) ** 2).mean())
    return out, y, xhat, x


def grad_norms(codec, X, tail, n_prefix=1, norm_mode="per_image"):
    codec.zero_grad(set_to_none=True)
    codec.train()
    with torch.no_grad():
        Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
        teacher = tail.forward_nograd(X)
    Y_hat, _ = codec(Y)
    X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
    loss = ((teacher - tail(X_hat)) ** 2).sum() / X.shape[0]
    loss.backward()
    groups = {
        "route_down": codec.route_down.parameters(),
        "pred_down": codec.pred_down.parameters(),
        "route_up": codec.route_up.parameters(),
        "pred_up": codec.pred_up.parameters(),
        "global_fc": codec.global_fc.parameters(),
    }
    norms = {}
    for name, params in groups.items():
        tot = 0.0
        for p in params:
            if p.grad is not None:
                tot += float(p.grad.detach().pow(2).sum().sqrt())
        norms[name] = tot
    norms["loss"] = float(loss.detach())
    codec.zero_grad(set_to_none=True)
    codec.eval()
    return norms


def ckpt_path(layer, generic):
    if generic:
        return CKPT_DIR / f"{layer}_s2_k5_generic_lr0.0003_ep50_s42.pt"
    return CKPT_DIR / f"{layer}_s2_k5_lr0.0003_ep50_s42.pt"


def main():
    device = torch.device("cuda")
    set_seed(42)
    feat_root = Path(PROJECT_ROOT) / "features"
    n_stat = 128
    n_grad = 32

    wrapper = Dinov2Wrapper(head_layers=1, model_name="dinov2_vitl14",
                            device=device)
    report = {}
    for layer in ("blk05", "blk20"):
        print(f"\n{'=' * 64}\n  {layer}\n{'=' * 64}")
        layer_idx = int(layer[-2:])
        test_dir = feat_root / "test" / "dinov2_vitl14" / layer
        train_dir = feat_root / "train" / "dinov2_vitl14" / layer
        feats_test, _ = preload_features(sorted(test_dir.glob("*.npy"))[:n_stat],
                                         num_workers=2)
        feats_train, _ = preload_features(
            sorted(train_dir.glob("*.npy"))[:n_grad], num_workers=2)
        Xte = torch.from_numpy(np.stack(feats_test)).float().to(device)
        Xtr = torch.from_numpy(np.stack(feats_train)).float().to(device)

        int_c, _ = load_spatial_reassembly(ckpt_path(layer, False), device)
        gen_c, _ = load_spatial_reassembly(ckpt_path(layer, True), device)
        int_c.eval()
        gen_c.eval()

        s_int, y_int, xhat_int, x = stats_on_batch(int_c, Xte)
        s_gen, y_gen, xhat_gen, _ = stats_on_batch(gen_c, Xte)
        print("  integer", json.dumps(s_int, indent=None))
        print("  generic", json.dumps(s_gen, indent=None))

        cross = {
            "mse_Y_int_vs_gen": float(((y_int - y_gen) ** 2).mean()),
            "mse_xhat_int_vs_gen": float(((xhat_int - xhat_gen) ** 2).mean()),
        }
        # decode swap on maps
        x_ig, _ = gen_c.decode_map(y_int)
        x_gi, _ = int_c.decode_map(y_gen)
        cross["mse_intY_genDec_vs_input"] = float(((x_ig - x) ** 2).mean())
        cross["mse_genY_intDec_vs_input"] = float(((x_gi - x) ** 2).mean())
        cross["mse_int_vs_input"] = s_int["mse_to_input"]
        cross["mse_gen_vs_input"] = s_gen["mse_to_input"]
        print("  cross", json.dumps(cross, indent=None))

        # generic trained, swap only up_win to repeat
        gen_c.up_win = "repeat"
        s_rep, _, _, _ = stats_on_batch(gen_c, Xte)
        gen_c.up_win = "center"
        print("  generic+repeat_win mse_in", s_rep["mse_to_input"],
              "kl", s_rep["beta_kl_bili"])

        # integer trained, center windows
        int_c.up_win = "center"
        s_ctr, _, _, _ = stats_on_batch(int_c, Xte)
        int_c.up_win = "repeat"
        print("  integer+center_win mse_in", s_ctr["mse_to_input"],
              "kl", s_ctr["beta_kl_bili"])

        tail = FrozenTail(list(wrapper.backbone.blocks[layer_idx + 1:]),
                          wrapper.backbone.norm, device=device)
        # init grads
        g_int = SpatialReassemblyCodec(
            int_c.D, n_prefix=1, scale=2, k=5, Cr=64, grid=(16, 16)).to(device)
        g_int.set_forward_mode("integer")
        g_int._init_bilinear()
        g_gen = SpatialReassemblyCodec(
            int_c.D, n_prefix=1, scale=2, k=5, Cr=64, grid=(16, 16)).to(device)
        g_gen.set_forward_mode("generic")
        g_gen._init_bilinear()

        gn_int = grad_norms(g_int, Xtr, tail)
        gn_gen = grad_norms(g_gen, Xtr, tail)
        print("  init-grad integer", gn_int)
        print("  init-grad generic", gn_gen)
        ratio = {k: (gn_gen[k] / gn_int[k] if gn_int[k] else None)
                 for k in gn_int}
        print("  init-grad gen/int", ratio)

        # generic without 4-phase mean: use sum (4x residual scale) at init
        # (grad through mean is 1/4)
        report[layer] = {
            "integer_trained": s_int,
            "generic_trained": s_gen,
            "cross": cross,
            "generic_repeat_win": {
                "mse_to_input": s_rep["mse_to_input"],
                "beta_kl_bili": s_rep["beta_kl_bili"],
            },
            "integer_center_win": {
                "mse_to_input": s_ctr["mse_to_input"],
                "beta_kl_bili": s_ctr["beta_kl_bili"],
            },
            "init_grad_integer": gn_int,
            "init_grad_generic": gn_gen,
            "init_grad_ratio_gen_over_int": ratio,
        }
        tail.to("cpu")
        del int_c, gen_c, g_int, g_gen, Xte, Xtr
        torch.cuda.empty_cache()

    out = CKPT_DIR / "diagnose_generic_gap.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
