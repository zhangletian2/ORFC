"""Fully convolutional spatial analysis–synthesis (CARAFE-style).

Patch tokens only (CLS identity).  Channel dim D is unchanged.  No side
info: upsample kernels are predicted from the transmitted coarse map Y.

Down (scale s=2 by default):
    U0 = Conv1x1(LN(X)) ∈ R^{H×W×C_r}
    dilated 3×3 convs at full resolution, then stride-s to H'×W'
    L^↓ ∈ R^{H'×W'×k²},  α = softmax_δ L^↓
    Y_{u,d} = Σ_δ α_{u,δ} X_{π(u)+δ, d}

Up:
    V0 = Conv1x1(LN(Y)) ∈ R^{H'×W'×C_r}
    dilated 3×3 + global average pooling
    L^↑ ∈ R^{H×W×k²} via pixel-shuffle,  β = softmax_δ L^↑
    X̂_{t,d} = Σ_δ β_{t,δ}(Y) Y_{π'(t)+δ, d}

Last kernel-predictor layers are zero-weight + bilinear-logit bias, so
epoch-0 equals bilinear down/up (align_corners=False, replicate borders).
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ORFC_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "orfc"))
if _ORFC_DIR not in sys.path:
    sys.path.insert(0, _ORFC_DIR)

from opq import batch_normalize_gpu, batch_inv_normalize_gpu  # noqa: E402


LOGIT_FLOOR = -20.0


def infer_patch_hw(n_patch, grid=None):
    if grid is not None:
        if isinstance(grid, (int, float)):
            h = w = int(grid)
        else:
            h, w = int(grid[0]), int(grid[1])
        if h * w != n_patch:
            raise ValueError(f"grid {h}x{w} != n_patch={n_patch}")
        return h, w
    h = int(round(math.sqrt(n_patch)))
    if h * h != n_patch:
        raise ValueError(
            f"n_patch={n_patch} is not a square grid; pass grid=(H,W)")
    return h, h


def reshape_map(tokens, H, W):
    """``[B, T, D] → [B, D, H, W]``."""
    B, T, D = tokens.shape
    if T != H * W:
        raise ValueError(f"T={T} != {H}x{W}")
    return tokens.transpose(1, 2).reshape(B, D, H, W)


def flatten_map(feat):
    """``[B, D, H, W] → [B, HW, D]``."""
    B, D, H, W = feat.shape
    return feat.reshape(B, D, H * W).transpose(1, 2).contiguous()


def bilinear_down_kernel(k, scale=2, scale_h=None, scale_w=None):
    """``k×k`` kernel: uniform ``1/(sh·sw)`` on the aligned ``sh×sw`` block."""
    sh = int(scale if scale_h is None else scale_h)
    sw = int(scale if scale_w is None else scale_w)
    ker = torch.zeros(k, k)
    r0 = k // 2
    if r0 + sh > k or r0 + sw > k:
        raise ValueError(
            f"k={k} cannot hold {sh}x{sw} block at offset {r0}")
    ker[r0:r0 + sh, r0:r0 + sw] = 1.0 / float(sh * sw)
    return ker


def _linear_weights_1d(k, offset):
    """Bilinear 1-D taps.  ``offset`` is relative to window centre (index k//2)."""
    w = torch.zeros(k)
    i0 = int(math.floor(offset))
    t = offset - i0
    for off, val in ((i0, 1.0 - t), (i0 + 1, t)):
        idx = k // 2 + off
        if 0 <= idx < k:
            w[idx] += val
    return w


def bilinear_up_kernel(k, dy, dx, scale=2, scale_h=None, scale_w=None):
    """``k×k`` bilinear kernel for fine-pixel phase ``(dy, dx)``.

    Matches ``F.interpolate(..., mode='bilinear', align_corners=False)``:
    relative sample vs parent is ``(phase + 0.5) / s - 0.5``.
    """
    sh = int(scale if scale_h is None else scale_h)
    sw = int(scale if scale_w is None else scale_w)
    oy = (dy + 0.5) / sh - 0.5
    ox = (dx + 0.5) / sw - 0.5
    wy = _linear_weights_1d(k, oy)
    wx = _linear_weights_1d(k, ox)
    return wy[:, None] * wx[None, :]


def bilinear_up_logit_map(H, W, Hm, Wm, k, device, floor=LOGIT_FLOOR):
    """Per-location bilinear logits for ``gather_kxk`` around ``floor(c)``.

    ``c = (p+0.5)*n_src/n_dst - 0.5`` (align_corners=False).  Softmax of
    the map times those windows equals ``F.interpolate`` bilinear, including
    the clamp-to-border behaviour of ``gather_kxk``.
    """
    cy = mesh_centers(Hm, H, device).view(H, 1)
    cx = mesh_centers(Wm, W, device).view(1, W)
    ty = cy - torch.floor(cy)
    tx = cx - torch.floor(cx)
    c = k // 2
    wy = cy.new_zeros(H, W, k)
    wx = cx.new_zeros(H, W, k)
    wy[:, :, c] = 1.0 - ty
    wx[:, :, c] = 1.0 - tx
    if c + 1 < k:
        wy[:, :, c + 1] = ty.expand(H, W)
        wx[:, :, c + 1] = tx.expand(H, W)
    ker = wy.unsqueeze(-1) * wx.unsqueeze(-2)
    bias = torch.full_like(ker, float(floor))
    pos = ker > 0
    bias[pos] = torch.log(ker[pos].clamp_min(1e-8))
    return bias.reshape(H, W, k * k).permute(2, 0, 1).unsqueeze(0).contiguous()


def mix_uniform_kernel(ker, eps):
    """``K_0 = (1-ε) K + ε/k²``.  ``eps=0`` returns ``K`` unchanged."""
    if eps is None or float(eps) <= 0:
        return ker
    eps = float(eps)
    return (1.0 - eps) * ker + eps / float(ker.numel())


def span_normalize_logits(logits, M, dim=2):
    """Cap per-position logit span before softmax.

    ``l̃ = (l - mean(l)) / max(1, (max(l)-min(l))/M)``.
    ``M<=0`` returns ``logits`` unchanged.  Mean/max/min are over ``dim``
    (the k² kernel axis).  Softmax is shift-invariant, so centering
    alone does not change weights; the divisor limits peakiness to
    span ≤ M.
    """
    if M is None or float(M) <= 0:
        return logits
    M = float(M)
    l_mean = logits.mean(dim=dim, keepdim=True)
    span = (logits.max(dim=dim, keepdim=True).values
            - logits.min(dim=dim, keepdim=True).values)
    denom = (span / M).clamp(min=1.0)
    return (logits - l_mean) / denom


def kernel_to_bias(ker, floor=LOGIT_FLOOR):
    """``log(w)`` on support, ``floor`` elsewhere → softmax ≈ ``w``."""
    bias = torch.full_like(ker, float(floor))
    pos = ker > 0
    bias[pos] = torch.log(ker[pos].clamp_min(1e-8))
    return bias


def unfold_windows(feat, k, stride):
    """``[B, D, H, W] → [B, D, k², H_out, W_out]`` with replicate pad."""
    pad = k // 2
    xp = F.pad(feat, (pad, pad, pad, pad), mode="replicate")
    B, D, H, W = feat.shape
    if isinstance(stride, (tuple, list)):
        stride_h, stride_w = int(stride[0]), int(stride[1])
    else:
        stride_h = stride_w = int(stride)
    H_out = (H + 2 * pad - k) // stride_h + 1
    W_out = (W + 2 * pad - k) // stride_w + 1
    unf = F.unfold(xp, kernel_size=k, stride=(stride_h, stride_w))
    return unf.view(B, D, k * k, H_out, W_out)


def phase_shuffle(x, scale_h, scale_w):
    """Anisotropic pixel-shuffle: ``[B, C·sh·sw, H, W] → [B, C, H·sh, W·sw]``.

    Channel layout matches ``F.pixel_shuffle``: ``c * sh * sw + dy * sw + dx``.
    """
    sh, sw = int(scale_h), int(scale_w)
    if sh == sw:
        return F.pixel_shuffle(x, sh)
    B, ch, H, W = x.shape
    nph = sh * sw
    if ch % nph != 0:
        raise ValueError(f"channels {ch} not divisible by {sh}x{sw}")
    C = ch // nph
    x = x.view(B, C, sh, sw, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(B, C, H * sh, W * sw)


def mesh_centers(n_src, n_dst, device):
    """Map destination index ``p`` to source pixel coord (align_corners=False)."""
    p = torch.arange(n_dst, device=device, dtype=torch.float32)
    return (p + 0.5) * float(n_src) / float(n_dst) - 0.5


def gather_kxk(feat, cy, cx, k):
    """Integer ``k×k`` windows around ``floor(c)``.

    ``feat [B,D,H,W]``, ``cy/cx [Ho,Wo]`` (pixel coords) → ``[B,D,k²,Ho,Wo]``.
    Clamp to the map (same border behaviour as replicate-pad unfold).
    """
    B, D, H, W = feat.shape
    off = torch.arange(k, device=feat.device, dtype=torch.float32) - (k // 2)
    iy = torch.floor(cy).unsqueeze(-1).unsqueeze(-1) + off.view(k, 1)
    ix = torch.floor(cx).unsqueeze(-1).unsqueeze(-1) + off.view(1, k)
    iy, ix = torch.broadcast_tensors(iy, ix)
    iy = iy.long().clamp(0, H - 1)
    ix = ix.long().clamp(0, W - 1)
    gathered = feat[:, :, iy, ix]
    Ho, Wo = cy.shape
    return gathered.reshape(B, D, Ho, Wo, k * k).permute(0, 1, 4, 2, 3).contiguous()


FORWARD_MODES = {
    "integer": dict(down_pool="avg", down_win="stride",
                    up_kern="shuffle", up_win="repeat"),
    "down_pool": dict(down_pool="bilinear", down_win="stride",
                      up_kern="shuffle", up_win="repeat"),
    "down_win": dict(down_pool="avg", down_win="center",
                     up_kern="shuffle", up_win="repeat"),
    "up_kern": dict(down_pool="avg", down_win="stride",
                    up_kern="interp", up_win="repeat"),
    "up_win": dict(down_pool="avg", down_win="stride",
                   up_kern="shuffle", up_win="center"),
    "generic": dict(down_pool="bilinear", down_win="center",
                    up_kern="interp", up_win="center"),
}


class _Routing(nn.Module):
    """LN → 1×1 compress → dilated 3×3 convs.  Stays at input spatial size."""

    def __init__(self, D, Cr=64, dilations=(1, 2)):
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.compress = nn.Conv2d(D, Cr, kernel_size=1, bias=True)
        layers = []
        for d in dilations:
            layers.extend([
                nn.Conv2d(Cr, Cr, kernel_size=3, padding=d, dilation=d, bias=True),
                nn.ReLU(inplace=True),
            ])
        self.enc = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, D, H, W]
        u = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.enc(self.compress(u))


class SpatialReassemblyCodec(nn.Module):
    """Content-adaptive spatial down/up, channel D unchanged."""

    def __init__(self, D, n_prefix=1, scale=2, k=5, Cr=64, grid=None,
                 dilations=(1, 2), scale_h=None, scale_w=None, init_eps=0.0,
                 n_groups=1, logit_span=0.0):
        super().__init__()
        if k % 2 == 0:
            raise ValueError(f"k must be odd, got {k}")
        n_groups = int(n_groups)
        if n_groups < 1:
            raise ValueError(f"n_groups must be >= 1, got {n_groups}")
        D = int(D)
        if D % n_groups != 0:
            raise ValueError(f"D={D} not divisible by n_groups={n_groups}")
        if scale_h is None and scale_w is None:
            if isinstance(scale, (tuple, list)):
                scale_h, scale_w = int(scale[0]), int(scale[1])
            else:
                scale_h = scale_w = int(scale)
        else:
            scale_h = int(scale if scale_h is None else scale_h)
            scale_w = int(scale if scale_w is None else scale_w)
        if scale_h < 1 or scale_w < 1:
            raise ValueError(f"scale_h={scale_h}, scale_w={scale_w} must be >= 1")
        if scale_h * scale_w < 2:
            raise ValueError(
                f"need scale_h*scale_w >= 2, got {scale_h}x{scale_w}")
        self.D = D
        self.n_prefix = int(n_prefix)
        self.scale_h = scale_h
        self.scale_w = scale_w
        # isotropic alias used by old ckpts / logs
        self.scale = scale_h if scale_h == scale_w else (scale_h, scale_w)
        self.k = int(k)
        self.Cr = int(Cr)
        self.grid = grid
        self.init_eps = float(init_eps)
        self.n_groups = n_groups
        self.group_channels = D // n_groups
        self.logit_span = float(logit_span)
        kk = self.k * self.k
        nph = scale_h * scale_w

        self.route_down = _Routing(D, Cr=Cr, dilations=dilations)
        self.pred_down = nn.Conv2d(Cr, n_groups * kk, kernel_size=1, bias=True)

        self.route_up = _Routing(D, Cr=Cr, dilations=dilations)
        self.global_fc = nn.Sequential(
            nn.Conv2d(Cr, Cr, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        # G groups × sh·sw phases of k² at coarse res → shuffle to H×W
        self.pred_up = nn.Conv2d(
            Cr, n_groups * nph * kk, kernel_size=1, bias=True)

        self.set_forward_mode("integer")
        self._init_bilinear()

    def set_forward_mode(self, name):
        """Switch integer vs generic sampling.  Weights unchanged."""
        if name not in FORWARD_MODES:
            raise ValueError(f"unknown forward mode {name!r}")
        for key, val in FORWARD_MODES[name].items():
            setattr(self, key, val)
        self.forward_mode = name
        return self

    def _init_bilinear(self):
        """Zero last-layer weights; bilinear logits so epoch-0 ≈ interpolate.

        Integer (shuffle): ``sh·sw`` phase kernels in ``pred_up.bias``.
        If ``init_eps>0``, mix uniform mass
        ``K_0=(1-ε)K_bili+ε/k²`` so off-support taps keep a small
        softmax probability (no ``-20`` floor).  ``ε=0`` is the old
        hard bilinear support.

        Generic (interp): ``pred_up.bias=0``; position-dependent bilinear
        prior is added in ``_up_logits`` (phase-average cannot represent
        bilinear upsample).
        """
        nn.init.zeros_(self.pred_down.weight)
        nn.init.zeros_(self.pred_up.weight)
        down_ker = mix_uniform_kernel(
            bilinear_down_kernel(
                self.k, scale_h=self.scale_h, scale_w=self.scale_w),
            self.init_eps)
        down_bias = kernel_to_bias(down_ker).reshape(-1)
        self.pred_down.bias.data.copy_(down_bias.repeat(self.n_groups))

        if getattr(self, "up_kern", "shuffle") == "interp":
            nn.init.zeros_(self.pred_up.bias)
            return

        sh, sw, kk = self.scale_h, self.scale_w, self.k * self.k
        nph = sh * sw
        # phase-shuffle: out[c, h*sh+dy, w*sw+dx] = in[c*nph + dy*sw + dx, h, w]
        # c = g * k² + tap, G copies of the same bilinear phase kernels.
        bias = torch.zeros(nph * kk)
        for dy in range(sh):
            for dx in range(sw):
                ker = mix_uniform_kernel(
                    bilinear_up_kernel(
                        self.k, dy, dx, scale_h=sh, scale_w=sw),
                    self.init_eps)
                b = kernel_to_bias(ker).reshape(-1)
                for c in range(kk):
                    bias[c * nph + dy * sw + dx] = b[c]
        self.pred_up.bias.data.copy_(bias.repeat(self.n_groups))

    def _hw(self, n_patch):
        return infer_patch_hw(n_patch, self.grid)

    def coded_tokens(self, T_full=None):
        if T_full is None:
            return None
        n_patch = int(T_full) - self.n_prefix
        H, W = self._hw(n_patch)
        return self.n_prefix + (H // self.scale_h) * (W // self.scale_w)

    def _coarse_hw(self, H, W):
        sh, sw = self.scale_h, self.scale_w
        if H % sh or W % sw:
            raise ValueError(
                f"H={H}, W={W} must be divisible by scale={sh}x{sw}")
        return H // sh, W // sw

    def _down_logits(self, x, Hm, Wm):
        u = self.route_down(x)
        sh, sw = self.scale_h, self.scale_w
        if self.down_pool == "avg":
            u = F.avg_pool2d(u, kernel_size=(sh, sw), stride=(sh, sw))
        elif self.down_pool == "bilinear":
            u = F.interpolate(
                u, size=(Hm, Wm), mode="bilinear", align_corners=False)
        else:
            raise ValueError(f"down_pool={self.down_pool!r}")
        return self.pred_down(u)

    def _up_logits(self, y, H, W):
        v = self.route_up(y)
        g = self.global_fc(v.mean(dim=(2, 3), keepdim=True))
        logits_low = self.pred_up(v + g)
        sh, sw = self.scale_h, self.scale_w
        if self.up_kern == "shuffle":
            return phase_shuffle(logits_low, sh, sw)
        if self.up_kern == "interp":
            B = logits_low.shape[0]
            kk = self.k * self.k
            G = self.n_groups
            Hm, Wm = y.shape[-2:]
            nph = sh * sw
            collapsed = logits_low.view(B, G * kk, nph, Hm, Wm)
            collapsed = collapsed.mean(dim=2)
            learned = F.interpolate(
                collapsed, size=(H, W), mode="bilinear", align_corners=False)
            prior = bilinear_up_logit_map(
                H, W, Hm, Wm, self.k, device=y.device).to(dtype=learned.dtype)
            if G > 1:
                prior = prior.repeat(1, G, 1, 1)
            return learned + prior
        raise ValueError(f"up_kern={self.up_kern!r}")

    def _down_windows(self, x, Hm, Wm):
        if self.down_win == "stride":
            return unfold_windows(
                x, self.k, stride=(self.scale_h, self.scale_w))
        if self.down_win == "center":
            H, W = x.shape[-2:]
            cy, cx = torch.meshgrid(
                mesh_centers(H, Hm, x.device),
                mesh_centers(W, Wm, x.device),
                indexing="ij")
            return gather_kxk(x, cy, cx, self.k)
        raise ValueError(f"down_win={self.down_win!r}")

    def _up_windows(self, y, H, W):
        sh, sw = self.scale_h, self.scale_w
        if self.up_win == "repeat":
            win = unfold_windows(y, self.k, stride=1)
            return win.repeat_interleave(sh, dim=3).repeat_interleave(sw, dim=4)
        if self.up_win == "center":
            Hm, Wm = y.shape[-2:]
            cy, cx = torch.meshgrid(
                mesh_centers(Hm, H, y.device),
                mesh_centers(Wm, W, y.device),
                indexing="ij")
            return gather_kxk(y, cy, cx, self.k)
        raise ValueError(f"up_win={self.up_win!r}")

    def _grouped_reassemble(self, win, logits):
        """Apply per-group softmax kernels.

        ``win [B,D,k²,H,W]``, ``logits [B,G·k²,H,W]``
        → ``feat [B,D,H,W]``, ``alpha [B,G,k²,H,W]``.
        """
        B, D, kk, H, W = win.shape
        G = self.n_groups
        if D != self.D:
            raise RuntimeError(f"win D={D} != codec D={self.D}")
        if logits.shape[1] != G * kk:
            raise RuntimeError(
                f"logits C={logits.shape[1]} != G·k²={G * kk}")
        Dg = self.group_channels
        l = span_normalize_logits(
            logits.view(B, G, kk, H, W), self.logit_span, dim=2)
        alpha = torch.softmax(l, dim=2)
        feat = (win.view(B, G, Dg, kk, H, W) * alpha.unsqueeze(2)).sum(dim=3)
        return feat.reshape(B, D, H, W), alpha

    def encode_map(self, x):
        """``x [B,D,H,W] → y [B,D,H',W'], alpha [B,G,k²,H',W']``."""
        H, W = x.shape[-2:]
        Hm, Wm = self._coarse_hw(H, W)
        logits = self._down_logits(x, Hm, Wm)
        win = self._down_windows(x, Hm, Wm)
        if win.shape[-2:] != (Hm, Wm):
            raise RuntimeError(
                f"down window {tuple(win.shape[-2:])} != {(Hm, Wm)}")
        y, alpha = self._grouped_reassemble(win, logits)
        return y, alpha

    def decode_map(self, y):
        """``y [B,D,H',W'] → xhat [B,D,H,W], beta [B,G,k²,H,W]``."""
        H = y.shape[-2] * self.scale_h
        W = y.shape[-1] * self.scale_w
        logits = self._up_logits(y, H, W)
        win_up = self._up_windows(y, H, W)
        if win_up.shape[-2:] != logits.shape[-2:]:
            raise RuntimeError(
                f"up window {tuple(win_up.shape[-2:])} != "
                f"logits {tuple(logits.shape[-2:])}")
        xhat, beta = self._grouped_reassemble(win_up, logits)
        return xhat, beta

    def encode(self, Y):
        p = self.n_prefix
        patch = Y[:, p:, :]
        n_patch = patch.shape[1]
        H, W = self._hw(n_patch)
        x = reshape_map(patch, H, W)
        y, alpha = self.encode_map(x)
        Hm, Wm = y.shape[-2:]
        body = flatten_map(y)
        seq = torch.cat([Y[:, :p, :], body], dim=1) if p else body
        aux = {
            "H": H, "W": W, "Hm": Hm, "Wm": Wm, "n_patch": n_patch,
            "alpha_pmax": alpha.max(dim=2).values.mean().detach(),
        }
        return seq, aux

    def decode(self, seq, aux):
        p = self.n_prefix
        Hm, Wm = aux["Hm"], aux["Wm"]
        body = seq[:, p:, :]
        y = reshape_map(body, Hm, Wm)
        xhat, beta = self.decode_map(y)
        patch = flatten_map(xhat)
        aux["beta_pmax"] = beta.max(dim=2).values.mean().detach()
        if p == 0:
            return patch
        return torch.cat([seq[:, :p, :], patch], dim=1)

    def forward(self, Y, **_kwargs):
        seq, aux = self.encode(Y)
        return self.decode(seq, aux), aux

    @torch.no_grad()
    def bilinear_roundtrip_map(self, x):
        """Reference bilinear down/up (same sizes as this codec)."""
        H, W = x.shape[-2:]
        Hm, Wm = H // self.scale_h, W // self.scale_w
        y = F.interpolate(x, size=(Hm, Wm), mode="bilinear", align_corners=False)
        xhat = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        return xhat, y


def save_spatial_reassembly(codec, path, meta_extra=None):
    meta = {
        "format": "spatial_reassembly_v1",
        "D": codec.D,
        "n_prefix": codec.n_prefix,
        "scale": codec.scale,
        "scale_h": codec.scale_h,
        "scale_w": codec.scale_w,
        "init_eps": getattr(codec, "init_eps", 0.0),
        "n_groups": getattr(codec, "n_groups", 1),
        "logit_span": getattr(codec, "logit_span", 0.0),
        "k": codec.k,
        "Cr": codec.Cr,
        "grid": codec.grid,
        "forward_mode": getattr(codec, "forward_mode", "integer"),
        "state_dict": codec.state_dict(),
    }
    if meta_extra:
        meta.update(meta_extra)
    torch.save(meta, path)


def load_spatial_reassembly(path, device="cuda"):
    meta = torch.load(path, map_location="cpu")
    fmt = meta.get("format")
    if fmt not in (None, "spatial_reassembly_v1"):
        raise ValueError(f"unexpected format {fmt}")
    scale_meta = meta.get("scale", 2)
    if "scale_h" in meta:
        scale_h, scale_w = int(meta["scale_h"]), int(meta["scale_w"])
    elif isinstance(scale_meta, (list, tuple)):
        scale_h, scale_w = int(scale_meta[0]), int(scale_meta[1])
    else:
        scale_h = scale_w = int(scale_meta)
    codec = SpatialReassemblyCodec(
        D=meta["D"], n_prefix=int(meta.get("n_prefix", 1)),
        scale_h=scale_h, scale_w=scale_w, k=int(meta.get("k", 5)),
        Cr=int(meta.get("Cr", 64)), grid=meta.get("grid"),
        init_eps=float(meta.get("init_eps", 0.0)),
        n_groups=int(meta.get("n_groups", 1)),
        logit_span=float(meta.get("logit_span", 0.0)))
    codec.load_state_dict(meta["state_dict"])
    codec.set_forward_mode(meta.get("forward_mode", "integer"))
    return codec.to(device).eval(), meta


def train_spatial_reassembly(
    features_train,
    tail,
    D,
    n_prefix=1,
    scale=2,
    scale_h=None,
    scale_w=None,
    k=5,
    Cr=64,
    grid=None,
    norm_mode="per_image",
    epochs=100,
    lr=3e-4,
    batch_size=32,
    device="cuda",
    seed=42,
    val_features=None,
    verbose=True,
    grad_clip=1.0,
    forward_mode="integer",
    init_eps=0.0,
    n_groups=1,
    logit_span=0.0,
    init_codec=None,
):
    """Train down/up reassembly with ΔL_ref.  No R, no PQ.

    ``init_codec``: optional warm-start module (already on ``device``).
    Bilinear init is skipped when it is provided.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if isinstance(features_train, np.ndarray) and features_train.ndim == 3:
        features_array = features_train
    else:
        features_array = np.stack(features_train)
    N_img = features_array.shape[0]
    T_full = features_array.shape[1]
    if D is None:
        D = features_array.shape[2]
    if init_codec is not None:
        merge = init_codec
        merge.logit_span = float(logit_span)
        merge.set_forward_mode(forward_mode)
    else:
        merge = SpatialReassemblyCodec(
            D, n_prefix=n_prefix, scale=scale, scale_h=scale_h,
            scale_w=scale_w, k=k, Cr=Cr, grid=grid, init_eps=init_eps,
            n_groups=n_groups, logit_span=logit_span).to(device)
        merge.set_forward_mode(forward_mode)
        merge._init_bilinear()
    n_patch = T_full - n_prefix
    H, W = merge._hw(n_patch)
    Tm = merge.coded_tokens(T_full)
    sh, sw = merge.scale_h, merge.scale_w

    trainable = [p for p in merge.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=lr * 0.01)
    if verbose:
        n_tr = sum(p.numel() for p in trainable)
        print(f"  Trainable params: {n_tr:,}  (spatial reassembly, no PQ)")
        print(f"  grid={H}x{W} -> {H // sh}x{W // sw}  ({sh}x{sw})  "
              f"Tm={Tm}/{T_full}  k={k} Cr={Cr}  G={merge.n_groups}  "
              f"mode={forward_mode}  init_eps={init_eps}  "
              f"spanM={merge.logit_span:g}  "
              f"{'warm-start' if init_codec is not None else 'from-init'}  "
              f"objective: ΔL_ref")

    teacher_cache = np.empty_like(features_array)
    with torch.no_grad():
        for s in range(0, N_img, batch_size):
            e = min(s + batch_size, N_img)
            Xc = torch.from_numpy(features_array[s:e]).float().to(device)
            teacher_cache[s:e] = tail.forward_nograd(Xc).cpu().numpy()
            del Xc
    torch.cuda.empty_cache()

    val_array = np.stack(val_features) if val_features else None
    history = []
    for epoch in range(epochs):
        t_ep = time.time()
        perm = np.random.permutation(N_img)
        total_d = 0.0
        merge.train()
        for s in range(0, N_img, batch_size):
            idx = perm[s:s + batch_size]
            X = torch.from_numpy(features_array[idx]).float().to(device)
            B = X.shape[0]
            with torch.no_grad():
                Y, Mu, Std = batch_normalize_gpu(
                    X, mode=norm_mode, n_prefix=n_prefix)
            Y_hat, _ = merge(Y)
            Y_teacher = torch.from_numpy(teacher_cache[idx]).float().to(device)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            distortion = ((Y_teacher - tail(X_hat)) ** 2).sum() / B
            optimizer.zero_grad()
            distortion.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            total_d += distortion.item() * B
            del X, Y, Mu, Std, Y_hat, Y_teacher, X_hat, distortion
        scheduler.step()

        avg_d = total_d / N_img
        val_loss = None
        if val_array is not None:
            merge.eval()
            vs_sum = 0.0
            n_val = val_array.shape[0]
            with torch.no_grad():
                for vs in range(0, n_val, batch_size):
                    ve = min(vs + batch_size, n_val)
                    Xv = torch.from_numpy(val_array[vs:ve]).float().to(device)
                    Yv, Muv, Stdv = batch_normalize_gpu(
                        Xv, mode=norm_mode, n_prefix=n_prefix)
                    Yhv, _ = merge(Yv)
                    Ytv = tail.forward_nograd(Xv)
                    Xhv = batch_inv_normalize_gpu(Yhv, Muv, Stdv)
                    vs_sum += ((Ytv - tail.forward_nograd(Xhv)) ** 2).sum().item()
                    del Xv, Yv, Muv, Stdv, Yhv, Ytv, Xhv
            val_loss = vs_sum / n_val

        info = {
            "epoch": epoch,
            "loss_distortion": avg_d,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
            "time": time.time() - t_ep,
        }
        history.append(info)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1 or epochs <= 20):
            vstr = f"  val={val_loss:.1f}" if val_loss is not None else ""
            print(f"  ep {epoch:3d}/{epochs}  D={avg_d:.1f}{vstr}  "
                  f"({info['time']:.1f}s)")
    return merge, history
