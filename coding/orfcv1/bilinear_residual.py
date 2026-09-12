"""Fixed 4× spatial codec + 4-channel residual guidance.

Main path:

    Z = E(X_patch)                   Conv2d(D,C,k,s=2)  k=2 or 3
    seq = [CLS, Z]                   65 tokens
    X_0 = U(Z)                       ConvTranspose2d(C,D,k,s=2)

Stage-1 pretrain: no ORFC / PQ.  Residual is patch tokens only:

    R = X - X_0
    G = R W_g                        Linear(D, c, bias=False)
    hat_X = X_0 + F_φ(X_0, G)

Ablations (decoder never sees matching indices):
    main    hat_X = U(E(X))
    recon0  hat_X = X_0 + F_φ(X_0, 0)
    full    hat_X = X_0 + F_φ(X_0, G)
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
_ORFC_DIR = os.path.normpath(os.path.join(HERE, "..", "orfc"))
if _ORFC_DIR not in sys.path:
    sys.path.insert(0, _ORFC_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from opq import batch_normalize_gpu  # noqa: E402
# -- inlined from spatial_reassembly.py --

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
    B, T, D = tokens.shape
    if T != H * W:
        raise ValueError(f"T={T} != {H}x{W}")
    return tokens.transpose(1, 2).reshape(B, D, H, W)


def flatten_map(feat):
    B, D, H, W = feat.shape
    return feat.reshape(B, D, H * W).transpose(1, 2).contiguous()

_GRID_HINT = None


def set_grid_hint(grid):
    """Per-image true patch grid, set by the eval loop from meta['token_hw'].

    Both BilinearSpatialCodec and ConvRecon call infer_token_hw; routing the
    truth through a hint avoids plumbing a grid arg through every forward.
    """
    global _GRID_HINT
    _GRID_HINT = None if grid is None else (int(grid[0]), int(grid[1]))


def infer_token_hw(n_patch, grid=None):
    """Patch grid for bilinear down/up.  Accepts non-square / odd sizes."""
    n_patch = int(n_patch)
    if grid is None and _GRID_HINT is not None and \
            _GRID_HINT[0] * _GRID_HINT[1] == n_patch:
        grid = _GRID_HINT
    if grid is not None:
        return infer_patch_hw(n_patch, grid)
    h = int(round(math.sqrt(n_patch)))
    if h * h == n_patch:
        return h, h
    # Anything non-square needs the real grid.  Factoring it here is a guess
    # that used to transpose non-square images silently.
    raise ValueError(
        f"n_patch={n_patch} is not square; pass grid= or call set_grid_hint()")

INTERP_KW = dict(mode="bilinear", align_corners=False)
N_LEVELS = 4
BITS_PER_CHANNEL = 2
SCALE_EPS = 1e-8
RESIDUAL_ABLATIONS = ("full", "main", "recon0")


def freeze_module(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


class BilinearSpatialCodec(nn.Module):
    """CLS identity + patch down/up with optional channel reduction.

    ``C``: latent channel width (default ``None`` -> same as ``D``).
    ``down``/``up``: ``'conv2'`` (2×2 stride 2) or ``'conv3'`` (3×3 stride 2 pad 1).
    Both produce the same output spatial size (16×16 → 8×8).
    CLS prefix: ``cls_mode='learned'`` uses ``Linear`` projections;
    ``cls_mode='identity'`` slices/zero-pads.  When ``C == D`` there is nothing
    to project, so the prefix passes through untouched (``'none'``).
    """

    _VALID_MODES = ("conv2", "conv3")

    def __init__(self, D, n_prefix=1, scale=2, grid=None,
                 down="conv2", up="conv2", C=None, cls_mode="learned"):
        super().__init__()
        if down not in self._VALID_MODES:
            raise ValueError(f"unknown spatial down {down!r}")
        if up not in self._VALID_MODES:
            raise ValueError(f"unknown spatial up {up!r}")
        if int(scale) != 2:
            raise ValueError("conv down/up requires scale=2")
        self.D = int(D)
        self.C = int(C) if C is not None else self.D
        self.n_prefix = int(n_prefix)
        self.scale_h = int(scale)
        self.scale_w = int(scale)
        self.scale = int(scale)
        self.grid = grid
        self.down = down
        self.up = up

        ks_e, pad_e = (2, 0) if down == "conv2" else (3, 1)
        ks_d, pad_d = (2, 0) if up == "conv2" else (3, 1)
        opad_d = 1 if up == "conv3" else 0

        self.analysis = nn.Conv2d(
            self.D, self.C, kernel_size=ks_e, stride=2,
            padding=pad_e, bias=False)
        if self.C == self.D:
            self.init_avgpool_analysis()

        self.synthesis = nn.ConvTranspose2d(
            self.C, self.D, kernel_size=ks_d, stride=2,
            padding=pad_d, output_padding=opad_d, bias=False)
        if self.C == self.D:
            self.init_repeat_synthesis()

        self.cls_mode = cls_mode if self.C != self.D else "none"
        self.prefix_down = None
        self.prefix_up = None
        if self.C != self.D and self.cls_mode == "learned":
            self.prefix_down = nn.Linear(self.D, self.C, bias=False)
            self.prefix_up = nn.Linear(self.C, self.D, bias=False)


    def init_avgpool_analysis(self):
        """Channel-wise mean-pool init: each output = mean of the receptive field."""
        if self.analysis is None or self.C != self.D:
            return
        ks = self.analysis.kernel_size[0]
        val = 1.0 / (ks * ks)
        with torch.no_grad():
            self.analysis.weight.zero_()
            idx = torch.arange(self.D)
            self.analysis.weight[idx, idx, :, :] = val

    def init_repeat_synthesis(self):
        """Channel-wise copy: each coarse value fills its receptive-field block."""
        if self.synthesis is None or self.C != self.D:
            return
        with torch.no_grad():
            self.synthesis.weight.zero_()
            idx = torch.arange(self.D)
            self.synthesis.weight[idx, idx, :, :] = 1.0

    def init_orthogonal_projection(self, seed=None):
        """Paired random orthogonal init for ``C < D``.

        Generates column-orthogonal ``P`` (D x C, ``P^T P = I_C``) and sets
        each spatial tap of E to ``P^T / k^2`` and of U to ``P^T``.
        """
        if self.C >= self.D or self.analysis is None or self.synthesis is None:
            return
        if seed is not None:
            torch.manual_seed(seed)
        P = torch.linalg.qr(torch.randn(self.D, self.C))[0]
        Pt = P.t().contiguous()
        ks_e = self.analysis.kernel_size[0]
        val = 1.0 / (ks_e * ks_e)
        with torch.no_grad():
            for i in range(ks_e):
                for j in range(ks_e):
                    self.analysis.weight.data[:, :, i, j] = val * Pt
            ks_d = self.synthesis.kernel_size[0]
            for i in range(ks_d):
                for j in range(ks_d):
                    self.synthesis.weight.data[:, :, i, j] = Pt


    def _pad_hw(self, H, W):
        """Replicate-pad so ``H,W`` are multiples of the stride."""
        pad_h = (self.scale_h - H % self.scale_h) % self.scale_h
        pad_w = (self.scale_w - W % self.scale_w) % self.scale_w
        return pad_h, pad_w

    def coded_tokens(self, T_full=None):
        if T_full is None:
            return None
        n_patch = int(T_full) - self.n_prefix
        H, W = infer_token_hw(n_patch, self.grid)
        pad_h, pad_w = self._pad_hw(H, W)
        Hm = (H + pad_h) // self.scale_h
        Wm = (W + pad_w) // self.scale_w
        prefix_in_seq = 0 if self.cls_mode == "identity" else self.n_prefix
        return prefix_in_seq + Hm * Wm

    def encode(self, Y, groups=None):
        """Encode spatial tokens, optionally reducing channels D -> C."""
        del groups
        p = self.n_prefix
        prefix, patch = Y[:, :p], Y[:, p:]
        H, W = infer_token_hw(patch.shape[1], self.grid)
        x = reshape_map(patch, H, W)
        pad_h, pad_w = self._pad_hw(H, W)
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        z = self.analysis(x)
        Hm, Wm = int(z.shape[-2]), int(z.shape[-1])
        aux = {
            "H": H, "W": W, "Hm": Hm, "Wm": Wm, "n_patch": patch.shape[1],
            "pad_h": pad_h, "pad_w": pad_w,
        }
        if self.cls_mode == "identity":
            aux["prefix"] = prefix
            seq = flatten_map(z)
        else:
            if self.prefix_down is not None:
                prefix = self.prefix_down(prefix)
            seq = torch.cat([prefix, flatten_map(z)], dim=1)
        return seq, aux

    def decode(self, seq, aux):
        """Decode spatial tokens back to D-dim, restore CLS."""
        if self.cls_mode == "identity":
            prefix = aux["prefix"]
            coarse = seq
        else:
            p = self.n_prefix
            prefix = seq[:, :p]
            if self.prefix_up is not None:
                prefix = self.prefix_up(prefix)
            coarse = seq[:, p:]
        H, W, Hm, Wm = aux["H"], aux["W"], aux["Hm"], aux["Wm"]
        if coarse.shape[1] != Hm * Wm:
            raise ValueError(
                f"coarse tokens {coarse.shape[1]} != {Hm}x{Wm}")
        z = reshape_map(coarse, Hm, Wm)
        xhat = self.synthesis(z)
        oh, ow = int(xhat.shape[-2]), int(xhat.shape[-1])
        if oh >= H and ow >= W:
            xhat = xhat[..., :H, :W]
        elif (oh, ow) != (H, W):
            xhat = F.interpolate(xhat, size=(H, W), **INTERP_KW)
        return torch.cat([prefix, flatten_map(xhat)], dim=1)

    def forward(self, Y, groups=None):
        seq, aux = self.encode(Y, groups)
        patch_seq = seq if self.cls_mode == "identity" else seq[:, self.n_prefix:]
        return self.decode(seq, aux), patch_seq

def four_level_quantize(G, scale, ste=False):
    """Hard 4-level mid-riser.  ``scale`` broadcasts over the last dim.

    ``q = clip(floor(G/s) + 2, 0, 3)``,  ``hat = s (q - 1.5)``.
    Reconstruction levels: ``{-1.5s, -0.5s, +0.5s, +1.5s}``.
    """
    s = scale.to(dtype=G.dtype, device=G.device)
    q = torch.floor(G / s) + 2
    q = q.clamp(0, N_LEVELS - 1)
    hat = s * (q - 1.5)
    if ste:
        hat = hat + (G - G.detach())
    return hat, q


def pack_two_bit_indices(q):
    """Pack values in ``{0,1,2,3}`` at 2 bits each.  Returns ``bytes``."""
    flat = np.asarray(q, dtype=np.uint8).ravel()
    n = int(flat.size)
    pad = (-n) % 4
    if pad:
        flat = np.pad(flat, (0, pad))
    packed = (
        flat[0::4]
        | (flat[1::4] << 2)
        | (flat[2::4] << 4)
        | (flat[3::4] << 6)
    ).astype(np.uint8)
    return packed.tobytes()


def unpack_two_bit_indices(buf, shape):
    """Inverse of ``pack_two_bit_indices``."""
    n = int(np.prod(shape))
    pad = (-n) % 4
    n_pad = n + pad
    packed = np.frombuffer(buf, dtype=np.uint8)[: n_pad // 4]
    out = np.empty(n_pad, dtype=np.uint8)
    out[0::4] = packed & 3
    out[1::4] = (packed >> 2) & 3
    out[2::4] = (packed >> 4) & 3
    out[3::4] = (packed >> 6) & 3
    return out[:n].reshape(shape)


def overflow_mask(G, scale):
    s = scale.to(dtype=G.dtype, device=G.device)
    return G.abs() > (2.0 * s)


class FourLevelScalarQuantizer(nn.Module):
    """Per-channel 2-bit scalar quantizer.  ``scale`` is a frozen buffer."""

    def __init__(self, c, scale=None):
        super().__init__()
        self.c = int(c)
        if scale is None:
            scale = torch.ones(self.c)
        else:
            scale = torch.as_tensor(scale, dtype=torch.float32).reshape(-1)
            if scale.numel() != self.c:
                raise ValueError(f"scale length {scale.numel()} != c={self.c}")
        self.register_buffer("scale", scale.clamp_min(SCALE_EPS).contiguous())

    def set_scale(self, scale):
        scale = torch.as_tensor(scale, dtype=torch.float32, device=self.scale.device)
        if scale.numel() != self.c:
            raise ValueError(f"scale length {scale.numel()} != c={self.c}")
        self.scale.copy_(scale.reshape(-1).clamp_min(SCALE_EPS))

    def forward(self, G):
        ste = self.training
        return four_level_quantize(G, self.scale, ste=ste)


class ResidualConvRecon(nn.Module):
    """``F_θ(X_0, G)`` on the patch grid.  Hidden width is ``D``.

    concat[X0, G] → 1×1 (D+c→D) → GELU → DW 3×3 + skip → 1×1 (D→D).
    Last 1×1 is zero-init so ``hat X = X_0`` at step 0.  Decoder uses only
    ``X_0`` and ``G`` (no matching indices).
    """

    def __init__(self, D, c):
        super().__init__()
        self.D = int(D)
        self.c = int(c)
        self.mix = nn.Conv2d(self.D + self.c, self.D, kernel_size=1, bias=False)
        self.dw = nn.Conv2d(
            self.D, self.D, kernel_size=3, padding=1, groups=self.D, bias=False)
        self.proj = nn.Conv2d(self.D, self.D, kernel_size=1, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, X0, G):
        if X0.shape[:2] != G.shape[:2]:
            raise ValueError(
                f"X0 {tuple(X0.shape)} vs G {tuple(G.shape)} token mismatch")
        if X0.shape[-1] != self.D or G.shape[-1] != self.c:
            raise ValueError(
                f"expected X0[..., {self.D}] G[..., {self.c}], "
                f"got {tuple(X0.shape)} {tuple(G.shape)}")
        H, W = infer_token_hw(X0.shape[1])
        h = F.gelu(self.mix(torch.cat(
            [reshape_map(X0, H, W), reshape_map(G, H, W)], dim=1)))
        return flatten_map(self.proj(h + self.dw(h)))


class ResidualLinearCodec(nn.Module):
    """``G = R W_g``.  Reconstruction is linear ``W_r``/``W_a`` or conv ``F_θ``.

    ``decoder='linear'``: ``hat R = G W_r + X_0 ⊙ (G W_a)``.
    ``decoder='conv'``: ``hat R = F_θ(X_0, G)`` (channel mix + 3×3 neighborhood).
    ``Q_G`` stays a frozen 4-level scalar when enabled.
    """

    def __init__(self, D, c, B0=None, scale=None, decoder="linear"):
        super().__init__()
        if decoder not in ("linear", "conv"):
            raise ValueError(f"unknown residual decoder {decoder!r}")
        self.D = int(D)
        self.c = int(c)
        self.decoder = decoder
        self.W_g = nn.Linear(self.D, self.c, bias=False)
        self.W_r = None
        self.W_a = None
        self.recon = None
        if decoder == "linear":
            self.W_r = nn.Linear(self.c, self.D, bias=False)
            self.W_a = nn.Linear(self.c, self.D, bias=False)
            nn.init.zeros_(self.W_a.weight)
        else:
            self.recon = ResidualConvRecon(self.D, self.c)
        self.quant = FourLevelScalarQuantizer(self.c, scale=scale)
        if B0 is not None:
            self.init_from_B0(B0)

    def init_from_B0(self, B0):
        B0 = torch.as_tensor(B0, dtype=torch.float32)
        if tuple(B0.shape) != (self.D, self.c):
            raise ValueError(f"B0 shape {tuple(B0.shape)} != {(self.D, self.c)}")
        with torch.no_grad():
            self.W_g.weight.copy_(B0.t().contiguous())
            if self.W_r is not None:
                self.W_r.weight.copy_(B0.contiguous())
            if self.W_a is not None:
                self.W_a.weight.zero_()

    def recon_parameters(self):
        if self.decoder == "conv":
            return list(self.recon.parameters())
        params = list(self.W_r.parameters()) + list(self.W_a.parameters())
        return params

    def set_mode(self, mode):
        """``fixed`` / ``recon`` / ``both``.

        ``recon`` trains the decoder (``W_r``/``W_a`` or ``F_θ``).
        ``both`` also trains the encoder ``W_g``.  ``Q_G`` stays frozen.
        """
        if mode not in ("fixed", "recon", "both"):
            raise ValueError(f"unknown residual mode {mode!r}")
        for p in self.W_g.parameters():
            p.requires_grad_(mode == "both")
        for p in self.recon_parameters():
            p.requires_grad_(mode in ("recon", "both"))
        return self

    def encode_residual(self, R):
        return self.W_g(R)

    def decode_residual(self, G_hat, X0=None):
        """Linear: ``G W_r + X_0 ⊙ (G W_a)``.  Conv: ``F_θ(X_0, G)``."""
        if self.decoder == "conv":
            if X0 is None:
                raise ValueError("conv residual decoder requires X0")
            return self.recon(X0, G_hat)
        linear = self.W_r(G_hat)
        if X0 is None:
            return linear
        if X0.shape[-1] != self.D:
            raise ValueError(
                f"X0 last dim {X0.shape[-1]} != D={self.D}")
        return linear + X0 * self.W_a(G_hat)

    def forward(self, R, X0=None, quantize=True):
        G = self.encode_residual(R)
        if quantize:
            G_hat, q = self.quant(G)
        else:
            G_hat, q = G, None
        R_hat = self.decode_residual(G_hat, X0)
        return R_hat, G, G_hat, q


def canonicalize_evecs(B0):
    """Flip each column so the max-abs entry is positive."""
    B0 = B0.clone()
    for j in range(B0.shape[1]):
        v = B0[:, j]
        i = int(torch.argmax(v.abs()).item())
        if v[i] < 0:
            B0[:, j] = -v
    return B0


def evecs_from_gram(C, c):
    """Top-``c`` eigenvectors of a symmetric Gram ``C [D, D]``."""
    evals, evecs = torch.linalg.eigh(C)
    idx = torch.argsort(evals, descending=True)[:c]
    return canonicalize_evecs(evecs[:, idx].contiguous()), evals[idx].contiguous()


def spatial_roundtrip(Y, spatial, orfc=None):
    """``X_0 = U(E(X))`` (and optional ORFC on the coded sequence).

    Differentiable.  Does not switch ``spatial`` to eval.
    """
    seq, aux = spatial.encode(Y)
    if orfc is None:
        Y0 = spatial.decode(seq, aux)
        return Y0, seq, seq, aux, None
    seq_hat, usage = orfc(seq)
    Y0 = spatial.decode(seq_hat, aux)
    return Y0, seq, seq_hat, aux, usage


@torch.no_grad()
def reconstruct_spatial(Y, spatial):
    """Spatial down/up only.  No ORFC / PQ."""
    spatial.eval()
    return spatial_roundtrip(Y, spatial, orfc=None)


@torch.no_grad()
def reconstruct_main(Y, spatial, orfc):
    """Hard ORFC on downsampled tokens.  ``orfc`` should be eval()."""
    spatial.eval()
    orfc.eval()
    return spatial_roundtrip(Y, spatial, orfc)


def reconstruct_base(Y, spatial, orfc=None):
    """``orfc=None`` → spatial only (stage-1 pretrain, no PQ)."""
    if orfc is None:
        return reconstruct_spatial(Y, spatial)
    return reconstruct_main(Y, spatial, orfc)


def apply_residual(Y, Y0, residual, n_prefix=1, quantize=True, ablation="full"):
    """``Y_hat`` for a residual ablation.  Decoder never sees raw ``Y``.

    ``main``: ``hat = X_0``.  ``recon0``: ``hat = X_0 + F_φ(X_0, 0)``.
    ``full``: ``hat = X_0 + F_φ(X_0, G)`` with ``G = (X-X_0) W_g``.
    Returns ``(Y_hat, G, G_hat, q)``; ``G`` is ``None`` when unused.
    """
    if ablation not in RESIDUAL_ABLATIONS:
        raise ValueError(f"unknown residual ablation {ablation!r}")
    if residual is None or ablation == "main":
        return Y0, None, None, None
    X0 = Y0[:, n_prefix:]
    if ablation == "recon0":
        G0 = X0.new_zeros(X0.shape[0], X0.shape[1], residual.c)
        R_hat = residual.decode_residual(G0, X0)
        return combine_residual(Y0, R_hat, n_prefix), G0, G0, None
    R = residual_from_main(Y, Y0, n_prefix=n_prefix)
    R_hat, G, G_hat, q = residual(R, X0, quantize=quantize)
    return combine_residual(Y0, R_hat, n_prefix), G, G_hat, q


def combine_residual(Y0, R_hat, n_prefix=1):
    """CLS from the main path; patches = ``X_0 + hat R``."""
    p = n_prefix
    return torch.cat([Y0[:, :p], Y0[:, p:] + R_hat], dim=1)


def residual_from_main(Y, Y0, n_prefix=1):
    return Y[:, n_prefix:] - Y0[:, n_prefix:]


class BilinearORFCWrapper(nn.Module):
    """Drop-in ``forward(Y) -> (Y_hat, aux)`` for VOC / NYU task heads.

    ``orfc=None`` is stage-1 spatial pretrain (bilinear ``X_0``, no PQ).
    """

    def __init__(self, spatial, orfc=None, residual=None, n_prefix=1,
                 residual_quantize=True, residual_ablation="full"):
        super().__init__()
        if residual_ablation not in RESIDUAL_ABLATIONS:
            raise ValueError(f"unknown residual ablation {residual_ablation!r}")
        self.spatial = freeze_module(spatial)
        self.orfc = None if orfc is None else freeze_module(orfc)
        self.residual = None if residual is None else freeze_module(residual)
        self.n_prefix = int(n_prefix)
        self.residual_quantize = bool(residual_quantize)
        self.residual_ablation = residual_ablation

    def forward(self, Y, groups=None):
        del groups
        Y0, _seq, _seq_hat, aux, usage = reconstruct_base(
            Y, self.spatial, self.orfc)
        Y_hat, _, _, _ = apply_residual(
            Y, Y0, self.residual, n_prefix=self.n_prefix,
            quantize=self.residual_quantize,
            ablation=self.residual_ablation)
        out_aux = dict(aux)
        out_aux["usage"] = usage
        return Y_hat, out_aux


@torch.no_grad()
def fit_residual_basis(features, spatial, orfc, norm_mode, device,
                       batch_size=32, c=4, n_prefix=1):
    """Uncentered Gram of train residuals → ``B0 [D, c]`` and ``s [c]``.

    Only patch tokens enter the residual.  ``orfc=None`` uses bilinear
    spatial reconstruction only (no PQ).
    """
    if isinstance(features, np.ndarray) and features.ndim == 3:
        feat = features
    else:
        feat = np.stack(features)
    N, T, D = feat.shape
    spatial = freeze_module(spatial.to(device))
    if orfc is not None:
        orfc = freeze_module(orfc.to(device))

    gram = torch.zeros(D, D, dtype=torch.float64, device=device)
    n_patch = 0
    energy = 0.0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        X = torch.from_numpy(feat[start:end]).float().to(device)
        Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
        Y0, _, _, _, _ = reconstruct_base(Y, spatial, orfc)
        R = residual_from_main(Y, Y0, n_prefix=n_prefix)
        flat = R.reshape(-1, D).double()
        gram += flat.t() @ flat
        energy += float((R.float() ** 2).sum().item())
        n_patch += int(flat.shape[0])
        del X, Y, Y0, R, flat

    C = gram / max(n_patch, 1)
    B0, evals = evecs_from_gram(C, c)
    B0_f = B0.float()

    g_chunks = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        X = torch.from_numpy(feat[start:end]).float().to(device)
        Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
        Y0, _, _, _, _ = reconstruct_base(Y, spatial, orfc)
        R = residual_from_main(Y, Y0, n_prefix=n_prefix)
        G0 = torch.matmul(R, B0_f)
        g_chunks.append(G0.reshape(-1, c).cpu())
        del X, Y, Y0, R, G0

    G0 = torch.cat(g_chunks, dim=0)
    scale = G0.pow(2).mean(dim=0).sqrt().clamp_min(SCALE_EPS)
    ov = overflow_mask(G0, scale)
    overflow = {
        "frac": float(ov.float().mean().item()),
        "frac_per_channel": ov.float().mean(dim=0).tolist(),
        "n_coeff": int(G0.shape[0] * c),
    }
    explained = float(evals.sum().item() / max(float(torch.trace(C).item()), 1e-30))
    stats = {
        "n_image": int(N),
        "n_patch": int(n_patch),
        "mse_residual": float(energy / max(n_patch * D, 1)),
        "eigenvalues": [float(x) for x in evals.cpu().tolist()],
        "trace": float(torch.trace(C).item()),
        "explained_topc": explained,
        "scale": [float(x) for x in scale.tolist()],
        "overflow": overflow,
    }
    return B0_f.cpu(), scale.cpu(), stats


def save_residual_init(path, B0, scale, stats, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": "bilinear_residual_init_v1",
        "B0": torch.as_tensor(B0).cpu(),
        "scale": torch.as_tensor(scale).cpu().reshape(-1),
        "stats": stats,
    }
    if extra:
        meta.update(extra)
    torch.save(meta, path)


def load_residual_init(path, device="cpu"):
    meta = torch.load(path, map_location="cpu")
    B0 = meta["B0"].float()
    scale = meta["scale"].float().reshape(-1)
    return B0.to(device), scale.to(device), meta


def save_residual_codec(codec, path, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    decoder = getattr(codec, "decoder", "linear")
    meta = {
        "format": "bilinear_residual_v3" if decoder == "conv" else "bilinear_residual_v2",
        "D": int(codec.D),
        "c": int(codec.c),
        "decoder": decoder,
        "state_dict": codec.state_dict(),
    }
    if extra:
        meta.update(extra)
    torch.save(meta, path)


def load_spatial_weights(spatial, meta):
    """Restore ``E_θ`` from a residual checkpoint extra dict."""
    sd = meta.get("spatial_state_dict")
    if not sd:
        return spatial
    spatial.load_state_dict(sd, strict=True)
    return spatial


def load_residual_codec(path, device="cuda"):
    meta = torch.load(path, map_location="cpu")
    decoder = meta.get("decoder", "linear")
    codec = ResidualLinearCodec(int(meta["D"]), int(meta["c"]), decoder=decoder)
    missing, unexpected = codec.load_state_dict(
        meta["state_dict"], strict=False)
    allowed_missing = {k for k in missing if k.startswith("W_a.")}
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"residual codec state mismatch in {path}: "
            f"missing={bad_missing} unexpected={list(unexpected)}")
    return codec.to(device).eval(), meta
