"""GAPC — Global Average Pooling-based Compressor.

Strict implementation of Algorithm 1 in
  Dhaouadi, Achir, Adjih. "Enhancing Split ViT Inference Through
  Sparsity-Driven Compression." VTC2025-Spring.

For each image with latent tensor L ∈ R^{N×D}:
    avg_feature[j] = (1/N) * sum_i |L[i, j]|
    selected[j]    = (avg_feature[j] >= threshold)
    L_reduced      = L[:, selected]              # drop low-activity columns
    bitstream      = ZIP(L_reduced)              # DEFLATE lossless
At the decoder side, the dropped columns are filled with zeros, so the
reconstructed tensor L_hat has shape [N, D] (with some columns zero).

Notes
-----
* Per-image dynamic mask (one mask per image). Mask overhead = D bits/image.
* The method operates directly on the raw ViT latent tensor (no normalisation
  is required or used — strictly as in the paper).
* Zero trainable parameters. No model retraining.
* Random-Zero / Top-K baselines from paper §V have been removed from this
  runner by request; the core GAPC thresholding path is the only mode.
"""

from __future__ import annotations

import math
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# ================================================================
#  Column-selection — strict paper Algorithm 1 (per-image, stateless)
# ================================================================


def gapc_mask(L: torch.Tensor, threshold: float) -> torch.Tensor:
    """GAPC column mask (Algorithm 1, lines 3-4).

    Args:
        L: [B, N, D] or [N, D] raw latent tensor (before any normalisation).
        threshold: float, columns with mean |L[:, j]| >= threshold are kept.

    Returns:
        mask: same prefix shape as L but with last dim D, dtype=bool.
              True means the column is *kept*.
    """
    avg = L.abs().mean(dim=-2, keepdim=True)           # [..., 1, D]
    return avg >= threshold                             # broadcast over tokens


# ================================================================
#  Codec modules (nn.Module so they slot into the FeatureCodec API)
# ================================================================


class GAPCCodec(nn.Module):
    """Strict per-image GAPC codec (zero parameters).

    The ``forward`` method returns the reconstructed latent with the same
    ``[B, T, D]`` shape; dropped columns are filled with zeros.  Compatible
    with ``soft_pq_encode_decode`` / ``evaluate_accuracy`` pipelines.

    Optional *post-GAPC* scalar quantisation (extension beyond the paper):
    each kept column is asymmetrically quantised over tokens to ``quant_bits``
    bits (``32`` = identity / paper-strict; ``16`` = IEEE fp16 round-to-
    nearest; ``8`` / ``4`` = per-column min/max int quantisation).  The
    forward pass performs the matching *fake* quant+dequant so that the
    downstream accuracy corresponds exactly to the bit-stream measured in
    :func:`measure_rate` with the same ``quant_bits``.
    """

    def __init__(self, D: int, threshold: float,
                 seed: int = 42, quant_bits: int = 32):
        super().__init__()
        assert quant_bits in (32, 16, 8, 4), \
            f"quant_bits must be 32/16/8/4, got {quant_bits}"
        self.D = D
        self.threshold = float(threshold)
        self.seed = seed
        self.quant_bits = int(quant_bits)
        self._last_mask: torch.Tensor | None = None     # [B, 1, D] bool

    def make_mask(self, X: torch.Tensor) -> torch.Tensor:
        """Return a [B, 1, D] (or [1, D]) bool column mask for X (GAPC only)."""
        return gapc_mask(X, self.threshold)

    def _fake_quant(self, X_masked: torch.Tensor,
                    mask_f: torch.Tensor) -> torch.Tensor:
        """Per-column asymmetric fake quant+dequant on GPU.

        Args:
            X_masked: [B, T, D] already zero on dropped columns.
            mask_f:   [B, 1, D] float mask.
        """
        bits = self.quant_bits
        if bits == 32:
            return X_masked
        if bits == 16:
            # IEEE half round-to-nearest, then back to fp32.
            return X_masked.to(torch.float16).to(X_masked.dtype) * mask_f

        qmax = float((1 << bits) - 1)
        # per-image, per-column range over tokens (axis=1)
        lo = X_masked.amin(dim=1, keepdim=True)          # [B, 1, D]
        hi = X_masked.amax(dim=1, keepdim=True)
        scale = (hi - lo) / qmax                         # [B, 1, D]
        # fp16 side-info transmitted in the rate model -> reproduce here.
        scale = scale.to(torch.float16).to(X_masked.dtype)
        lo_t = lo.to(torch.float16).to(X_masked.dtype)
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        codes = torch.round((X_masked - lo_t) / safe).clamp_(0.0, qmax)
        X_dq = codes * safe + lo_t
        return X_dq * mask_f

    def forward(self, X: torch.Tensor):
        """Apply GAPC column mask (and optional post-quantisation) to L.

        Args:
            X: [B, T, D] raw latent (no normalisation expected).

        Returns:
            (X_hat, None): X_hat with same shape, dropped cols zeroed,
            kept cols fake-quantised to ``self.quant_bits`` bits.
        """
        mask = self.make_mask(X)                         # [B, 1, D] bool
        self._last_mask = mask.detach()
        mask_f = mask.to(X.dtype)
        X_masked = X * mask_f
        X_hat = self._fake_quant(X_masked, mask_f)
        return X_hat, None


# ================================================================
#  Rate / compression-ratio measurement (strict ZIP/DEFLATE)
# ================================================================


@dataclass
class RateRecord:
    bits_ref: int              # ZIP(L_full, fp32) size in bits  (baseline)
    bits_reduced: int          # ZIP(payload) size in bits
    bits_sideinfo: int         # uncompressed (scale, zero_point) bits
    bits_mask: int             # D bits/image, mask overhead
    bits_total: int            # reduced + sideinfo + mask
    num_tokens: int            # N (incl. CLS)
    feat_dim: int              # D
    kept_cols: int             # |selected|
    quant_bits: int            # 32 / 16 / 8 / 4
    cr: float                  # compression ratio = bits_ref / bits_total
    bpt: float                 # bits per token = bits_total / N
    bpfp: float                # bits per feature = bits_total / (N * D)


def _zip_bits(arr: np.ndarray, level: int = 6) -> int:
    """Return DEFLATE bit-count for a contiguous fp32 array."""
    if arr.size == 0:
        return 0
    buf = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
    return len(zlib.compress(buf, level)) * 8


def _zip_bits_bytes(buf: bytes, level: int = 6) -> int:
    if not buf:
        return 0
    return len(zlib.compress(buf, level)) * 8


def _quantize_per_col(L_red: np.ndarray, bits: int
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column asymmetric quant of L_red [N, K_kept] to ``bits`` (4 or 8).

    Returns (codes_packed_bytes, scale_fp16, lo_fp16).
    """
    assert bits in (4, 8)
    if L_red.size == 0:
        return (np.zeros(0, dtype=np.uint8),
                np.zeros(0, dtype=np.float16),
                np.zeros(0, dtype=np.float16))
    lo = L_red.min(axis=0).astype(np.float32)             # [K]
    hi = L_red.max(axis=0).astype(np.float32)
    qmax = float((1 << bits) - 1)
    scale = (hi - lo) / qmax
    scale_safe = np.where(scale > 0, scale, 1.0).astype(np.float32)
    # Reproduce what is actually transmitted: scale/lo are fp16 on the wire.
    scale_fp16 = scale.astype(np.float16)
    lo_fp16 = lo.astype(np.float16)
    scale_send = scale_fp16.astype(np.float32)
    lo_send = lo_fp16.astype(np.float32)
    scale_send = np.where(scale_send > 0, scale_send, 1.0)
    codes = np.round((L_red - lo_send[None, :]) / scale_send[None, :])
    codes = np.clip(codes, 0, qmax).astype(np.uint8)       # [N, K]
    if bits == 8:
        return codes.tobytes(order="C"), scale_fp16, lo_fp16
    # 4-bit: pack two codes per byte (little-endian nibble order).
    flat = codes.ravel()
    if flat.size % 2 == 1:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
    packed = (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)
    return packed.tobytes(order="C"), scale_fp16, lo_fp16


def measure_rate(L_np: np.ndarray, mask_1d: np.ndarray,
                 zip_level: int = 6, count_mask: bool = True,
                 quant_bits: int = 32) -> RateRecord:
    """Measure DEFLATE-compressed bit-rate for one image.

    Args:
        L_np: [N, D] fp32 raw latent tensor of ONE image.
        mask_1d: [D] bool — True for columns kept.
        zip_level: 0..9 zlib compression level (6 is default).
        count_mask: include the D-bit mask overhead if True.
        quant_bits: ``32`` (paper-strict fp32+zlib), ``16`` (fp16+zlib),
            ``8`` / ``4`` (per-column asymmetric int quant + zlib;
            scale/zero-point transmitted uncompressed as fp16).
    """
    assert L_np.ndim == 2 and mask_1d.ndim == 1
    assert quant_bits in (32, 16, 8, 4)
    N, D = L_np.shape
    kept = int(mask_1d.sum())
    L_red = L_np[:, mask_1d]
    bits_ref = _zip_bits(L_np, level=zip_level)

    bits_sideinfo = 0
    if quant_bits == 32:
        bits_reduced = _zip_bits(L_red, level=zip_level)
    elif quant_bits == 16:
        if L_red.size == 0:
            bits_reduced = 0
        else:
            buf = np.ascontiguousarray(
                L_red.astype(np.float16)
            ).tobytes()
            bits_reduced = _zip_bits_bytes(buf, level=zip_level)
    else:  # 8 or 4
        codes_buf, scale_fp16, lo_fp16 = _quantize_per_col(L_red, quant_bits)
        bits_reduced = _zip_bits_bytes(codes_buf, level=zip_level)
        bits_sideinfo = (scale_fp16.nbytes + lo_fp16.nbytes) * 8

    bits_mask = D if count_mask else 0
    bits_total = bits_reduced + bits_sideinfo + bits_mask
    cr = bits_ref / max(bits_total, 1)
    bpt = bits_total / N
    bpfp = bits_total / (N * D)
    return RateRecord(
        bits_ref=bits_ref,
        bits_reduced=bits_reduced,
        bits_sideinfo=bits_sideinfo,
        bits_mask=bits_mask,
        bits_total=bits_total,
        num_tokens=N,
        feat_dim=D,
        kept_cols=kept,
        quant_bits=quant_bits,
        cr=cr,
        bpt=bpt,
        bpfp=bpfp,
    )


# ================================================================
#  Encode/decode loop aligned with run_soft_pq's API
# ================================================================


def gapc_encode_decode(features: List[np.ndarray], codec: GAPCCodec,
                       device: str = "cuda",
                       chunk_images: int = 200,
                       zip_level: int = 6,
                       count_mask: bool = True) -> Tuple[List[np.ndarray], List[RateRecord]]:
    """Run GAPC over a list of per-image feature arrays.

    Returns:
        xhat_list: list of [N, D] np.float32 reconstructed tensors.
        rate_records: list of RateRecord, one per image.
    """
    codec.eval()
    N = len(features)
    xhat_list: List[np.ndarray] = []
    rate_records: List[RateRecord] = []
    with torch.no_grad():
        for start in range(0, N, chunk_images):
            end = min(start + chunk_images, N)
            X = torch.from_numpy(
                np.stack(features[start:end])
            ).float().to(device)
            X_hat, _ = codec(X)
            mask = codec._last_mask                    # [B, 1, D] bool
            X_np = X.cpu().numpy()
            mask_np = mask.squeeze(-2).cpu().numpy()   # [B, D]
            X_hat_np = X_hat.cpu().numpy()
            for i in range(X_hat_np.shape[0]):
                xhat_list.append(X_hat_np[i].astype(np.float32))
                rate_records.append(
                    measure_rate(X_np[i], mask_np[i],
                                 zip_level=zip_level,
                                 count_mask=count_mask,
                                 quant_bits=int(codec.quant_bits))
                )
            del X, X_hat
    return xhat_list, rate_records


def aggregate_rate(records: List[RateRecord]) -> dict:
    """Average the per-image rate records into scalar metrics."""
    if not records:
        return {}
    n = len(records)
    agg = {
        "avg_bits_ref": sum(r.bits_ref for r in records) / n,
        "avg_bits_reduced": sum(r.bits_reduced for r in records) / n,
        "avg_bits_sideinfo": sum(r.bits_sideinfo for r in records) / n,
        "avg_bits_total": sum(r.bits_total for r in records) / n,
        "avg_kept_cols": sum(r.kept_cols for r in records) / n,
        "avg_cr": sum(r.cr for r in records) / n,
        "avg_bpt": sum(r.bpt for r in records) / n,
        "avg_bpfp": sum(r.bpfp for r in records) / n,
        "num_images": n,
        "feat_dim": records[0].feat_dim,
        "num_tokens": records[0].num_tokens,
        "quant_bits": records[0].quant_bits,
    }
    # Also report the *pooled* compression ratio (sum-bits basis, usually
    # slightly different from the per-image-averaged cr).
    agg["pooled_cr"] = (
        sum(r.bits_ref for r in records) /
        max(sum(r.bits_total for r in records), 1)
    )
    return agg


# ================================================================
#  Fast streaming sweep: GPU batch mask + batched classifier + CPU-
#  threaded zlib, all overlapped. No xhat materialisation.
# ================================================================


def _measure_rate_for_batch(
    X_cpu: np.ndarray,          # [B, N, D]
    mask_cpu: np.ndarray,       # [B, D] bool
    zip_level: int,
    count_mask: bool,
    quant_bits: int,
    executor: ThreadPoolExecutor,
) -> List[RateRecord]:
    """Parallel zlib across images in a batch (GIL is released by zlib)."""
    B = X_cpu.shape[0]
    futures = [
        executor.submit(measure_rate, X_cpu[i], mask_cpu[i],
                        zip_level, count_mask, quant_bits)
        for i in range(B)
    ]
    return [f.result() for f in futures]


@torch.no_grad()
def run_sweep_point(
    features: List[np.ndarray],
    basenames: List[str],
    gt: Dict[str, int],
    wrapper,                   # Dinov2Wrapper-like, needs forward_from_tokens
    layer_idx: int,
    codec: "GAPCCodec",
    device: torch.device,
    chunk_images: int = 64,
    zip_level: int = 6,
    count_mask: bool = True,
    zip_workers: int = 8,
) -> Tuple[float, dict, List[RateRecord]]:
    """Single-pass streaming evaluation for one (θ, dataset) sweep point.

    Pipeline per chunk:
        (a) Upload [B, T, D] to GPU.
        (b) GAPC masking on GPU -> X_hat.
        (c) ``wrapper.forward_from_tokens(X_hat, layer_idx)`` — batched,
            single GPU forward over the tail + head.
        (d) In parallel: copy X and mask to CPU; dispatch zlib for every
            image in the batch through a thread pool (releases GIL).

    Returns:
        acc: classification accuracy over images with valid ground truth.
        agg: aggregated rate metrics dict (same keys as aggregate_rate).
        records: per-image RateRecord list (length == len(features)).
    """
    codec.eval()
    N = len(features)
    correct = 0
    total = 0
    rate_records: List[RateRecord] = []

    with ThreadPoolExecutor(max_workers=zip_workers) as executor:
        for start in range(0, N, chunk_images):
            end = min(start + chunk_images, N)
            X_cpu_np = np.stack(features[start:end])                # [B, T, D]
            X_gpu = torch.from_numpy(X_cpu_np).to(
                device, non_blocking=True
            )
            X_hat_gpu, _ = codec(X_gpu)
            mask_gpu = codec._last_mask                              # [B, 1, D]

            # --- GPU: batched tail + head -> predictions ---
            logits = wrapper.forward_from_tokens(X_hat_gpu, layer_idx)
            preds = logits.argmax(dim=1).cpu().numpy()

            # --- CPU: parallel zlib (with codec.quant_bits) ---
            mask_cpu = mask_gpu.squeeze(-2).cpu().numpy()            # [B, D]
            batch_records = _measure_rate_for_batch(
                X_cpu_np, mask_cpu, zip_level, count_mask,
                int(codec.quant_bits), executor,
            )
            rate_records.extend(batch_records)

            for i, bn in enumerate(basenames[start:end]):
                if bn in gt:
                    total += 1
                    if int(preds[i]) == int(gt[bn]):
                        correct += 1

            del X_gpu, X_hat_gpu, mask_gpu, logits

    acc = correct / total if total > 0 else 0.0
    agg = aggregate_rate(rate_records)
    return float(acc), agg, rate_records


@torch.no_grad()
def zip_bits_reference(
    features: List[np.ndarray], zip_level: int = 6, zip_workers: int = 8
) -> Tuple[float, float]:
    """ZIP(raw latent) statistics — needed for the compression-ratio baseline.

    Returns:
        avg_bits: mean DEFLATE bits per image over ``features``.
        avg_bpfp: mean bits per feature (total_bits / (T*D)).
    """
    if not features:
        return 0.0, 0.0
    T, D = features[0].shape
    with ThreadPoolExecutor(max_workers=zip_workers) as executor:
        futures = [executor.submit(_zip_bits, f, zip_level) for f in features]
        bits_list = [f.result() for f in futures]
    avg_bits = float(np.mean(bits_list))
    return avg_bits, avg_bits / (T * D)
