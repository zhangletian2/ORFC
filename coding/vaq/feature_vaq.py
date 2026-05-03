#!/usr/bin/env python
"""
Variance-Aware Quantization as a feature coding codec.

This module ports the codec-relevant part of the VAQ paper/code to Python:
PCA variance ordering, partial subspace balancing, adaptive bit allocation,
per-subspace k-means, hard assignment, and feature reconstruction.

The ANN query pruning pieces from the original C++ VAQ implementation are not
used here because the ORFC experiment evaluates reconstructed ViT tokens.
"""

import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


VAQ_ROOT = os.path.dirname(os.path.abspath(__file__))
ORFC_ROOT = os.path.normpath(os.path.join(VAQ_ROOT, "..", "orfc"))
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

from opq import batch_inv_normalize_gpu, batch_normalize_gpu  # noqa: E402
from soft_pq import _rans_encode_bpt  # noqa: E402


def _next_pow2_floor(x: float) -> int:
    if x == 0 or not np.isfinite(x):
        return 0
    return int(2 ** math.floor(math.log2(abs(float(x)))))


def _pad_matrix(x: torch.Tensor, padded_dim: int) -> torch.Tensor:
    if x.shape[1] == padded_dim:
        return x
    pad_cols = padded_dim - x.shape[1]
    return torch.cat([x, x.new_zeros(x.shape[0], pad_cols)], dim=1)


def _all_same_shape(features: Sequence[np.ndarray], start: int, end: int) -> bool:
    shape = features[start].shape
    return all(features[i].shape == shape for i in range(start + 1, end))


def sample_normalized_vectors(
    features: Sequence[np.ndarray],
    norm_mode: str,
    max_vectors: int,
    device: torch.device,
    chunk_images: int = 64,
    seed: int = 42,
    verbose: bool = True,
) -> np.ndarray:
    """Sample flattened normalized tokens from a list of [T, D] features."""
    if not features:
        raise ValueError("features is empty")

    lengths = np.asarray([f.shape[0] for f in features], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    total = int(offsets[-1])
    if max_vectors <= 0 or max_vectors >= total:
        selected = np.arange(total, dtype=np.int64)
    else:
        rng = np.random.RandomState(seed)
        selected = np.sort(rng.choice(total, max_vectors, replace=False))

    if verbose:
        print(f"  VAQ fit vectors: {len(selected):,}/{total:,} tokens")

    chunks = []
    sel_pos = 0
    for start in range(0, len(features), chunk_images):
        end = min(start + chunk_images, len(features))
        chunk_begin = int(offsets[start])
        chunk_end = int(offsets[end])

        left = np.searchsorted(selected, chunk_begin, side="left")
        right = np.searchsorted(selected, chunk_end, side="left")
        if right <= left:
            continue
        wanted = selected[left:right] - chunk_begin

        if _all_same_shape(features, start, end):
            X_np = np.stack(features[start:end])
            X = torch.from_numpy(X_np).float().to(device)
            with torch.no_grad():
                Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
                flat = Y.reshape(-1, Y.shape[-1])
                chunks.append(flat[wanted].cpu().numpy().astype(np.float32))
            del X, Y, flat
        else:
            local_parts = []
            local_offsets = np.concatenate(
                [[0], np.cumsum([features[i].shape[0] for i in range(start, end)])]
            )
            for i in range(start, end):
                img_begin = int(local_offsets[i - start])
                img_end = int(local_offsets[i - start + 1])
                l_i = np.searchsorted(wanted, img_begin, side="left")
                r_i = np.searchsorted(wanted, img_end, side="left")
                if r_i <= l_i:
                    continue
                local_idx = wanted[l_i:r_i] - img_begin
                X = torch.from_numpy(features[i]).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
                    local_parts.append(Y.reshape(-1, Y.shape[-1])[local_idx].cpu().numpy())
                del X, Y
            if local_parts:
                chunks.append(np.concatenate(local_parts, axis=0).astype(np.float32))

        sel_pos = right
        if verbose and sel_pos == len(selected):
            pass

    if not chunks:
        raise RuntimeError("no normalized vectors were sampled")
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def _fit_pca_rotation(
    vectors: np.ndarray,
    padded_dim: int,
    device: torch.device,
    chunk_vectors: int = 65536,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute eigenvectors of X^T X, matching the released C++ VAQ code."""
    n, dim = vectors.shape
    cov = torch.zeros((padded_dim, padded_dim), dtype=torch.float32, device=device)
    if verbose:
        print(f"  PCA: N={n:,}, D={dim}, padded_D={padded_dim}")

    with torch.no_grad():
        for start in range(0, n, chunk_vectors):
            end = min(start + chunk_vectors, n)
            X = torch.from_numpy(np.ascontiguousarray(vectors[start:end])).float().to(device)
            X = _pad_matrix(X, padded_dim)
            cov.add_(X.t().matmul(X))
            del X
        cov.div_(max(n, 1))
        eigvals, eigvecs = torch.linalg.eigh(cov)
        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order].clamp_min(1e-12)
        eigvecs = eigvecs[:, order].contiguous()

    return eigvecs.cpu().numpy().astype(np.float32), eigvals.cpu().numpy().astype(np.float32)


def _subspace_variances(eigvals: np.ndarray, num_subspaces: int, subspace_len: int) -> np.ndarray:
    out = np.zeros(num_subspaces, dtype=np.float64)
    for g in range(num_subspaces):
        s = g * subspace_len
        e = s + subspace_len
        out[g] = float(eigvals[s:e].sum())
    return out


def _partial_balance(
    eigvecs: np.ndarray,
    eigvals: np.ndarray,
    num_subspaces: int,
    subspace_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror the partial balancing strategy in bitvecengine/VAQ.cpp."""
    values = eigvals.copy()
    vectors = eigvecs.copy()

    def ordered(vals: np.ndarray) -> bool:
        var_per_sub = _subspace_variances(vals, num_subspaces, subspace_len)
        return bool(np.all(var_per_sub[:-1] >= var_per_sub[1:] - 1e-12))

    max_swap = min(subspace_len, num_subspaces)
    for i in range(1, max_swap):
        j = i * subspace_len + (subspace_len - 1)
        if j >= len(values):
            break
        values[[i, j]] = values[[j, i]]
        if not ordered(values):
            values[[i, j]] = values[[j, i]]
            break
        vectors[:, [i, j]] = vectors[:, [j, i]]
    return vectors, values


def _bit_score(var: float, bits: int, subspace_len: int, objective: str) -> float:
    """Per-subspace utility used by the bit-allocation DP.

    - "linear": original VAQ paper / GLPK objective `var * bits`.
      Linear in bits, so under a wide [min_bits, max_bits] range and a heavy-tailed
      variance spectrum the optimum saturates to extremes (top->max, tail->min).
    - "rd": high-rate Lloyd rate-distortion proxy.
      Distortion D_g ~ var_g * 2^{-2 b_g / d}, so utility (variance recovered)
      U_g(b) = var_g * (1 - 4^{-b/d}). Concave in b => natural diminishing returns.
    """
    if objective == "linear":
        return float(var) * float(bits)
    if objective == "rd":
        return float(var) * (1.0 - 4.0 ** (-float(bits) / max(subspace_len, 1)))
    raise ValueError(f"unknown bit-allocation objective: {objective}")


def allocate_bits_vaq(
    var_per_subspace: np.ndarray,
    bit_budget: int,
    min_bits: int,
    max_bits: int,
    percent_var: float,
    cum_var: np.ndarray,
    subspace_len: int,
    objective: str = "rd",
) -> List[int]:
    """Allocate integer bits with the VAQ constraints and a configurable objective."""
    highest = len(var_per_subspace)
    lower = []
    for g in range(highest):
        if percent_var >= 1.0 or cum_var[g] <= percent_var:
            lower.append(min_bits)
        else:
            lower.append(0)
    upper = [max_bits] * highest

    if bit_budget < sum(lower) or bit_budget > sum(upper):
        raise ValueError(
            f"infeasible bit budget {bit_budget}; bounds allow "
            f"[{sum(lower)}, {sum(upper)}]"
        )

    allowed = [list(range(lower[g], upper[g] + 1)) for g in range(highest)]
    gaps = []
    for g in range(highest - 1):
        ratio = var_per_subspace[g] / max(var_per_subspace[g + 1], 1e-30)
        gaps.append(_next_pow2_floor(ratio))

    # dp[g][(sum_bits, curr_bits)] = (score, prev_sum, prev_bits)
    dp: Dict[Tuple[int, int], Tuple[float, Optional[int], Optional[int]]] = {}
    for b in allowed[0]:
        dp[(b, b)] = (
            _bit_score(var_per_subspace[0], b, subspace_len, objective),
            None,
            None,
        )

    parents: List[Dict[Tuple[int, int], Tuple[float, Optional[int], Optional[int]]]] = [dp]
    for g in range(1, highest):
        ndp: Dict[Tuple[int, int], Tuple[float, Optional[int], Optional[int]]] = {}
        for (prev_sum, prev_b), (score, _, _) in dp.items():
            for b in allowed[g]:
                if prev_b - b > gaps[g - 1]:
                    continue
                new_sum = prev_sum + b
                if new_sum > bit_budget:
                    continue
                key = (new_sum, b)
                new_score = score + _bit_score(
                    var_per_subspace[g], b, subspace_len, objective
                )
                if key not in ndp or new_score > ndp[key][0]:
                    ndp[key] = (new_score, prev_sum, prev_b)
        if not ndp:
            raise RuntimeError("VAQ bit allocation became infeasible")
        dp = ndp
        parents.append(dp)

    candidates = [(key, val) for key, val in dp.items() if key[0] == bit_budget]
    if not candidates:
        raise RuntimeError(f"no exact VAQ allocation for bit budget {bit_budget}")
    (sum_bits, curr_b), _ = max(candidates, key=lambda item: item[1][0])

    bits = [0] * highest
    bits[-1] = curr_b
    for g in range(highest - 1, 0, -1):
        _, prev_sum, prev_b = parents[g][(sum_bits, bits[g])]
        if prev_sum is None or prev_b is None:
            raise RuntimeError("broken VAQ bit-allocation backtrace")
        bits[g - 1] = prev_b
        sum_bits = prev_sum
    return [int(b) for b in bits]


def _kmeans_single(
    data: np.ndarray,
    k: int,
    max_iter: int,
    device: torch.device,
    seed: int,
    verbose: bool = False,
) -> np.ndarray:
    """Chunked torch k-means for one VAQ subspace."""
    data = np.ascontiguousarray(data.astype(np.float32, copy=False))
    n, dim = data.shape
    if k <= 1:
        return data.mean(axis=0, keepdims=True).astype(np.float32)
    if n == 0:
        raise ValueError("cannot run k-means on an empty subspace")

    rng = np.random.RandomState(seed)
    init_idx = rng.choice(n, k, replace=n < k)
    centroids = torch.from_numpy(data[init_idx]).float().to(device)
    max_mem = 768 * 1024**2
    chunk = max(1, min(n, max_mem // max(k * 4, 1)))

    for it in range(max_iter):
        sums = torch.zeros((k, dim), dtype=torch.float32, device=device)
        counts = torch.zeros(k, dtype=torch.float32, device=device)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            batch = torch.from_numpy(data[start:end]).float().to(device)
            with torch.no_grad():
                labels = torch.cdist(batch, centroids).argmin(dim=1)
            sums.scatter_add_(0, labels[:, None].expand(-1, dim), batch)
            counts.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
            del batch, labels

        empty = counts == 0
        new_centroids = sums / counts.clamp_min(1.0)[:, None]
        if empty.any():
            empty_idx = empty.nonzero(as_tuple=False).flatten()
            repl = rng.choice(n, int(empty_idx.numel()), replace=n < int(empty_idx.numel()))
            new_centroids[empty_idx] = torch.from_numpy(data[repl]).float().to(device)

        shift = (new_centroids - centroids).norm(dim=1).max().item()
        centroids = new_centroids
        if verbose and ((it + 1) % 10 == 0 or it == 0):
            print(f"    kmeans K={k} iter {it + 1}/{max_iter}: shift={shift:.6f}")
        if shift < 1e-4:
            break

    return centroids.cpu().numpy().astype(np.float32)


@torch.no_grad()
def _assign_labels_chunked(
    data: np.ndarray,
    centroids: torch.Tensor,
    device: torch.device,
    chunk: int = 65536,
) -> np.ndarray:
    """Hard nearest-centroid assignment for a [N, d] np.ndarray."""
    out = np.empty(data.shape[0], dtype=np.int64)
    k = centroids.shape[0]
    for s in range(0, data.shape[0], chunk):
        e = min(s + chunk, data.shape[0])
        batch = torch.from_numpy(np.ascontiguousarray(data[s:e])).float().to(device)
        out[s:e] = torch.cdist(batch, centroids).argmin(dim=1).cpu().numpy()
        del batch
    return out


def _kmeans_hierarchical(
    data: np.ndarray,
    k: int,
    max_iter: int,
    device: torch.device,
    seed: int,
    flat_threshold: int = 1024,
    branching: int = 64,
    verbose: bool = False,
) -> np.ndarray:
    """Hierarchical k-means matching VAQ paper Section III-D.

    For K <= ``flat_threshold`` (default 1024) use flat k-means.
    For K > flat_threshold split into a 2-level tree:
      level-1 with K1=branching (default 64),
      level-2 with K2=ceil(K/K1) inside each level-1 cluster.
    Returned centroids are flattened to shape [K, d] (K1*K2 == K when K is divisible
    by branching). Encoding remains FLAT (search over all K centroids) so that
    accuracy matches a vanilla flat k-means with K codes; the hierarchy only helps
    initialization and convergence on huge K.
    """
    data = np.ascontiguousarray(data.astype(np.float32, copy=False))
    n, dim = data.shape
    if k <= flat_threshold or k <= 1:
        return _kmeans_single(data, k, max_iter, device, seed, verbose=verbose)

    k1 = min(branching, k)
    k2 = (k + k1 - 1) // k1
    if k1 * k2 != k:
        # round up branching so K1*K2 == K (only matters for non-power-of-two K)
        # find a K1 that divides k cleanly when possible
        for cand in (branching, 32, 16, 8):
            if k % cand == 0:
                k1 = cand
                k2 = k // cand
                break
        else:
            k2 = (k + k1 - 1) // k1
    if verbose:
        print(f"    hierarchical kmeans K={k}: K1={k1} x K2={k2}")

    centroids_l1 = _kmeans_single(data, k1, max_iter, device, seed, verbose=False)
    cent_l1_t = torch.from_numpy(centroids_l1).float().to(device)
    labels_l1 = _assign_labels_chunked(data, cent_l1_t, device)

    flat_centroids: List[np.ndarray] = []
    rng = np.random.RandomState(seed + 9999)
    for c in range(k1):
        mask = labels_l1 == c
        cluster = data[mask]
        if cluster.shape[0] >= k2:
            sub_cent = _kmeans_single(
                cluster, k2, max_iter, device, seed + c + 1, verbose=False
            )
        elif cluster.shape[0] == 0:
            # Empty L1 cluster: replicate the L1 centroid k2 times with tiny jitter
            base = centroids_l1[c]
            jitter = rng.normal(scale=1e-4, size=(k2, dim)).astype(np.float32)
            sub_cent = base[None, :] + jitter
        else:
            # Too few points: pad by sampling within the cluster + jitter
            need = k2 - cluster.shape[0]
            extra = cluster[rng.randint(0, cluster.shape[0], size=need)] + (
                rng.normal(scale=1e-4, size=(need, dim)).astype(np.float32)
            )
            sub_cent = np.concatenate([cluster, extra], axis=0).astype(np.float32)
        flat_centroids.append(sub_cent)

    centroids = np.concatenate(flat_centroids, axis=0).astype(np.float32)
    if centroids.shape[0] != k:
        # Truncate or pad to exact K
        if centroids.shape[0] > k:
            centroids = centroids[:k]
        else:
            need = k - centroids.shape[0]
            pad_idx = rng.choice(n, need, replace=n < need)
            centroids = np.concatenate([centroids, data[pad_idx]], axis=0)
    return centroids


@dataclass
class VAQConfig:
    bit_budget: int
    num_subspaces: int
    min_bits: int = 1
    max_bits: int = 13
    percent_var: float = 1.0
    kmeans_iter: int = 50
    seed: int = 42
    bit_alloc_objective: str = "rd"  # {"rd", "linear"} - DP scoring rule
    kmeans_hierarchical_threshold: int = 1024  # K above this uses hierarchical k-means
    kmeans_hierarchical_branching: int = 64    # paper Section III-D uses K1=64


class VarianceAwareQuantizer:
    """Hard VAQ codec for normalized [B, T, D] feature tensors."""

    def __init__(self, config: VAQConfig):
        self.config = config
        self.original_dim: Optional[int] = None
        self.padded_dim: Optional[int] = None
        self.subspace_len: Optional[int] = None
        self.highest_subspaces: Optional[int] = None
        self.rotation: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.variance_per_subspace: Optional[np.ndarray] = None
        self.bits_alloc: Optional[List[int]] = None
        self.centroids: Optional[List[np.ndarray]] = None

    @property
    def k_per_group(self) -> List[int]:
        if self.bits_alloc is None:
            raise RuntimeError("VAQ codec is not fitted")
        return [1 << int(b) for b in self.bits_alloc]

    @property
    def max_rate_bpt(self) -> float:
        if self.bits_alloc is None:
            raise RuntimeError("VAQ codec is not fitted")
        return float(sum(self.bits_alloc))

    def fit_rotation_only(
        self,
        vectors: np.ndarray,
        device: torch.device,
        verbose: bool = True,
    ) -> "VarianceAwareQuantizer":
        """Fit only PCA rotation + partial balancing, skip k-means.

        Used for the "PCA-only" diagnostic baseline that isolates the loss caused
        by the orthogonal rotation alone (no quantization).
        """
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be [N, D], got {vectors.shape}")
        cfg = self.config
        n, dim = vectors.shape
        self.original_dim = int(dim)
        self.subspace_len = int(math.ceil(dim / cfg.num_subspaces))
        self.padded_dim = int(self.subspace_len * cfg.num_subspaces)

        eigvecs, eigvals = _fit_pca_rotation(
            vectors, self.padded_dim, device=device, verbose=verbose
        )
        eigvecs, eigvals = _partial_balance(
            eigvecs, eigvals, cfg.num_subspaces, self.subspace_len
        )
        self.rotation = eigvecs.astype(np.float32)
        self.eigenvalues = eigvals.astype(np.float32)
        var_dim = eigvals.astype(np.float64)
        var_dim = np.maximum(var_dim, 1e-12) / np.maximum(var_dim, 1e-12).sum()
        self.variance_per_subspace = _subspace_variances(
            var_dim, cfg.num_subspaces, self.subspace_len
        ).astype(np.float32)
        self.highest_subspaces = cfg.num_subspaces
        if verbose:
            print(
                f"  [PCA-only] D={self.original_dim} padded={self.padded_dim} "
                f"G={cfg.num_subspaces} subspace_len={self.subspace_len}"
            )
            print(
                f"  [PCA-only] variance/subspace (first 10): "
                f"{self.variance_per_subspace[:10]}"
            )
        return self

    def fit(self, vectors: np.ndarray, device: torch.device, verbose: bool = True) -> "VarianceAwareQuantizer":
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be [N, D], got {vectors.shape}")
        cfg = self.config
        n, dim = vectors.shape
        self.original_dim = int(dim)
        self.subspace_len = int(math.ceil(dim / cfg.num_subspaces))
        self.padded_dim = int(self.subspace_len * cfg.num_subspaces)

        eigvecs, eigvals = _fit_pca_rotation(
            vectors, self.padded_dim, device=device, verbose=verbose
        )
        eigvecs, eigvals = _partial_balance(
            eigvecs, eigvals, cfg.num_subspaces, self.subspace_len
        )

        var_dim = eigvals.astype(np.float64)
        var_dim = np.maximum(var_dim, 1e-12)
        var_dim = var_dim / var_dim.sum()
        var_per_sub = _subspace_variances(var_dim, cfg.num_subspaces, self.subspace_len)
        cum_var = np.cumsum(var_per_sub)
        if cfg.percent_var < 1.0:
            highest = 1
            for i, val in enumerate(cum_var):
                if val <= cfg.percent_var:
                    highest = i + 1
            highest = min(highest + 1, cfg.num_subspaces)
        else:
            highest = cfg.num_subspaces

        bits = allocate_bits_vaq(
            var_per_sub[:highest],
            cfg.bit_budget,
            cfg.min_bits,
            cfg.max_bits,
            cfg.percent_var,
            cum_var[:highest],
            subspace_len=self.subspace_len,
            objective=cfg.bit_alloc_objective,
        )
        if verbose:
            print(f"  VAQ variance/subspace (first 10): {var_per_sub[:10]}")
            print(f"  VAQ highest_subspaces={highest}/{cfg.num_subspaces}")
            print(f"  VAQ objective={cfg.bit_alloc_objective}")
            print(f"  VAQ bits: {bits} (sum={sum(bits)})")

        X = torch.from_numpy(np.ascontiguousarray(vectors)).float().to(device)
        with torch.no_grad():
            X_pad = _pad_matrix(X, self.padded_dim)
            R = torch.from_numpy(eigvecs).float().to(device)
            Z = X_pad.matmul(R).cpu().numpy().astype(np.float32)
        del X, X_pad, R

        centroids = []
        for g, b in enumerate(bits):
            k = 1 << int(b)
            start = g * self.subspace_len
            end = start + self.subspace_len
            mode = "hier" if k > cfg.kmeans_hierarchical_threshold else "flat"
            if verbose:
                print(f"  VAQ k-means group {g + 1}/{highest}: bits={b}, K={k} ({mode})")
            if mode == "hier":
                c = _kmeans_hierarchical(
                    Z[:, start:end],
                    k,
                    max_iter=cfg.kmeans_iter,
                    device=device,
                    seed=cfg.seed + g,
                    flat_threshold=cfg.kmeans_hierarchical_threshold,
                    branching=cfg.kmeans_hierarchical_branching,
                    verbose=verbose,
                )
            else:
                c = _kmeans_single(
                    Z[:, start:end],
                    k,
                    max_iter=cfg.kmeans_iter,
                    device=device,
                    seed=cfg.seed + g,
                    verbose=False,
                )
            centroids.append(c)

        self.rotation = eigvecs.astype(np.float32)
        self.eigenvalues = eigvals.astype(np.float32)
        self.variance_per_subspace = var_per_sub.astype(np.float32)
        self.highest_subspaces = int(highest)
        self.bits_alloc = bits
        self.centroids = centroids
        return self

    def _check_ready(self, require_centroids: bool = True) -> None:
        rotation_ready = (
            self.original_dim is not None
            and self.padded_dim is not None
            and self.subspace_len is not None
            and self.rotation is not None
        )
        if not rotation_ready:
            raise RuntimeError("VAQ codec rotation is not fitted")
        if require_centroids and (self.centroids is None or self.bits_alloc is None):
            raise RuntimeError("VAQ codec centroids are not fitted")

    @torch.no_grad()
    def reconstruct_normalized(
        self,
        y_norm: torch.Tensor,
        return_labels: bool = False,
        quantize: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Reconstruct normalized features. y_norm is [B, T, D].

        When ``quantize=False`` the per-subspace k-means step is skipped and the
        reconstruction is just PCA round-trip (used as a "PCA-only" diagnostic
        baseline that isolates the loss caused by the rotation/dim cap).
        """
        self._check_ready(require_centroids=quantize)
        orig_shape = y_norm.shape
        flat = y_norm.reshape(-1, orig_shape[-1]).float()
        if flat.shape[1] != self.original_dim:
            raise ValueError(f"expected D={self.original_dim}, got {flat.shape[1]}")

        device = flat.device
        R = torch.from_numpy(self.rotation).float().to(device)
        z = _pad_matrix(flat, self.padded_dim).matmul(R)

        if not quantize:
            y_hat = z.matmul(R.t())[:, : self.original_dim].reshape(orig_shape)
            return y_hat, None

        z_hat = torch.zeros_like(z)
        labels_out = []
        for g, cent_np in enumerate(self.centroids):
            start = g * self.subspace_len
            end = start + self.subspace_len
            sub = z[:, start:end]
            cent = torch.from_numpy(cent_np).float().to(device)
            max_mem = 768 * 1024**2
            chunk = max(1, min(sub.shape[0], max_mem // max(cent.shape[0] * 4, 1)))
            group_labels = []
            group_recon = []
            for s in range(0, sub.shape[0], chunk):
                e = min(s + chunk, sub.shape[0])
                dists = torch.cdist(sub[s:e], cent)
                labels = dists.argmin(dim=1)
                group_labels.append(labels)
                group_recon.append(cent[labels])
            labels_g = torch.cat(group_labels, dim=0)
            z_hat[:, start:end] = torch.cat(group_recon, dim=0)
            if return_labels:
                labels_out.append(labels_g)

        y_hat = z_hat.matmul(R.t())[:, : self.original_dim]
        y_hat = y_hat.reshape(orig_shape)
        if return_labels:
            return y_hat, torch.stack(labels_out, dim=0)
        return y_hat, None

    def state_dict(self) -> Dict[str, object]:
        self._check_ready()
        return {
            "config": self.config.__dict__,
            "original_dim": self.original_dim,
            "padded_dim": self.padded_dim,
            "subspace_len": self.subspace_len,
            "highest_subspaces": self.highest_subspaces,
            "rotation": self.rotation,
            "eigenvalues": self.eigenvalues,
            "variance_per_subspace": self.variance_per_subspace,
            "bits_alloc": self.bits_alloc,
            "centroids": self.centroids,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "VarianceAwareQuantizer":
        cfg_kwargs = dict(state["config"])
        cfg_kwargs.setdefault("bit_alloc_objective", "linear")
        cfg_kwargs.setdefault("kmeans_hierarchical_threshold", 10**9)  # legacy: flat
        cfg_kwargs.setdefault("kmeans_hierarchical_branching", 64)
        codec = cls(VAQConfig(**cfg_kwargs))
        codec.original_dim = int(state["original_dim"])
        codec.padded_dim = int(state["padded_dim"])
        codec.subspace_len = int(state["subspace_len"])
        codec.highest_subspaces = int(state["highest_subspaces"])
        codec.rotation = state["rotation"].astype(np.float32)
        codec.eigenvalues = state["eigenvalues"].astype(np.float32)
        codec.variance_per_subspace = state["variance_per_subspace"].astype(np.float32)
        codec.bits_alloc = [int(b) for b in state["bits_alloc"]]
        codec.centroids = [c.astype(np.float32) for c in state["centroids"]]
        return codec


def train_vaq_codec(
    features_train: Sequence[np.ndarray],
    norm_mode: str,
    bit_budget: int,
    num_subspaces: int,
    min_bits: int,
    max_bits: int,
    percent_var: float,
    max_fit_vectors: int,
    kmeans_iter: int,
    device: torch.device,
    seed: int,
    batch_size: int = 64,
    verbose: bool = True,
    bit_alloc_objective: str = "rd",
    kmeans_hierarchical_threshold: int = 1024,
    kmeans_hierarchical_branching: int = 64,
) -> VarianceAwareQuantizer:
    vectors = sample_normalized_vectors(
        features_train,
        norm_mode=norm_mode,
        max_vectors=max_fit_vectors,
        device=device,
        chunk_images=batch_size,
        seed=seed,
        verbose=verbose,
    )
    config = VAQConfig(
        bit_budget=bit_budget,
        num_subspaces=num_subspaces,
        min_bits=min_bits,
        max_bits=max_bits,
        percent_var=percent_var,
        kmeans_iter=kmeans_iter,
        seed=seed,
        bit_alloc_objective=bit_alloc_objective,
        kmeans_hierarchical_threshold=kmeans_hierarchical_threshold,
        kmeans_hierarchical_branching=kmeans_hierarchical_branching,
    )
    return VarianceAwareQuantizer(config).fit(vectors, device=device, verbose=verbose)


@torch.no_grad()
def vaq_encode_decode_features(
    features: Sequence[np.ndarray],
    codec: VarianceAwareQuantizer,
    norm_mode: str,
    device: torch.device,
    batch_size: int = 32,
) -> List[np.ndarray]:
    out = []
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        Y, mu, std = batch_normalize_gpu(X, mode=norm_mode)
        Y_hat, _ = codec.reconstruct_normalized(Y, return_labels=False)
        X_hat = batch_inv_normalize_gpu(Y_hat, mu, std)
        out.extend([x.cpu().numpy().astype(np.float32) for x in X_hat])
        del X, Y, mu, std, Y_hat, X_hat
    return out


@torch.no_grad()
def collect_vaq_labels(
    features: Sequence[np.ndarray],
    codec: VarianceAwareQuantizer,
    norm_mode: str,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    labels_all = []
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
        _, labels = codec.reconstruct_normalized(Y, return_labels=True)
        labels_all.append(labels.cpu())
        del X, Y, labels
    return torch.cat(labels_all, dim=1).numpy()


def histogram_pmfs(labels: np.ndarray, k_per_group: Sequence[int], smoothing: float = 1.0) -> List[np.ndarray]:
    pmfs = []
    for g, k in enumerate(k_per_group):
        counts = np.zeros(int(k), dtype=np.float64)
        np.add.at(counts, labels[g], 1)
        counts += smoothing
        pmfs.append(counts / counts.sum())
    return pmfs


def _cross_entropy_bpt(labels: np.ndarray, pmfs: Sequence[np.ndarray]) -> float:
    total = 0.0
    n = labels.shape[1]
    for g, p in enumerate(pmfs):
        total += -np.log2(p[labels[g]] + 1e-30).sum()
    return float(total / max(n, 1))


def _empirical_entropy_bpt(labels: np.ndarray, k_per_group: Sequence[int]) -> float:
    pmfs = histogram_pmfs(labels, k_per_group, smoothing=0.0)
    total = 0.0
    for p in pmfs:
        nz = p[p > 0]
        total += float(-(nz * np.log2(nz)).sum())
    return total


@torch.no_grad()
def diagnose_vaq_codec(
    features: Sequence[np.ndarray],
    codec: VarianceAwareQuantizer,
    norm_mode: str,
    device: torch.device,
    batch_size: int = 32,
    max_images: int = 200,
) -> Dict[str, object]:
    """Per-subspace MSE + codebook usage diagnostics.

    Computes for each of the G subspaces (in PCA-rotated normalized space):
      - input_var: variance of the subspace's PCA coords on the diagnosed split
      - mse:        mean squared reconstruction error after argmin lookup
      - snr_db:     10 * log10(input_var / mse)
      - K:          codebook size (= 2^bits)
      - used:       number of distinct centroids actually picked
      - used_frac:  used / K
      - top1_frac:  fraction of tokens assigned to the most popular centroid
      - entropy:    empirical Shannon entropy in bits (max = bits)

    A "healthy" codebook should have used_frac ~ 1.0, top1_frac small, and
    entropy close to bits. A collapsed/under-trained codebook shows used_frac << 1
    and/or top1_frac near 1.
    """
    codec._check_ready(require_centroids=True)
    feats = features[:max_images]
    # When percent_var < 1.0, only the first `highest_subspaces` of the G PCA
    # subspaces actually have codebooks; the trailing ones get zero bits and
    # are reconstructed as zeros. Diagnose only the subspaces with codebooks.
    G_total = codec.config.num_subspaces
    G = len(codec.centroids)  # = codec.highest_subspaces
    sub_len = codec.subspace_len
    R = torch.from_numpy(codec.rotation).float().to(device)

    sum_sq_err = np.zeros(G, dtype=np.float64)
    sum_sq_in = np.zeros(G, dtype=np.float64)
    sum_in = np.zeros(G, dtype=np.float64)
    n_tokens = 0
    label_counts: List[np.ndarray] = [
        np.zeros(int(k), dtype=np.int64) for k in codec.k_per_group
    ]
    # Also accumulate zero-bit (skipped) tail subspace input variance / MSE for
    # the reconstruction-as-zero error contribution.
    tail_sum_sq_in = 0.0
    tail_sum_in = 0.0
    tail_n = 0

    for start in range(0, len(feats), batch_size):
        end = min(start + batch_size, len(feats))
        X = torch.from_numpy(np.stack(feats[start:end])).float().to(device)
        Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
        flat = Y.reshape(-1, Y.shape[-1])
        z = _pad_matrix(flat, codec.padded_dim).matmul(R)
        n_tokens += z.shape[0]
        for g, cent_np in enumerate(codec.centroids):
            s, e_ = g * sub_len, (g + 1) * sub_len
            sub = z[:, s:e_]
            cent = torch.from_numpy(cent_np).float().to(device)
            dists = torch.cdist(sub, cent)
            labels = dists.argmin(dim=1)
            recon = cent[labels]
            err = (sub - recon).pow(2).sum().item()
            sum_sq_err[g] += err
            sum_sq_in[g] += sub.pow(2).sum().item()
            sum_in[g] += sub.sum().item()
            np.add.at(
                label_counts[g],
                labels.cpu().numpy().astype(np.int64),
                1,
            )
        if G < G_total:
            tail = z[:, G * sub_len :]
            tail_sum_sq_in += float(tail.pow(2).sum().item())
            tail_sum_in += float(tail.sum().item())
            tail_n += int(tail.numel())
        del X, Y, flat, z

    per_sub: List[Dict[str, float]] = []
    for g in range(G):
        var_in = max(sum_sq_in[g] / max(n_tokens * sub_len, 1)
                     - (sum_in[g] / max(n_tokens * sub_len, 1)) ** 2,
                     1e-30)
        mse = sum_sq_err[g] / max(n_tokens * sub_len, 1)
        K = int(codec.k_per_group[g])
        counts = label_counts[g]
        used = int((counts > 0).sum())
        top1_frac = float(counts.max() / max(counts.sum(), 1))
        nz = counts[counts > 0].astype(np.float64)
        nz = nz / nz.sum()
        H = float(-(nz * np.log2(nz)).sum()) if nz.size > 0 else 0.0
        snr = 10.0 * math.log10(var_in / max(mse, 1e-30))
        per_sub.append(
            {
                "g": g,
                "bits": int(codec.bits_alloc[g]),
                "K": K,
                "input_var": float(var_in),
                "mse": float(mse),
                "snr_db": snr,
                "used": used,
                "used_frac": used / max(K, 1),
                "top1_frac": top1_frac,
                "entropy": H,
            }
        )

    # Add a synthetic "tail (skipped)" entry summarizing zero-bit subspaces so
    # the printed table tells the whole story.
    if G < G_total and tail_n > 0:
        tail_var = max(
            tail_sum_sq_in / tail_n - (tail_sum_in / tail_n) ** 2, 1e-30
        )
        # MSE if reconstructed as zeros == E[x^2] = tail_sum_sq_in / tail_n
        tail_mse = tail_sum_sq_in / tail_n
        per_sub.append(
            {
                "g": -1,  # marker for "aggregated tail"
                "bits": 0,
                "K": 0,
                "input_var": float(tail_var),
                "mse": float(tail_mse),
                "snr_db": 10.0 * math.log10(max(tail_var, 1e-30) / max(tail_mse, 1e-30)),
                "used": 0,
                "used_frac": 0.0,
                "top1_frac": 0.0,
                "entropy": 0.0,
                "n_skipped_subspaces": int(G_total - G),
            }
        )

    total_var = float(sum(d["input_var"] for d in per_sub))
    weighted_mse = float(sum(d["mse"] for d in per_sub))
    return {
        "n_tokens": n_tokens,
        "per_subspace": per_sub,
        "total_input_var": total_var,
        "total_mse": weighted_mse,
        "global_snr_db": 10.0 * math.log10(max(total_var, 1e-30) / max(weighted_mse, 1e-30)),
    }


def evaluate_vaq_rate(
    features: Sequence[np.ndarray],
    codec: VarianceAwareQuantizer,
    norm_mode: str,
    device: torch.device,
    batch_size: int = 32,
    train_pmfs: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, float]:
    labels = collect_vaq_labels(features, codec, norm_mode, device, batch_size=batch_size)
    k_per_group = codec.k_per_group
    g_count = len(k_per_group)
    test_pmfs = histogram_pmfs(labels, k_per_group, smoothing=1.0)
    primary_pmfs = train_pmfs or test_pmfs
    # compressai's rANS encoder has a per-symbol alphabet cap (~2^14-2^16 in
    # practice). For very large K_g (e.g. K=65536 when max_bits=16) it raises
    # a TypeError + dumps the full input array, which bombs the log. Skip
    # rANS gracefully in that case and fall back to xent/H_emp for rate.
    rans_bpt = None
    rans_train_bpt = None
    max_k = max(int(k) for k in k_per_group)
    rans_safe_max_k = 1 << 14  # 16384 - safely below compressai's internal cap
    if max_k <= rans_safe_max_k:
        try:
            rans_bpt = _rans_encode_bpt(labels, primary_pmfs, g_count, k_per_group)
            if train_pmfs is not None:
                rans_train_bpt = _rans_encode_bpt(
                    labels, train_pmfs, g_count, k_per_group
                )
        except Exception as exc:  # pragma: no cover
            print(f"  [warn] rANS encoding failed ({type(exc).__name__}); "
                  "skipping rans_bpt. xent/H_emp are still valid rate proxies.")
            rans_bpt = None
            rans_train_bpt = None
    else:
        print(
            f"  [warn] max K_g = {max_k} exceeds rANS-safe cap {rans_safe_max_k}; "
            "skipping rans_bpt (use xent/H_emp instead)."
        )
    result = {
        "max_rate_bpt": float(sum(math.log2(k) for k in k_per_group)),
        "empirical_entropy_bpt": _empirical_entropy_bpt(labels, k_per_group),
        "xent_rate_bpt": _cross_entropy_bpt(labels, primary_pmfs),
    }
    if train_pmfs is not None:
        result["xent_train_bpt"] = _cross_entropy_bpt(labels, train_pmfs)
    if rans_bpt is not None:
        result["rans_bpt"] = float(rans_bpt)
    if rans_train_bpt is not None:
        result["rans_train_bpt"] = float(rans_train_bpt)
    return result


@torch.no_grad()
def evaluate_delta_l_ref_vaq(
    features: Sequence[np.ndarray],
    tail,
    codec: VarianceAwareQuantizer,
    norm_mode: str,
    device: torch.device,
    batch_size: int = 4,
) -> float:
    total = 0.0
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        teacher = tail.forward_nograd(X)
        Y, mu, std = batch_normalize_gpu(X, mode=norm_mode)
        Y_hat, _ = codec.reconstruct_normalized(Y, return_labels=False)
        X_hat = batch_inv_normalize_gpu(Y_hat, mu, std)
        student = tail.forward_nograd(X_hat)
        loss = ((teacher - student) ** 2).sum().item() / X.shape[0]
        total += loss * X.shape[0]
        del X, teacher, Y, mu, std, Y_hat, X_hat, student
    return float(total / max(len(features), 1))


def save_vaq_codec(codec: VarianceAwareQuantizer, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(codec.state_dict(), path)


def load_vaq_codec(path: str) -> VarianceAwareQuantizer:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return VarianceAwareQuantizer.from_state_dict(state)

