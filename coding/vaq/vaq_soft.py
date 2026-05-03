#!/usr/bin/env python
"""
VAQ-initialised Soft-PQ codec.

Combines:
  * VAQ's PCA rotation + non-uniform bit allocation + per-subspace K_g
  * Soft-PQ's Cayley orthogonal rotation + straight-through softmax +
    learnable codebooks

Stage 1 (initialiser, no learning): VarianceAwareQuantizer.fit() produces
    R   in R^{D_pad x D_pad}        (rotation, partial-balanced PCA)
    bits[g] for g in 0..G-1         (non-uniform bit allocation, sum = budget)
    c_g in R^{K_g x d}              (per-subspace k-means centroids)
We always run with ``percent_var=1.0`` so every subspace gets a real codebook
(K_g = 2^bits[g] >= 2 since min_bits >= 1). No "skipped" / zero-bit groups.

Stage 2 (this module, learnable): pack the per-group codebooks into a single
padded ``nn.Parameter`` of shape ``[G, K_max, d]`` with K_max = max_g K_g, mask
out positions ``[K_g .. K_max)`` of each row, and finetune end-to-end with
reconstruction MSE in the normalised feature space.

The masked "ghost" centroids are excluded from:
  * argmin distance       (we add +inf to invalid columns)
  * soft assignment       (same +inf -> 0 softmax weight)
  * usage / dead-entry stats

Memory: soft-PQ allocates a [G, B*T, K_max] softmax tensor. With the project
default K_max = 1024 (= 2^10), G = 32, B = 32, T = 257 this is ~1.1 GB per
tensor (fwd) + same for grads -> well within 24 GB GPUs.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


VAQ_ROOT = os.path.dirname(os.path.abspath(__file__))
ORFC_ROOT = os.path.normpath(os.path.join(VAQ_ROOT, "..", "orfc"))
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

from opq import batch_inv_normalize_gpu, batch_normalize_gpu  # noqa: E402
from soft_pq import OrthogonalTransform, _rans_encode_bpt  # noqa: E402

from feature_vaq import (  # noqa: E402
    VAQConfig,
    VarianceAwareQuantizer,
    _empirical_entropy_bpt,
    _cross_entropy_bpt,
    histogram_pmfs,
)


# ============================================================
#                VAQ-friendly orthogonal transform
# ============================================================

class VAQOrthogonalTransform(nn.Module):
    """OrthogonalTransform wrapper that handles VAQ-style D -> D_padded.

    encode([..., D]) zero-pads to D_padded and applies the Cayley rotation R.
    decode([..., D_padded]) applies R.T then truncates back to D.

    Internally holds a standard ``OrthogonalTransform`` of size D_padded so
    R^T R = I is guaranteed by construction.
    """

    def __init__(self, original_dim: int, padded_dim: int):
        super().__init__()
        if padded_dim < original_dim:
            raise ValueError(f"padded_dim={padded_dim} < original_dim={original_dim}")
        self.original_dim = int(original_dim)
        self.padded_dim = int(padded_dim)
        self.transform = OrthogonalTransform(self.padded_dim)

    @property
    def D(self) -> int:
        return self.padded_dim

    def get_rotation(self) -> torch.Tensor:
        return self.transform.get_rotation()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.padded_dim:
            return self.transform.encode(x)
        if x.shape[-1] != self.original_dim:
            raise ValueError(
                f"encode expects last dim {self.original_dim} or "
                f"{self.padded_dim}, got {x.shape[-1]}"
            )
        pad = x.new_zeros(*x.shape[:-1], self.padded_dim - self.original_dim)
        x_pad = torch.cat([x, pad], dim=-1)
        return self.transform.encode(x_pad)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.padded_dim:
            raise ValueError(
                f"decode expects last dim {self.padded_dim}, got {z.shape[-1]}"
            )
        x_pad = self.transform.decode(z)
        return x_pad[..., : self.original_dim]

    def init_from_opq(self, R_np: np.ndarray) -> None:
        if R_np.shape != (self.padded_dim, self.padded_dim):
            raise ValueError(
                f"R must be [{self.padded_dim}, {self.padded_dim}], "
                f"got {R_np.shape}"
            )
        self.transform.init_from_opq(R_np)

    def orth_error(self) -> float:
        return self.transform.orth_error()


# ============================================================
#                VAQ-aware Soft PQ quantiser
# ============================================================

class VAQSoftPQ(nn.Module):
    """Soft-PQ with per-subspace variable codebook size K_g.

    All G subspaces are real (K_g >= 2). Codebooks are packed as a single
    padded ``nn.Parameter`` of shape ``[G, K_max, d]``; positions
    ``[K_g .. K_max)`` of row g are masked out of distance / softmax / usage.
    """

    INF: float = 1.0e9

    def __init__(
        self,
        K_per_group: Sequence[int],
        d: int,
        lmbda: float = 0.0,
        prior_floor: float = 0.0,
    ):
        super().__init__()
        self.K_per_group: List[int] = [int(k) for k in K_per_group]
        if any(k < 2 for k in self.K_per_group):
            raise ValueError(
                f"K_g must be >= 2 (use percent_var=1.0 + min_bits>=1); "
                f"got K_per_group={self.K_per_group}"
            )
        self.G = len(self.K_per_group)
        self.K_max = int(max(self.K_per_group))
        self.d = int(d)
        self.D = self.G * self.d

        # --- Rate-distortion knobs (mirror orfc.soft_pq.SoftPQ) ---
        self.lmbda = float(lmbda)
        self.use_rate = self.lmbda > 0
        self.prior_floor = float(prior_floor)

        self.codebooks = nn.Parameter(torch.zeros(self.G, self.K_max, self.d))
        nn.init.normal_(self.codebooks, mean=0.0, std=0.01)

        mask = torch.zeros(self.G, self.K_max, dtype=torch.float32)
        for g, k in enumerate(self.K_per_group):
            mask[g, :k] = 1.0
        self.register_buffer("cb_mask", mask)
        # +inf added to invalid distances; precompute for cheap reuse.
        self.register_buffer("dist_pad", (1.0 - mask) * self.INF)

        # log_prior is a learnable per-group categorical logit. Padding slots
        # (K_g..K_max) must be excluded from the softmax. We achieve this by
        # *adding* prior_pad to the logits at every forward (constant -INF on
        # padding rows, 0 elsewhere). Initial values are 0 -> uniform prior.
        if self.use_rate:
            self.log_prior = nn.Parameter(torch.zeros(self.G, self.K_max))
            self.register_buffer("prior_pad", (1.0 - mask) * (-self.INF))
        else:
            # Keep attribute for safe getattr access elsewhere.
            self.log_prior = None
            self.register_buffer("prior_pad", torch.zeros(self.G, self.K_max))

        self.temperature: float = 0.0
        self._last_labels: Optional[torch.Tensor] = None
        # Rate accumulators (set by every _quantise call when use_rate).
        self._last_rate: Optional[torch.Tensor] = None
        self._last_rate_per_group: Optional[torch.Tensor] = None

    @property
    def k_per_group(self) -> List[int]:
        return list(self.K_per_group)

    @property
    def max_rate_bpt(self) -> float:
        return float(sum(math.log2(max(k, 1)) for k in self.K_per_group))

    def init_from_centroids(self, centroids: Sequence[np.ndarray]) -> None:
        """Copy a list of [K_g, d] arrays into the padded codebooks.

        Padded slots are filled with the per-group mean (close to zero in
        centred PCA coords) plus tiny jitter; they are masked out at runtime
        but a sane init avoids NaNs in any path that may touch them.
        """
        if len(centroids) != self.G:
            raise ValueError(
                f"got {len(centroids)} centroid groups, expected G={self.G}"
            )
        with torch.no_grad():
            for g, c in enumerate(centroids):
                k = self.K_per_group[g]
                if c.shape != (k, self.d):
                    raise ValueError(
                        f"group {g}: centroid shape {c.shape} != expected ({k}, {self.d})"
                    )
                c_t = torch.from_numpy(c.astype(np.float32))
                self.codebooks.data[g, :k].copy_(c_t)
                pad_n = self.K_max - k
                if pad_n > 0:
                    m = c_t.mean(dim=0)
                    self.codebooks.data[g, k:].copy_(
                        m.unsqueeze(0).expand(pad_n, -1)
                        + 1e-4 * torch.randn(pad_n, self.d)
                    )

    @torch.no_grad()
    def init_prior_from_freq(self, usage_counts, smoothing: float = 1.0) -> None:
        """Initialise log_prior from empirical assignment frequency.

        Parameters
        ----------
        usage_counts : np.ndarray | torch.Tensor | Sequence[np.ndarray]
            Either a [G, K_max] padded array (padding entries should be 0) or
            a list of length-K_g arrays. Counts are smoothed (Laplace) and
            normalised per group; padding rows of log_prior are set to -INF.
        """
        if not self.use_rate:
            return
        # Coerce to padded [G, K_max] numpy array.
        if isinstance(usage_counts, (list, tuple)):
            arr = np.zeros((self.G, self.K_max), dtype=np.float64)
            for g, c in enumerate(usage_counts):
                k = self.K_per_group[g]
                arr[g, :k] = np.asarray(c, dtype=np.float64)[:k]
        else:
            if isinstance(usage_counts, torch.Tensor):
                usage_counts = usage_counts.detach().cpu().numpy()
            arr = np.asarray(usage_counts, dtype=np.float64)
            if arr.shape != (self.G, self.K_max):
                raise ValueError(
                    f"usage_counts shape {arr.shape} != ({self.G}, {self.K_max})"
                )
        # Apply Laplace smoothing only on valid slots; renormalise per group.
        mask_np = self.cb_mask.detach().cpu().numpy()
        arr = arr * mask_np  # zero out padding (defensive)
        arr = arr + smoothing * mask_np  # smoothing only on valid slots
        denom = arr.sum(axis=-1, keepdims=True)
        denom = np.maximum(denom, 1e-12)
        freq = arr / denom
        # log(0) = -inf for padding -> hard mask.
        with np.errstate(divide="ignore"):
            lp = np.where(mask_np > 0, np.log(np.maximum(freq, 1e-30)), -self.INF)
        self.log_prior.data.copy_(torch.from_numpy(lp.astype(np.float32)))

    @torch.no_grad()
    def get_prior_pmf(self) -> List[np.ndarray]:
        """Return a list of length-K_g PMF arrays (one per group)."""
        if not self.use_rate:
            return [np.full(int(k), 1.0 / int(k), dtype=np.float32)
                    for k in self.K_per_group]
        logits = self.log_prior + self.prior_pad
        pmf_full = F.softmax(logits, dim=-1).detach().cpu().numpy()
        return [pmf_full[g, :self.K_per_group[g]].astype(np.float32)
                for g in range(self.G)]

    def _quantise(self, Z_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Z_flat: [N, G*d] -> Z_hat_flat: [N, G*d], usage: [G, K_max].

        With ``use_rate`` we follow the ECVQ rule of soft_pq:

            cost_k = ||z_g - c_k||^2 + (-log2 p_k) / lambda

        Distances are computed via the L2 expansion ||a||^2 + ||b||^2 - 2 a.b
        instead of ``torch.cdist`` to keep autograd from materialising the
        [G, N, K_max, d] difference tensor (which OOMs at typical sizes).
        """
        N = Z_flat.shape[0]
        sub_g = Z_flat.reshape(N, self.G, self.d).permute(1, 0, 2).contiguous()
        x_sq = (sub_g * sub_g).sum(dim=-1, keepdim=True)             # [G, N, 1]
        c_sq = (self.codebooks * self.codebooks).sum(dim=-1).unsqueeze(1)  # [G, 1, K_max]
        xc = torch.einsum("gnd,gkd->gnk", sub_g, self.codebooks)     # [G, N, K_max]
        dists_sq = (x_sq + c_sq - 2.0 * xc).clamp_min(0.0)
        dists_sq = dists_sq + self.dist_pad.unsqueeze(1)

        log2_pmf = None
        if self.use_rate:
            # Padding-safe log-softmax: -INF on padded slots.
            log_p = F.log_softmax(self.log_prior + self.prior_pad, dim=-1)
            if self.prior_floor > 0:
                p = log_p.exp()
                # Distribute floor mass uniformly across *valid* slots only.
                k_valid = self.cb_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                p = (1.0 - self.prior_floor) * p + (self.prior_floor / k_valid) * self.cb_mask
                log2_pmf = -(p + 1e-30).log() / math.log(2)
            else:
                log2_pmf = -log_p / math.log(2)
            # log2_pmf is +INF on padding; this naturally rules them out via cost.
            cost = dists_sq + log2_pmf.unsqueeze(1) / self.lmbda
        else:
            cost = dists_sq

        labels = cost.argmin(dim=-1)
        self._last_labels = labels.detach()

        if self.training and self.temperature > 0:
            logits = -cost / self.temperature
            soft = F.softmax(logits, dim=-1)
            hard = torch.zeros_like(soft).scatter_(
                -1, labels.unsqueeze(-1), 1.0
            )
            weights = hard - soft.detach() + soft
            Z_hat_g = torch.einsum("gnk,gkd->gnd", weights, self.codebooks)
        else:
            # Fancy-indexing avoids materialising the [G, N, K_max, d] view
            # that ``torch.gather`` would otherwise build (and which OOMs at
            # G=32, N=B*T~8k, K_max=1024). Indexing produces only [G, N, d].
            group_idx = torch.arange(
                self.G, device=labels.device
            ).unsqueeze(1).expand(-1, N)
            Z_hat_g = self.codebooks[group_idx, labels]  # [G, N, d]

        Z_hat_flat = Z_hat_g.permute(1, 0, 2).reshape(N, self.D)

        usage = torch.zeros(self.G, self.K_max, device=Z_flat.device)
        usage.scatter_add_(
            1,
            labels,
            torch.ones_like(labels, dtype=torch.float32),
        )
        usage = usage * self.cb_mask

        if self.use_rate:
            # gathered_rate[g, n] = log2_pmf[g, labels[g, n]] (bits/token/group)
            gathered = torch.gather(log2_pmf, 1, labels)            # [G, N]
            self._last_rate = gathered.sum(0).mean()                # bits/token (avg over N)
            self._last_rate_per_group = gathered.mean(1)            # [G]
        else:
            self._last_rate = torch.tensor(0.0, device=Z_flat.device)
            self._last_rate_per_group = None

        return Z_hat_flat, usage

    def forward(self, Z_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Z_norm: [B, T, D'] -> (Z_hat, usage)."""
        B, T, Dp = Z_norm.shape
        if Dp != self.D:
            raise ValueError(f"expected D'={self.D}, got {Dp}")
        Z_hat, usage = self._quantise(Z_norm.reshape(B * T, Dp))
        return Z_hat.reshape(B, T, Dp), usage


# ============================================================
#                Composed feature codec
# ============================================================

class VAQSoftFeatureCodec(nn.Module):
    """Pipeline:  Y_norm --[transform.encode]--> Z --[pq]--> Z_hat
                  --[transform.decode]--> Y_hat
    """

    def __init__(self, transform: VAQOrthogonalTransform, pq: VAQSoftPQ):
        super().__init__()
        self.transform = transform
        self.pq = pq

    def forward(self, Y_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = Y_norm.shape
        flat = Y_norm.reshape(B * T, D)
        Z = self.transform.encode(flat)
        Z_hat, usage = self.pq._quantise(Z)
        Y_hat = self.transform.decode(Z_hat)
        return Y_hat.reshape(B, T, D), usage

    @property
    def k_per_group(self) -> List[int]:
        return self.pq.k_per_group


# ============================================================
#         Build a VAQ-Soft codec from a fitted VAQ codec
# ============================================================

def build_codec_from_vaq(
    vaq: VarianceAwareQuantizer,
    device: torch.device,
    lmbda: float = 0.0,
    prior_floor: float = 0.0,
) -> VAQSoftFeatureCodec:
    """Construct a learnable VAQSoftFeatureCodec initialised from a fitted VAQ.

    Requires ``percent_var=1.0`` (every subspace has a real codebook).

    - Rotation: VAQ's PCA + partial-balanced rotation R [D_pad, D_pad] is
      copied into a Cayley-parametrised ``OrthogonalTransform``.
    - Codebooks: VAQ's per-group centroids [K_g, d] are packed into a padded
      [G, K_max, d] tensor; padding columns are masked at runtime.
    """
    vaq._check_ready(require_centroids=True)
    G_total = int(vaq.config.num_subspaces)
    G_fit = int(vaq.highest_subspaces or G_total)
    if G_fit != G_total:
        raise ValueError(
            f"VAQSoftFeatureCodec requires percent_var=1.0 so every subspace "
            f"is fitted; got highest_subspaces={G_fit} < num_subspaces={G_total}."
        )
    d = int(vaq.subspace_len)
    K_per_group: List[int] = [int(k) for k in vaq.k_per_group]
    centroids: List[np.ndarray] = [c.astype(np.float32) for c in vaq.centroids]

    transform = VAQOrthogonalTransform(
        original_dim=int(vaq.original_dim),
        padded_dim=int(vaq.padded_dim),
    ).to(device)
    transform.init_from_opq(vaq.rotation.astype(np.float32))

    pq = VAQSoftPQ(
        K_per_group=K_per_group,
        d=d,
        lmbda=lmbda,
        prior_floor=prior_floor,
    ).to(device)
    pq.init_from_centroids(centroids)

    return VAQSoftFeatureCodec(transform=transform, pq=pq).to(device)


# ============================================================
#                Training loop (MSE in normalised space)
# ============================================================

def _stack_features(features: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack(features).astype(np.float32, copy=False)


@torch.no_grad()
def _eval_mse(
    codec: "VAQSoftFeatureCodec",
    features_array: np.ndarray,
    norm_mode: str,
    device: torch.device,
    batch_size: int,
) -> float:
    """Compute average per-image MSE in normalised space (no backbone needed)."""
    codec.eval()
    total = 0.0
    n = 0
    for s in range(0, features_array.shape[0], batch_size):
        e = min(s + batch_size, features_array.shape[0])
        X = torch.from_numpy(features_array[s:e]).to(device)
        Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
        Y_hat, _ = codec(Y)
        total += ((Y - Y_hat) ** 2).sum().item()
        n += X.shape[0]
        del X, Y, Y_hat
    return total / max(n, 1)


def train_vaq_soft(
    codec: VAQSoftFeatureCodec,
    features_train: Sequence[np.ndarray],
    norm_mode: str,
    epochs: int = 30,
    lr: float = 1.0e-4,
    batch_size: int = 32,
    device: torch.device = torch.device("cuda"),
    seed: int = 42,
    val_features: Optional[Sequence[np.ndarray]] = None,
    tau_start: float = 0.0,
    tau_end: float = 0.0,
    tau_schedule: str = "exponential",
    grad_clip: float = 1.0,
    freeze_transform: bool = False,
    freeze_codebooks: bool = False,
    verbose: bool = True,
    tail=None,
    use_lref: bool = False,
    snapshot_epochs: Optional[List[int]] = None,
) -> Tuple[List[Dict[str, float]], Dict[int, dict]]:
    """End-to-end finetune of (transform, codebooks) on top of VAQ init.

    By default both the codebooks and the Cayley orthogonal rotation are
    trainable; ``freeze_transform=True`` reduces to "stage 2 only" finetuning
    of codebooks over the fixed VAQ-PCA rotation.

    With ``tau_start <= 0`` the codebooks are updated through the (still
    differentiable) hard ``torch.gather`` path -> mini-batch online k-means
    refinement of the VAQ init. With ``tau_start > 0`` the standard
    straight-through softmax is used.

    Distortion term (``D``):
      - ``use_lref=False`` (default): MSE in normalised feature space,
        ``D = ||Y - Y_hat||^2``.
      - ``use_lref=True``: ΔL_ref distillation through the frozen ViT tail,
        ``D = ||tail(X) - tail(inv_norm(Y_hat))||^2``. Requires ``tail``
        (a ``soft_pq.FrozenTail``) to be passed in. Mirrors the reference
        implementation in ``orfc/run_soft_pq.py``: pre-computes teacher
        outputs once at the start so each batch only forwards the *student*
        through the (frozen, but autograd-enabled) tail. Uses ``batch_size``
        for both teacher pre-compute and per-step student forward.

    ``snapshot_epochs``: if provided, save a deep copy of the codec
    state_dict at each listed epoch (0-based, after optimizer step).
    Returned as ``{epoch: state_dict}`` alongside the history.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if freeze_transform:
        for p in codec.transform.parameters():
            p.requires_grad_(False)
    if freeze_codebooks:
        codec.pq.codebooks.requires_grad_(False)

    _snap_epochs = set(snapshot_epochs) if snapshot_epochs else set()
    snapshots: Dict[int, dict] = {}

    trainable = [p for p in codec.parameters() if p.requires_grad]
    if not trainable:
        if verbose:
            print("  [train_vaq_soft] no trainable params - skipping training.")
        return [], snapshots

    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=lr * 0.01
    )

    use_soft = tau_start > 0
    use_rate = bool(getattr(codec.pq, "use_rate", False))
    lmbda = float(getattr(codec.pq, "lmbda", 0.0))
    T_tokens = int(features_train[0].shape[0])
    if use_lref and tail is None:
        raise ValueError("train_vaq_soft: use_lref=True requires `tail`")
    if verbose:
        n_trainable = sum(p.numel() for p in trainable)
        print(f"  Trainable params: {n_trainable:,}")
        print(
            f"  freeze_transform={freeze_transform}  "
            f"freeze_codebooks={freeze_codebooks}"
        )
        print(f"  K_per_group (first 16): {codec.pq.K_per_group[:16]}  "
              f"K_max={codec.pq.K_max}")
        print(f"  orth_error(R-init)={codec.transform.orth_error():.2e}")
        if use_soft:
            print(f"  Soft PQ: tau {tau_start} -> {tau_end} ({tau_schedule})")
        else:
            print("  Hard PQ (tau=0)")
        if use_lref:
            print(
                "  Distortion: ΔL_ref distillation through frozen ViT tail"
            )
        else:
            print("  Distortion: MSE in normalised feature space")
        if use_rate:
            print(
                f"  Rate-aware: lambda={lmbda}  prior_floor={codec.pq.prior_floor}  "
                f"loss = R*T + lambda*D  (T={T_tokens})"
            )

    features_array = _stack_features(features_train)
    val_array = _stack_features(val_features) if val_features else None
    N_img = features_array.shape[0]

    teacher_cache = None
    val_teacher_cache = None
    if use_lref:
        if verbose:
            print(f"  Pre-computing teacher outputs (train, {N_img} images)...")
        t_pre = time.time()
        teacher_cache = np.empty_like(features_array)
        with torch.no_grad():
            for s in range(0, N_img, batch_size):
                e = min(s + batch_size, N_img)
                X_chunk = torch.from_numpy(features_array[s:e]).float().to(device)
                teacher_cache[s:e] = tail.forward_nograd(X_chunk).cpu().numpy()
                del X_chunk
        torch.cuda.empty_cache()
        if verbose:
            cache_gb = teacher_cache.nbytes / 1e9
            print(
                f"  Teacher cache (train): {cache_gb:.2f} GB CPU "
                f"({time.time() - t_pre:.1f}s)"
            )
        if val_array is not None:
            val_teacher_cache = np.empty_like(val_array)
            with torch.no_grad():
                for s in range(0, val_array.shape[0], batch_size):
                    e = min(s + batch_size, val_array.shape[0])
                    X_v = torch.from_numpy(val_array[s:e]).float().to(device)
                    val_teacher_cache[s:e] = tail.forward_nograd(X_v).cpu().numpy()
                    del X_v
            torch.cuda.empty_cache()

    history: List[Dict[str, float]] = []

    # Pre-training baseline distortion (so we can tell if training actually
    # improves over the VAQ-init checkpoint). For the L_ref path, we just
    # log the MSE baseline too -- a "true" L_ref baseline would re-forward
    # tail per batch and is expensive; the per-epoch logs below report the
    # actual L_ref values.
    init_train_mse = _eval_mse(codec, features_array, norm_mode, device, batch_size)
    init_val_mse = (
        _eval_mse(codec, val_array, norm_mode, device, batch_size)
        if val_array is not None else None
    )
    if verbose:
        v = f"  val_D={init_val_mse:.2f}" if init_val_mse is not None else ""
        d_tag = "MSE_baseline"
        print(f"  init        D({d_tag})={init_train_mse:.2f}{v}  (VAQ-init)")

    indices = np.arange(N_img)
    rng = np.random.RandomState(seed)

    for epoch in range(epochs):
        t_epoch = time.time()
        if use_soft and epochs > 1:
            progress = epoch / (epochs - 1)
            if tau_schedule == "linear":
                tau = tau_start + (tau_end - tau_start) * progress
            else:
                tau = tau_start * (tau_end / max(tau_start, 1e-12)) ** progress
            codec.pq.temperature = float(tau)
        elif use_soft:
            codec.pq.temperature = float(tau_start)
        else:
            codec.pq.temperature = 0.0

        rng.shuffle(indices)
        codec.train()
        total_distortion = 0.0
        total_rate = 0.0
        total_imgs = 0
        usage_acc = torch.zeros(codec.pq.G, codec.pq.K_max, device=device)

        for start in range(0, N_img, batch_size):
            end = min(start + batch_size, N_img)
            batch_idx = indices[start:end]
            X = torch.from_numpy(features_array[batch_idx]).to(device)
            B = X.shape[0]
            with torch.no_grad():
                Y, mu, std = batch_normalize_gpu(X, mode=norm_mode)
            Y_hat, usage = codec(Y)
            if use_lref:
                Y_teacher = torch.from_numpy(
                    teacher_cache[batch_idx]
                ).to(device, non_blocking=True)
                X_hat = batch_inv_normalize_gpu(Y_hat, mu, std)
                Y_student = tail(X_hat)
                distortion = ((Y_teacher - Y_student) ** 2).sum() / B
                del Y_teacher, X_hat, Y_student
            else:
                distortion = ((Y - Y_hat) ** 2).sum() / B
            if use_rate:
                # Rate is in bits/token (averaged over all N=B*T flattened
                # codewords, summed over G groups) -> multiply by T to scale
                # to per-image bits, then add lambda * distortion.
                rate_bits = codec.pq._last_rate * T_tokens
                loss = rate_bits + lmbda * distortion
            else:
                loss = distortion
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(codec.parameters(), grad_clip)
            optimizer.step()

            total_distortion += distortion.item() * B
            if use_rate:
                total_rate += codec.pq._last_rate.item() * B
            total_imgs += B
            usage_acc += usage.detach()
            del X, Y, mu, std, Y_hat, usage, distortion, loss

        scheduler.step()
        avg_d = total_distortion / max(total_imgs, 1)
        avg_r = total_rate / max(total_imgs, 1) if use_rate else 0.0

        usage_pmf = usage_acc / usage_acc.sum(dim=-1, keepdim=True).clamp_min(1.0)
        ppl = (-(usage_pmf.clamp_min(1e-30).log() * usage_pmf).sum(dim=-1)).exp().mean().item()
        # Mask out padding slots (always 0 by construction) before counting
        # truly dead codebook entries.
        valid_mask = codec.pq.cb_mask.bool()
        dead = int(((usage_acc == 0) & valid_mask).sum().item())

        val_loss = None
        if val_array is not None:
            codec.eval()
            with torch.no_grad():
                vd_sum = 0.0
                vc = 0
                for s in range(0, val_array.shape[0], batch_size):
                    e = min(s + batch_size, val_array.shape[0])
                    Xv = torch.from_numpy(val_array[s:e]).to(device)
                    Bv = Xv.shape[0]
                    Yv, muv, stdv = batch_normalize_gpu(Xv, mode=norm_mode)
                    Yhv, _ = codec(Yv)
                    if use_lref:
                        Yt_v = torch.from_numpy(
                            val_teacher_cache[s:e]
                        ).to(device, non_blocking=True)
                        Xh_v = batch_inv_normalize_gpu(Yhv, muv, stdv)
                        Yo_v = tail.forward_nograd(Xh_v)
                        vd_sum += ((Yt_v - Yo_v) ** 2).sum().item()
                        del Yt_v, Xh_v, Yo_v
                    else:
                        vd_sum += ((Yv - Yhv) ** 2).sum().item()
                    vc += Bv
                    del Xv, Yv, muv, stdv, Yhv
                val_loss = vd_sum / max(vc, 1)

        info = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "loss_distortion": avg_d,
            "rate_bpt": avg_r,
            "perplexity": ppl,
            "dead_entries": dead,
            "temperature": codec.pq.temperature,
            "val_loss": val_loss,
            "orth_error": codec.transform.orth_error(),
            "time": time.time() - t_epoch,
        }
        if use_rate and codec.pq._last_rate_per_group is not None:
            info["rate_per_group"] = (
                codec.pq._last_rate_per_group.detach().cpu().tolist()
            )
        history.append(info)

        if epoch in _snap_epochs:
            import copy as _copy
            snapshots[epoch] = _copy.deepcopy(codec.state_dict())
            if verbose:
                print(f"  ** snapshot saved at epoch {epoch}")

        if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
            val_str = f"  val_D={val_loss:.2f}" if val_loss is not None else ""
            tau_str = f"  tau={codec.pq.temperature:.4f}" if use_soft else ""
            rate_str = f"  R={avg_r:.2f}bpt  lD={lmbda * avg_d:.1f}" if use_rate else ""
            print(
                f"  ep {epoch:3d}/{epochs}  D={avg_d:.2f}{rate_str}  ppl={ppl:.1f}  "
                f"dead={dead}{tau_str}{val_str}  ({time.time() - t_epoch:.1f}s)"
            )

    return history, snapshots


# ============================================================
#                Inference / evaluation helpers
# ============================================================

@torch.no_grad()
def encode_decode_features(
    features: Sequence[np.ndarray],
    codec: VAQSoftFeatureCodec,
    norm_mode: str,
    device: torch.device,
    batch_size: int = 32,
) -> List[np.ndarray]:
    codec.eval()
    out: List[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        Y, mu, std = batch_normalize_gpu(X, mode=norm_mode)
        Y_hat, _ = codec(Y)
        X_hat = batch_inv_normalize_gpu(Y_hat, mu, std)
        for i in range(X_hat.shape[0]):
            out.append(X_hat[i].cpu().numpy().astype(np.float32))
        del X, Y, mu, std, Y_hat, X_hat
    return out


@torch.no_grad()
def collect_labels(
    features: Sequence[np.ndarray],
    codec: VAQSoftFeatureCodec,
    norm_mode: str,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    codec.eval()
    chunks: List[torch.Tensor] = []
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
        _ = codec(Y)
        chunks.append(codec.pq._last_labels.cpu())
        del X, Y
    return torch.cat(chunks, dim=1).numpy()


@torch.no_grad()
def evaluate_delta_l_ref(
    features: Sequence[np.ndarray],
    tail,
    codec: VAQSoftFeatureCodec,
    norm_mode: str,
    device: torch.device,
    batch_size: int = 4,
) -> float:
    codec.eval()
    total = 0.0
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        teacher = tail.forward_nograd(X)
        Y, mu, std = batch_normalize_gpu(X, mode=norm_mode)
        Y_hat, _ = codec(Y)
        X_hat = batch_inv_normalize_gpu(Y_hat, mu, std)
        student = tail.forward_nograd(X_hat)
        loss = ((teacher - student) ** 2).sum().item() / X.shape[0]
        total += loss * X.shape[0]
        del X, Y, mu, std, Y_hat, X_hat, teacher, student
    return float(total / max(len(features), 1))


def evaluate_rate(
    features: Sequence[np.ndarray],
    codec: VAQSoftFeatureCodec,
    norm_mode: str,
    device: torch.device,
    batch_size: int = 32,
    train_pmfs: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, float]:
    labels = collect_labels(features, codec, norm_mode, device, batch_size)
    k_per_group = codec.k_per_group
    g_count = len(k_per_group)
    test_pmfs = histogram_pmfs(labels, k_per_group, smoothing=1.0)
    primary_pmfs = train_pmfs or test_pmfs
    rans_bpt = None
    rans_train_bpt = None
    rans_safe_max_k = 1 << 14
    max_k = max(int(k) for k in k_per_group)
    if max_k <= rans_safe_max_k:
        try:
            rans_bpt = _rans_encode_bpt(labels, primary_pmfs, g_count, k_per_group)
            if train_pmfs is not None:
                rans_train_bpt = _rans_encode_bpt(
                    labels, train_pmfs, g_count, k_per_group
                )
        except Exception as exc:  # pragma: no cover
            print(
                f"  [warn] rANS encoding failed ({type(exc).__name__}); "
                "skipping rans_bpt."
            )
            rans_bpt = None
            rans_train_bpt = None
    else:
        print(
            f"  [warn] max K_g = {max_k} exceeds rANS-safe cap {rans_safe_max_k}; "
            "skipping rans_bpt."
        )
    result: Dict[str, float] = {
        "max_rate_bpt": float(sum(math.log2(max(k, 1)) for k in k_per_group)),
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


# ============================================================
#                Save / load
# ============================================================

def save_codec(
    codec: VAQSoftFeatureCodec,
    path: str,
    extra_meta: Optional[Dict] = None,
) -> None:
    """Persist a VAQSoftFeatureCodec to disk.

    The .pt file is fully self-describing for **architecture** (K_per_group,
    d, original/padded dim are enough to reconstruct the modules). Optional
    ``extra_meta`` lets callers persist **training context** (e.g. layer,
    backbone, norm_mode, bit_budget, args dict, bits_alloc, k_per_group),
    making downstream ``--eval_only`` invocations self-sufficient.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = {
        "K_per_group": list(codec.pq.K_per_group),
        "d": int(codec.pq.d),
        "original_dim": int(codec.transform.original_dim),
        "padded_dim": int(codec.transform.padded_dim),
        "lmbda": float(codec.pq.lmbda),
        "prior_floor": float(codec.pq.prior_floor),
        "state_dict": codec.state_dict(),
        "format_version": 3,
    }
    if extra_meta:
        # Avoid clobbering core architecture / state_dict keys.
        for k, v in extra_meta.items():
            if k in ("state_dict", "K_per_group", "d", "original_dim", "padded_dim"):
                continue
            meta[k] = v
    torch.save(meta, path)


def load_codec_with_meta(
    path: str,
    device: torch.device = torch.device("cuda"),
) -> Tuple[VAQSoftFeatureCodec, Dict]:
    """Load a codec **and** return the full meta dict (for eval-only flows)."""
    meta = torch.load(path, map_location="cpu")
    transform = VAQOrthogonalTransform(
        original_dim=int(meta["original_dim"]),
        padded_dim=int(meta["padded_dim"]),
    )
    pq = VAQSoftPQ(
        K_per_group=list(meta["K_per_group"]),
        d=int(meta["d"]),
        lmbda=float(meta.get("lmbda", 0.0)),
        prior_floor=float(meta.get("prior_floor", 0.0)),
    )
    codec = VAQSoftFeatureCodec(transform=transform, pq=pq)
    # Non-strict load: format_version=2 ckpts lack the new ``pq.prior_pad``
    # buffer (and ``pq.log_prior`` when use_rate=False), but those are fully
    # determined by K_per_group / lmbda which we already pass to the constructor.
    missing, unexpected = codec.load_state_dict(meta["state_dict"], strict=False)
    expected_missing = {"pq.prior_pad"}
    if pq.use_rate:
        # log_prior was registered fresh; if ckpt is rate-aware it'll be in
        # the state_dict already. If not (loading old non-rate ckpt as rate-
        # aware -> doesn't make sense, will leave log_prior at zero = uniform).
        expected_missing.discard("pq.log_prior")
    real_missing = [k for k in missing if k not in expected_missing]
    if real_missing:
        raise RuntimeError(f"Missing keys in ckpt state_dict: {real_missing}")
    if unexpected:
        # log_prior present in ckpt but lmbda=0 in meta: warn but allow.
        print(f"  [load_codec] ignoring unexpected ckpt keys: {unexpected}")
    return codec.to(device).eval(), meta


def load_codec(
    path: str,
    device: torch.device = torch.device("cuda"),
) -> VAQSoftFeatureCodec:
    """Backward-compatible loader (returns codec only)."""
    codec, _meta = load_codec_with_meta(path, device=device)
    return codec


# ============================================================
#                Convenience: full fit pipeline
# ============================================================

def fit_vaq_then_finetune(
    features_train: Sequence[np.ndarray],
    norm_mode: str,
    bit_budget: int,
    num_subspaces: int,
    min_bits: int,
    max_bits: int,
    max_fit_vectors: int,
    kmeans_iter: int,
    bit_alloc_objective: str,
    kmeans_hier_threshold: int,
    kmeans_hier_branching: int,
    epochs: int,
    lr: float,
    batch_size: int,
    tau_start: float,
    tau_end: float,
    tau_schedule: str,
    grad_clip: float,
    freeze_transform: bool,
    freeze_codebooks: bool,
    device: torch.device,
    seed: int,
    val_features: Optional[Sequence[np.ndarray]] = None,
    verbose: bool = True,
    lmbda: float = 0.0,
    prior_floor: float = 0.0,
    init_prior_from_vaq_usage: bool = True,
    tail=None,
    use_lref: bool = False,
    snapshot_epochs: Optional[List[int]] = None,
) -> Tuple[VAQSoftFeatureCodec, VarianceAwareQuantizer, List[Dict[str, float]], Dict[int, dict]]:
    """One-shot helper: VAQ fit -> wrap into VAQSoftFeatureCodec -> MSE finetune.

    ``percent_var`` is internally fixed to 1.0 so every subspace has a real
    codebook (K_g = 2^bits[g] >= 2). Bit allocation feasibility therefore
    requires ``num_subspaces * min_bits <= bit_budget <= num_subspaces *
    max_bits``.

    Returns ``(codec, vaq, history, snapshots)`` where *snapshots* is a dict
    mapping epoch number to a deep-copied ``state_dict`` captured at the end
    of that epoch (empty if ``snapshot_epochs`` is None).
    """
    from feature_vaq import sample_normalized_vectors  # local import to avoid cycles

    if verbose:
        print("\n[stage 1] VAQ initialisation (PCA + bit alloc + k-means)...")
    vectors = sample_normalized_vectors(
        features_train,
        norm_mode=norm_mode,
        max_vectors=max_fit_vectors,
        device=device,
        chunk_images=batch_size,
        seed=seed,
        verbose=verbose,
    )
    cfg = VAQConfig(
        bit_budget=bit_budget,
        num_subspaces=num_subspaces,
        min_bits=min_bits,
        max_bits=max_bits,
        percent_var=1.0,  # enforced: no skipped subspaces
        kmeans_iter=kmeans_iter,
        seed=seed,
        bit_alloc_objective=bit_alloc_objective,
        kmeans_hierarchical_threshold=kmeans_hier_threshold,
        kmeans_hierarchical_branching=kmeans_hier_branching,
    )
    vaq = VarianceAwareQuantizer(cfg).fit(vectors, device=device, verbose=verbose)
    del vectors

    if verbose:
        print(
            f"  VAQ bits: {vaq.bits_alloc} (sum={sum(vaq.bits_alloc)}),"
            f" K_g (first 16): {vaq.k_per_group[:16]} K_max={max(vaq.k_per_group)}"
        )
        loss_tag = "ΔL_ref" if use_lref else "MSE"
        print(f"\n[stage 2] Wrapping into VAQ-Soft codec and finetuning with {loss_tag}...")

    codec = build_codec_from_vaq(
        vaq, device=device, lmbda=lmbda, prior_floor=prior_floor,
    )

    # Bootstrap log_prior from VAQ argmin usage frequencies (mirrors soft_pq's
    # prior_init_counts path; gives the rate-aware quantiser a sensible
    # starting categorical so the first epoch isn't dominated by the uniform
    # prior penalty).
    if codec.pq.use_rate and init_prior_from_vaq_usage:
        if verbose:
            print("  Initialising log_prior from VAQ argmin usage...")
        from feature_vaq import collect_vaq_labels as _collect_vaq_labels
        vaq_labels = _collect_vaq_labels(
            features_train, vaq, norm_mode, device, batch_size=batch_size,
        )
        # vaq_labels: [G, N*T]. Build per-group counts of length K_g.
        usage_counts: List[np.ndarray] = []
        for g, k in enumerate(vaq.k_per_group):
            counts = np.zeros(int(k), dtype=np.float64)
            np.add.at(counts, vaq_labels[g], 1)
            usage_counts.append(counts)
        codec.pq.init_prior_from_freq(usage_counts, smoothing=1.0)
        del vaq_labels, usage_counts

    history, snapshots = train_vaq_soft(
        codec,
        features_train=features_train,
        norm_mode=norm_mode,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        seed=seed,
        val_features=val_features,
        tau_start=tau_start,
        tau_end=tau_end,
        tau_schedule=tau_schedule,
        grad_clip=grad_clip,
        freeze_transform=freeze_transform,
        freeze_codebooks=freeze_codebooks,
        verbose=verbose,
        tail=tail,
        use_lref=use_lref,
        snapshot_epochs=snapshot_epochs,
    )
    return codec, vaq, history, snapshots
