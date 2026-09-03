"""
Scalable two-stage feature codec: frozen base codebook + task residual codebook.

Stage 1 (already trained, see orfc/run_soft_pq.py):
    base codec = OrthogonalTransform R1 + SoftPQ, optimised for
    tail MSE (ΔL_ref = ||F(H) - F(Ĥ)||², F = frozen ViT tail).
    Encodes generic semantics. Frozen here — its bitstream never changes.

Stage 2 (this file):
    residual codec = OrthogonalTransform R2 + SoftPQ, applied to
    R = Y - Ŷ_base and optimised under a *task* loss (classification CE).
    R2 is independent of R1, so training it leaves the base bitstream and
    the base-only reconstruction bit-exact: the stream stays scalable
    (base only = generic, base + residual = task-enhanced).

Rate is reported per stage: base bits are a fixed cost, residual bits are
the task-specific increment.
"""

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_fn
from torch.utils.data import DataLoader, Dataset

ORFCV2_ROOT = os.path.dirname(os.path.abspath(__file__))
CODING_ROOT = os.path.dirname(ORFCV2_ROOT)
ORFC_ROOT = os.path.join(CODING_ROOT, "orfc")
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

from opq import batch_inv_normalize_gpu, batch_normalize_gpu
from soft_pq import (
    FeatureCodec, FeatureTransform, OrthogonalTransform, SoftPQ,
    compute_perplexity, load_codec,
)


def _cuda_mem(device, tag=""):
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    peak_rsv = torch.cuda.max_memory_reserved(device) / 1e9
    print(f"  CUDA{tag}: alloc={alloc:.2f}GB reserved={reserved:.2f}GB "
          f"peak_alloc={peak:.2f}GB peak_reserved={peak_rsv:.2f}GB",
          flush=True)


# ================================================================
#                    Two-stage codec
# ================================================================

class ResidualFeatureCodec(nn.Module):
    """Frozen base codec + trainable residual codec.

    Y_norm --[base, frozen]--> Ŷ_base
    R = Y_norm - Ŷ_base --[residual]--> R̂
    Ŷ = Ŷ_base + R̂
    """

    def __init__(self, base_codec, res_codec):
        super().__init__()
        self.base = base_codec
        self.res = res_codec
        self.freeze_base()

    def freeze_base(self):
        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()             # base must never see train-mode soft PQ
        return self

    def base_forward(self, Y_norm):
        with torch.no_grad():
            Y_base, _ = self.base(Y_norm)
        return Y_base

    def forward(self, Y_norm):
        """Y_norm [B, T, D] -> (Ŷ [B, T, D], residual usage [G, K])."""
        Y_base = self.base_forward(Y_norm)
        R_hat, usage = self.res(Y_norm - Y_base)
        return Y_base + R_hat, usage

    # --- rate bookkeeping (residual stage only; base is a fixed cost) ---

    @property
    def use_rate(self):
        return self.res.use_rate

    @property
    def _last_rate(self):
        return self.res._last_rate

    @property
    def pq(self):
        return self.res.pq

    @torch.no_grad()
    def get_prior_pmf(self):
        return self.res.get_prior_pmf()


def share_base_transform(res_codec, base_codec, verbose=True):
    """Copy the frozen ORFC rotation into the residual codec (R2 := R1).

    Residual PQ then lives in the same orthogonal coordinates as the base
    stage. The caller must freeze the residual transform afterwards.
    """
    bt, rt = base_codec.transform, res_codec.transform
    if bt is None or rt is None:
        raise ValueError("share_base_transform needs a transform on both stages")
    if type(bt) is not type(rt):
        raise ValueError(
            f"transform type mismatch: base={type(bt).__name__}, "
            f"res={type(rt).__name__}")
    if hasattr(bt, 'D') and hasattr(rt, 'D') and bt.D != rt.D:
        raise ValueError(f"transform dim mismatch: base D={bt.D}, res D={rt.D}")
    rt.load_state_dict(bt.state_dict())
    if verbose and hasattr(bt, 'get_rotation'):
        with torch.no_grad():
            diff = (bt.get_rotation() - rt.get_rotation()).norm().item()
            orth = rt.orth_error() if hasattr(rt, 'orth_error') else float('nan')
        print(f"  Shared ORFC R: copied Cayley params, "
              f"||R_res-R_base||={diff:.2e}, ||R'R-I||={orth:.2e}")


# ================================================================
#                    Differentiable frozen task heads
# ================================================================

class _TaskTail:
    """Base for frozen-but-differentiable task heads on top of tail blocks.

    Unlike Dinov2Wrapper.forward_from_tokens / forward_from_tokens_seg (both
    @torch.no_grad), these let a task loss backpropagate into the codec.
    Parameters are frozen; only the input tokens carry gradient.
    """

    task_kind = None

    def __init__(self, blocks, norm_layer, head, device='cuda', n_prefix=1):
        self.blocks = list(blocks)
        self.norm = norm_layer
        self.head = head
        self.n_prefix = n_prefix
        self.device = device
        for mod in self.blocks + [self.norm, self.head]:
            mod.to(device).eval()
            for p in mod.parameters():
                p.requires_grad_(False)

    def _trunk(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    @torch.no_grad()
    def forward_nograd(self, x):
        return self(x)

    def to(self, device):
        for mod in self.blocks + [self.norm, self.head]:
            mod.to(device)
        self.device = device
        return self


class ClsTaskTail(_TaskTail):
    """Tail blocks + norm + linear classification head.

    Mirrors the official LinearClassifierWrapper(layers=1) input:
        cat([x_norm_clstoken, mean(x_norm_patchtokens)])
    """

    task_kind = 'cls'

    def __call__(self, x):
        """x [B, T, D] tokens at the cut layer -> logits [B, n_classes]."""
        x = self._trunk(x)
        cls_token = x[:, 0]
        mean_patch = x[:, self.n_prefix:].mean(dim=1)
        return self.head(torch.cat([cls_token, mean_patch], dim=1))


class SegTaskTail(_TaskTail):
    """Tail blocks + norm + VOC linear segmentation head (BN + 1x1 conv).

    The CLS/register prefix tokens are dropped before the head, matching
    Dinov2Wrapper.forward_from_tokens_seg. The patch grid is inferred from the
    token count, so the same instance works for 16x16 ImageNet features and
    37x37 VOC slide crops.
    """

    task_kind = 'seg'

    def __init__(self, blocks, norm_layer, head, device='cuda', n_prefix=1,
                 token_hw=None):
        super().__init__(blocks, norm_layer, head, device=device,
                         n_prefix=n_prefix)
        self.token_hw = token_hw

    def __call__(self, x):
        """x [B, T, D] -> seg logits [B, n_classes, H, W]."""
        x = self._trunk(x)
        patch = x[:, self.n_prefix:]
        B, N, D = patch.shape
        h, w = _resolve_token_hw(N, self.token_hw)
        return self.head(patch.reshape(B, h, w, D).permute(0, 3, 1, 2))


class DepthTaskTail(_TaskTail):
    """Tail blocks + NYU depth head (BNHead), tapping the per-bin logits.

    Two things differ from the cls/seg tails:
      * No final LayerNorm. DINOv2's depth pipeline runs the tail with
        norm=False (see tools/dinov2_depth_pipeline.decode_depth); the depth
        head was trained on the raw last-block output. We pass norm=Identity.
      * The KD signal is the raw ``conv_depth`` logits [B, n_bins, H, W]
        (before the head's relu+eps+L1 bin aggregation), so KD reuses the
        segmentation-style softmax-KL with n_bins acting as the class axis.

    conv_depth is a 1x1 conv, so it commutes with the head's internal bilinear
    upsample; we therefore tap the logits at the native patch grid
    (H×W, square or rectangular) instead of the 4x-upsampled grid.
    """

    task_kind = 'depth'

    def __init__(self, blocks, head, device='cuda', n_prefix=1, token_hw=None):
        # norm=Identity: depth uses the un-normed last-block output.
        super().__init__(blocks, nn.Identity(), head, device=device,
                         n_prefix=n_prefix)
        self.token_hw = token_hw

    def __call__(self, x):
        """x [B, T, D] -> per-bin depth logits [B, n_bins, H, W]."""
        x = self._trunk(x)
        cls_token = x[:, 0]
        patch = x[:, self.n_prefix:]
        B, N, D = patch.shape
        h, w = _resolve_token_hw(N, self.token_hw)
        patch_map = patch.reshape(B, h, w, D).permute(0, 3, 1, 2)
        # Replicate BNHead._forward_feature's cls-token concat, then apply
        # conv_depth directly at native resolution (skips the 4x upsample;
        # equivalent for a 1x1 conv).
        cls_map = cls_token[:, :, None, None].expand_as(patch_map)
        feat = torch.cat([patch_map, cls_map], dim=1)
        return self.head.conv_depth(feat)


# ================================================================
#                    Feature IO (path-mode lazy load)
# ================================================================

_KNOWN_TOKEN_HW = (
    (16, 16), (14, 14), (32, 32), (37, 37),
    (37, 49), (49, 37), (32, 43), (43, 32),
    (64, 85), (85, 64),
)


def infer_token_hw(n_tokens, n_prefix=1, known=_KNOWN_TOKEN_HW):
    """Factor T - n_prefix into (H, W); prefer known ADE/ImageNet/VOC layouts."""
    n_patch = int(n_tokens) - int(n_prefix)
    if n_patch <= 0:
        raise ValueError(f"n_tokens={n_tokens} n_prefix={n_prefix} has no patches")
    for h, w in known:
        if h * w == n_patch:
            return (h, w)
    best, best_score = None, 1e18
    for h in range(1, int(n_patch ** 0.5) + 1):
        if n_patch % h == 0:
            w = n_patch // h
            score = abs(h - w)
            if score < best_score:
                best_score = score
                best = (h, w)
    return best


def _resolve_token_hw(n_patch, token_hw=None):
    if token_hw is not None:
        h, w = int(token_hw[0]), int(token_hw[1])
        if h * w != n_patch:
            raise ValueError(f"patch count {n_patch} != token_hw {h}x{w}")
        return h, w
    h = int(round(math.sqrt(n_patch)))
    if h * h != n_patch:
        raise ValueError(
            f"patch count {n_patch} is not a square grid; pass token_hw=(H,W)")
    return h, h


def _is_path_collection(features):
    if not isinstance(features, (list, tuple)) or not features:
        return False
    x = features[0]
    return isinstance(x, (str, Path)) or hasattr(x, '__fspath__')


def _as_td(arr):
    arr = np.asarray(arr, dtype=np.float32)
    while arr.ndim > 2:
        arr = np.squeeze(arr, axis=0)
    if arr.ndim != 2:
        raise ValueError(f"expected [T, D] feature, got {arr.shape}")
    return arr


def _load_feature_item(features, idx):
    if isinstance(features, np.ndarray):
        return _as_td(features[idx])
    item = features[idx]
    if isinstance(item, (str, Path)) or hasattr(item, '__fspath__'):
        return _as_td(np.load(os.fspath(item)))
    return _as_td(item)


def _feature_count(features):
    if isinstance(features, np.ndarray):
        return int(features.shape[0])
    return len(features)


def _feature_token_dim(features):
    if isinstance(features, np.ndarray) and features.ndim == 3:
        return int(features.shape[1]), int(features.shape[2])
    arr = _load_feature_item(features, 0)
    return int(arr.shape[0]), int(arr.shape[1])


def _stack_slice(features, start, end):
    if isinstance(features, np.ndarray) and features.ndim == 3:
        return features[start:end]
    return np.stack([_load_feature_item(features, i) for i in range(start, end)])


# ================================================================
#                    Dataset
# ================================================================

class ResidualDataset(Dataset):
    """Serves (features, label, teacher_logits, teacher_tail_feat) as needed.

    ``features`` may be a stacked ndarray ``[N, T, D]``, a list of arrays, or
    a list of ``.npy`` paths (lazy disk reads; no teacher cache in path mode).
    """

    def __init__(self, features, labels=None,
                 teacher_logits=None, teacher_tail=None):
        self.features = features
        self.path_mode = _is_path_collection(features)
        self.labels = labels
        self.teacher_logits = teacher_logits
        self.teacher_tail = teacher_tail

    def __len__(self):
        return _feature_count(self.features)

    def __getitem__(self, idx):
        out = [torch.from_numpy(_load_feature_item(self.features, idx)).float()]
        out.append(int(self.labels[idx]) if self.labels is not None else -1)
        if self.teacher_logits is not None:
            out.append(torch.from_numpy(self.teacher_logits[idx]).float())
        if self.teacher_tail is not None:
            out.append(torch.from_numpy(self.teacher_tail[idx]).float())
        return tuple(out)


# ================================================================
#                    Residual statistics helpers
# ================================================================

@torch.no_grad()
def collect_residuals(features, base_codec, norm_mode, device,
                      max_vectors=2_000_000, chunk_images=None,
                      seed=42, n_prefix=0):
    """Sample flattened residual vectors R = Y - Ŷ_base as [N, D] numpy.

    ``features`` may be a stacked ndarray, a list of arrays, or ``.npy`` paths.
    Used for OPQ warm-start / k-means init of the residual codebooks.
    """
    base_codec.eval()
    N_img = _feature_count(features)
    tokens_per_img, D = _feature_token_dim(features)
    if chunk_images is None:
        chunk_images = 32 if _is_path_collection(features) else 100
    keep_ratio = min(1.0, max_vectors / float(max(N_img * tokens_per_img, 1)))
    rng = np.random.RandomState(seed)
    order = rng.permutation(N_img)

    chunks = []
    n_have = 0
    for start in range(0, N_img, chunk_images):
        if n_have >= max_vectors:
            break
        idx = order[start:min(start + chunk_images, N_img)]
        batch = np.stack([_load_feature_item(features, int(i)) for i in idx])
        X = torch.from_numpy(batch).float().to(device)
        Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
        Y_base, _ = base_codec(Y)
        Rres = (Y - Y_base).reshape(-1, D).cpu().numpy()
        if keep_ratio < 1.0:
            n_keep = max(1, int(round(Rres.shape[0] * keep_ratio)))
            Rres = Rres[rng.choice(Rres.shape[0], n_keep, replace=False)]
        chunks.append(Rres)
        n_have += Rres.shape[0]
        del X, Y, Y_base, batch
    torch.cuda.empty_cache()
    out = np.concatenate(chunks, axis=0)
    if out.shape[0] > max_vectors:
        out = out[rng.choice(out.shape[0], max_vectors, replace=False)]
    return out


def build_residual_codec(D, G, K, d, ecvq_lmbda=0.0, prior_floor=0.0,
                         transform_kind='orthogonal', bottleneck_dim=0):
    """Create the residual FeatureCodec (own transform, own PQ, own prior)."""
    if transform_kind == 'orthogonal':
        transform = OrthogonalTransform(D)
    elif transform_kind == 'lowrank':
        assert bottleneck_dim > 0, "lowrank transform needs bottleneck_dim > 0"
        transform = FeatureTransform(D, bottleneck_dim)
    elif transform_kind == 'none':
        transform = None
    else:
        raise ValueError(f"Unknown transform_kind: {transform_kind}")
    pq = SoftPQ(G, K, d, lmbda=ecvq_lmbda, prior_floor=prior_floor)
    return FeatureCodec(pq, transform)


# ================================================================
#                    Training
# ================================================================

def train_residual_pq(
    features_train,
    labels_train,
    base_codec,
    res_codec,
    task_tail: ClsTaskTail = None,
    mse_tail=None,
    res_loss='task',
    norm_mode='per_image',
    epochs=100,
    lr=3e-4,
    batch_size=32,
    device='cuda',
    seed=42,
    val_features=None,
    val_labels=None,
    lmbda_rate=0.0,
    ce_weight=1.0,
    lmbda_kd=0.0,
    kd_temperature=2.0,
    beta_mse=0.0,
    grad_clip=1.0,
    freeze_res_transform=False,
    freeze_res_codebooks=False,
    tau_start=0.5,
    tau_end=0.005,
    tau_schedule='exponential',
    n_prefix=0,
    verbose=True,
):
    """Train the residual codec on top of a frozen base codec.

    Loss:  J = D + lmbda_rate * R_bits(residual)
      res_loss='task':     D = ce_weight * CE(logits(Ŷ), y)
                               + lmbda_kd * KD(teacher logits)
                             ce_weight=0 with lmbda_kd>0 gives pure logit
                             distillation, which needs no labels at all.
      res_loss='mse_tail': D = ||F(H) - F(Ĥ)||² / B      (control experiment)
      res_loss='mse':      D = ||Y - Ŷ||² / B            (control experiment)
      beta_mse > 0 adds beta_mse * ||Y - Ŷ||² / B as a regulariser that keeps
      the residual from trading away generic fidelity for task accuracy.

    Note on λ: orfc/soft_pq.py writes J = R + λ·D, which only makes sense when
    D is an MSE-scale quantity. Here D can be a cross-entropy (O(1)), so the
    rate weight sits on R instead. lmbda_rate=0 disables the rate term (the
    residual then costs a fixed G·log2(K) bits/token).

    Returns:
        codec: trained ResidualFeatureCodec.
        history: list of per-epoch dicts.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    path_mode = _is_path_collection(features_train)
    if isinstance(features_train, np.ndarray) and features_train.ndim == 3:
        features_ref = features_train
        N_img, T_tokens, D = features_train.shape
    elif path_mode:
        features_ref = features_train
        N_img = len(features_train)
        T_tokens, D = _feature_token_dim(features_train)
    else:
        features_ref = np.stack(features_train)
        N_img, T_tokens, D = features_ref.shape

    task_kind = getattr(task_tail, 'task_kind', None)
    if res_loss == 'task':
        assert task_tail is not None, "res_loss='task' needs a task tail"
        assert ce_weight > 0 or lmbda_kd > 0, \
            "res_loss='task' needs ce_weight > 0 or lmbda_kd > 0"
        if task_kind in ('seg', 'depth'):
            assert ce_weight == 0, \
                f"{task_kind} supervision is distillation-only: use --ce_weight 0"
        if ce_weight > 0:
            assert labels_train is not None, "ce_weight > 0 needs labels"
        if labels_train is not None:
            labels_train = np.asarray(labels_train, dtype=np.int64)
            assert len(labels_train) == N_img
    if res_loss == 'mse_tail':
        assert mse_tail is not None, "res_loss='mse_tail' needs a FrozenTail"

    codec = ResidualFeatureCodec(base_codec, res_codec).to(device)
    res = codec.res
    pq = res.pq
    transform = res.transform
    use_soft = (tau_start > 0)

    if freeze_res_transform and transform is not None:
        for p in transform.parameters():
            p.requires_grad_(False)
    if freeze_res_codebooks:
        pq.codebooks.requires_grad_(False)
        if pq.use_rate:
            pq.log_prior.requires_grad_(False)

    trainable = [p for p in res.parameters() if p.requires_grad]
    assert trainable, "residual codec has no trainable parameter left"
    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    if verbose:
        n_base = sum(p.numel() for p in codec.base.parameters())
        print(f"  Base codec: {n_base:,} params (frozen), "
              f"G={codec.base.pq.G}, K={codec.base.pq.K}, d={codec.base.pq.d}")
        print(f"  Residual codec: G={pq.G}, K={pq.K}, d={pq.d}, "
              f"{sum(p.numel() for p in trainable):,} trainable params")
        if freeze_res_transform:
            print(f"  Frozen: residual transform")
        if freeze_res_codebooks:
            print(f"  Frozen: residual codebooks")
        if transform is not None and hasattr(transform, 'orth_error'):
            print(f"  Residual transform ||R'R-I||={transform.orth_error():.2e}")
        if res_loss != 'task':
            loss_desc = res_loss
        elif ce_weight > 0:
            loss_desc = f"{ce_weight}*CE"
        else:
            loss_desc = ""
        kd_desc = f"{lmbda_kd}*KD(T={kd_temperature})" if lmbda_kd > 0 else ""
        print(f"  Loss: {loss_desc}"
              + ((" + " if loss_desc else "") + kd_desc if kd_desc else "")
              + (f" + {beta_mse}*MSE" if beta_mse > 0 else "")
              + (f" + {lmbda_rate}*R_bits" if lmbda_rate > 0 else ""))
        print(f"  Soft PQ: τ {tau_start:.3f} → {tau_end:.4f} ({tau_schedule})"
              if use_soft else "  Hard PQ (τ=0)")
        if path_mode:
            print(f"  Features: path-mode lazy load ({N_img} files, "
                  f"T={T_tokens}, D={D})")

    # --- teacher caches (CPU; features may stay path-mode) ---
    teacher_logits = None
    if lmbda_kd > 0:
        if verbose:
            print(f"  Pre-computing teacher logits ({N_img} images"
                  f"{', path-mode' if path_mode else ''})...")
        t_pre = time.time()
        teacher_logits = _precompute_logits(
            features_ref, task_tail, batch_size, device)
        if verbose:
            gb = teacher_logits.nbytes / 1e9
            print(f"  Teacher logits: {tuple(teacher_logits.shape)}  "
                  f"{gb:.2f} GB CPU  ({time.time() - t_pre:.1f}s)")

    teacher_tail = None
    if res_loss == 'mse_tail':
        if path_mode:
            raise ValueError("mse_tail teacher cache needs stacked features, "
                             "not path-mode")
        if verbose:
            gb = features_ref.nbytes / 1e9
            print(f"  Pre-computing teacher tail features "
                  f"({N_img} images, ~{gb:.1f} GB CPU)...")
        teacher_tail = np.empty_like(features_ref)
        with torch.no_grad():
            for s in range(0, N_img, batch_size):
                e = min(s + batch_size, N_img)
                Xc = torch.from_numpy(features_ref[s:e]).float().to(device)
                teacher_tail[s:e] = mse_tail.forward_nograd(Xc).cpu().numpy()
                del Xc
        torch.cuda.empty_cache()

    dataset = ResidualDataset(features_ref, labels=labels_train,
                              teacher_logits=teacher_logits,
                              teacher_tail=teacher_tail)
    nw = min(4 if path_mode else 2, max(0, N_img // 8))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=nw, pin_memory=True,
                        persistent_workers=(nw > 0))

    # --- validation set ---
    val_array = None
    if val_features is not None and len(val_features) > 0:
        val_array = np.stack(val_features)
        val_labels = (np.asarray(val_labels, dtype=np.int64)
                      if val_labels is not None else None)

    history = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        _cuda_mem(device, " before train")
    first_step = True
    for epoch in range(epochs):
        t_epoch = time.time()
        pq.temperature = _anneal_tau(epoch, epochs, tau_start, tau_end,
                                     tau_schedule) if use_soft else 0.0

        codec.train()
        sums = dict(loss=0.0, distortion=0.0, ce=0.0, kd=0.0, mse=0.0,
                    rate=0.0, correct=0.0)
        usage_acc = torch.zeros(pq.G, pq.K, device=device)

        for batch in loader:
            X, y = batch[0], batch[1]
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            extras = list(batch[2:])
            B = X.shape[0]

            with torch.no_grad():
                Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode,
                                                 n_prefix=n_prefix)
            Y_hat, usage = codec(Y)

            ce = kd = mse = None
            if res_loss == 'mse':
                distortion = ((Y - Y_hat) ** 2).sum() / B
                mse = distortion
            else:
                X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
                if res_loss == 'mse_tail':
                    if extras:
                        Y_teacher = extras.pop(0).to(device, non_blocking=True)
                    else:
                        Y_teacher = mse_tail.forward_nograd(X)
                    distortion = ((Y_teacher - mse_tail(X_hat)) ** 2).sum() / B
                else:
                    logits = task_tail(X_hat)
                    distortion = 0.0
                    if ce_weight > 0:
                        ce = F_fn.cross_entropy(logits, y)
                        distortion = ce_weight * ce
                    if lmbda_kd > 0:
                        if extras:
                            t_logits = extras.pop(0).to(device, non_blocking=True)
                        else:
                            t_logits = task_tail.forward_nograd(X)
                        kd = _kd_loss(logits, t_logits, kd_temperature)
                        distortion = distortion + lmbda_kd * kd
                    if task_kind == 'cls' and labels_train is not None:
                        sums['correct'] += (logits.argmax(1) == y).sum().item()
                del X_hat

            if beta_mse > 0 and res_loss != 'mse':
                mse = ((Y - Y_hat) ** 2).sum() / B
                distortion = distortion + beta_mse * mse

            loss = distortion
            if lmbda_rate > 0 and res.use_rate:
                loss = loss + lmbda_rate * res._last_rate * T_tokens

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            if first_step:
                first_step = False
                _cuda_mem(device, " after first backward")

            sums['loss'] += loss.item() * B
            sums['distortion'] += distortion.item() * B
            if ce is not None:
                sums['ce'] += ce.item() * B
            if kd is not None:
                sums['kd'] += kd.item() * B
            if mse is not None:
                sums['mse'] += mse.item() * B
            if res.use_rate:
                sums['rate'] += res._last_rate.item() * B
            usage_acc += usage.detach()
            del X, Y, Mu, Std, Y_hat, loss, distortion

        scheduler.step()

        is_task = (res_loss == 'task')
        info = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'loss': sums['loss'] / N_img,
            'distortion': sums['distortion'] / N_img,
            'ce': sums['ce'] / N_img if (is_task and ce_weight > 0) else None,
            'kd': sums['kd'] / N_img if lmbda_kd > 0 else None,
            'mse': (sums['mse'] / N_img
                    if (res_loss == 'mse' or beta_mse > 0) else None),
            'rate_bpt': sums['rate'] / N_img if res.use_rate else 0.0,
            'train_acc': (sums['correct'] / N_img
                          if (is_task and labels_train is not None) else None),
            'perplexity': compute_perplexity(usage_acc),
            'dead_entries': int((usage_acc == 0).sum().item()),
            'temperature': pq.temperature,
            'time': time.time() - t_epoch,
        }
        if transform is not None and hasattr(transform, 'orth_error'):
            info['orth_error'] = transform.orth_error()
        if val_array is not None:
            info.update(_validate(codec, val_array, val_labels, res_loss,
                                  task_tail, mse_tail, norm_mode, device,
                                  batch_size, n_prefix,
                                  kd_temperature=kd_temperature))
        history.append(info)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            parts = [f"D={info['distortion']:.4f}"]
            if info['ce'] is not None:
                parts.append(f"CE={info['ce']:.4f}")
            if info['train_acc'] is not None:
                parts.append(f"acc={info['train_acc']:.3f}")
            if info['mse'] is not None:
                parts.append(f"MSE={info['mse']:.1f}")
            if res.use_rate:
                parts.append(f"R={info['rate_bpt']:.2f}b/t")
            parts.append(f"ppl={info['perplexity']:.1f}")
            parts.append(f"dead={info['dead_entries']}")
            if use_soft:
                parts.append(f"τ={info['temperature']:.4f}")
            if info.get('val_distortion') is not None:
                parts.append(f"val={info['val_distortion']:.4f}")
            if info.get('val_acc') is not None:
                parts.append(f"val_acc={info['val_acc']:.3f}")
            if info.get('val_teacher_pixel_agree') is not None:
                parts.append(f"val_agree={info['val_teacher_pixel_agree']:.3f}")
            print(f"  ep {epoch:3d}/{epochs}  " + "  ".join(parts)
                  + f"  ({info['time']:.1f}s)")
            _cuda_mem(device, f" after ep {epoch}")

    _cuda_mem(device, " train done")
    return codec, history


def _anneal_tau(epoch, epochs, tau_start, tau_end, schedule):
    if epochs <= 1:
        return tau_start
    progress = epoch / (epochs - 1)
    if schedule == 'linear':
        return tau_start + (tau_end - tau_start) * progress
    return tau_start * (tau_end / tau_start) ** progress


@torch.no_grad()
def _precompute_logits(features, task_tail, batch_size, device):
    """Teacher logits from unquantised features.

    Shape is [N, C] for classification and [N, C, H, W] for seg/depth.
    ``features`` may be a stacked ndarray or a path list.
    """
    N = _feature_count(features)
    out = None
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        X = torch.from_numpy(_stack_slice(features, s, e)).float().to(device)
        lg = task_tail.forward_nograd(X).cpu().numpy()
        if out is None:
            out = np.empty((N,) + lg.shape[1:], dtype=np.float32)
        out[s:e] = lg
        del X
    torch.cuda.empty_cache()
    return out


def _kd_loss(student_logits, teacher_logits, temperature):
    """Temperature-scaled KL(teacher || student), scaled back by T².

    Accepts [B, C] classification logits or [B, C, H, W] segmentation logits;
    the latter is averaged over pixels.
    """
    if student_logits.dim() == 4:
        C = student_logits.shape[1]
        student_logits = student_logits.permute(0, 2, 3, 1).reshape(-1, C)
        teacher_logits = teacher_logits.permute(0, 2, 3, 1).reshape(-1, C)
    t = temperature
    log_p_s = F_fn.log_softmax(student_logits / t, dim=-1)
    p_t = F_fn.softmax(teacher_logits / t, dim=-1)
    return F_fn.kl_div(log_p_s, p_t, reduction='batchmean') * (t * t)


@torch.no_grad()
def _validate(codec, val_array, val_labels, res_loss, task_tail, mse_tail,
              norm_mode, device, batch_size, n_prefix, kd_temperature=2.0):
    """Held-out monitoring metric matching the training objective.

    For segmentation there are no labels, so the metric is the teacher KD
    divergence; pixel accuracy against the teacher's argmax is also reported.
    """
    codec.eval()
    task_kind = getattr(task_tail, 'task_kind', None)
    n_val = val_array.shape[0]
    total_d, correct, n_items = 0.0, 0, 0
    for s in range(0, n_val, batch_size):
        e = min(s + batch_size, n_val)
        X = torch.from_numpy(val_array[s:e]).float().to(device)
        Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
        Y_hat, _ = codec(Y)
        if res_loss == 'mse':
            total_d += ((Y - Y_hat) ** 2).sum().item()
        else:
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            if res_loss == 'mse_tail':
                ref = mse_tail.forward_nograd(X)
                total_d += ((ref - mse_tail.forward_nograd(X_hat)) ** 2).sum().item()
            elif task_kind in ('seg', 'depth'):
                t_logits = task_tail.forward_nograd(X)
                s_logits = task_tail.forward_nograd(X_hat)
                B = X.shape[0]
                total_d += _kd_loss(s_logits, t_logits,
                                    kd_temperature).item() * B
                correct += (s_logits.argmax(1) == t_logits.argmax(1)
                            ).float().mean().item() * B
                n_items += B
                del t_logits, s_logits
            else:
                logits = task_tail.forward_nograd(X_hat)
                if val_labels is not None:
                    yb = torch.from_numpy(val_labels[s:e]).to(device)
                    total_d += F_fn.cross_entropy(
                        logits, yb, reduction='sum').item()
                    correct += (logits.argmax(1) == yb).sum().item()
            del X_hat
        del X, Y, Mu, Std, Y_hat
    codec.train()
    out = {'val_distortion': total_d / n_val}
    if res_loss == 'task':
        if task_kind in ('seg', 'depth') and n_items:
            out['val_teacher_pixel_agree'] = correct / n_items
        elif val_labels is not None:
            out['val_acc'] = correct / n_val
    return out


# ================================================================
#                    Inference
# ================================================================

@torch.no_grad()
def residual_encode_decode(features, codec, norm_mode, device,
                           chunk_images=None, n_prefix=0, base_only=False):
    """Encode/decode with hard PQ. base_only=True drops the residual stream."""
    codec.eval()
    N = _feature_count(features)
    if chunk_images is None:
        tokens_per_img, _ = _feature_token_dim(features)
        G = codec.res.pq.G if not base_only else codec.base.pq.G
        K = codec.res.pq.K if not base_only else codec.base.pq.K
        max_tokens = max(tokens_per_img, int(3e9 / (G * K * 4)))
        chunk_images = max(1, min(200, max_tokens // tokens_per_img))

    all_xhat = []
    for start in range(0, N, chunk_images):
        end = min(start + chunk_images, N)
        X = torch.from_numpy(_stack_slice(features, start, end)).float().to(device)
        Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
        if base_only:
            Y_hat = codec.base_forward(Y)
        else:
            Y_hat, _ = codec(Y)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        for i in range(X_hat.shape[0]):
            all_xhat.append(X_hat[i].cpu().numpy())
        del X, Y, Mu, Std, Y_hat, X_hat
    torch.cuda.empty_cache()
    return all_xhat


@torch.no_grad()
def collect_stage_labels(features, codec, norm_mode, device, batch_size=32,
                        n_prefix=0):
    """Return (base_labels [G_b, N], residual_labels [G_r, N]) hard indices."""
    codec.eval()
    base_labels, res_labels = [], []
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(_stack_slice(features, start, end)).float().to(device)
        Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
        _ = codec(Y)
        base_labels.append(codec.base.pq._last_labels.cpu())
        res_labels.append(codec.res.pq._last_labels.cpu())
        del X, Y
    return (torch.cat(base_labels, dim=1).numpy(),
            torch.cat(res_labels, dim=1).numpy())


# ================================================================
#                    Save / load
# ================================================================

def save_residual_codec(codec, path, base_ckpt_path, extra=None):
    """Save only the residual stage; the base is referenced by path."""
    res = codec.res
    meta = {
        'base_ckpt': os.path.abspath(base_ckpt_path),
        'G': res.pq.G, 'K': res.pq.K, 'd': res.pq.d,
        'ecvq_lmbda': res.pq.lmbda, 'prior_floor': res.pq.prior_floor,
        'transform_type': (type(res.transform).__name__
                           if res.transform is not None else None),
        'state_dict': res.state_dict(),
    }
    if isinstance(res.transform, OrthogonalTransform):
        meta['D'] = res.transform.D
    elif isinstance(res.transform, FeatureTransform):
        meta['D_in'] = res.transform.D_in
        meta['D_out'] = res.transform.D_out
    if extra:
        meta['extra'] = extra
    torch.save(meta, path)


def load_residual_codec(path, device='cuda', base_ckpt_path=None):
    """Rebuild a ResidualFeatureCodec from a residual checkpoint."""
    meta = torch.load(path, map_location='cpu')
    base_codec = load_codec(base_ckpt_path or meta['base_ckpt'], device=device)

    ttype = meta.get('transform_type')
    if ttype == 'OrthogonalTransform':
        transform = OrthogonalTransform(meta['D'])
    elif ttype == 'FeatureTransform':
        transform = FeatureTransform(meta['D_in'], meta['D_out'])
    else:
        transform = None
    pq = SoftPQ(meta['G'], meta['K'], meta['d'],
                lmbda=meta.get('ecvq_lmbda', 0.0),
                prior_floor=meta.get('prior_floor', 0.0))
    res_codec = FeatureCodec(pq, transform)
    res_codec.load_state_dict(meta['state_dict'])
    codec = ResidualFeatureCodec(base_codec, res_codec)
    return codec.to(device).eval()


def load_any_codec(path, device='cuda', base_ckpt_path=None):
    """Dispatch FeatureCodec vs ResidualFeatureCodec by checkpoint keys."""
    meta = torch.load(path, map_location='cpu')
    if 'base_ckpt' in meta:
        return load_residual_codec(path, device=device,
                                   base_ckpt_path=base_ckpt_path)
    return load_codec(path, device=device)
