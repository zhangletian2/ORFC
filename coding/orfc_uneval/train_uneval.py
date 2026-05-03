"""
Training loop for non-uniform bit-allocation PQ with frozen OPQ rotation.

Codec structure:  FeatureCodec(pq=VAQSoftPQ, transform=OrthogonalTransform)
  - OrthogonalTransform is frozen (no learnable rotation)
  - VAQSoftPQ.K_per_group is fixed (no learnable allocation)
  - Only VAQSoftPQ.codebooks (+ optional log_prior) are trained

Distortion:
  - default: ΔL_ref = ||tail(X) - tail(inv_norm(Y_hat))||^2
  - fallback: MSE in normalised feature space (when tail is None)

Follows the same conventions as ``train_vaq_soft`` in vaq_soft.py and
``train_soft_pq`` in soft_pq.py.
"""

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _stack_features(features: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack(features).astype(np.float32, copy=False)


@torch.no_grad()
def _eval_mse(codec, features_array, norm_mode, device, batch_size,
              batch_normalize_gpu):
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


def train_uneval_pq(
    codec,
    features_train: Sequence[np.ndarray],
    norm_mode: str,
    batch_normalize_gpu,
    batch_inv_normalize_gpu,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: torch.device = torch.device("cuda"),
    seed: int = 42,
    val_features: Optional[Sequence[np.ndarray]] = None,
    tau_start: float = 0.5,
    tau_end: float = 0.005,
    tau_schedule: str = "exponential",
    grad_clip: float = 1.0,
    freeze_transform: bool = True,
    freeze_codebooks: bool = False,
    verbose: bool = True,
    tail=None,
    snapshot_epochs: Optional[List[int]] = None,
) -> Tuple[List[Dict[str, float]], Dict[int, dict]]:
    """Fine-tune codebooks of a FeatureCodec(VAQSoftPQ, OrthogonalTransform).

    Parameters
    ----------
    codec : FeatureCodec
        Must have ``codec.pq`` = VAQSoftPQ and ``codec.transform`` =
        OrthogonalTransform (frozen by default).
    features_train : list of [T, D] arrays
    tail : FrozenTail or None
        If provided, uses ΔL_ref distortion; otherwise falls back to MSE.
    batch_normalize_gpu, batch_inv_normalize_gpu : callables
        Normalization helpers from ``opq.py``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    use_lref = tail is not None

    if freeze_transform and codec.transform is not None:
        for p in codec.transform.parameters():
            p.requires_grad_(False)
    if freeze_codebooks:
        codec.pq.codebooks.requires_grad_(False)

    _snap_epochs = set(snapshot_epochs) if snapshot_epochs else set()
    snapshots: Dict[int, dict] = {}

    trainable = [p for p in codec.parameters() if p.requires_grad]
    if not trainable:
        if verbose:
            print("  [train_uneval_pq] no trainable params — skipping.")
        return [], snapshots

    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=lr * 0.01
    )

    use_soft = tau_start > 0
    use_rate = bool(getattr(codec.pq, "use_rate", False))
    lmbda = float(getattr(codec.pq, "lmbda", 0.0))
    T_tokens = int(features_train[0].shape[0])
    pq = codec.pq
    K_per_group = pq.K_per_group

    if verbose:
        n_trainable = sum(p.numel() for p in trainable)
        print(f"  Trainable params: {n_trainable:,}")
        print(f"  freeze_transform={freeze_transform}  "
              f"freeze_codebooks={freeze_codebooks}")
        print(f"  K_per_group (first 16): {K_per_group[:16]}  "
              f"K_max={pq.K_max}")
        if codec.transform is not None and hasattr(codec.transform, 'orth_error'):
            print(f"  orth_error(R-init)={codec.transform.orth_error():.2e}")
        if use_soft:
            print(f"  Soft PQ: tau {tau_start} -> {tau_end} ({tau_schedule})")
        else:
            print("  Hard PQ (tau=0)")
        dist_tag = "ΔL_ref" if use_lref else "MSE"
        print(f"  Distortion: {dist_tag}")
        if use_rate:
            print(f"  Rate-aware: lambda={lmbda}  "
                  f"loss = R*T + lambda*D  (T={T_tokens})")

    features_array = _stack_features(features_train)
    val_array = _stack_features(val_features) if val_features else None
    N_img = features_array.shape[0]

    # Pre-compute teacher outputs for ΔL_ref
    teacher_cache = None
    val_teacher_cache = None
    if use_lref:
        if verbose:
            print(f"  Pre-computing teacher outputs ({N_img} images)...")
        t_pre = time.time()
        teacher_cache = np.empty_like(features_array)
        with torch.no_grad():
            for s in range(0, N_img, batch_size):
                e = min(s + batch_size, N_img)
                X = torch.from_numpy(features_array[s:e]).float().to(device)
                teacher_cache[s:e] = tail.forward_nograd(X).cpu().numpy()
                del X
        torch.cuda.empty_cache()
        if verbose:
            gb = teacher_cache.nbytes / 1e9
            print(f"  Teacher cache: {gb:.2f} GB ({time.time() - t_pre:.1f}s)")
        if val_array is not None:
            val_teacher_cache = np.empty_like(val_array)
            with torch.no_grad():
                for s in range(0, val_array.shape[0], batch_size):
                    e = min(s + batch_size, val_array.shape[0])
                    X = torch.from_numpy(val_array[s:e]).float().to(device)
                    val_teacher_cache[s:e] = tail.forward_nograd(X).cpu().numpy()
                    del X
            torch.cuda.empty_cache()

    init_train_mse = _eval_mse(
        codec, features_array, norm_mode, device, batch_size,
        batch_normalize_gpu)
    init_val_mse = (
        _eval_mse(codec, val_array, norm_mode, device, batch_size,
                  batch_normalize_gpu)
        if val_array is not None else None
    )
    if verbose:
        v = f"  val_D={init_val_mse:.2f}" if init_val_mse is not None else ""
        print(f"  init  D(MSE)={init_train_mse:.2f}{v}")

    history: List[Dict[str, float]] = []
    indices = np.arange(N_img)
    rng = np.random.RandomState(seed)

    for epoch in range(epochs):
        t_epoch = time.time()

        # Temperature schedule
        if use_soft and epochs > 1:
            progress = epoch / (epochs - 1)
            if tau_schedule == "linear":
                tau = tau_start + (tau_end - tau_start) * progress
            else:
                tau = tau_start * (tau_end / max(tau_start, 1e-12)) ** progress
            pq.temperature = float(tau)
        elif use_soft:
            pq.temperature = float(tau_start)
        else:
            pq.temperature = 0.0

        rng.shuffle(indices)
        codec.train()
        total_distortion = 0.0
        total_rate = 0.0
        total_imgs = 0
        usage_acc = torch.zeros(pq.G, pq.K_max, device=device)

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
                rate_bits = pq._last_rate * T_tokens
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
                total_rate += pq._last_rate.item() * B
            total_imgs += B
            usage_acc += usage.detach()
            del X, Y, mu, std, Y_hat, usage, distortion, loss

        scheduler.step()
        avg_d = total_distortion / max(total_imgs, 1)
        avg_r = total_rate / max(total_imgs, 1) if use_rate else 0.0

        usage_pmf = usage_acc / usage_acc.sum(dim=-1, keepdim=True).clamp_min(1.0)
        ppl = (-(usage_pmf.clamp_min(1e-30).log() * usage_pmf)
               .sum(dim=-1)).exp().mean().item()
        valid_mask = pq.cb_mask.bool()
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

        orth_err = None
        if codec.transform is not None and hasattr(codec.transform, 'orth_error'):
            orth_err = codec.transform.orth_error()

        info: Dict[str, float] = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "loss_distortion": avg_d,
            "rate_bpt": avg_r,
            "perplexity": ppl,
            "dead_entries": dead,
            "temperature": pq.temperature,
            "val_loss": val_loss,
            "time": time.time() - t_epoch,
        }
        if orth_err is not None:
            info["orth_error"] = orth_err
        if use_rate and pq._last_rate_per_group is not None:
            info["rate_per_group"] = (
                pq._last_rate_per_group.detach().cpu().tolist()
            )
        history.append(info)

        if epoch in _snap_epochs:
            import copy as _copy
            snapshots[epoch] = _copy.deepcopy(codec.state_dict())

        if verbose and (epoch % max(epochs // 10, 1) == 0
                        or epoch == epochs - 1):
            val_str = f"  val_D={val_loss:.2f}" if val_loss is not None else ""
            tau_str = f"  tau={pq.temperature:.4f}" if use_soft else ""
            rate_str = (f"  R={avg_r:.2f}bpt  lD={lmbda * avg_d:.1f}"
                        if use_rate else "")
            d_tag = "ΔLref" if use_lref else "MSE"
            print(f"  ep{epoch:>3d}  D({d_tag})={avg_d:.2f}{rate_str}"
                  f"  ppl={ppl:.1f}  dead={dead}{tau_str}{val_str}"
                  f"  ({time.time() - t_epoch:.1f}s)")

    pq.temperature = 0.0
    codec.eval()
    return history, snapshots
