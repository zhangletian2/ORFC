#!/usr/bin/env python
"""Train ORFC (R + SoftPQ) on frozen Haar-joint 65-token sequences.

Y(257) --normalize--> Haar.encode (frozen) --> seq(65)
     --> FeatureCodec (trainable R + PQ) --> seq_hat
     --> Haar.decode --> Y_hat --inv_norm--> X_hat
loss = (_last_rate * 65 if λ>0 else 0) + λ · ||tail(X) - tail(X_hat)||²

OPQ warm-start is computed on the Haar-coded 65 tokens, not the original 257.
Training hyper-parameters follow the original SoftPQ recipe
(``run_batch_train.DEFAULTS_DINO_L``) plus per-(layer, K) overrides from
``orfc/best config.csv``.

Usage:
    CUDA_VISIBLE_DEVICES=4 python -u run_haar_orfc_train.py \\
        --layer blk05 --K 16 --lmbda 0.5 --lr 3e-4 --epochs 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
_ORFC_DIR = os.path.normpath(os.path.join(HERE, "..", "orfc"))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if _ORFC_DIR not in sys.path:
    sys.path.insert(0, _ORFC_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from backbone.wrapper import Dinov2Wrapper  # noqa: E402
from opq import (  # noqa: E402
    batch_inv_normalize_gpu,
    batch_normalize_gpu,
    batched_assign,
    learn_opq_rotation,
)
from run_multilayer_calibrator import (  # noqa: E402
    evaluate_accuracy,
    load_gt,
    preload_features,
    set_seed,
)
from soft_pq import (  # noqa: E402
    FeatureCodec,
    FrozenTail,
    OrthogonalTransform,
    SoftPQ,
    compute_perplexity,
    load_codec,
    save_codec,
)

from eval_haar_orfc import (  # noqa: E402
    GROUPING_BITS_PERM,
    _rate_from_labels,
    collect_orfc_labels,
    eval_cascade,
    haar_encode_all,
    pack_image_rate,
)
from run_global_residual import build_groups, load_global_detail  # noqa: E402

warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")


def default_haar_ckpt(layer, backbone="dinov2_vitl14"):
    return (
        Path(HERE) / "results" / "global_residual" / backbone
        / f"{layer}_global_haar_joint_D_lr0.0003_ep30_s42.pt"
    )


def codec_stem(args):
    bt_tag = f"bt{args.bottleneck_dim}" if args.bottleneck_dim > 0 else "noBt"
    ws_tag = "ws" if args.warm_start_opq else "km"
    rate_tag = f"_lmbda{args.lmbda}"
    tau_tag = f"_tau{args.tau_start}" if args.tau_start > 0 else ""
    tau_end_tag = (
        f"_te{args.tau_end}"
        if args.tau_start > 0 and args.tau_end != 0.005
        else ""
    )
    tau_sched_tag = (
        f"_ts{args.tau_schedule[:3]}"
        if args.tau_schedule != "exponential"
        else ""
    )
    return (
        f"{args.layer}_haar65_K{args.K}_emb{args.embedding_dim}"
        f"_{bt_tag}_{ws_tag}{rate_tag}{tau_tag}{tau_end_tag}{tau_sched_tag}"
        f"_lr{args.lr}_ep{args.epochs}_n{args.max_train_images}_s{args.seed}"
    )


def codec_path(args):
    out_dir = Path(args.ckpt_dir) / args.backbone
    return out_dir / f"{codec_stem(args)}.pt"


class HaarTrainDataset(Dataset):
    """Raw 257-token features + frozen-tail teacher + cached Haar groups."""

    def __init__(self, features, teacher, groups):
        self.features = features
        self.teacher = teacher
        self.groups = groups

    def __len__(self):
        return int(self.features.shape[0])

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).float()
        t = torch.from_numpy(self.teacher[idx]).float()
        g = torch.from_numpy(np.asarray(self.groups[idx], dtype=np.int64))
        return x, t, g


def _groups_path(cache_dir, layer, split, n, seed):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{layer}_{split}_n{n}_s{seed}.npy"


def load_or_build_groups(cache_path, features, norm_mode, device, batch_size):
    cache_path = Path(cache_path)
    if cache_path.is_file():
        groups = np.load(cache_path)
        if groups.shape[0] == len(features):
            print(f"  groups cache hit: {cache_path.name}  {groups.shape}")
            return groups
        print(f"  groups cache stale ({groups.shape[0]} vs {len(features)}), rebuild")
    print(f"  building groups ({len(features)} images)...")
    t0 = time.time()
    groups = build_groups(features, norm_mode, device, batch_size)
    np.save(cache_path, groups)
    print(f"  groups saved {cache_path.name}  {groups.shape}  ({time.time() - t0:.1f}s)")
    return groups


def freeze_haar(haar):
    haar.eval()
    for p in haar.parameters():
        p.requires_grad_(False)
    return haar


def _rebuild_train_tail(wrapper, layer_idx, device):
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


@torch.no_grad()
def eval_haar_opq(name, features, basenames, groups, haar, codebooks, R,
                  embedding_dim, tail, wrapper, layer_idx, gt, norm_mode,
                  device, batch_size):
    """Haar encode → hard OPQ on 65 tokens → Haar decode."""
    haar.eval()
    cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
    R_t = torch.from_numpy(R).float().to(device)
    G = len(codebooks)
    all_xhat = []
    delta_l = 0.0
    mse = 0.0
    n_elem = 0
    t0 = time.time()
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        g = torch.from_numpy(groups[start:end].astype(np.int64)).to(device)
        Y, Mu, Std = batch_normalize_gpu(
            X, mode=norm_mode, n_prefix=haar.n_prefix)
        seq, aux = haar.encode(Y, g)
        B, Tm, D = seq.shape
        flat = seq.reshape(B * Tm, D) @ R_t
        z_3d = flat.reshape(B * Tm, G, embedding_dim).permute(1, 0, 2).contiguous()
        z_hat_3d, _ = batched_assign(z_3d, cb_t, device=device)
        seq_hat = (z_hat_3d.permute(1, 0, 2).reshape(B * Tm, D) @ R_t.T).reshape(B, Tm, D)
        Y_hat = haar.decode(seq_hat, aux)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        teacher = tail.forward_nograd(X)
        student = tail.forward_nograd(X_hat)
        delta_l += ((teacher - student) ** 2).sum().item()
        diff = Y[:, haar.n_prefix:] - Y_hat[:, haar.n_prefix:]
        mse += (diff ** 2).sum().item()
        n_elem += diff.numel()
        all_xhat.extend(X_hat.cpu().numpy())
        del X, Y, Mu, Std, seq, seq_hat, Y_hat, X_hat, teacher, student, g
        del flat, z_3d, z_hat_3d
    acc = evaluate_accuracy(all_xhat, basenames, gt, wrapper, layer_idx, device)
    row = {
        "name": name,
        "acc": float(acc),
        "delta_l": float(delta_l / len(features)),
        "mse_patch": float(mse / n_elem),
        "t_s": time.time() - t0,
    }
    print(f"  {name:28s} Acc={row['acc']:.4f}  ΔL={row['delta_l']:.1f}  "
          f"MSE_p={row['mse_patch']:.6f}")
    return row


def collect_haar_opq_vectors(features, groups, haar, norm_mode, device, batch_size):
    """Flatten Haar-coded tokens for OPQ. Returns [N_tok, D] float32 numpy."""
    chunks = []
    n_prefix = haar.n_prefix
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        g = torch.from_numpy(groups[start:end].astype(np.int64)).to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
            seq, _ = haar.encode(Y, g)
        chunks.append(seq.reshape(-1, seq.shape[-1]).cpu().numpy())
        del X, Y, g, seq
    vectors = np.concatenate(chunks, axis=0)
    del chunks
    return vectors


def subsample_vectors(vectors, opq_groups, kmeans_max_samples, seed):
    max_flat = kmeans_max_samples // max(opq_groups, 1)
    if vectors.shape[0] > max_flat:
        rng = np.random.RandomState(seed)
        idx = rng.choice(vectors.shape[0], max_flat, replace=False)
        vectors = vectors[idx]
    return vectors


def opq_usage_on_haar(features, groups, haar, R, codebooks, embedding_dim,
                      norm_mode, device, batch_size):
    R_t = torch.from_numpy(R).float().to(device)
    cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
    G = len(codebooks)
    K = codebooks[0].shape[0]
    counts = np.zeros((G, K), dtype=np.float64)
    n_prefix = haar.n_prefix
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        g = torch.from_numpy(groups[start:end].astype(np.int64)).to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
            seq, _ = haar.encode(Y, g)
            flat = seq.reshape(-1, seq.shape[-1]) @ R_t
            sub = flat.reshape(-1, G, embedding_dim).permute(1, 0, 2).contiguous()
            labels = torch.cdist(sub, cb_t).argmin(dim=-1)
            for gi in range(G):
                np.add.at(counts[gi], labels[gi].cpu().numpy(), 1)
        del X, Y, g, seq, flat, sub, labels
    del R_t, cb_t
    torch.cuda.empty_cache()
    return counts


def train_haar_orfc(
    features_train,
    groups_train,
    haar,
    tail,
    G, K, d,
    norm_mode="per_image",
    epochs=100,
    lr=3e-4,
    batch_size=32,
    device="cuda",
    seed=42,
    val_features=None,
    val_groups=None,
    verbose=True,
    transform=None,
    R_init=None,
    codebooks_init=None,
    kmeans_max_samples=2_000_000,
    lmbda=0.0,
    prior_init_counts=None,
    grad_clip=1.0,
    tau_start=0.5,
    tau_end=0.005,
    tau_schedule="exponential",
    n_prefix=1,
):
    """Train FeatureCodec on Haar-coded tokens. Haar stays frozen.

    Rate term uses Tm = Haar sequence length (65), not the original 257.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if isinstance(features_train, np.ndarray) and features_train.ndim == 3:
        features_array = features_train
    else:
        features_array = np.stack(features_train)
    N_img = features_array.shape[0]
    D = features_array.shape[2]
    groups_array = np.asarray(groups_train)
    assert groups_array.shape[0] == N_img, "groups / features length mismatch"

    freeze_haar(haar)
    pq = SoftPQ(G, K, d, lmbda=lmbda).to(device)
    if transform is not None:
        transform = transform.to(device)
    codec = FeatureCodec(pq, transform).to(device)
    use_soft = tau_start > 0

    if R_init is not None and codebooks_init is not None and transform is not None:
        if verbose:
            print("  Warm-start from OPQ on Haar-65 tokens")
        transform.init_from_opq(R_init)
        pq.init_codebooks(codebooks_init)
        if pq.use_rate and prior_init_counts is not None:
            pq.init_prior_from_freq(prior_init_counts)
            if verbose:
                print("  log_prior init from OPQ empirical frequency")
    elif codebooks_init is not None:
        pq.init_codebooks(codebooks_init)
        if pq.use_rate and prior_init_counts is not None:
            pq.init_prior_from_freq(prior_init_counts)
    else:
        if verbose:
            print(f"  K-means init on Haar tokens ({N_img} images)...")
        all_Z = []
        for start in range(0, N_img, 200):
            end = min(start + 200, N_img)
            X = torch.from_numpy(features_array[start:end]).float().to(device)
            g = torch.from_numpy(groups_array[start:end].astype(np.int64)).to(device)
            with torch.no_grad():
                Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
                seq, _ = haar.encode(Y, g)
                flat = seq.reshape(-1, D)
                Z = transform.encode(flat) if transform is not None else flat
            all_Z.append(Z.cpu())
            del X, Y, g, seq, flat, Z
        Z_flat = torch.cat(all_Z, dim=0)
        if Z_flat.shape[0] > kmeans_max_samples:
            idx = np.random.choice(Z_flat.shape[0], kmeans_max_samples, replace=False)
            Z_flat = Z_flat[idx]
        pq.init_from_kmeans(Z_flat, device=device)
        del all_Z, Z_flat
        torch.cuda.empty_cache()

    trainable = [p for p in codec.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01,
    )
    if verbose:
        n_tr = sum(p.numel() for p in trainable)
        print(f"  Trainable params: {n_tr:,}  (Haar frozen)")
        if transform is not None and hasattr(transform, "orth_error"):
            print(f"  Orthogonal transform: ||R'R-I||={transform.orth_error():.2e}")
        if use_soft:
            print(f"  Soft PQ: τ {tau_start:.2f} → {tau_end:.4f} ({tau_schedule})")

    if verbose:
        print(f"  Pre-computing teacher outputs ({N_img} images)...")
    t_pre = time.time()
    teacher_cache = np.empty_like(features_array)
    with torch.no_grad():
        for start in range(0, N_img, batch_size):
            end = min(start + batch_size, N_img)
            X = torch.from_numpy(features_array[start:end]).float().to(device)
            teacher_cache[start:end] = tail.forward_nograd(X).cpu().numpy()
            del X
    torch.cuda.empty_cache()
    if verbose:
        print(f"  Teacher cache: {teacher_cache.nbytes / 1e9:.1f} GB CPU "
              f"({time.time() - t_pre:.1f}s)")

    val_array = None
    val_teacher = None
    val_groups_arr = None
    if val_features is not None and len(val_features) > 0:
        val_array = np.stack(val_features) if not isinstance(val_features, np.ndarray) else val_features
        val_groups_arr = np.asarray(val_groups)
        n_val = val_array.shape[0]
        val_teacher = np.empty_like(val_array)
        with torch.no_grad():
            for vs in range(0, n_val, batch_size):
                ve = min(vs + batch_size, n_val)
                X_v = torch.from_numpy(val_array[vs:ve]).float().to(device)
                val_teacher[vs:ve] = tail.forward_nograd(X_v).cpu().numpy()
                del X_v
        torch.cuda.empty_cache()

    loader = DataLoader(
        HaarTrainDataset(features_array, teacher_cache, groups_array),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    history = []
    Tm_rate = None

    for epoch in range(epochs):
        t_epoch = time.time()
        if use_soft and epochs > 1:
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
        codec.train()
        haar.eval()

        for X, Y_teacher, groups in loader:
            X = X.to(device, non_blocking=True)
            Y_teacher = Y_teacher.to(device, non_blocking=True)
            groups = groups.to(device, non_blocking=True)
            B = X.shape[0]

            with torch.no_grad():
                Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
                seq, aux = haar.encode(Y, groups)
            Tm = seq.shape[1]
            if Tm_rate is None:
                Tm_rate = int(Tm)

            seq_hat, usage = codec(seq)
            Y_hat = haar.decode(seq_hat, aux)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            X_hat_out = tail(X_hat)
            distortion = ((Y_teacher - X_hat_out) ** 2).sum() / B

            if codec.use_rate:
                rate_bits = codec._last_rate * Tm
                loss = rate_bits + lmbda * distortion
            else:
                loss = distortion

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(codec.parameters(), grad_clip)
            optimizer.step()

            total_distortion += distortion.item() * B
            if codec.use_rate:
                total_rate += codec._last_rate.item() * B
            usage_acc += usage.detach()
            del X, Y, Mu, Std, seq, seq_hat, Y_hat, X_hat, X_hat_out
            del Y_teacher, groups, loss, distortion

        scheduler.step()
        avg_distortion = total_distortion / N_img
        avg_rate = total_rate / N_img if codec.use_rate else 0.0
        perplexity = compute_perplexity(usage_acc)
        dead_entries = int((usage_acc == 0).sum().item())

        val_loss = None
        if val_array is not None:
            val_sum = 0.0
            n_val = val_array.shape[0]
            codec.eval()
            with torch.no_grad():
                for vs in range(0, n_val, batch_size):
                    ve = min(vs + batch_size, n_val)
                    X_v = torch.from_numpy(val_array[vs:ve]).float().to(device)
                    g_v = torch.from_numpy(
                        val_groups_arr[vs:ve].astype(np.int64)).to(device)
                    Bv = X_v.shape[0]
                    Y_v, Mu_v, Std_v = batch_normalize_gpu(
                        X_v, mode=norm_mode, n_prefix=n_prefix)
                    seq_v, aux_v = haar.encode(Y_v, g_v)
                    seqh_v, _ = codec(seq_v)
                    Yh_v = haar.decode(seqh_v, aux_v)
                    Yt_v = torch.from_numpy(val_teacher[vs:ve]).float().to(device)
                    Xh_v = batch_inv_normalize_gpu(Yh_v, Mu_v, Std_v)
                    Xo_v = tail.forward_nograd(Xh_v)
                    val_sum += ((Yt_v - Xo_v) ** 2).sum().item() / Bv * Bv
                    del X_v, Y_v, Mu_v, Std_v, seq_v, seqh_v, Yh_v, Yt_v, Xh_v, Xo_v, g_v
            val_loss = val_sum / n_val

        Tm_tokens = Tm_rate if Tm_rate is not None else 65
        epoch_info = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "loss_distortion": avg_distortion,
            "perplexity": perplexity,
            "val_loss": val_loss,
            "rate_bits": avg_rate,
            "rate_per_image": avg_rate * Tm_tokens if codec.use_rate else 0.0,
            "n_coded": Tm_tokens,
            "dead_entries": dead_entries,
            "temperature": pq.temperature,
            "time": time.time() - t_epoch,
        }
        if transform is not None and hasattr(transform, "orth_error"):
            epoch_info["orth_error"] = transform.orth_error()
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
                  f"{rate_str}{tau_str}{val_str}  ({time.time() - t_epoch:.1f}s)")

    return codec, history


def parse_args():
    p = argparse.ArgumentParser(
        description="Train ORFC on frozen Haar-joint 65-token sequences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--layer", default="blk20")
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--embedding_dim", type=int, default=32)
    p.add_argument("--bottleneck_dim", type=int, default=1024)
    p.add_argument("--norm_mode", default="per_image")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lmbda", type=float, default=0.5)
    p.add_argument("--warm_start_opq", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--init_rotation", default="opq",
                   choices=["opq", "random_orth", "pca", "identity"])
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--tau_start", type=float, default=0.5)
    p.add_argument("--tau_end", type=float, default=0.005)
    p.add_argument("--tau_schedule", default="exponential",
                   choices=["exponential", "linear"])
    p.add_argument("--max_train_images", type=int, default=5000)
    p.add_argument("--kmeans_max_samples", type=int, default=2_000_000)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", default="dinov2_vitl14")
    p.add_argument("--feat_root", default=os.path.join(PROJECT_ROOT, "features"))
    p.add_argument("--train_subset", default="train")
    p.add_argument("--test_subset", default="test")
    p.add_argument("--gt_path", default=os.path.join(
        PROJECT_ROOT, "utils", "imagenet_selected_label500.txt"))
    p.add_argument("--haar_ckpt", default=None)
    p.add_argument("--ckpt_dir", default=os.path.join(HERE, "checkpoints"))
    p.add_argument("--result_dir", default=os.path.join(HERE, "results", "haar_orfc"))
    p.add_argument("--groups_cache", default=os.path.join(
        HERE, "results", "global_residual", "dinov2_vitl14", "groups"))
    p.add_argument("--skip_baselines", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--ckpt_path", default="")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    layer_idx = int(args.layer[-2:])

    haar_path = Path(args.haar_ckpt) if args.haar_ckpt else default_haar_ckpt(
        args.layer, args.backbone)
    if not haar_path.is_file():
        raise FileNotFoundError(haar_path)

    print(f"\n{'#' * 70}")
    print("# Haar-65 + trainable ORFC")
    print(f"# layer={args.layer} (idx={layer_idx})  K={args.K}  e={args.embedding_dim}"
          f"  bt={args.bottleneck_dim}")
    print(f"# λ={args.lmbda}  lr={args.lr}  ep={args.epochs}"
          f"  τ={args.tau_start}→{args.tau_end}")
    print(f"# haar={haar_path.name}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    train_dir = Path(args.feat_root) / args.train_subset / args.backbone / args.layer
    test_dir = Path(args.feat_root) / args.test_subset / args.backbone / args.layer
    train_files = sorted(train_dir.glob("*.npy"))
    test_files = sorted(test_dir.glob("*.npy"))
    if not train_files or not test_files:
        raise FileNotFoundError(f"missing features in {train_dir} or {test_dir}")
    features_train, _ = preload_features(train_files, num_workers=4)
    features_test, basenames_test = preload_features(test_files, num_workers=4)
    gt_test = load_gt(args.gt_path)
    D = int(features_train[0].shape[1])
    T = int(features_train[0].shape[0])
    bt_dim = args.bottleneck_dim if args.bottleneck_dim > 0 else D
    G = bt_dim // args.embedding_dim
    print(f"  D={D} T={T}  G={G} d={args.embedding_dim}  "
          f"train={len(features_train)} test={len(features_test)}")

    if args.max_train_images > 0 and len(features_train) > args.max_train_images:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(features_train), args.max_train_images, replace=False)
        features_train_sub = [features_train[i] for i in idx]
        print(f"  Training subset: {len(features_train_sub)}")
    else:
        features_train_sub = features_train

    n_val = min(args.n_val, len(features_train))
    rng_val = np.random.RandomState(args.seed + 1)
    val_idx = rng_val.choice(len(features_train), n_val, replace=False)
    val_features = [features_train[i] for i in val_idx]

    haar, _haar_meta = load_global_detail(str(haar_path), device=device)
    if not haar.joint:
        raise ValueError(f"Haar ckpt is not joint-65: {haar_path}")
    freeze_haar(haar)
    n_coded = haar.coded_tokens(T)
    print(f"  Haar joint={haar.joint}  coded_tokens={n_coded}")

    cache_dir = Path(args.groups_cache)
    train_groups = load_or_build_groups(
        _groups_path(cache_dir, args.layer, "train", len(features_train_sub), args.seed),
        features_train_sub, args.norm_mode, device, args.batch_size)
    test_groups = load_or_build_groups(
        _groups_path(cache_dir, args.layer, "test", len(features_test), args.seed),
        features_test, args.norm_mode, device, args.batch_size)
    print("  building val groups...")
    val_groups = build_groups(val_features, args.norm_mode, device, args.batch_size)

    print(f"\nLoading {args.backbone}...")
    wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone, device=device)
    tail = FrozenTail(
        list(wrapper.backbone.blocks[layer_idx + 1:]),
        wrapper.backbone.norm, device=device)
    print(f"  Tail: {len(wrapper.backbone.blocks) - layer_idx - 1} blocks")

    results = {
        "config": vars(args),
        "haar_ckpt": str(haar_path),
        "n_coded": int(n_coded),
        "haar_joint": bool(haar.joint),
        "haar_D": int(haar.D),
    }

    R_std, codebooks_std, opq_usage_counts = None, None, None
    need_opq = (not args.eval_only) or (not args.skip_baselines)
    if need_opq:
        print(f"\n{'=' * 60}")
        print("  [OPQ] on frozen Haar-65 tokens")
        print(f"{'=' * 60}")
        wrapper.backbone.cpu()
        if wrapper.head is not None:
            wrapper.head.cpu()
        torch.cuda.empty_cache()

        t0 = time.time()
        vectors = collect_haar_opq_vectors(
            features_train_sub, train_groups, haar, args.norm_mode, device,
            batch_size=200)
        vectors = subsample_vectors(vectors, G, args.kmeans_max_samples, args.seed)
        R_std, codebooks_std, hist_std = learn_opq_rotation(
            vectors, G, args.embedding_dim, args.K,
            max_iter_opq=20, max_iter_kmeans=100,
            device=device, verbose=False,
        )
        del vectors
        torch.cuda.empty_cache()
        print(f"    OPQ done: MSE={hist_std[-1][0]:.8f} ({time.time() - t0:.1f}s)")

        if args.lmbda > 0 and args.warm_start_opq and not args.eval_only:
            opq_usage_counts = opq_usage_on_haar(
                features_train_sub, train_groups, haar, R_std, codebooks_std,
                args.embedding_dim, args.norm_mode, device, 200)
            ppl_opq = np.exp(
                -(opq_usage_counts / opq_usage_counts.sum(-1, keepdims=True)
                  * np.log(opq_usage_counts / opq_usage_counts.sum(-1, keepdims=True)
                           + 1e-30)).sum(-1)
            ).mean()
            print(f"    OPQ empirical ppl={ppl_opq:.1f} (for prior init)")

        wrapper.backbone.to(device)
        if wrapper.head is not None:
            wrapper.head.to(device)
        tail = FrozenTail(
            list(wrapper.backbone.blocks[layer_idx + 1:]),
            wrapper.backbone.norm, device=device)

        if not args.skip_baselines and R_std is not None:
            results["haar_only"] = eval_cascade(
                "haar-only", features_test, basenames_test, test_groups,
                haar, None, tail, wrapper, layer_idx, gt_test,
                args.norm_mode, device, args.batch_size)
            results["haar_opq"] = eval_haar_opq(
                "haar+OPQ", features_test, basenames_test, test_groups,
                haar, codebooks_std, R_std, args.embedding_dim,
                tail, wrapper, layer_idx, gt_test,
                args.norm_mode, device, args.batch_size)

    ckpt = Path(args.ckpt_path) if args.eval_only and args.ckpt_path else codec_path(args)
    if args.eval_only:
        print(f"\n  [Eval-Only] {ckpt}")
        codec = load_codec(str(ckpt), device=device)
        history = []
        train_time = 0.0
    else:
        print(f"\n{'=' * 60}")
        print(f"  [Codec] Haar-65  K={args.K}  λ={args.lmbda}  "
              f"lr={args.lr}  ep={args.epochs}")
        print(f"{'=' * 60}")
        tail = _rebuild_train_tail(wrapper, layer_idx, device)

        transform = OrthogonalTransform(D) if bt_dim == D else None
        R_ws, C_ws = None, None
        if args.warm_start_opq and args.init_rotation == "opq" and transform is not None:
            R_ws = R_std.copy()
            C_ws = [c.copy() for c in codebooks_std]
            if np.linalg.det(R_ws) < 0:
                R_ws[:, -1] *= -1
                C_ws[-1][:, -1] *= -1
                print("    det(R_opq)<0: flipped last col to SO(D)")

        t0 = time.time()
        codec, history = train_haar_orfc(
            features_train=features_train_sub,
            groups_train=train_groups,
            haar=haar,
            tail=tail,
            G=G, K=args.K, d=args.embedding_dim,
            norm_mode=args.norm_mode,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            val_features=val_features,
            val_groups=val_groups,
            verbose=True,
            transform=transform,
            R_init=R_ws,
            codebooks_init=C_ws,
            kmeans_max_samples=args.kmeans_max_samples,
            lmbda=args.lmbda,
            prior_init_counts=opq_usage_counts,
            grad_clip=args.grad_clip,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
            tau_schedule=args.tau_schedule,
            n_prefix=haar.n_prefix,
        )
        train_time = time.time() - t0
        print(f"  Codec training: {train_time:.1f}s")

        ckpt.parent.mkdir(parents=True, exist_ok=True)
        save_codec(codec, str(ckpt))
        extra = torch.load(str(ckpt), map_location="cpu")
        extra["haar_ckpt"] = str(haar_path)
        extra["n_coded"] = int(n_coded)
        extra["layer"] = args.layer
        extra["joint"] = True
        torch.save(extra, str(ckpt))
        print(f"  Codec saved: {ckpt}")

    tail.to("cpu")
    torch.cuda.empty_cache()
    wrapper.backbone.to(device)
    if wrapper.head is not None:
        wrapper.head.to(device)
    tail = FrozenTail(
        list(wrapper.backbone.blocks[layer_idx + 1:]),
        wrapper.backbone.norm, device=device)

    codec.eval()
    results["haar_orfc"] = eval_cascade(
        "haar+ORFC", features_test, basenames_test, test_groups,
        haar, codec, tail, wrapper, layer_idx, gt_test,
        args.norm_mode, device, args.batch_size)

    print("\n  Rate (PQ on 65 tokens + grouping + norm)...")
    train_seq = haar_encode_all(
        features_train_sub, train_groups, haar, args.norm_mode, device, args.batch_size)
    test_seq = haar_encode_all(
        features_test, test_groups, haar, args.norm_mode, device, args.batch_size)
    train_labels = collect_orfc_labels(train_seq, codec, device, args.batch_size)
    test_labels = collect_orfc_labels(test_seq, codec, device, args.batch_size)
    bpt, _ = _rate_from_labels(test_labels, train_labels, G, args.K)
    results["rate"] = pack_image_rate(
        bpt, n_coded=int(test_seq.shape[1]), grouping_bits=GROUPING_BITS_PERM)
    print(f"  pq_bpt={results['rate']['pq_bpt']:.3f}  "
          f"bits/img={results['rate']['bits_per_image']:.1f}  "
          f"BPFP={results['rate']['bpfp']:.4f}")

    results["history"] = history
    results["train_time_s"] = float(train_time)
    results["ckpt"] = str(ckpt)
    out_dir = Path(args.result_dir) / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{codec_stem(args)}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_json}")


if __name__ == "__main__":
    main()
