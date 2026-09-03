#!/usr/bin/env python
"""Joint fine-tune: unfreeze Haar-joint analysis/synthesis + ORFC R/codebooks.

Loads frozen-Haar ORFC checkpoints and continues training with quantized
ΔL_ref (same cascade as ``run_haar_orfc_train.py``, but Haar encode stays
in the graph).

Usage:
    CUDA_VISIBLE_DEVICES=2 python -u run_haar_orfc_finetune.py \\
        --layer blk05 --K 16 --tau_start 0.5 --epochs 50
    CUDA_VISIBLE_DEVICES=2 python -u run_haar_orfc_finetune.py \\
        --layer blk05 --K 16 --tau_start 1.5 --epochs 50
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
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
_ORFC_DIR = os.path.normpath(os.path.join(HERE, "..", "orfc"))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if _ORFC_DIR not in sys.path:
    sys.path.insert(0, _ORFC_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from backbone.wrapper import Dinov2Wrapper  # noqa: E402
from opq import batch_inv_normalize_gpu, batch_normalize_gpu  # noqa: E402
from run_multilayer_calibrator import (  # noqa: E402
    load_gt, preload_features, set_seed,
)
from soft_pq import FrozenTail, compute_perplexity, load_codec, save_codec  # noqa: E402

from eval_haar_orfc import (  # noqa: E402
    GROUPING_BITS_PERM, _rate_from_labels, collect_orfc_labels,
    eval_cascade, haar_encode_all, pack_image_rate,
)
from run_global_residual import load_global_detail, save_global_detail  # noqa: E402
from run_haar_orfc_train import (  # noqa: E402
    HaarTrainDataset, _groups_path, _rebuild_train_tail,
    default_haar_ckpt, load_or_build_groups,
)

warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")


def default_orfc_ckpt(layer, K, backbone="dinov2_vitl14", ckpt_dir=None,
                      lmbda=0.5, lr=3e-4, epochs=100, n=5000, seed=42):
    ckpt_dir = Path(ckpt_dir or Path(HERE) / "checkpoints")
    name = (
        f"{layer}_haar65_K{K}_emb32_bt1024_ws_lmbda{lmbda}"
        f"_tau0.5_lr{lr}_ep{epochs}_n{n}_s{seed}.pt"
    )
    return ckpt_dir / backbone / name


def finetune_stem(args):
    bt_tag = f"bt{args.bottleneck_dim}" if args.bottleneck_dim > 0 else "noBt"
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
        f"{args.layer}_haar65_jointft_K{args.K}_emb{args.embedding_dim}"
        f"_{bt_tag}_ws{rate_tag}{tau_tag}{tau_end_tag}{tau_sched_tag}"
        f"_lr{args.lr}_ep{args.epochs}_n{args.max_train_images}_s{args.seed}"
    )


def unfreeze_haar(haar):
    haar.train()
    for p in haar.parameters():
        p.requires_grad_(True)
    return haar


def train_joint_finetune(
    features_train, groups_train, haar, codec, tail,
    epochs, lr, batch_size, device, seed,
    val_features=None, val_groups=None,
    lmbda=0.5, grad_clip=1.0,
    tau_start=0.5, tau_end=0.005, tau_schedule="exponential",
    n_prefix=1, max_steps=0, verbose=True,
):
    """Haar encode stays in-graph.  Optimizer = Haar Linear + ORFC R/PQ/prior."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if isinstance(features_train, np.ndarray) and features_train.ndim == 3:
        features_array = features_train
    else:
        features_array = np.stack(features_train)
    N_img = features_array.shape[0]
    groups_array = np.asarray(groups_train)
    pq = codec.pq
    G, K = pq.G, pq.K
    use_soft = tau_start > 0

    unfreeze_haar(haar)
    codec.train()
    trainable = [p for p in list(haar.parameters()) + list(codec.parameters())
                 if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=lr * 0.01)
    if verbose:
        n_h = sum(p.numel() for p in haar.parameters() if p.requires_grad)
        n_c = sum(p.numel() for p in codec.parameters() if p.requires_grad)
        print(f"  Trainable: Haar {n_h:,} + ORFC {n_c:,} = {n_h + n_c:,}")
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

    val_array = val_teacher = val_groups_arr = None
    if val_features is not None and len(val_features) > 0:
        val_array = (val_features if isinstance(val_features, np.ndarray)
                     else np.stack(val_features))
        val_groups_arr = np.asarray(val_groups)
        val_teacher = np.empty_like(val_array)
        with torch.no_grad():
            for vs in range(0, val_array.shape[0], batch_size):
                ve = min(vs + batch_size, val_array.shape[0])
                X_v = torch.from_numpy(val_array[vs:ve]).float().to(device)
                val_teacher[vs:ve] = tail.forward_nograd(X_v).cpu().numpy()
                del X_v
        torch.cuda.empty_cache()

    loader = DataLoader(
        HaarTrainDataset(features_array, teacher_cache, groups_array),
        batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )
    history = []
    global_step = 0
    stop = False

    for epoch in range(epochs):
        if stop:
            break
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
        n_seen = 0
        usage_acc = torch.zeros(G, K, device=device)
        codec.train()
        haar.train()
        Tm_rate = 65

        for X, Y_teacher, groups in loader:
            X = X.to(device, non_blocking=True)
            Y_teacher = Y_teacher.to(device, non_blocking=True)
            groups = groups.to(device, non_blocking=True)
            B = X.shape[0]

            with torch.no_grad():
                Y, Mu, Std = batch_normalize_gpu(
                    X, mode="per_image", n_prefix=n_prefix)
            seq, aux = haar.encode(Y, groups)
            Tm_rate = int(seq.shape[1])
            seq_hat, usage = codec(seq)
            Y_hat = haar.decode(seq_hat, aux)
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
            n_seen += B
            if codec.use_rate:
                total_rate += codec._last_rate.item() * B
            usage_acc += usage.detach()
            global_step += 1
            del X, Y, Mu, Std, seq, seq_hat, Y_hat, X_hat
            del Y_teacher, groups, loss, distortion

            if max_steps > 0 and global_step >= max_steps:
                stop = True
                break

        scheduler.step()
        denom = max(n_seen, 1)
        avg_distortion = total_distortion / denom
        avg_rate = total_rate / denom if codec.use_rate else 0.0
        perplexity = compute_perplexity(usage_acc)
        dead_entries = int((usage_acc == 0).sum().item())

        val_loss = None
        if val_array is not None and not stop:
            val_sum = 0.0
            n_val = val_array.shape[0]
            codec.eval()
            haar.eval()
            with torch.no_grad():
                for vs in range(0, n_val, batch_size):
                    ve = min(vs + batch_size, n_val)
                    X_v = torch.from_numpy(val_array[vs:ve]).float().to(device)
                    g_v = torch.from_numpy(
                        val_groups_arr[vs:ve].astype(np.int64)).to(device)
                    Bv = X_v.shape[0]
                    Y_v, Mu_v, Std_v = batch_normalize_gpu(
                        X_v, mode="per_image", n_prefix=n_prefix)
                    seq_v, aux_v = haar.encode(Y_v, g_v)
                    seqh_v, _ = codec(seq_v)
                    Yh_v = haar.decode(seqh_v, aux_v)
                    Yt_v = torch.from_numpy(val_teacher[vs:ve]).float().to(device)
                    Xh_v = batch_inv_normalize_gpu(Yh_v, Mu_v, Std_v)
                    Xo_v = tail.forward_nograd(Xh_v)
                    val_sum += ((Yt_v - Xo_v) ** 2).sum().item()
                    del X_v, Y_v, Mu_v, Std_v, seq_v, seqh_v, Yh_v
                    del Yt_v, Xh_v, Xo_v, g_v
            val_loss = val_sum / n_val

        epoch_info = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "loss_distortion": avg_distortion,
            "perplexity": perplexity,
            "val_loss": val_loss,
            "rate_bits": avg_rate,
            "dead_entries": dead_entries,
            "temperature": pq.temperature,
            "time": time.time() - t_epoch,
            "steps": global_step,
        }
        history.append(epoch_info)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1 or stop):
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
        if stop:
            break

    return haar, codec, history


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--layer", default="blk05")
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--embedding_dim", type=int, default=32)
    p.add_argument("--bottleneck_dim", type=int, default=1024)
    p.add_argument("--norm_mode", default="per_image")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lmbda", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--tau_start", type=float, default=0.5)
    p.add_argument("--tau_end", type=float, default=0.005)
    p.add_argument("--tau_schedule", default="exponential",
                   choices=["exponential", "linear"])
    p.add_argument("--max_train_images", type=int, default=5000)
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
    p.add_argument("--orfc_ckpt", default=None)
    p.add_argument("--orfc_train_epochs", type=int, default=100,
                   help="Epoch tag of the frozen-Haar ORFC checkpoint to load")
    p.add_argument("--ckpt_dir", default=os.path.join(HERE, "checkpoints"))
    p.add_argument("--result_dir", default=os.path.join(
        HERE, "results", "haar_orfc_jointft"))
    p.add_argument("--groups_cache", default=os.path.join(
        HERE, "results", "global_residual", "dinov2_vitl14", "groups"))
    p.add_argument("--skip_baselines", action="store_true")
    p.add_argument("--max_steps", type=int, default=0,
                   help="Stop after N optimizer steps (0 = full epochs)")
    p.add_argument("--mem_probe", action="store_true",
                   help="3-step memory probe: skip eval/save")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mem_probe:
        args.max_steps = args.max_steps or 3
        args.skip_baselines = True
        args.n_val = 0
        args.max_train_images = min(args.max_train_images, 96)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    layer_idx = int(args.layer[-2:])

    haar_path = Path(args.haar_ckpt) if args.haar_ckpt else default_haar_ckpt(
        args.layer, args.backbone)
    orfc_path = Path(args.orfc_ckpt) if args.orfc_ckpt else default_orfc_ckpt(
        args.layer, args.K, args.backbone, args.ckpt_dir,
        lmbda=args.lmbda, lr=args.lr, epochs=args.orfc_train_epochs,
        n=5000, seed=args.seed)
    if not haar_path.is_file():
        raise FileNotFoundError(haar_path)
    if not orfc_path.is_file():
        raise FileNotFoundError(orfc_path)

    print(f"\n{'#' * 70}")
    print("# Haar-65 + ORFC joint fine-tune  (unfreeze analysis/synthesis + R/PQ)")
    print(f"# layer={args.layer}  K={args.K}  λ={args.lmbda}  lr={args.lr}  "
          f"ep={args.epochs}  τ={args.tau_start}→{args.tau_end}")
    print(f"# haar={haar_path.name}")
    print(f"# orfc={orfc_path.name}")
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
    G = args.bottleneck_dim // args.embedding_dim
    print(f"  D={D} T={T}  G={G} d={args.embedding_dim}")

    if args.max_train_images > 0 and len(features_train) > args.max_train_images:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(features_train), args.max_train_images, replace=False)
        features_train_sub = [features_train[i] for i in idx]
        print(f"  Training subset: {len(features_train_sub)}")
    else:
        features_train_sub = features_train

    n_val = min(args.n_val, len(features_train))
    if n_val > 0:
        rng_val = np.random.RandomState(args.seed + 1)
        val_idx = rng_val.choice(len(features_train), n_val, replace=False)
        val_features = [features_train[i] for i in val_idx]
    else:
        val_features = []

    haar, _ = load_global_detail(str(haar_path), device=device)
    if not haar.joint:
        raise ValueError(f"need joint-65 Haar: {haar_path}")
    codec = load_codec(str(orfc_path), device=device)
    if codec.pq.K != args.K:
        raise ValueError(f"ORFC ckpt K={codec.pq.K} != --K {args.K}")
    n_coded = haar.coded_tokens(T)
    print(f"  Haar joint coded={n_coded}  ORFC G={codec.pq.G} K={codec.pq.K} "
          f"λ={codec.pq.lmbda}")

    cache_dir = Path(args.groups_cache)
    train_groups = load_or_build_groups(
        _groups_path(cache_dir, args.layer, "train",
                     len(features_train_sub), args.seed),
        features_train_sub, args.norm_mode, device, args.batch_size)
    test_groups = load_or_build_groups(
        _groups_path(cache_dir, args.layer, "test",
                     len(features_test), args.seed),
        features_test, args.norm_mode, device, args.batch_size)
    val_groups = None
    if val_features:
        print("  building val groups...")
        from run_global_residual import build_groups
        val_groups = build_groups(
            val_features, args.norm_mode, device, args.batch_size)

    print(f"\nLoading {args.backbone}...")
    wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone, device=device)
    results = {
        "config": vars(args),
        "haar_ckpt": str(haar_path),
        "orfc_ckpt": str(orfc_path),
        "n_coded": int(n_coded),
    }

    if not args.skip_baselines:
        tail = FrozenTail(
            list(wrapper.backbone.blocks[layer_idx + 1:]),
            wrapper.backbone.norm, device=device)
        haar.eval()
        codec.eval()
        results["before"] = eval_cascade(
            "before-ft", features_test, basenames_test, test_groups,
            haar, codec, tail, wrapper, layer_idx, gt_test,
            args.norm_mode, device, args.batch_size)

    print(f"\n{'=' * 60}")
    print(f"  [Joint FT] K={args.K}  λ={args.lmbda}  lr={args.lr}  "
          f"ep={args.epochs}  τ={args.tau_start}→{args.tau_end}")
    print(f"{'=' * 60}")
    tail = _rebuild_train_tail(wrapper, layer_idx, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    t0 = time.time()
    haar, codec, history = train_joint_finetune(
        features_train=features_train_sub,
        groups_train=train_groups,
        haar=haar,
        codec=codec,
        tail=tail,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        val_features=val_features if val_features else None,
        val_groups=val_groups,
        lmbda=args.lmbda,
        grad_clip=args.grad_clip,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        tau_schedule=args.tau_schedule,
        n_prefix=haar.n_prefix,
        max_steps=args.max_steps,
        verbose=True,
    )
    train_time = time.time() - t0
    peak_gb = None
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  peak CUDA alloc: {peak_gb:.2f} GB  "
              f"(batch={args.batch_size})")
    print(f"  Joint FT: {train_time:.1f}s")

    if args.mem_probe:
        print("  mem_probe done, skip save/eval")
        results["peak_gb"] = peak_gb
        results["batch_size"] = args.batch_size
        print(json.dumps({"peak_gb": peak_gb, "batch_size": args.batch_size}, indent=2))
        return

    stem = finetune_stem(args)
    out_ckpt_dir = Path(args.ckpt_dir) / args.backbone
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)
    orfc_out = out_ckpt_dir / f"{stem}.pt"
    haar_out = out_ckpt_dir / f"{stem}_haar.pt"
    save_codec(codec, str(orfc_out))
    extra = torch.load(str(orfc_out), map_location="cpu")
    extra.update({
        "haar_ckpt": str(haar_out),
        "orfc_init": str(orfc_path),
        "n_coded": int(n_coded),
        "layer": args.layer,
        "joint": True,
        "joint_finetune": True,
        "tau_start": args.tau_start,
        "tau_end": args.tau_end,
    })
    torch.save(extra, str(orfc_out))
    save_global_detail(
        haar, str(haar_out),
        extra={"layer": args.layer, "joint": True, "joint_finetune": True,
               "orfc_ckpt": str(orfc_out)})
    print(f"  saved ORFC {orfc_out.name}")
    print(f"  saved Haar {haar_out.name}")

    tail.to("cpu")
    torch.cuda.empty_cache()
    wrapper.backbone.to(device)
    if wrapper.head is not None:
        wrapper.head.to(device)
    tail = FrozenTail(
        list(wrapper.backbone.blocks[layer_idx + 1:]),
        wrapper.backbone.norm, device=device)
    haar.eval()
    codec.eval()
    results["after"] = eval_cascade(
        "after-ft", features_test, basenames_test, test_groups,
        haar, codec, tail, wrapper, layer_idx, gt_test,
        args.norm_mode, device, args.batch_size)

    print("\n  Rate (PQ on 65 tokens + grouping + norm)...")
    train_seq = haar_encode_all(
        features_train_sub, train_groups, haar, args.norm_mode,
        device, args.batch_size)
    test_seq = haar_encode_all(
        features_test, test_groups, haar, args.norm_mode,
        device, args.batch_size)
    train_labels = collect_orfc_labels(train_seq, codec, device, args.batch_size)
    test_labels = collect_orfc_labels(test_seq, codec, device, args.batch_size)
    bpt, _ = _rate_from_labels(test_labels, train_labels, G, args.K)
    results["rate"] = pack_image_rate(
        bpt, n_coded=int(test_seq.shape[1]), grouping_bits=GROUPING_BITS_PERM)
    print(f"  pq_bpt={results['rate']['pq_bpt']:.3f}  "
          f"BPFP={results['rate']['bpfp']:.4f}")

    results["history"] = history
    results["train_time_s"] = float(train_time)
    results["peak_gb"] = peak_gb
    results["ckpt"] = str(orfc_out)
    results["haar_out"] = str(haar_out)
    out_dir = Path(args.result_dir) / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{stem}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_json}")


if __name__ == "__main__":
    main()
