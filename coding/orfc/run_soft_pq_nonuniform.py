#!/usr/bin/env python
"""Non-uniform-allocation Soft-PQ ground-truth runner.

Reuses ORFC's exact soft_pq training path (``train_soft_pq``) and evaluation
(``evaluate_delta_l_ref`` / classification / segmentation).  The *only*
difference from ``run_soft_pq.py`` is that each group gets its own codebook
size ``K_g`` (a fixed non-uniform bit allocation), realised by
``NonUniformSoftPQ`` (independent per-group codebooks, no nested tree).

Allocation model (R64 @ G=32, menu bits (1,2,3) → sizes (2,4,8)):
    histogram t ∈ [0, 16]:  t groups at K=8, (32-2t) at K=4, t groups at K=2
    → nominal rate = 3t + 2(32-2t) + t = 64 bits/token for every t.
Groups are assigned **descending from group 0** (highest rate first), which
lines the high-rate slots up with OPQ's high-energy leading groups at init.
``t=0`` reproduces the uniform ORFC baseline through this exact code path
(same-口径 control).

Init: OPQ rotation ``R`` learned at the uniform base K, then each group's
codebook is k-means-fit at its assigned ``K_g`` in the ``R`` frame; ``R`` and
the codebooks are jointly trained afterwards (transform is NOT frozen) so the
result approximates V(m)=min_{U,Θ} D for that histogram.

Usage:
    python run_soft_pq_nonuniform.py --layer blk20 --t 4 --eval_seg
"""

import os, sys, argparse, json, math, time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

ORFC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))

from run_multilayer_calibrator import (
    set_seed, preload_features, load_gt, evaluate_accuracy,
)
from opq import batch_normalize_gpu, learn_opq_rotation, batched_kmeans
from backbone.wrapper import Dinov2Wrapper
from soft_pq import (
    NonUniformSoftPQ, OrthogonalTransform, FeatureCodec, FrozenTail,
    train_soft_pq, soft_pq_encode_decode,
)
from run_soft_pq import (
    evaluate_delta_l_ref, pq_encode_decode_features,
    CodecSegmentationEvaluator, OPQSegmentationEvaluator,
    _codec_labels, _histogram_pmf, _rans_encode_bpt,
)

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")


# ------------------------------------------------------------------ allocation

BITS = (1, 2, 3)          # menu bit-widths (mode 0/1/2)
SIZES = (2, 4, 8)         # codebook sizes 2**bits


def build_allocation(t, G, base_bits):
    """Descending K_per_group for histogram ``t`` at rate G*base_bits.

    ``t`` groups step up one mode, ``t`` groups step down one mode, the rest
    stay uniform.  Requires a symmetric 3-mode menu (which (1,2,3) is).
    """
    t = int(t)
    if not 0 <= t <= G // 2:
        raise ValueError(f"t={t} out of range [0,{G // 2}]")
    up = SIZES[2]     # K=8  (mode 2)
    mid = SIZES[1]    # K=4  (mode 1, uniform)
    down = SIZES[0]   # K=2  (mode 0)
    if mid != 2 ** base_bits:
        raise ValueError("base uniform mode must be the middle mode")
    return [up] * t + [mid] * (G - 2 * t) + [down] * t


def per_group_kmeans_init(vectors, R, G, d, K_per_group, device,
                          max_iter=100, max_samples=500_000, seed=42):
    """K-means each group at its own K in the OPQ-rotated frame."""
    X = torch.as_tensor(vectors, device=device, dtype=torch.float32)
    if X.shape[0] > max_samples:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(X.shape[0], generator=g)[:max_samples]
        X = X[idx.to(device)]
    R_t = torch.as_tensor(R, device=device, dtype=torch.float32)
    Z = X @ R_t
    sub = Z.reshape(-1, G, d).permute(1, 0, 2).contiguous()   # [G, N, d]
    books = [None] * G
    for Kval in sorted(set(int(k) for k in K_per_group)):
        gidx = [g for g in range(G) if int(K_per_group[g]) == Kval]
        cent = batched_kmeans(sub[gidx], Kval, max_iter=max_iter,
                              device=device, verbose=False)     # [len, Kval, d]
        for i, g in enumerate(gidx):
            books[g] = cent[i].detach().cpu().numpy().astype(np.float32)
    del X, Z, sub
    torch.cuda.empty_cache()
    return books


def save_nonuniform_codec(codec, K_per_group, args, nominal_bpt, path):
    """Persist a trained NonUniformSoftPQ FeatureCodec for later paired eval."""
    meta = {
        "state_dict": codec.state_dict(),
        "K_per_group": [int(k) for k in K_per_group],
        "G": int(codec.pq.G), "d": int(codec.pq.d), "Kmax": int(codec.pq.K),
        "has_transform": codec.transform is not None,
        "transform_type": (type(codec.transform).__name__
                           if codec.transform else None),
        "D": int(getattr(codec.transform, "D", 0)) or None,
        "t": int(args.t), "nominal_bpt": int(nominal_bpt),
        "norm_mode": args.norm_mode, "embedding_dim": int(args.embedding_dim),
        "layer": args.layer, "seed": int(args.seed),
    }
    torch.save(meta, path)


def load_nonuniform_codec(path, device="cuda"):
    """Reconstruct a NonUniformSoftPQ FeatureCodec saved by the runner."""
    meta = torch.load(path, map_location="cpu")
    pq = NonUniformSoftPQ(meta["G"], meta["K_per_group"], meta["d"], lmbda=0.0)
    transform = None
    if meta.get("has_transform") and meta.get("transform_type") == "OrthogonalTransform":
        transform = OrthogonalTransform(meta["D"])
    codec = FeatureCodec(pq, transform)
    codec.load_state_dict(meta["state_dict"])
    return codec.to(device).eval(), meta


# ------------------------------------------------------------------ experiment

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    layer_idx = int(args.layer[-2:])

    G = 1024 // args.embedding_dim if args.feat_dim == 1024 else \
        args.feat_dim // args.embedding_dim
    base_bits = int(round(math.log2(args.base_K)))
    K_per_group = build_allocation(args.t, G, base_bits)
    nominal_bpt = sum(int(round(math.log2(k))) for k in K_per_group)

    print("#" * 70)
    print(f"# Non-uniform Soft-PQ  layer={args.layer} t={args.t}  G={G}")
    print(f"# K_per_group counts: "
          f"K8={K_per_group.count(8)} K4={K_per_group.count(4)} "
          f"K2={K_per_group.count(2)}  nominal={nominal_bpt} bpt")
    print(f"# lr={args.lr} lmbda={args.lmbda} ep={args.epochs} seed={args.seed} "
          f"tau={args.tau_start}->{args.tau_end}")
    print(f"# {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("#" * 70)

    # ---- data ----
    train_dir = Path(args.feat_root) / args.train_subset / args.backbone / args.layer
    test_dir = Path(args.feat_root) / args.test_subset / args.backbone / args.layer
    train_files = sorted(train_dir.glob("*.npy"))
    test_files = sorted(test_dir.glob("*.npy"))
    if not train_files or not test_files:
        raise FileNotFoundError(f"no features under {train_dir} / {test_dir}")
    features_train, _ = preload_features(train_files, num_workers=4)
    features_test, basenames_test = preload_features(test_files, num_workers=4)
    gt_test = load_gt(args.gt_path)
    D = features_train[0].shape[1]
    assert D == G * args.embedding_dim, f"D={D} != G*d={G * args.embedding_dim}"

    if args.max_train_images > 0 and len(features_train) > args.max_train_images:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(features_train), args.max_train_images, replace=False)
        features_train_sub = [features_train[i] for i in idx]
    else:
        features_train_sub = features_train
    n_val = min(args.n_val, len(features_train))
    rng_val = np.random.RandomState(args.seed + 1)
    val_idx = rng_val.choice(len(features_train), n_val, replace=False)
    val_features = [features_train[i] for i in val_idx]

    # ---- backbone (tail built after OPQ to avoid the .cpu() shuffle) ----
    wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone, device=device)

    results = {"config": vars(args), "allocation": {
        "t": args.t, "K_per_group": K_per_group, "nominal_bpt": nominal_bpt,
        "counts": {"K8": K_per_group.count(8), "K4": K_per_group.count(4),
                   "K2": K_per_group.count(2)}}}

    # ---- OPQ rotation at the uniform base K (offload backbone to CPU) ----
    wrapper.backbone.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()

    t0 = time.time()
    chunks = []
    for start in range(0, len(features_train_sub), 200):
        end = min(start + 200, len(features_train_sub))
        X = torch.from_numpy(np.stack(features_train_sub[start:end])).float().to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode=args.norm_mode)
        chunks.append(Y.reshape(-1, D).cpu().numpy())
        del X, Y
    full_vectors = np.concatenate(chunks, axis=0)
    del chunks
    # Cap OPQ token count exactly like run_soft_pq (kmeans_max_samples // G).
    max_flat = max(1, args.kmeans_max_samples // G)
    if full_vectors.shape[0] > max_flat:
        rng2 = np.random.RandomState(args.seed)
        full_vectors = full_vectors[
            rng2.choice(full_vectors.shape[0], max_flat, replace=False)]

    R_std, codebooks_std, hist_std = learn_opq_rotation(
        full_vectors, G, args.embedding_dim, args.base_K,
        max_iter_opq=args.opq_iter, max_iter_kmeans=args.opq_kmeans_iter,
        device=device, verbose=False)
    print(f"  OPQ(base K={args.base_K}) MSE={hist_std[-1][0]:.8f} "
          f"({time.time() - t0:.1f}s)")

    # per-group k-means init at the assigned sizes, in the OPQ frame
    codebooks_init = per_group_kmeans_init(
        full_vectors, R_std, G, args.embedding_dim, K_per_group, device,
        max_iter=args.opq_kmeans_iter, seed=args.seed)
    del full_vectors
    torch.cuda.empty_cache()

    # ---- build the frozen tail on device (backbone tail blocks + norm) ----
    tail = FrozenTail(list(wrapper.backbone.blocks[layer_idx + 1:]),
                      wrapper.backbone.norm, device=device)

    # ---- standard OPQ (uniform, untrained) ΔL_ref reference ----
    std_delta_l = evaluate_delta_l_ref(
        features_test, tail, args.norm_mode, device,
        codebooks=codebooks_std, R=R_std,
        embedding_dim=args.embedding_dim, batch_size=args.batch_size)
    results["std_opq_delta_l"] = float(std_delta_l)
    print(f"  Standard OPQ (uniform K={args.base_K}) ΔL_ref = {std_delta_l:.1f}")

    # ---- train the non-uniform codec (transform + per-group PQ) ----
    for i, blk in enumerate(wrapper.backbone.blocks):
        if i <= layer_idx:
            blk.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()

    pq = NonUniformSoftPQ(G, K_per_group, args.embedding_dim, lmbda=args.lmbda)
    transform = OrthogonalTransform(D)
    t1 = time.time()
    codec, history = train_soft_pq(
        features_train=features_train_sub, tail=tail,
        G=G, K=pq.K, d=args.embedding_dim,
        norm_mode=args.norm_mode, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, device=device, seed=args.seed,
        val_features=val_features, verbose=True,
        transform=transform, R_init=R_std, codebooks_init=codebooks_init,
        kmeans_max_samples=args.kmeans_max_samples, use_mse_loss=False,
        lmbda=args.lmbda, grad_clip=args.grad_clip,
        tau_start=args.tau_start, tau_end=args.tau_end,
        tau_schedule=args.tau_schedule, pq=pq)
    train_time = time.time() - t1
    print(f"  Codec training: {train_time:.1f}s")

    if args.save_codec:
        ckpt_dir = Path(ORFC_ROOT) / "results" / "soft_pq_nonuniform" / args.backbone
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_tag = (f"{args.layer}_R{nominal_bpt}_t{args.t:02d}_baseK{args.base_K}"
                    f"_emb{args.embedding_dim}_lmbda{args.lmbda}"
                    f"_lr{args.lr}_ep{args.epochs}_s{args.seed}")
        ckpt_path = ckpt_dir / f"{ckpt_tag}.pt"
        save_nonuniform_codec(codec, K_per_group, args, nominal_bpt, ckpt_path)
        print(f"  Saved codec: {ckpt_path}")

    # ---- ΔL_ref (Tail MSE) on test (reuse the on-device tail) ----
    codec_delta_l = evaluate_delta_l_ref(
        features_test, tail, args.norm_mode, device,
        codec=codec, batch_size=args.batch_size)
    results["codec_delta_l"] = float(codec_delta_l)
    print(f"  Codec ΔL_ref (Tail MSE) = {codec_delta_l:.1f} "
          f"(d vs OPQ = {codec_delta_l - std_delta_l:+.1f})")
    tail.to("cpu")
    torch.cuda.empty_cache()

    # ---- classification ----
    wrapper.backbone.to(device)
    if wrapper.head is not None:
        wrapper.head.to(device)
    torch.cuda.empty_cache()
    xhat = soft_pq_encode_decode(features_test, codec, args.norm_mode, device)
    acc = evaluate_accuracy(xhat, basenames_test, gt_test, wrapper,
                            layer_idx, device)
    results["codec_acc"] = float(acc)
    print(f"  Codec Acc = {acc:.4f}")
    del xhat

    # ---- rate (per-group K, real rANS) ----
    codec.to(device)
    train_labels = _codec_labels(features_train_sub, codec, args.norm_mode,
                                 device, batch_size=args.batch_size)
    train_pmf = _histogram_pmf(train_labels, G, K_per_group)
    test_labels = _codec_labels(features_test, codec, args.norm_mode,
                                device, batch_size=args.batch_size)
    rans_bpt = _rans_encode_bpt(test_labels, train_pmf, G, K_per_group)
    results["rate"] = {"nominal_bpt": nominal_bpt,
                       "rans_bpt": float(rans_bpt) if rans_bpt else None}
    print(f"  Rate: nominal={nominal_bpt}  rANS={results['rate']['rans_bpt']}")

    # ---- segmentation ----
    if args.eval_seg:
        seg_feat_dir = Path(args.seg_feat_root) / args.backbone / args.layer
        if seg_feat_dir.exists():
            wrapper.backbone.cpu()
            if wrapper.head is not None:
                wrapper.head.cpu()
            torch.cuda.empty_cache()
            codec_seg = CodecSegmentationEvaluator(
                codec=codec, norm_mode=args.norm_mode, layer_idx=layer_idx,
                voc_root=args.voc_root, weights_root=wrapper.weights_root,
                device=device, feat_dim=D, model_name=args.backbone)
            seg = codec_seg.evaluate(seg_feat_dir=str(seg_feat_dir),
                                     image_list=args.seg_image_list, verbose=False)
            results["codec_miou"] = float(seg["miou"])
            print(f"  Codec mIoU = {seg['miou']:.4f}")
            del codec_seg
            torch.cuda.empty_cache()
        else:
            print(f"  [seg] skipped (missing {seg_feat_dir})")

    results["history"] = history
    results["train_time"] = float(train_time)

    out_dir = Path(ORFC_ROOT) / "results" / "soft_pq_nonuniform" / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"{args.layer}_R{nominal_bpt}_t{args.t:02d}_baseK{args.base_K}"
           f"_emb{args.embedding_dim}_lmbda{args.lmbda}"
           f"_lr{args.lr}_ep{args.epochs}_s{args.seed}")
    out_path = out_dir / f"{tag}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {out_path}")
    print(f"SUMMARY t={args.t} TailMSE={results['codec_delta_l']:.1f} "
          f"Acc={results['codec_acc']:.4f} "
          f"mIoU={results.get('codec_miou', 'NA')}")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--layer", type=str, default="blk20")
    p.add_argument("--t", type=int, required=True,
                   help="histogram index 0..16 (0=uniform control)")
    p.add_argument("--base_K", type=int, default=4,
                   help="uniform base codebook size (R64 -> 4)")
    p.add_argument("--embedding_dim", type=int, default=32)
    p.add_argument("--feat_dim", type=int, default=1024)
    p.add_argument("--norm_mode", type=str, default="per_image")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--lmbda", type=float, default=0.0)
    p.add_argument("--tau_start", type=float, default=0.5)
    p.add_argument("--tau_end", type=float, default=0.005)
    p.add_argument("--tau_schedule", type=str, default="exponential")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max_train_images", type=int, default=5000)
    p.add_argument("--kmeans_max_samples", type=int, default=2_000_000)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--opq_iter", type=int, default=20)
    p.add_argument("--opq_kmeans_iter", type=int, default=100)

    p.add_argument("--backbone", type=str, default="dinov2_vitl14")
    p.add_argument("--feat_root", type=str,
                   default=os.path.join(PROJECT_ROOT, "features"))
    p.add_argument("--train_subset", type=str, default="train")
    p.add_argument("--test_subset", type=str, default="test")
    p.add_argument("--gt_path", type=str,
                   default=os.path.join(PROJECT_ROOT, "utils",
                                        "imagenet_selected_label500.txt"))
    p.add_argument("--eval_seg", action="store_true")
    p.add_argument("--save_codec", action="store_true",
                   help="persist the trained codec (.pt) for paired eval")
    p.add_argument("--seg_feat_root", type=str,
                   default=os.path.join(PROJECT_ROOT, "features", "voc2012_100"))
    p.add_argument("--voc_root", type=str,
                   default=os.path.join(PROJECT_ROOT, "data", "VOCdevkit", "VOC2012"))
    p.add_argument("--seg_image_list", type=str,
                   default=os.path.join(PROJECT_ROOT, "utils", "voc2012_val_100.txt"))
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
