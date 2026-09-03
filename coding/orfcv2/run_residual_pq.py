#!/usr/bin/env python
"""
Scalable feature codec runner: frozen base codebook + task residual codebook.

Stage 1 (reused, not retrained): a base codec trained by orfc/run_soft_pq.py
under the tail MSE (ΔL_ref) objective. It encodes generic semantics and is
loaded from orfc/checkpoints/<backbone>/*.pt.

Stage 2 (trained here): a residual codec with its *own* orthogonal transform,
codebooks and prior, trained on R = Y - Ŷ_base under the classification loss.
The base bitstream is untouched, so the stream is scalable:
    base only        -> generic reconstruction (identical to stage 1)
    base + residual  -> task-enhanced reconstruction

Usage:
    python run_residual_pq.py \
        --base_ckpt ../orfc/checkpoints/dinov2_vitl14/blk20_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt \
        --layer blk20 --K 16 --embedding_dim 32 --epochs 100

    # smoke test
    python run_residual_pq.py --base_ckpt <...>.pt --layer blk20 --K 16 \
        --epochs 3 --max_train_images 100 --n_val 20
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ORFCV2_ROOT = os.path.dirname(os.path.abspath(__file__))
CODING_ROOT = os.path.dirname(ORFCV2_ROOT)
ORFC_ROOT = os.path.join(CODING_ROOT, "orfc")
PROJECT_ROOT = os.path.dirname(CODING_ROOT)           # .../featcodec/ORFC
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

from opq import (
    batch_inv_normalize_gpu, batch_normalize_gpu, batched_assign,
    learn_opq_rotation,
)
from run_multilayer_calibrator import (
    evaluate_accuracy, load_gt, preload_features, set_seed,
)
from run_soft_pq import (
    _histogram_pmf, _rans_encode_bpt, evaluate_delta_l_ref,
)
from soft_pq import FrozenTail, load_codec
from backbone.wrapper import Dinov2Wrapper, SegmentationEvaluator

from residual_pq import (
    ClsTaskTail, SegTaskTail, DepthTaskTail, build_residual_codec,
    collect_residuals, collect_stage_labels, infer_token_hw,
    load_residual_codec, residual_encode_decode, save_residual_codec,
    share_base_transform, train_residual_pq,
)


# ================================================================
#                    Rate helpers
# ================================================================

def stage_rate(labels, K, train_pmf=None):
    """Rate metrics for one stage's labels [G, N]."""
    G, N = labels.shape
    pmf = train_pmf if train_pmf is not None else _histogram_pmf(labels, G, K)
    xent = 0.0
    for g in range(G):
        xent += -np.log2(pmf[g] + 1e-30)[labels[g]].sum()
    emp_pmf = _histogram_pmf(labels, G, K, smoothing=0)
    entropy = 0.0
    for g in range(G):
        p = emp_pmf[g]
        p = p[p > 0]
        entropy += -np.sum(p * np.log2(p))
    out = {
        'xent_bpt': float(xent / N),
        'empirical_entropy_bpt': float(entropy),
        'max_rate_bpt': float(G * math.log2(K)),
    }
    rans = _rans_encode_bpt(labels, pmf, G, K)
    if rans is not None:
        out['rans_bpt'] = float(rans)
    return out


class ResidualSegEvaluator(SegmentationEvaluator):
    """VOC2012 slide-inference mIoU with the two-stage codec.

    base_only=True drops the residual stream, which reproduces the stage-1
    (generic) reconstruction exactly.
    """

    def __init__(self, codec, norm_mode, layer_idx, voc_root, weights_root,
                 device='cuda', feat_dim=1024, model_name='dinov2_vitl14',
                 base_only=False):
        self.codec = codec
        self.base_only = base_only
        self.norm_mode = norm_mode
        self.layer_idx = layer_idx
        self.voc_root = voc_root
        self.weights_root = weights_root
        self.device = device
        self.feat_dim = feat_dim
        self.model_name = model_name

    @torch.no_grad()
    def quantize_tokens(self, tokens_np):
        X = torch.from_numpy(tokens_np).float().unsqueeze(0).to(self.device)
        Y, Mu, Std = batch_normalize_gpu(X, mode=self.norm_mode)
        Y_hat = (self.codec.base_forward(Y) if self.base_only
                 else self.codec(Y)[0])
        return batch_inv_normalize_gpu(Y_hat, Mu, Std).squeeze(0)


def load_seg_features(seg_feat_dir, image_list):
    """Flatten VOC per-image slide features into a list of [T, D] arrays."""
    with open(image_list) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    flat = []
    for name in names:
        fp = os.path.join(str(seg_feat_dir), f"{name}.npy")
        if os.path.exists(fp):
            arr = np.load(fp)
            for s in range(arr.shape[0]):
                flat.append(arr[s].astype(np.float32))
    return flat


def fmt_rate(name, info, feat_dim):
    rans = info.get('rans_bpt')
    rans_s = f"rANS={rans:.2f}" if rans else "rANS=N/A"
    bpfp_s = f"BPFP={rans / feat_dim:.4f}" if rans else ""
    return (f"{name}: {rans_s}  xent={info['xent_bpt']:.2f}  "
            f"H_emp={info['empirical_entropy_bpt']:.2f}  "
            f"max={info['max_rate_bpt']:.0f}  {bpfp_s}")


# ================================================================
#                    Residual codebook initialisation
# ================================================================

def _prior_from_latents(pq, Z_lat, G, K, d, device, verbose=True, tag="k-means"):
    """Fit log_prior from nearest-neighbour assignments of already-rotated Z."""
    if not pq.use_rate:
        return None
    counts = np.zeros((G, K), dtype=np.float64)
    chunk = 200_000
    C = pq.codebooks.detach().to(device)
    if not isinstance(Z_lat, torch.Tensor):
        Z_lat = torch.from_numpy(Z_lat).float()
    with torch.no_grad():
        for s in range(0, Z_lat.shape[0], chunk):
            e = min(s + chunk, Z_lat.shape[0])
            Zg = (Z_lat[s:e].to(device)
                  .reshape(-1, G, d).permute(1, 0, 2).contiguous())
            labels = torch.cdist(Zg, C).argmin(dim=-1).cpu().numpy()
            for g in range(G):
                np.add.at(counts[g], labels[g], 1)
            del Zg
    pq.init_prior_from_freq(counts)
    if verbose:
        p = counts / counts.sum(-1, keepdims=True)
        ppl = np.exp(-(p * np.log(p + 1e-30)).sum(-1)).mean()
        print(f"    residual prior init from {tag} frequency (ppl={ppl:.1f})")
    return counts


def init_residual_codec(res_codec, residual_vectors, G, K, d, device,
                        warm_start_opq=True, seed=42, verbose=True):
    """Warm-start the residual transform + codebooks (+ prior) from OPQ.

    Falls back to plain k-means when warm_start_opq is False or the transform
    is not a same-dim rotation.  If the residual transform was already set
    (e.g. copied from the frozen ORFC R), k-means runs in that coordinate
    system.
    """
    transform = res_codec.transform
    pq = res_codec.pq
    can_warmstart = (
        warm_start_opq and transform is not None
        and hasattr(transform, 'init_from_opq')
        and getattr(transform, 'D', None) == residual_vectors.shape[1]
    )

    if not can_warmstart:
        space = ("residual-transform space" if transform is not None
                 else "ambient space")
        if verbose:
            print(f"  Residual init: k-means on {residual_vectors.shape[0]:,} "
                  f"residual vectors ({space})")
        Z = torch.from_numpy(residual_vectors).float()
        if transform is not None:
            with torch.no_grad():
                Z = transform.to('cpu').encode(Z)
            transform.to(device)
        pq.init_from_kmeans(Z, device=device)
        return _prior_from_latents(
            pq, Z, G, K, d, device, verbose=verbose, tag="k-means")

    if verbose:
        print(f"  Residual init: OPQ on {residual_vectors.shape[0]:,} "
              f"residual vectors (G={G}, d={d}, K={K})")
    t0 = time.time()
    R_res, cb_res, hist = learn_opq_rotation(
        residual_vectors, G, d, K,
        max_iter_opq=20, max_iter_kmeans=100, device=device, verbose=False,
    )
    R_res = R_res.copy()
    cb_res = [c.copy() for c in cb_res]
    if np.linalg.det(R_res) < 0:
        R_res[:, -1] *= -1
        cb_res[-1][:, -1] *= -1
        if verbose:
            print(f"    det(R_res)<0: flipped last column into SO(D)")
    transform.init_from_opq(R_res)
    pq.init_codebooks(cb_res)
    if verbose:
        print(f"    residual OPQ MSE={hist[-1][0]:.8f} ({time.time() - t0:.1f}s)")

    if not pq.use_rate:
        return None

    # Empirical assignment frequency -> prior init
    R_t = torch.from_numpy(R_res).float().to(device)
    cb_t = torch.from_numpy(np.stack(cb_res)).float().to(device)
    counts = np.zeros((G, K), dtype=np.float64)
    chunk = 200_000
    with torch.no_grad():
        for s in range(0, residual_vectors.shape[0], chunk):
            e = min(s + chunk, residual_vectors.shape[0])
            V = torch.from_numpy(residual_vectors[s:e]).float().to(device)
            Z = (V @ R_t).reshape(-1, G, d).permute(1, 0, 2).contiguous()
            labels = torch.cdist(Z, cb_t).argmin(dim=-1).cpu().numpy()
            for g in range(G):
                np.add.at(counts[g], labels[g], 1)
            del V, Z
    del R_t, cb_t
    torch.cuda.empty_cache()
    pq.init_prior_from_freq(counts)
    if verbose:
        p = counts / counts.sum(-1, keepdims=True)
        ppl = np.exp(-(p * np.log(p + 1e-30)).sum(-1)).mean()
        print(f"    residual prior init from OPQ frequency (ppl={ppl:.1f})")
    return counts


# ================================================================
#                    Main experiment
# ================================================================

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    layer_idx = int(args.layer[-2:])

    assert args.backbone.startswith("dinov2"), \
        "classification task supervision is implemented for DINOv2 only"

    if args.share_base_transform:
        args.warm_start_opq = False
        args.freeze_res_transform = True
    if args.task == 'seg':
        args.eval_seg = True

    print(f"\n{'#' * 70}")
    print(f"# Scalable codec: frozen base (tail-MSE) + task residual")
    print(f"# layer={args.layer} (idx={layer_idx})  backbone={args.backbone}"
          f"  task={args.task}")
    print(f"# base_ckpt={os.path.basename(args.base_ckpt)}")
    print(f"# residual: K={args.K}, emb={args.embedding_dim}, "
          f"transform={args.res_transform}, loss={args.res_loss}"
          f"{'  shareR' if args.share_base_transform else ''}"
          f"{'  freezeR' if args.freeze_res_transform else ''}")
    print(f"# epochs={args.epochs}, lr={args.lr}, "
          f"lmbda_rate={args.lmbda_rate}, ecvq_lmbda={args.ecvq_lmbda}, "
          f"lmbda_kd={args.lmbda_kd}, beta_mse={args.beta_mse}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    # ---- Features + labels ----
    if args.train_feat_root:
        train_dir = Path(args.train_feat_root) / args.layer
    else:
        train_dir = Path(args.feat_root) / args.train_subset / args.backbone / args.layer
    train_files = sorted(train_dir.glob("*.npy"))
    if not train_files:
        raise FileNotFoundError(f"No train features in {train_dir}")

    need_cls_eval = (args.task == 'cls')
    use_path_mode = bool(args.path_mode)
    features_test, basenames_test, gt_test = [], [], {}
    if need_cls_eval:
        test_dir = Path(args.feat_root) / args.test_subset / args.backbone / args.layer
        test_files = sorted(test_dir.glob("*.npy"))
        if not test_files:
            raise FileNotFoundError(f"No test features in {test_dir}")
        print(f"\nData: train={len(train_files)}, test={len(test_files)}")
        features_test, basenames_test = preload_features(test_files, num_workers=4)
        gt_test = load_gt(args.gt_path)
    else:
        print(f"\nData: train={len(train_files)} in {train_dir}  "
              f"(path-mode={use_path_mode})")

    rng = np.random.RandomState(args.seed)

    if use_path_mode:
        n_total = len(train_files)
        n_val = min(args.n_val, max(1, n_total // 5))
        perm = rng.permutation(n_total)
        val_idx, pool_idx = perm[:n_val], perm[n_val:]
        if args.max_train_images > 0 and len(pool_idx) > args.max_train_images:
            pool_idx = pool_idx[:args.max_train_images]
        train_paths = [train_files[i] for i in pool_idx]
        val_paths = [train_files[i] for i in val_idx]
        sample = np.load(train_paths[0])
        while sample.ndim > 2:
            sample = np.squeeze(sample, axis=0)
        T, D = int(sample.shape[0]), int(sample.shape[1])
        del sample
        features_sub = [str(p) for p in train_paths]
        labels_sub = None
        val_features, _ = preload_features(
            val_paths, num_workers=min(8, max(1, len(val_paths))),
            verbose=False)
        val_features = [
            (v.squeeze(0) if getattr(v, 'ndim', 0) == 3 else v).astype(np.float32)
            for v in val_features]
        val_labels = None
        print(f"  path-mode lazy train={len(features_sub)}, "
              f"val={len(val_features)} (preloaded)  D={D}, T={T}")
    else:
        print(f"\nData: train={len(train_files)}"
              + (f", test={len(features_test)}" if need_cls_eval else ""))
        features_train, basenames_train = preload_features(
            train_files, num_workers=4)
        gt_train = load_gt(args.train_gt_path)
        if args.ce_weight > 0:
            keep = [i for i, b in enumerate(basenames_train) if b in gt_train]
            if len(keep) < len(basenames_train):
                print(f"  [warn] {len(basenames_train) - len(keep)} train features "
                      f"have no label, dropped")
            features_train = [features_train[i] for i in keep]
            labels_train_all = np.array(
                [gt_train[basenames_train[i]] for i in keep], dtype=np.int64)
        else:
            labels_train_all = None
            print("  ce_weight=0: skipping ImageNet label filter")

        D = features_train[0].shape[1]
        T = features_train[0].shape[0]
        if labels_train_all is not None:
            print(f"  D={D}, T={T}, labels: "
                  f"{len(set(labels_train_all.tolist()))} classes")
        else:
            print(f"  D={D}, T={T}, labels: none (pure KD)")

        perm = rng.permutation(len(features_train))
        n_val = min(args.n_val, len(features_train) // 5)
        val_idx, pool_idx = perm[:n_val], perm[n_val:]
        if args.max_train_images > 0 and len(pool_idx) > args.max_train_images:
            pool_idx = pool_idx[:args.max_train_images]
        features_sub = [features_train[i] for i in pool_idx]
        labels_sub = (None if labels_train_all is None
                      else labels_train_all[pool_idx])
        val_features = [features_train[i] for i in val_idx]
        val_labels = (None if labels_train_all is None
                      else labels_train_all[val_idx])
        print(f"  train={len(features_sub)}, val={len(val_features)} "
              f"(held out from train pool)")

    token_hw = None
    if args.token_hw:
        parts = [int(x) for x in args.token_hw.replace('x', ',').split(',')]
        if len(parts) != 2:
            raise ValueError(f"--token_hw expects H,W got {args.token_hw!r}")
        token_hw = tuple(parts)
    elif args.task in ('depth', 'seg'):
        token_hw = infer_token_hw(T, n_prefix=1)
        print(f"  inferred token_hw={token_hw} from T={T}")

    # ---- Backbone + differentiable task head ----
    print(f"\nLoading DINOv2 ({args.backbone})...")
    wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone,
                            device=device)
    tail_blocks = list(wrapper.backbone.blocks[layer_idx + 1:])
    print(f"  Tail: {len(tail_blocks)} blocks "
          f"(blk{layer_idx + 1}..blk{layer_idx + len(tail_blocks)}) + norm")

    if args.task == 'seg':
        wrapper.load_segmentation_head()
        task_tail = SegTaskTail(tail_blocks, wrapper.backbone.norm,
                                wrapper.seg_head, device=device,
                                token_hw=token_hw)
        print(f"  Task head: VOC2012 segmentation (BN + 1x1 conv, 21 classes)"
              f"  token_hw={token_hw}")
    elif args.task == 'depth':
        wrapper.load_depth_head(args.depth_head_path or None)
        task_tail = DepthTaskTail(tail_blocks, wrapper.depth_head,
                                  device=device, token_hw=token_hw)
        print(f"  Task head: NYU depth (BNHead bin logits, no final norm)"
              f"  token_hw={token_hw}")
    else:
        task_tail = ClsTaskTail(tail_blocks, wrapper.backbone.norm,
                                wrapper.head, device=device)
        print(f"  Task head: ImageNet linear classifier (1000 classes)")
    mse_tail = FrozenTail(tail_blocks, wrapper.backbone.norm, device=device)

    results = {'config': vars(args)}

    # ================================================================
    #   (A) Base codec alone  (stage-1 bitstream)
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  [Base] frozen codec from stage 1 (tail-MSE optimised)")
    print(f"{'=' * 60}")

    base_codec = load_codec(args.base_ckpt, device=device)
    for p in base_codec.parameters():
        p.requires_grad_(False)
    base_codec.eval()
    G_b, K_b, d_b = base_codec.pq.G, base_codec.pq.K, base_codec.pq.d
    print(f"  base: G={G_b}, K={K_b}, d={d_b}, "
          f"transform={type(base_codec.transform).__name__}, "
          f"fixed rate={G_b * math.log2(K_b):.0f} bits/token")

    # ---- residual codec ----
    Dp = args.res_bottleneck_dim if args.res_bottleneck_dim > 0 else D
    G_r = Dp // args.embedding_dim
    res_codec = build_residual_codec(
        D, G_r, args.K, args.embedding_dim,
        ecvq_lmbda=args.ecvq_lmbda, prior_floor=args.prior_floor,
        transform_kind=args.res_transform,
        bottleneck_dim=args.res_bottleneck_dim,
    ).to(device)
    print(f"  residual: G={G_r}, K={args.K}, d={args.embedding_dim}, "
          f"fixed rate={G_r * math.log2(args.K):.0f} bits/token")

    if args.share_base_transform:
        if args.res_transform != "orthogonal":
            raise ValueError(
                "--share_base_transform requires --res_transform orthogonal")
        share_base_transform(res_codec, base_codec)
        args.warm_start_opq = False
        args.freeze_res_transform = True

    if args.eval_only and args.res_ckpt:
        print(f"\n  [Eval-Only] loading residual codec: {args.res_ckpt}")
        codec = load_residual_codec(args.res_ckpt, device=device,
                                    base_ckpt_path=args.base_ckpt)
        history, train_time = [], 0.0
    else:
        # kmeans_max_samples counts sub-vectors across all G groups (orfc
        # convention), so the number of D-dim vectors is capped at /G.
        max_vectors = max(1, args.kmeans_max_samples // G_r)
        print(f"\n  Collecting residuals R = Y - Ŷ_base for initialisation "
              f"(≤{max_vectors:,} vectors)...")
        residual_vectors = collect_residuals(
            features_sub, base_codec, args.norm_mode, device,
            max_vectors=max_vectors, seed=args.seed,
        )
        r_energy = float((residual_vectors ** 2).mean())
        print(f"  residual vectors: {residual_vectors.shape}, "
              f"mean square = {r_energy:.6f}")
        results['residual_mean_square'] = r_energy

        init_residual_codec(
            res_codec, residual_vectors, G_r, args.K, args.embedding_dim,
            device, warm_start_opq=args.warm_start_opq, seed=args.seed,
        )
        del residual_vectors
        torch.cuda.empty_cache()

        print(f"\n{'=' * 60}")
        print(f"  [Residual] training under '{args.res_loss}' loss")
        print(f"{'=' * 60}")
        t0 = time.time()
        codec, history = train_residual_pq(
            features_train=features_sub,
            labels_train=labels_sub,
            base_codec=base_codec,
            res_codec=res_codec,
            task_tail=task_tail,
            mse_tail=mse_tail,
            res_loss=args.res_loss,
            norm_mode=args.norm_mode,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            val_features=val_features,
            val_labels=val_labels,
            lmbda_rate=args.lmbda_rate,
            ce_weight=args.ce_weight,
            lmbda_kd=args.lmbda_kd,
            kd_temperature=args.kd_temperature,
            beta_mse=args.beta_mse,
            grad_clip=args.grad_clip,
            freeze_res_transform=args.freeze_res_transform,
            freeze_res_codebooks=args.freeze_res_codebooks,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
            tau_schedule=args.tau_schedule,
        )
        train_time = time.time() - t0
        print(f"  Residual training: {train_time:.1f}s")

        ckpt_dir = os.path.join(ORFCV2_ROOT, 'checkpoints', args.backbone)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{experiment_tag(args)}.pt")
        save_residual_codec(codec, ckpt_path, args.base_ckpt,
                            extra={'layer': args.layer})
        print(f"  Residual codec saved: {ckpt_path}")
        results['res_ckpt'] = ckpt_path

    codec.eval()

    # ================================================================
    #   (B) Evaluation
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  [Eval] {'NYU RMSE' if args.task == 'depth' else 'classification'} / rate")
    print(f"{'=' * 60}")

    acc_base = acc_full = dl_base = dl_full = None
    nyu_feats = None
    if need_cls_eval:
        xhat_base = residual_encode_decode(
            features_test, codec, args.norm_mode, device, base_only=True)
        acc_base = evaluate_accuracy(xhat_base, basenames_test, gt_test,
                                     wrapper, layer_idx, device)
        del xhat_base
        xhat_full = residual_encode_decode(
            features_test, codec, args.norm_mode, device)
        acc_full = evaluate_accuracy(xhat_full, basenames_test, gt_test,
                                     wrapper, layer_idx, device)
        del xhat_full
        torch.cuda.empty_cache()

        dl_base = evaluate_delta_l_ref(features_test, mse_tail, args.norm_mode,
                                       device, codec=codec.base,
                                       batch_size=args.batch_size)
        dl_full = evaluate_delta_l_ref(features_test, mse_tail, args.norm_mode,
                                       device, codec=codec,
                                       batch_size=args.batch_size)

        print(f"  * base only       Acc={acc_base:.4f}  ΔL_ref={dl_base:.1f}")
        print(f"  * base + residual Acc={acc_full:.4f}  ΔL_ref={dl_full:.1f}")
        print(f"  * Δ(Acc)={acc_full - acc_base:+.4f}  "
              f"Δ(ΔL_ref)={dl_full - dl_base:+.1f}")
        results.update({
            'base_acc': float(acc_base),
            'full_acc': float(acc_full),
            'delta_acc': float(acc_full - acc_base),
            'base_delta_l': float(dl_base),
            'full_delta_l': float(dl_full),
            'delta_delta_l': float(dl_full - dl_base),
        })

    if args.task == 'depth' or args.eval_depth:
        tools_dir = os.path.join(PROJECT_ROOT, "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        from eval_residual_depth import (
            evaluate_nyu_residual, load_feats, prepare_nyu_samples,
        )
        samples, sample_meta = prepare_nyu_samples(
            args.nyu_data_root, args.nyu_split_file)
        nyu_feats = load_feats(Path(args.nyu_feat_root) / args.layer, samples)
        print(f"  NYU test80: {len(nyu_feats)} feats  shape={nyu_feats[0].shape}")
        if getattr(wrapper, 'depth_head', None) is None:
            wrapper.load_depth_head(args.depth_head_path or None)
        nyu_metrics = evaluate_nyu_residual(
            codec, nyu_feats, layer_idx, sample_meta,
            wrapper.backbone, wrapper.depth_head, device, args.norm_mode)
        results.update(nyu_metrics)

    # ---- rate, per stage, with train-fitted PMFs ----
    print(f"  Computing train PMFs and test rate...")
    rate_bs = min(args.batch_size, 8)
    base_lab_tr, res_lab_tr = collect_stage_labels(
        features_sub, codec, args.norm_mode, device, batch_size=rate_bs)
    pmf_base = _histogram_pmf(base_lab_tr, G_b, K_b)
    pmf_res = _histogram_pmf(res_lab_tr, G_r, args.K)
    if nyu_feats is not None:
        rate_test_feats = nyu_feats
    elif features_test:
        rate_test_feats = features_test
    else:
        # seg-only: no ImageNet/NYU test split; rate on held-out train slides
        rate_test_feats = val_features if val_features else features_sub
    base_lab_te, res_lab_te = collect_stage_labels(
        rate_test_feats, codec, args.norm_mode, device, batch_size=rate_bs)

    rate_base = stage_rate(base_lab_te, K_b, train_pmf=pmf_base)
    rate_res = stage_rate(res_lab_te, args.K, train_pmf=pmf_res)
    results['rate_base'] = rate_base
    results['rate_residual'] = rate_res
    if 'rans_bpt' in rate_base and 'rans_bpt' in rate_res:
        total = rate_base['rans_bpt'] + rate_res['rans_bpt']
        results['rate_total_rans_bpt'] = float(total)
        results['rate_total_bpfp'] = float(total / D)
    print(f"  * {fmt_rate('base    ', rate_base, D)}")
    print(f"  * {fmt_rate('residual', rate_res, D)}")
    if 'rate_total_rans_bpt' in results:
        print(f"  * total: rANS={results['rate_total_rans_bpt']:.2f} b/t  "
              f"BPFP={results['rate_total_bpfp']:.4f}")

    # ================================================================
    #   (C) VOC2012 segmentation mIoU (optional)
    # ================================================================
    if args.eval_seg:
        seg_feat_dir = Path(args.seg_feat_root) / args.backbone / args.layer
        print(f"\n{'=' * 60}")
        print(f"  [Segmentation] VOC2012 mIoU  ({seg_feat_dir})")
        print(f"{'=' * 60}")

        wrapper.backbone.cpu()
        if wrapper.head is not None:
            wrapper.head.cpu()
        mse_tail.to('cpu')
        torch.cuda.empty_cache()

        miou = {}
        for name, base_only in (('base', True), ('full', False)):
            ev = ResidualSegEvaluator(
                codec=codec, norm_mode=args.norm_mode, layer_idx=layer_idx,
                voc_root=args.voc_root, weights_root=wrapper.weights_root,
                device=device, feat_dim=D, model_name=args.backbone,
                base_only=base_only)
            out = ev.evaluate(seg_feat_dir=str(seg_feat_dir),
                              image_list=args.seg_image_list, verbose=True)
            miou[name] = float(out['miou'])
            results[f'{name}_miou'] = miou[name]
            results[f'{name}_seg_acc'] = float(out['acc'])
            print(f"  * {'base only' if base_only else 'base + residual'}"
                  f"  mIoU={miou[name]:.4f}  aAcc={out['acc']:.4f}")
            del ev
            torch.cuda.empty_cache()
        results['delta_miou'] = miou['full'] - miou['base']
        print(f"  * Δ(mIoU) = {results['delta_miou']:+.4f}")

        # rate on the VOC domain (PMFs still fitted on the training features)
        seg_feats = load_seg_features(seg_feat_dir, args.seg_image_list)
        if seg_feats:
            b_lab, r_lab = collect_stage_labels(
                seg_feats, codec, args.norm_mode, device, batch_size=1)
            voc_base = stage_rate(b_lab, K_b, train_pmf=pmf_base)
            voc_res = stage_rate(r_lab, args.K, train_pmf=pmf_res)
            results['voc_rate_base'] = voc_base
            results['voc_rate_residual'] = voc_res
            print(f"  * VOC {fmt_rate('base    ', voc_base, D)}")
            print(f"  * VOC {fmt_rate('residual', voc_res, D)}")
            if 'rans_bpt' in voc_base and 'rans_bpt' in voc_res:
                tot = voc_base['rans_bpt'] + voc_res['rans_bpt']
                results['voc_rate_total_rans_bpt'] = float(tot)
                print(f"  * VOC total: rANS={tot:.2f} b/t  "
                      f"BPFP={tot / D:.4f}")
            del seg_feats

    # ================================================================
    #   Summary & save
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  Summary: {args.layer}  base K={K_b}/d={d_b}  "
          f"residual K={args.K}/d={args.embedding_dim}  loss={args.res_loss}")
    if acc_base is not None:
        print(f"  Acc     : base={acc_base:.4f} -> full={acc_full:.4f} "
              f"({acc_full - acc_base:+.4f})")
        print(f"  ΔL_ref  : base={dl_base:.1f} -> full={dl_full:.1f} "
              f"({dl_full - dl_base:+.1f})")
    if 'base_rmse' in results:
        print(f"  NYU RMSE: base={results['base_rmse']:.4f} -> "
              f"full={results['full_rmse']:.4f} "
              f"({results['delta_rmse_vs_base']:+.4f})  "
              f"anchor={results['anchor_rmse']:.4f}")
    if 'base_miou' in results:
        print(f"  mIoU    : base={results['base_miou']:.4f} -> "
              f"full={results['full_miou']:.4f} "
              f"({results['delta_miou']:+.4f})")
    if 'rate_total_rans_bpt' in results:
        print(f"  Rate    : base={rate_base['rans_bpt']:.2f} + "
              f"residual={rate_res['rans_bpt']:.2f} = "
              f"{results['rate_total_rans_bpt']:.2f} bits/token")
    print(f"  Train time: {train_time:.1f}s")
    print(f"{'=' * 60}")

    results['history'] = history
    results['train_time'] = float(train_time)

    out_dir = os.path.join(ORFCV2_ROOT, 'results', args.backbone)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{experiment_tag(args)}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")
    return results


def experiment_tag(args):
    base_tag = os.path.splitext(os.path.basename(args.base_ckpt))[0]
    parts = [
        f"base-{base_tag}",
        f"task-{args.task}",
        f"res_{args.res_loss}",
        f"K{args.K}", f"emb{args.embedding_dim}",
        args.res_transform,
    ]
    if args.res_bottleneck_dim > 0:
        parts.append(f"bt{args.res_bottleneck_dim}")
    if args.share_base_transform:
        parts.append("shareR")
    if getattr(args, "train_feat_root", ""):
        parts.append(Path(args.train_feat_root).name)
    if not args.warm_start_opq:
        parts.append("km")
    if args.freeze_res_transform:
        parts.append("fzR")
    if args.freeze_res_codebooks:
        parts.append("fzC")
    if args.lmbda_rate > 0:
        parts.append(f"rate{args.lmbda_rate}")
    if args.ce_weight != 1.0:
        parts.append(f"ce{args.ce_weight}")
    if args.lmbda_kd > 0:
        parts.append(f"kd{args.lmbda_kd}T{args.kd_temperature}")
    if args.beta_mse > 0:
        parts.append(f"bmse{args.beta_mse}")
    if args.ecvq_lmbda > 0:
        parts.append(f"ecvq{args.ecvq_lmbda}")
    parts += [f"tau{args.tau_start}", f"lr{args.lr}", f"ep{args.epochs}",
              f"n{args.max_train_images}", f"s{args.seed}"]
    tag = "_".join(str(p) for p in parts)
    if args.result_suffix:
        safe = ''.join(c if (c.isalnum() or c in '-_') else '_'
                       for c in args.result_suffix)
        tag = f"{tag}_{safe}"
    return tag


def main():
    p = argparse.ArgumentParser(
        description="Frozen base codebook + task residual codebook",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--base_ckpt", type=str, required=True,
                   help="Stage-1 codec checkpoint (orfc/checkpoints/...)")
    p.add_argument("--layer", type=str, default="blk20")
    p.add_argument("--backbone", type=str, default="dinov2_vitl14")
    p.add_argument("--norm_mode", type=str, default="per_image")
    p.add_argument("--task", type=str, default="cls",
                   choices=["cls", "seg", "depth"],
                   help="Which frozen head drives the task loss: ImageNet "
                        "classifier / VOC2012 segmentation / NYU depth head")
    p.add_argument("--depth_head_path", type=str, default="",
                   help="Override path to the NYU depth head checkpoint "
                        "(default: registry entry)")

    # residual codec
    p.add_argument("--K", type=int, default=16, help="Residual codebook size")
    p.add_argument("--embedding_dim", type=int, default=32,
                   help="Residual PQ sub-vector dim")
    p.add_argument("--res_transform", type=str, default="orthogonal",
                   choices=["orthogonal", "lowrank", "none"],
                   help="Residual transform: own SO(D) rotation / low-rank "
                        "projection onto a task subspace / none")
    p.add_argument("--res_bottleneck_dim", type=int, default=0,
                   help="Output dim for --res_transform lowrank")
    p.add_argument("--warm_start_opq", action="store_true", default=True,
                   help="Warm-start residual transform+codebooks from OPQ on "
                        "the residuals")
    p.add_argument("--no_warm_start_opq", dest="warm_start_opq",
                   action="store_false")
    p.add_argument("--share_base_transform", action="store_true",
                   help="Copy the frozen ORFC rotation into the residual "
                        "codec (R2 := R1), k-means the residual codebooks in "
                        "that frame, and freeze R2. Implies --no_warm_start_opq "
                        "and --freeze_res_transform")
    p.add_argument("--freeze_res_transform", action="store_true",
                   help="Keep the residual rotation fixed and train codebooks "
                        "only. Implied by --share_base_transform")
    p.add_argument("--freeze_res_codebooks", action="store_true",
                   help="Ablation: train the residual rotation only")

    # loss
    p.add_argument("--res_loss", type=str, default="task",
                   choices=["task", "mse_tail", "mse"],
                   help="task=classification CE; mse_tail/mse are controls")
    p.add_argument("--lmbda_rate", type=float, default=0.0,
                   help="Weight on residual rate in J = D + lmbda_rate*R_bits "
                        "(0 = fixed-length residual, no rate term)")
    p.add_argument("--ce_weight", type=float, default=1.0,
                   help="Weight on the label cross-entropy; set 0 together "
                        "with --lmbda_kd > 0 for pure logit distillation "
                        "(needs no labels)")
    p.add_argument("--lmbda_kd", type=float, default=0.0,
                   help="Weight on KD against the unquantised teacher logits "
                        "(0 = pure CE)")
    p.add_argument("--kd_temperature", type=float, default=2.0)
    p.add_argument("--beta_mse", type=float, default=0.0,
                   help="Weight on ||Y-Ŷ||² added to the task loss as a "
                        "fidelity regulariser")
    p.add_argument("--ecvq_lmbda", type=float, default=0.0,
                   help="ECVQ assignment trade-off inside SoftPQ "
                        "(cost = d² + (-log2 p)/ecvq_lmbda); >0 also enables "
                        "the learned per-group prior")
    p.add_argument("--prior_floor", type=float, default=0.0)

    # optimisation
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--tau_start", type=float, default=0.5)
    p.add_argument("--tau_end", type=float, default=0.005)
    p.add_argument("--tau_schedule", type=str, default="exponential",
                   choices=["exponential", "linear"])

    # data
    p.add_argument("--feat_root", type=str,
                   default=os.path.join(PROJECT_ROOT, "features"))
    p.add_argument("--train_feat_root", type=str, default="",
                   help="Override train feature directory parent "
                        "(e.g. features/train/dinov2_vitl14_ade). "
                        "When set, train_dir = train_feat_root / layer, "
                        "labels are not required, and features are lazy-loaded.")
    p.add_argument("--path_mode", action="store_true",
                   help="Lazy-load train .npy paths instead of stacking in RAM")
    p.add_argument("--token_hw", type=str, default="",
                   help="Patch grid H,W (e.g. 37,49 for ADE). "
                        "Inferred from token count when omitted.")
    p.add_argument("--train_subset", type=str, default="train")
    p.add_argument("--test_subset", type=str, default="test")
    p.add_argument("--max_train_images", type=int, default=5000)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--kmeans_max_samples", type=int, default=2_000_000)
    p.add_argument("--train_gt_path", type=str,
                   default=os.path.join(PROJECT_ROOT, "utils",
                                        "imagenet_selected_label5000.txt"))
    p.add_argument("--gt_path", type=str,
                   default=os.path.join(PROJECT_ROOT, "utils",
                                        "imagenet_selected_label500.txt"))
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--eval_depth", action="store_true",
                   help="Evaluate NYU test80 RMSE (implied by --task depth)")
    p.add_argument("--nyu_feat_root", type=str,
                   default=os.path.join(PROJECT_ROOT, "features",
                                        "nyu_depth_80", "dinov2_vitl14"))
    p.add_argument("--nyu_data_root", type=str,
                   default="/data4/workspace/zlt/featcodec/CoFAI/data/NYU")
    p.add_argument("--nyu_split_file", type=str,
                   default=os.path.join(PROJECT_ROOT, "utils",
                                        "nyu_test_80.txt"))

    p.add_argument("--eval_seg", action="store_true",
                   help="Evaluate VOC2012 mIoU (base only vs base+residual)")
    p.add_argument("--seg_feat_root", type=str,
                   default=os.path.join(PROJECT_ROOT, "features", "voc2012_100"))
    p.add_argument("--voc_root", type=str,
                   default=os.path.join(PROJECT_ROOT, "data", "VOCdevkit",
                                        "VOC2012"))
    p.add_argument("--seg_image_list", type=str,
                   default=os.path.join(PROJECT_ROOT, "utils",
                                        "voc2012_val_100.txt"))

    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--res_ckpt", type=str, default="",
                   help="Residual checkpoint for --eval_only")
    p.add_argument("--result_suffix", type=str, default="")

    args = p.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
