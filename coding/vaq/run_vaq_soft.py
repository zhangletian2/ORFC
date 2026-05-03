#!/usr/bin/env python
"""
Run VAQ-initialised Soft-PQ on ORFC intermediate ViT features.

Pipeline:
  Stage 1: VAQ.fit() with non-uniform bit allocation; percent_var is forced
           to 1.0 so every subspace gets a real codebook (K_g = 2^bits[g] >= 2).
  Stage 2: Wrap into VAQSoftFeatureCodec and finetune (codebooks +/- rotation)
           by minimising reconstruction MSE in the normalised feature space.

Reports both stage-1 (VAQ argmin baseline) and stage-2 (finetuned) metrics:
  - classification accuracy on test split
  - ΔL_ref (||F(H) - F(Ĥ)||²)
  - rate: max, xent (train PMF), empirical entropy, rANS

Defaults are project-wide:
  --max_bits 10 (K_max=1024)   --batch_size 32   --lr 1e-4   --tau_start 0

Example:
    CUDA_VISIBLE_DEVICES=0 python run_vaq_soft.py \
        --backbone dinov2_vitl14 --layer blk20 \
        --bit_budget 256 --num_subspaces 32 --min_bits 1 \
        --bit_alloc_objective linear \
        --epochs 30 --max_train_images 5000
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

VAQ_ROOT = os.path.dirname(os.path.abspath(__file__))
ORFC_CODEC_ROOT = os.path.normpath(os.path.join(VAQ_ROOT, "..", "orfc"))
PROJECT_ROOT = os.path.normpath(os.path.join(VAQ_ROOT, "..", ".."))
if ORFC_CODEC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_CODEC_ROOT)
if VAQ_ROOT not in sys.path:
    sys.path.insert(0, VAQ_ROOT)

from backbone.wrapper import Dinov2Wrapper, SegmentationEvaluator  # noqa: E402
from opq import batch_normalize_gpu, batch_inv_normalize_gpu  # noqa: E402
from soft_pq import FrozenTail  # noqa: E402

from feature_vaq import (  # noqa: E402
    collect_vaq_labels,
    evaluate_delta_l_ref_vaq,
    evaluate_vaq_rate,
    histogram_pmfs,
    vaq_encode_decode_features,
)
from vaq_soft import (  # noqa: E402
    build_codec_from_vaq,
    collect_labels,
    encode_decode_features,
    evaluate_delta_l_ref,
    evaluate_rate,
    fit_vaq_then_finetune,
    load_codec,
    load_codec_with_meta,
    save_codec,
)


# ----------------------------------------------------------------
#         VAQSoftFeatureCodec -> SegmentationEvaluator adapter
# ----------------------------------------------------------------
#
# Mirrors orfc/run_soft_pq.py::CodecSegmentationEvaluator: only overrides the
# constructor and ``quantize_tokens`` of the wrapper.SegmentationEvaluator so
# the slide-inference / mIoU pipeline is reused unchanged.
# Works with any codec exposing ``codec(Y) -> (Y_hat, _)`` over [B, T, D]
# normalised features, which our VAQSoftFeatureCodec does.
# ----------------------------------------------------------------
class VAQSoftSegmentationEvaluator(SegmentationEvaluator):
    """Slide-inference mIoU using a VAQ(-Soft) codec as quantiser."""

    def __init__(self, codec, norm_mode, layer_idx,
                 voc_root, weights_root, device='cuda', feat_dim=1024,
                 model_name='dinov2_vitl14'):
        self.codec = codec
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
        Y_hat, _ = self.codec(Y)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        return X_hat.squeeze(0)


def _collect_seg_features_flat(seg_feat_dir, image_list_path):
    """Read all VOC slide features into a flat list of [1+N, D] arrays."""
    with open(image_list_path) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    flat = []
    for sname in names:
        fp = os.path.join(seg_feat_dir, f"{sname}.npy")
        if not os.path.exists(fp):
            continue
        darr = np.load(fp)
        for si in range(darr.shape[0]):
            flat.append(darr[si])
    return flat


# ----------------------------------------------------------------
#                Local helpers (no mmcv dependency)
# ----------------------------------------------------------------

def set_seed(seed: int) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_single_feature(path: Path):
    return np.load(path).astype(np.float32), path.stem


def preload_features(feat_files, num_workers: int = 8):
    if num_workers <= 1:
        feats, names = [], []
        for p in feat_files:
            f, n = _load_single_feature(p)
            feats.append(f)
            names.append(n)
        return feats, names
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        loaded = list(pool.map(_load_single_feature, feat_files))
    return [x[0] for x in loaded], [x[1] for x in loaded]


def load_gt(gt_path: str):
    gt = {}
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                gt[parts[0]] = int(parts[1])
    return gt


def evaluate_accuracy(xhat_list, basenames, gt_dict, dino_wrapper, layer_idx, device):
    correct = 0
    total = 0
    for x_hat, basename in zip(xhat_list, basenames):
        if basename not in gt_dict:
            continue
        label = gt_dict[basename]
        feat_tensor = torch.from_numpy(x_hat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = dino_wrapper.forward_from_tokens(feat_tensor, layer_idx)
            pred = torch.argmax(logits, dim=1).item()
        correct += int(pred == label)
        total += 1
    return correct / total if total > 0 else 0.0


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _safe_suffix(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in text)


def _select_train_subset(features, max_train_images: int, seed: int):
    if max_train_images > 0 and len(features) > max_train_images:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(features), max_train_images, replace=False)
        return [features[i] for i in idx]
    return features


# ----------------------------------------------------------------
#                Main experiment
# ----------------------------------------------------------------

def _peek_ckpt_meta(ckpt_path: str) -> Dict:
    """Lightweight meta-only read (we still load full state below; this is just
    to recover training-context fields *before* we know layer/backbone)."""
    return torch.load(ckpt_path, map_location="cpu")


def _restore_args_from_ckpt(args, ckpt_meta: Dict, cli_keys: set) -> None:
    """For eval-only mode: fill in args fields from ckpt meta when the user
    did NOT pass them explicitly on the CLI. Prints a one-line summary of
    every restored field so the run remains traceable.
    """
    train_ctx = ckpt_meta.get("train_context", {})
    saved_args = ckpt_meta.get("args", {})
    if not train_ctx and not saved_args:
        return

    restored = []

    # (1) Pipeline-critical: must match ckpt or downstream modules break.
    for key in (
        "layer", "backbone", "norm_mode",
        "bit_budget", "num_subspaces", "min_bits", "max_bits",
        "bit_alloc_objective", "embedding_dim", "K",
        "lmbda", "prior_floor", "use_lref",
    ):
        if key in cli_keys or key not in train_ctx:
            continue
        old = getattr(args, key, None)
        new = train_ctx[key]
        if old != new:
            setattr(args, key, new)
            restored.append(f"{key}={new}")

    # (2) Cosmetic-only: training hyperparams used in result-file naming.
    # Without these, eval-only result files would be tagged with default
    # training params instead of the ckpt's. We never override CLI.
    for key in (
        "epochs", "lr", "tau_start", "tau_end", "tau_schedule",
        "freeze_transform", "freeze_codebooks",
        "max_train_images", "seed",
    ):
        if key in cli_keys or key not in saved_args:
            continue
        old = getattr(args, key, None)
        new = saved_args[key]
        if old != new:
            setattr(args, key, new)
            restored.append(f"{key}={new}")

    if restored:
        print(f"  [ckpt] restored from meta ({len(restored)} fields): "
              f"{', '.join(restored)}")


def run_experiment(args, cli_keys: Optional[set] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    if args.eval_only and args.ckpt_path:
        _ckpt_meta_preview = _peek_ckpt_meta(args.ckpt_path)
        _restore_args_from_ckpt(args, _ckpt_meta_preview, cli_keys or set())
        del _ckpt_meta_preview

    layer_idx = int(args.layer[-2:])
    print(f"\n{'#' * 70}")
    print(f"# VAQ-Soft Feature Codec  [VAQ-init + MSE finetune]")
    print(f"# backbone={args.backbone}, layer={args.layer} (idx={layer_idx})")
    print(f"# bit_budget={args.bit_budget}, num_subspaces={args.num_subspaces}, "
          f"min_bits={args.min_bits}, max_bits={args.max_bits} (K_max={1 << args.max_bits})")
    print(f"# percent_var=1.0 (enforced), objective={args.bit_alloc_objective}")
    print(f"# epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}, "
          f"tau={args.tau_start}->{args.tau_end} ({args.tau_schedule})")
    print(f"# freeze_transform={args.freeze_transform}, "
          f"freeze_codebooks={args.freeze_codebooks}")
    if args.lmbda > 0:
        print(f"# RD: lambda={args.lmbda}, prior_floor={args.prior_floor}, "
              f"prior_init={'VAQ usage' if not args.no_prior_init else 'uniform'}")
    if args.use_lref:
        print(f"# Distortion: ΔL_ref distillation through frozen ViT tail")
    else:
        print(f"# Distortion: MSE in normalised feature space")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    # -- Load features --
    train_dir = Path(args.feat_root) / args.train_subset / args.backbone / args.layer
    test_dir = Path(args.feat_root) / args.test_subset / args.backbone / args.layer
    train_files = sorted(train_dir.glob("*.npy"))
    test_files = sorted(test_dir.glob("*.npy"))
    if not train_files:
        raise FileNotFoundError(f"No train features found in {train_dir}")
    if not test_files:
        raise FileNotFoundError(f"No test features found in {test_dir}")

    print(f"\nData: train={len(train_files)}, test={len(test_files)}")
    print(f"  train_dir={train_dir}\n  test_dir={test_dir}")
    features_train, _ = preload_features(train_files, num_workers=args.num_workers)
    features_test, basenames_test = preload_features(test_files, num_workers=args.num_workers)
    gt_test = load_gt(args.gt_path)

    D = features_train[0].shape[1]
    T = features_train[0].shape[0]

    if args.num_subspaces > 0:
        num_subspaces = args.num_subspaces
        if args.bit_budget > 0:
            bit_budget = args.bit_budget
        else:
            if not _is_power_of_two(args.K):
                raise ValueError("--K must be a power of two when --bit_budget is omitted")
            bit_budget = int(num_subspaces * math.log2(args.K))
    else:
        if D % args.embedding_dim != 0:
            raise ValueError(f"D={D} not divisible by embedding_dim={args.embedding_dim}")
        num_subspaces = D // args.embedding_dim
        if args.bit_budget > 0:
            bit_budget = args.bit_budget
        else:
            if not _is_power_of_two(args.K):
                raise ValueError("--K must be a power of two when --bit_budget is omitted")
            bit_budget = int(num_subspaces * math.log2(args.K))

    delta_batch_size = args.delta_batch_size if args.delta_batch_size > 0 else args.batch_size
    print(f"  D={D}, T={T}, subspaces={num_subspaces}, "
          f"bit_budget={bit_budget} bits/token  (BPFP={bit_budget / D:.6f})")

    features_train_sub = _select_train_subset(features_train, args.max_train_images, args.seed)
    if len(features_train_sub) != len(features_train):
        print(f"  Training subset: {len(features_train_sub)} images")

    n_val = min(args.n_val, len(features_train))
    val_features = None
    if n_val > 0:
        rng_val = np.random.RandomState(args.seed + 1)
        val_idx = rng_val.choice(len(features_train), n_val, replace=False)
        val_features = [features_train[i] for i in val_idx]

    # -- (optional) frozen ViT tail for ΔL_ref training --
    train_tail = None
    train_wrapper = None
    if args.use_lref and not args.eval_only:
        print(f"\n[ΔL_ref] Loading wrapper for tail (layer_idx={layer_idx})...")
        train_wrapper = Dinov2Wrapper(
            head_layers=1, model_name=args.backbone, device=device,
        )
        tail_blocks_train = list(train_wrapper.backbone.blocks[layer_idx + 1:])
        train_tail = FrozenTail(
            tail_blocks_train, train_wrapper.backbone.norm, device=device,
        )
        print(f"  tail: {len(tail_blocks_train)} blocks + final norm "
              f"(all frozen, no_grad teacher cache)")

    # -- Stage 1+2: fit VAQ then finetune --
    if args.eval_only:
        if not args.ckpt_path:
            raise ValueError("--eval_only requires --ckpt_path")
        print(f"\nLoading VAQ-Soft codec: {args.ckpt_path}")
        codec, ckpt_meta = load_codec_with_meta(args.ckpt_path, device=device)
        vaq = None
        history = list(ckpt_meta.get("history", []))
        snapshots = {}
        train_time = float(ckpt_meta.get("train_time", 0.0))
        ckpt_path = args.ckpt_path
        print(f"  ckpt format_version={ckpt_meta.get('format_version', 1)}, "
              f"K_max={max(ckpt_meta['K_per_group'])}, "
              f"#groups={len(ckpt_meta['K_per_group'])}")
    else:
        eval_at = None
        if getattr(args, "eval_at_epochs", None):
            eval_at = [int(x) for x in args.eval_at_epochs.split(",") if x.strip()]

        t0 = time.time()
        codec, vaq, history, snapshots = fit_vaq_then_finetune(
            features_train=features_train_sub,
            norm_mode=args.norm_mode,
            bit_budget=bit_budget,
            num_subspaces=num_subspaces,
            min_bits=args.min_bits,
            max_bits=args.max_bits,
            max_fit_vectors=args.max_fit_vectors,
            kmeans_iter=args.kmeans_iter,
            bit_alloc_objective=args.bit_alloc_objective,
            kmeans_hier_threshold=args.kmeans_hier_threshold,
            kmeans_hier_branching=args.kmeans_hier_branching,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
            tau_schedule=args.tau_schedule,
            grad_clip=args.grad_clip,
            freeze_transform=args.freeze_transform,
            freeze_codebooks=args.freeze_codebooks,
            device=device,
            seed=args.seed,
            val_features=val_features,
            verbose=True,
            lmbda=args.lmbda,
            prior_floor=args.prior_floor,
            init_prior_from_vaq_usage=not args.no_prior_init,
            tail=train_tail,
            use_lref=args.use_lref,
            snapshot_epochs=eval_at,
        )
        train_time = time.time() - t0

        # Release training-time tail/wrapper before downstream eval to avoid
        # holding two backbones in GPU memory simultaneously.
        if train_tail is not None:
            train_tail.to("cpu")
            del train_tail
            train_tail = None
        if train_wrapper is not None:
            del train_wrapper
            train_wrapper = None
        torch.cuda.empty_cache()

        ckpt_dir = os.path.join(VAQ_ROOT, "checkpoints_soft", args.backbone)
        os.makedirs(ckpt_dir, exist_ok=True)
        lmbda_tag = f"_lmbda{args.lmbda}" if args.lmbda > 0 else ""
        pf_tag = f"_pf{args.prior_floor}" if args.prior_floor > 0 else ""
        lref_tag = "_lref" if args.use_lref else ""
        tag = (
            f"{args.layer}_vaqsoft_B{bit_budget}_m{num_subspaces}"
            f"_min{args.min_bits}_max{args.max_bits}"
            f"_obj{args.bit_alloc_objective}"
            f"{lref_tag}{lmbda_tag}{pf_tag}"
            f"_ep{args.epochs}_lr{args.lr}_tau{args.tau_start}-{args.tau_end}"
            f"_fzR{int(args.freeze_transform)}_fzC{int(args.freeze_codebooks)}"
            f"_n{args.max_train_images}_s{args.seed}"
        )
        if args.result_suffix:
            tag += f"_{_safe_suffix(args.result_suffix)}"
        ckpt_path = os.path.join(ckpt_dir, f"{tag}.pt")
        if args.no_save_ckpt:
            print(f"  Codec NOT saved (--no_save_ckpt).  Would have been: {ckpt_path}")
            ckpt_path = ""
        else:
            extra_meta = {
                "train_context": {
                    "layer": args.layer,
                    "backbone": args.backbone,
                    "norm_mode": args.norm_mode,
                    "bit_budget": int(bit_budget),
                    "num_subspaces": int(num_subspaces),
                    "min_bits": int(args.min_bits),
                    "max_bits": int(args.max_bits),
                    "bit_alloc_objective": args.bit_alloc_objective,
                    "embedding_dim": int(args.embedding_dim),
                    "K": int(args.K),
                    "lmbda": float(args.lmbda),
                    "prior_floor": float(args.prior_floor),
                    "use_lref": bool(args.use_lref),
                    "feat_dim": int(D),
                    "num_tokens": int(T),
                },
                "args": vars(args),
                "bits_alloc": list(map(int, vaq.bits_alloc)),
                "k_per_group": list(map(int, vaq.k_per_group)),
                "train_time": float(train_time),
                "history": history,
            }
            save_codec(codec, ckpt_path, extra_meta=extra_meta)
            print(f"  Codec saved: {ckpt_path}")

    # -- Backbone for downstream eval --
    print("\nLoading DINOv2 wrapper...")
    wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone, device=device)

    # ============================================================
    # (A) Stage-1 VAQ argmin baseline (only if we just fitted)
    # ============================================================
    vaq_metrics: Dict = {}
    if vaq is not None and not args.skip_vaq_baseline:
        print(f"\n{'=' * 60}")
        print("  [Stage 1] VAQ argmin baseline")
        print(f"{'=' * 60}")
        xhat_vaq = vaq_encode_decode_features(
            features_test,
            vaq,
            norm_mode=args.norm_mode,
            device=device,
            batch_size=args.batch_size,
        )
        acc_vaq = evaluate_accuracy(
            xhat_vaq, basenames_test, gt_test, wrapper, layer_idx, device
        )
        print(f"  * VAQ Acc = {acc_vaq:.4f}")
        del xhat_vaq

        if not args.skip_delta_l_ref:
            tail_blocks = list(wrapper.backbone.blocks[layer_idx + 1:])
            tail = FrozenTail(tail_blocks, wrapper.backbone.norm, device=device)
            d_l_vaq = evaluate_delta_l_ref_vaq(
                features_test, tail, vaq, args.norm_mode, device,
                batch_size=delta_batch_size,
            )
            tail.to("cpu")
            # tail.blocks alias backbone.blocks[layer_idx+1:]; restore to keep
            # the wrapper usable for the next evaluate_accuracy call.
            wrapper.backbone.to(device)
            torch.cuda.empty_cache()
            print(f"  * VAQ Delta L_ref = {d_l_vaq:.1f}")
        else:
            d_l_vaq = None

        train_labels = collect_vaq_labels(
            features_train_sub, vaq, args.norm_mode, device,
            batch_size=args.batch_size,
        )
        train_pmfs = histogram_pmfs(train_labels, vaq.k_per_group, smoothing=1.0)
        rate_vaq = evaluate_vaq_rate(
            features_test, vaq, args.norm_mode, device,
            batch_size=args.batch_size, train_pmfs=train_pmfs,
        )
        print(
            f"  * VAQ Rate: xent={rate_vaq.get('xent_rate_bpt', float('nan')):.2f}  "
            f"H_emp={rate_vaq.get('empirical_entropy_bpt', float('nan')):.2f}  "
            f"max={rate_vaq.get('max_rate_bpt', 0):.0f} bits/token"
        )
        if "rans_bpt" in rate_vaq:
            print(
                f"  * VAQ rANS={rate_vaq['rans_bpt']:.2f}  "
                f"rANS_train={rate_vaq.get('rans_train_bpt', float('nan')):.2f}"
            )
        vaq_metrics = {
            "vaq_acc": float(acc_vaq),
            "vaq_delta_l_ref": None if d_l_vaq is None else float(d_l_vaq),
            "vaq_rate": rate_vaq,
            "bits_alloc": list(map(int, vaq.bits_alloc)),
            "K_per_group": list(map(int, vaq.k_per_group)),
        }

    # ============================================================
    # (B) Stage-2 VAQ-Soft (finetuned) evaluation
    # ============================================================
    print(f"\n{'=' * 60}")
    print("  [Stage 2] VAQ-Soft (finetuned) evaluation")
    print(f"{'=' * 60}")
    xhat_soft = encode_decode_features(
        features_test, codec, args.norm_mode, device,
        batch_size=args.batch_size,
    )
    acc_soft = evaluate_accuracy(
        xhat_soft, basenames_test, gt_test, wrapper, layer_idx, device
    )
    print(f"  * VAQ-Soft Acc = {acc_soft:.4f}")
    del xhat_soft

    d_l_soft = None
    if not args.skip_delta_l_ref:
        tail_blocks2 = list(wrapper.backbone.blocks[layer_idx + 1:])
        tail2 = FrozenTail(tail_blocks2, wrapper.backbone.norm, device=device)
        d_l_soft = evaluate_delta_l_ref(
            features_test, tail2, codec, args.norm_mode, device,
            batch_size=delta_batch_size,
        )
        tail2.to("cpu")
        wrapper.backbone.to(device)
        torch.cuda.empty_cache()
        print(f"  * VAQ-Soft Delta L_ref = {d_l_soft:.1f}")

    # Train PMFs from finetuned codec for rate eval
    if codec.pq.use_rate:
        # Use the *learned* categorical prior as encoding distribution: this
        # is what an actual rANS encoder paired with the rate-aware codec
        # would use (matches soft_pq's eval path).
        train_pmfs_s = codec.pq.get_prior_pmf()
        if not args.eval_only:
            print(f"  [rate eval] using learned log_prior (lambda={codec.pq.lmbda})")
    else:
        train_labels_s = collect_labels(
            features_train_sub, codec, args.norm_mode, device,
            batch_size=args.batch_size,
        )
        train_pmfs_s = histogram_pmfs(train_labels_s, codec.k_per_group, smoothing=1.0)
    rate_soft = evaluate_rate(
        features_test, codec, args.norm_mode, device,
        batch_size=args.batch_size, train_pmfs=train_pmfs_s,
    )
    print(
        f"  * Rate: xent={rate_soft.get('xent_rate_bpt', float('nan')):.2f}  "
        f"H_emp={rate_soft.get('empirical_entropy_bpt', float('nan')):.2f}  "
        f"max={rate_soft.get('max_rate_bpt', 0):.0f} bits/token"
    )
    if "rans_bpt" in rate_soft:
        print(
            f"  * rANS={rate_soft['rans_bpt']:.2f}  "
            f"rANS_train={rate_soft.get('rans_train_bpt', float('nan')):.2f}"
        )

    # ============================================================
    # (C) Segmentation evaluation (optional)
    # ============================================================
    seg_metrics: Dict = {}
    if args.eval_seg:
        seg_feat_dir = os.path.join(args.seg_feat_root, args.backbone, args.layer)
        if not os.path.isdir(seg_feat_dir):
            print(f"\n  [Segmentation] Skipped: {seg_feat_dir} not found")
        else:
            print(f"\n{'=' * 60}")
            print(f"  [Segmentation] VOC2012 mIoU evaluation")
            print(f"  seg features: {seg_feat_dir}")
            print(f"{'=' * 60}")

            # Free wrapper backbone before seg evaluators (which load their
            # own backbone + seg head). Not restored: nothing else uses it.
            wrapper.backbone.cpu()
            if getattr(wrapper, "head", None) is not None:
                wrapper.head.cpu()
            torch.cuda.empty_cache()

            # ---- Stage-1 baseline mIoU (VAQ argmin) ----
            if vaq is not None and not args.skip_vaq_baseline:
                vaq_init_codec = build_codec_from_vaq(vaq, device=device)
                vaq_init_codec.eval()
                vaq_seg_eval = VAQSoftSegmentationEvaluator(
                    codec=vaq_init_codec,
                    norm_mode=args.norm_mode, layer_idx=layer_idx,
                    voc_root=args.voc_root, weights_root=wrapper.weights_root,
                    device=device, feat_dim=D, model_name=args.backbone,
                )
                seg_vaq = vaq_seg_eval.evaluate(
                    seg_feat_dir=seg_feat_dir,
                    image_list=args.seg_image_list, verbose=True,
                )
                seg_metrics["vaq_miou"] = float(seg_vaq["miou"])
                seg_metrics["vaq_aacc"] = float(seg_vaq.get("acc", 0.0))
                print(f"  * VAQ        mIoU = {seg_vaq['miou']:.4f}")
                del vaq_seg_eval, vaq_init_codec
                torch.cuda.empty_cache()

            # ---- Stage-2 mIoU (VAQ-Soft finetuned) ----
            codec.eval()
            soft_seg_eval = VAQSoftSegmentationEvaluator(
                codec=codec,
                norm_mode=args.norm_mode, layer_idx=layer_idx,
                voc_root=args.voc_root, weights_root=wrapper.weights_root,
                device=device, feat_dim=D, model_name=args.backbone,
            )
            seg_soft = soft_seg_eval.evaluate(
                seg_feat_dir=seg_feat_dir,
                image_list=args.seg_image_list, verbose=True,
            )
            seg_metrics["vaq_soft_miou"] = float(seg_soft["miou"])
            seg_metrics["vaq_soft_aacc"] = float(seg_soft.get("acc", 0.0))
            if "vaq_miou" in seg_metrics:
                seg_metrics["delta_miou"] = (
                    seg_metrics["vaq_soft_miou"] - seg_metrics["vaq_miou"]
                )
            print(f"  * VAQ-Soft   mIoU = {seg_soft['miou']:.4f}")
            if "delta_miou" in seg_metrics:
                print(f"  * Δ(mIoU) = {seg_metrics['delta_miou']:+.4f}")
            del soft_seg_eval
            torch.cuda.empty_cache()

            # ---- VOC seg rate (Stage-1 + Stage-2) ----
            print(f"  Computing VOC seg rate...")
            seg_features_flat = _collect_seg_features_flat(
                seg_feat_dir, args.seg_image_list
            )
            if seg_features_flat:
                # Stage-1 (VAQ) seg rate
                if vaq is not None and not args.skip_vaq_baseline:
                    seg_rate_vaq = evaluate_vaq_rate(
                        seg_features_flat, vaq, args.norm_mode, device,
                        batch_size=1, train_pmfs=train_pmfs,
                    )
                    seg_metrics["vaq_seg_rate"] = seg_rate_vaq
                    print(
                        f"  * VAQ      VOC Rate: "
                        f"xent={seg_rate_vaq.get('xent_rate_bpt', float('nan')):.2f}  "
                        f"H_emp={seg_rate_vaq.get('empirical_entropy_bpt', float('nan')):.2f}  "
                        f"max={seg_rate_vaq.get('max_rate_bpt', 0):.0f} bits/token"
                    )
                    if "rans_bpt" in seg_rate_vaq:
                        print(
                            f"  * VAQ      VOC rANS={seg_rate_vaq['rans_bpt']:.2f}  "
                            f"rANS_train={seg_rate_vaq.get('rans_train_bpt', float('nan')):.2f}"
                        )

                # Stage-2 (VAQ-Soft) seg rate
                seg_rate_soft = evaluate_rate(
                    seg_features_flat, codec, args.norm_mode, device,
                    batch_size=1, train_pmfs=train_pmfs_s,
                )
                seg_metrics["vaq_soft_seg_rate"] = seg_rate_soft
                print(
                    f"  * VAQ-Soft VOC Rate: "
                    f"xent={seg_rate_soft.get('xent_rate_bpt', float('nan')):.2f}  "
                    f"H_emp={seg_rate_soft.get('empirical_entropy_bpt', float('nan')):.2f}  "
                    f"max={seg_rate_soft.get('max_rate_bpt', 0):.0f} bits/token"
                )
                if "rans_bpt" in seg_rate_soft:
                    print(
                        f"  * VAQ-Soft VOC rANS={seg_rate_soft['rans_bpt']:.2f}  "
                        f"rANS_train={seg_rate_soft.get('rans_train_bpt', float('nan')):.2f}"
                    )
                del seg_features_flat
            else:
                print(f"  * VOC Rate: no features found")

    # ============================================================
    # (D) Mid-training snapshot evaluations (--eval_at_epochs)
    # ============================================================
    snapshot_results: Dict[int, Dict] = {}
    final_state = None
    if snapshots:
        wrapper.backbone.to(device)
        if getattr(wrapper, "head", None) is not None:
            wrapper.head.to(device)
        torch.cuda.empty_cache()

        final_state = {k: v.clone() for k, v in codec.state_dict().items()}
        for snap_ep in sorted(snapshots.keys()):
            print(f"\n{'=' * 60}")
            print(f"  [Snapshot ep={snap_ep}] downstream evaluation")
            print(f"{'=' * 60}")
            codec.load_state_dict(snapshots[snap_ep])
            codec.eval()

            xhat_snap = encode_decode_features(
                features_test, codec, args.norm_mode, device,
                batch_size=args.batch_size,
            )
            acc_snap = evaluate_accuracy(
                xhat_snap, basenames_test, gt_test, wrapper, layer_idx, device
            )
            print(f"  * Acc@ep{snap_ep} = {acc_snap:.4f}")
            del xhat_snap

            d_l_snap = None
            if not args.skip_delta_l_ref:
                tail_snap_blocks = list(wrapper.backbone.blocks[layer_idx + 1:])
                tail_snap = FrozenTail(
                    tail_snap_blocks, wrapper.backbone.norm, device=device
                )
                d_l_snap = evaluate_delta_l_ref(
                    features_test, tail_snap, codec, args.norm_mode, device,
                    batch_size=delta_batch_size,
                )
                tail_snap.to("cpu")
                wrapper.backbone.to(device)
                torch.cuda.empty_cache()
                print(f"  * ΔL_ref@ep{snap_ep} = {d_l_snap:.1f}")

            snap_info = {
                "epoch": snap_ep,
                "acc": float(acc_snap),
                "delta_l_ref": None if d_l_snap is None else float(d_l_snap),
            }

            if args.eval_seg:
                seg_feat_dir_s = os.path.join(
                    args.seg_feat_root, args.backbone, args.layer
                )
                if os.path.isdir(seg_feat_dir_s):
                    wrapper.backbone.cpu()
                    if getattr(wrapper, "head", None) is not None:
                        wrapper.head.cpu()
                    torch.cuda.empty_cache()
                    snap_seg_eval = VAQSoftSegmentationEvaluator(
                        codec=codec,
                        norm_mode=args.norm_mode, layer_idx=layer_idx,
                        voc_root=args.voc_root,
                        weights_root=wrapper.weights_root,
                        device=device, feat_dim=D,
                        model_name=args.backbone,
                    )
                    seg_snap = snap_seg_eval.evaluate(
                        seg_feat_dir=seg_feat_dir_s,
                        image_list=args.seg_image_list, verbose=True,
                    )
                    snap_info["miou"] = float(seg_snap["miou"])
                    print(f"  * mIoU@ep{snap_ep} = {seg_snap['miou']:.4f}")
                    del snap_seg_eval
                    torch.cuda.empty_cache()
                    wrapper.backbone.to(device)
                    if getattr(wrapper, "head", None) is not None:
                        wrapper.head.to(device)

            snapshot_results[snap_ep] = snap_info

        codec.load_state_dict(final_state)
        codec.eval()
        del final_state

    # ============================================================
    # Summary + save
    # ============================================================
    print(f"\n{'=' * 60}")
    print(f"  Summary  (layer={args.layer}  bit_budget={bit_budget})")
    if vaq_metrics:
        d_acc = acc_soft - vaq_metrics["vaq_acc"]
        print(f"  Stage1 VAQ      Acc={vaq_metrics['vaq_acc']:.4f}")
        print(f"  Stage2 VAQ-Soft Acc={acc_soft:.4f}  (delta={d_acc:+.4f})")
        if vaq_metrics.get("vaq_delta_l_ref") is not None and d_l_soft is not None:
            d_dl = d_l_soft - vaq_metrics["vaq_delta_l_ref"]
            print(f"  Stage1 ΔL_ref={vaq_metrics['vaq_delta_l_ref']:.1f}  "
                  f"Stage2 ΔL_ref={d_l_soft:.1f}  (delta={d_dl:+.1f})")
    else:
        print(f"  Stage2 VAQ-Soft Acc={acc_soft:.4f}")
        if d_l_soft is not None:
            print(f"  Stage2 ΔL_ref={d_l_soft:.1f}")
    if seg_metrics:
        if "vaq_miou" in seg_metrics:
            print(f"  Stage1 VAQ      mIoU={seg_metrics['vaq_miou']:.4f}")
        if "vaq_soft_miou" in seg_metrics:
            extra = (
                f"  (delta={seg_metrics['delta_miou']:+.4f})"
                if "delta_miou" in seg_metrics else ""
            )
            print(f"  Stage2 VAQ-Soft mIoU={seg_metrics['vaq_soft_miou']:.4f}{extra}")
    if snapshot_results:
        for sep, si in sorted(snapshot_results.items()):
            dl_str = f"  ΔL_ref={si['delta_l_ref']:.1f}" if si.get("delta_l_ref") is not None else ""
            miou_str = f"  mIoU={si['miou']:.4f}" if "miou" in si else ""
            print(f"  Snap@ep{sep}: Acc={si['acc']:.4f}{dl_str}{miou_str}")
    print(f"  Train time={train_time:.1f}s  ({len(history)} epochs)")
    print(f"{'=' * 60}")

    results = {
        "config": vars(args),
        "ckpt_path": ckpt_path,
        "layer_idx": layer_idx,
        "D": D,
        "T": T,
        "num_subspaces": num_subspaces,
        "bit_budget": bit_budget,
        "vaq_baseline": vaq_metrics,
        "vaq_soft_acc": float(acc_soft),
        "vaq_soft_delta_l_ref": None if d_l_soft is None else float(d_l_soft),
        "vaq_soft_rate": rate_soft,
        "seg": seg_metrics,
        "snapshot_evals": {str(k): v for k, v in snapshot_results.items()},
        "K_per_group": list(map(int, codec.k_per_group)),
        "K_max": int(codec.pq.K_max),
        "train_time": float(train_time),
        "history": history,
    }
    out_dir = os.path.join(VAQ_ROOT, "results", "vaq_soft", args.backbone)
    os.makedirs(out_dir, exist_ok=True)
    lmbda_tag = f"_lmbda{args.lmbda}" if args.lmbda > 0 else ""
    pf_tag = f"_pf{args.prior_floor}" if args.prior_floor > 0 else ""
    lref_tag = "_lref" if args.use_lref else ""
    out_tag = (
        f"{args.layer}_vaqsoft_B{bit_budget}_m{num_subspaces}"
        f"_min{args.min_bits}_max{args.max_bits}"
        f"_obj{args.bit_alloc_objective}"
        f"{lref_tag}{lmbda_tag}{pf_tag}"
        f"_ep{args.epochs}_lr{args.lr}_tau{args.tau_start}-{args.tau_end}"
        f"_fzR{int(args.freeze_transform)}_fzC{int(args.freeze_codebooks)}"
        f"_n{args.max_train_images}_s{args.seed}"
    )
    if args.result_suffix:
        out_tag += f"_{_safe_suffix(args.result_suffix)}"
    out_path = os.path.join(out_dir, f"{out_tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")
    return results


# ----------------------------------------------------------------
#                CLI
# ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VAQ-initialised Soft-PQ (MSE) experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--layer", type=str, default="blk20")
    parser.add_argument("--backbone", type=str, default="dinov2_vitl14")
    parser.add_argument("--K", type=int, default=4,
                        help="Used only to derive bit_budget when --bit_budget=0 (uniform fallback)")
    parser.add_argument("--embedding_dim", type=int, default=32,
                        help="Used to derive num_subspaces when --num_subspaces=0")

    # VAQ allocation
    parser.add_argument("--num_subspaces", type=int, default=0,
                        help="VAQ subspace count; 0 means D / embedding_dim")
    parser.add_argument("--bit_budget", type=int, default=0,
                        help="Total VAQ bits/token; 0 means num_subspaces * log2(K)")
    parser.add_argument("--min_bits", type=int, default=1)
    parser.add_argument("--max_bits", type=int, default=10,
                        help="Cap on per-subspace bits. K_max=2^max_bits bounds the soft-PQ "
                             "softmax memory cost. Default 10 (K_max=1024).")
    parser.add_argument("--bit_alloc_objective", type=str, default="linear",
                        choices=["rd", "linear"])
    parser.add_argument("--kmeans_iter", type=int, default=50)
    parser.add_argument("--kmeans_hier_threshold", type=int, default=1024)
    parser.add_argument("--kmeans_hier_branching", type=int, default=64)
    parser.add_argument("--max_fit_vectors", type=int, default=2_000_000)
    parser.add_argument("--norm_mode", type=str, default="per_image",
                        choices=["per_image", "per_token_ln"])

    # Soft-PQ training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Adam lr. 1e-4 is conservative enough not to wreck "
                             "the (already k-means optimal) VAQ-init codebooks.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--delta_batch_size", type=int, default=0)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--tau_start", type=float, default=0.0,
                        help="Soft-PQ start temperature. <=0 disables soft "
                             "assignment (pure hard PQ + gather-gradient).")
    parser.add_argument("--tau_end", type=float, default=0.0)
    parser.add_argument("--tau_schedule", type=str, default="exponential",
                        choices=["exponential", "linear"])
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--freeze_transform", action="store_true",
                        help="Freeze the (VAQ-initialised) Cayley orthogonal rotation. "
                             "Default OFF: rotation is also finetuned by MSE.")
    parser.add_argument("--freeze_codebooks", action="store_true")

    # Rate-distortion (mirror orfc/run_soft_pq.py)
    parser.add_argument("--lmbda", type=float, default=0.0,
                        help="Rate-distortion Lagrange multiplier. >0 enables "
                             "rate-aware ECVQ: cost=||x-c||^2 + (-log2 p)/lambda, "
                             "loss = R*T + lambda*D. Default 0 (no rate term).")
    parser.add_argument("--prior_floor", type=float, default=0.0,
                        help="Mix uniform mass into the learned prior to bound "
                             "max -log2(p). 0 disables.")
    parser.add_argument("--no_prior_init", action="store_true",
                        help="Skip initialising log_prior from VAQ argmin usage; "
                             "start from uniform.")

    # ΔL_ref training (mirrors orfc/run_soft_pq.py)
    parser.add_argument("--use_lref", action="store_true",
                        help="Train with ΔL_ref distillation loss instead of "
                             "MSE in normalised feature space. Pre-computes "
                             "teacher (frozen ViT tail) outputs once, then "
                             "minimises ||tail(X) - tail(inv_norm(codec(Y)))||² "
                             "per batch. Memory-heavy: keeps a CPU teacher "
                             "cache of size N*T*D*4 bytes.")

    # Data / runtime
    parser.add_argument("--max_train_images", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features"))
    parser.add_argument("--train_subset", type=str, default="train")
    parser.add_argument("--test_subset", type=str, default="test")
    parser.add_argument("--gt_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "utils",
                                             "imagenet_selected_label500.txt"))

    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--no_save_ckpt", action="store_true",
                        help="Skip checkpoint serialization after training "
                             "(useful for grid-sweeps where disk would fill up).")
    parser.add_argument("--skip_delta_l_ref", action="store_true")
    parser.add_argument("--skip_vaq_baseline", action="store_true",
                        help="Skip the stage-1 (argmin VAQ) evaluation; only report finetuned codec.")
    parser.add_argument("--result_suffix", type=str, default="")
    parser.add_argument("--eval_at_epochs", type=str, default="",
                        help="Comma-separated epoch numbers (0-based) at which to "
                             "snapshot the codec and run full downstream evaluation "
                             "(Acc + mIoU) *in addition to* the final-epoch eval. "
                             "E.g. --eval_at_epochs 49 evaluates after epoch 49 "
                             "(= 50th epoch). Only effective when training (not eval_only).")

    # Segmentation (VOC2012 mIoU + seg-rate) -- mirrors run_soft_pq.py
    parser.add_argument("--eval_seg", action="store_true",
                        help="Evaluate VOC2012 segmentation mIoU + seg-rate "
                             "for Stage-1 (VAQ argmin) and Stage-2 (finetuned).")
    parser.add_argument("--seg_feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features", "voc2012_100"),
                        help="VOC pre-extracted slide features: "
                             "{seg_feat_root}/{backbone}/{layer}/<name>.npy")
    parser.add_argument("--voc_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "VOCdevkit", "VOC2012"))
    parser.add_argument("--seg_image_list", type=str,
                        default=os.path.join(PROJECT_ROOT, "utils", "voc2012_val_100.txt"))

    args = parser.parse_args()

    # Track which keys the user passed explicitly on the CLI so
    # _restore_args_from_ckpt() can avoid clobbering them in eval-only mode.
    cli_keys = set()
    for tok in sys.argv[1:]:
        if tok.startswith("--"):
            cli_keys.add(tok.lstrip("-").split("=", 1)[0].replace("-", "_"))

    run_experiment(args, cli_keys=cli_keys)


if __name__ == "__main__":
    main()
