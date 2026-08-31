#!/usr/bin/env python
"""Frozen similarity-merge + r_ψ, train R and PQ from scratch on y_m.

Loads a trained ``SimilarityMergeCodec``, freezes matching and ``r_ψ``,
quantises the short sequence ``CLS + y_m`` (129 tokens), OPQ-inits then
trains channel ``R`` + SoftPQ + factorized prior with ΔL_ref.

BPFP is normalised by original ``T_full · D``.  Matching side-info
``2 r log2(T_patch)`` bits/image is added on top of rANS.

Usage (blk05, K=4, GPU 2):
    MERGE=results/similarity_merge/dinov2_vitl14/blk05_r128_h256_lr0.0003_ep100_s42.pt
    CUDA_VISIBLE_DEVICES=2 python -u run_similarity_merge_pq.py --layer blk05 \
        --K 4 --merge_ckpt $MERGE --epochs 100 --save_codec
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_ORFC_DIR = os.path.normpath(os.path.join(HERE, '..', 'orfc'))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
if _ORFC_DIR not in sys.path:
    sys.path.insert(0, _ORFC_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from run_multilayer_calibrator import (  # noqa: E402
    set_seed, preload_features, load_gt, evaluate_accuracy,
)
from backbone.wrapper import Dinov2Wrapper  # noqa: E402
from soft_pq import FrozenTail, load_codec as load_orfc_codec  # noqa: E402
from run_soft_pq import (  # noqa: E402
    evaluate_delta_l_ref, evaluate_rate, _codec_labels, _histogram_pmf,
)
from soft_pq import soft_pq_encode_decode  # noqa: E402

from similarity_merge import (  # noqa: E402
    SimilarityMergeCodec, load_merge_codec, merge_side_bits,
    train_merge_pq_codec, save_merge_pq_codec,
)
from eval_similarity_merge_seg import _r_from_frac  # noqa: E402

import warnings  # noqa: E402
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")


def _bpfp(rans_bpt, Tm, T_full, D, side_bits=0.0):
    if rans_bpt is None:
        return None
    return float(rans_bpt) * Tm / (T_full * D) + side_bits / (T_full * D)


def _rebuild_train_tail(wrapper, layer_idx, device):
    for i, blk in enumerate(wrapper.backbone.blocks):
        if i <= layer_idx:
            blk.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()
    return FrozenTail(list(wrapper.backbone.blocks[layer_idx + 1:]),
                      wrapper.backbone.norm, device=device)


def _eval_acc_dl_rate(codec, features_test, basenames, gt, wrapper, layer_idx,
                      device, norm_mode, batch_size, train_features, Tm,
                      T_full, D, side_bits):
    wrapper.backbone.to(device)
    if wrapper.head is not None:
        wrapper.head.to(device)
    torch.cuda.empty_cache()
    xhat = soft_pq_encode_decode(features_test, codec, norm_mode, device,
                                 chunk_images=batch_size)
    acc = evaluate_accuracy(xhat, basenames, gt, wrapper, layer_idx, device)
    del xhat
    tail = FrozenTail(list(wrapper.backbone.blocks[layer_idx + 1:]),
                      wrapper.backbone.norm, device=device)
    dl = evaluate_delta_l_ref(features_test, tail, norm_mode, device,
                              codec=codec, batch_size=batch_size)
    tail.to('cpu')
    torch.cuda.empty_cache()

    G, K = codec.pq.G, codec.pq.K
    ref = train_features[:min(len(train_features), 1000)]
    train_labels = _codec_labels(ref, codec, norm_mode, device,
                                 batch_size=batch_size)
    train_pmf = _histogram_pmf(train_labels, G, K)
    rate = evaluate_rate(features_test, codec, norm_mode, device,
                         batch_size=batch_size, train_pmf=train_pmf)
    rbpt = rate.get('rans_train_bpt') or rate.get('rans_bpt')
    bpfp = _bpfp(rbpt, Tm, T_full, D, side_bits=0.0)
    bpfp_side = _bpfp(rbpt, Tm, T_full, D, side_bits=side_bits)
    return {
        'acc': float(acc), 'delta_l': float(dl), 'rate': rate,
        'rans_bpt': None if rbpt is None else float(rbpt),
        'bpfp': bpfp, 'bpfp_with_match_side': bpfp_side,
        'side_bits': float(side_bits), 'coded_tokens': Tm,
    }


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--layer", type=str, default="blk05")
    p.add_argument("--K", type=int, required=True,
                   choices=[2, 4, 8, 16, 32, 64, 256])
    p.add_argument("--emb", type=int, default=32)
    p.add_argument("--merge_ckpt", type=str, default="",
                   help="trained SimilarityMergeCodec .pt (r_ψ); unused for tome")
    p.add_argument("--match_mode", type=str, default="cosine",
                   choices=["cosine", "tome"])
    p.add_argument("--merge_frac", type=float, default=0.25,
                   help="r / T_patch when match_mode=tome")
    p.add_argument("--orfc_ckpt", type=str, default="",
                   help="optional ORFC checkpoint for the same K (reference)")
    p.add_argument("--n_prefix", type=int, default=1)
    p.add_argument("--norm_mode", type=str, default="per_image",
                   choices=["per_image", "split_cls_patch"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lmbda", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--tau_start", type=float, default=0.5)
    p.add_argument("--tau_end", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_images", type=int, default=5000)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--save_codec", action="store_true")
    p.add_argument("--skip_ceiling", action="store_true")
    p.add_argument("--feat_root", type=str,
                   default=os.path.join(PROJECT_ROOT, "features"))
    p.add_argument("--train_subset", type=str, default="train")
    p.add_argument("--test_subset", type=str, default="test")
    p.add_argument("--backbone", type=str, default="dinov2_vitl14")
    p.add_argument("--gt_path", type=str,
                   default=os.path.join(PROJECT_ROOT, "utils",
                                        "imagenet_selected_label500.txt"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    layer_idx = int(args.layer[-2:])

    print(f"\n{'#' * 70}")
    print(f"# Frozen merge  +  train R/PQ from scratch")
    print(f"# {args.layer}  match={args.match_mode}  K={args.K} emb={args.emb} "
          f"lmbda={args.lmbda}  ep={args.epochs} lr={args.lr}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    train_dir = Path(args.feat_root) / args.train_subset / args.backbone / args.layer
    test_dir = Path(args.feat_root) / args.test_subset / args.backbone / args.layer
    train_files = sorted(train_dir.glob("*.npy"))
    test_files = sorted(test_dir.glob("*.npy"))
    if not train_files or not test_files:
        raise FileNotFoundError(f"features missing: {train_dir} / {test_dir}")
    features_train, _ = preload_features(train_files, num_workers=4)
    features_test, basenames_test = preload_features(test_files, num_workers=4)
    gt_test = load_gt(args.gt_path)

    T_full, D = features_train[0].shape
    T_patch = T_full - args.n_prefix
    G = D // args.emb
    d = args.emb
    print(f"\nData: train={len(features_train)} test={len(features_test)}  "
          f"D={D} T_full={T_full} T_patch={T_patch}  G={G} d={d} K={args.K}")

    merge_meta = {}
    if args.match_mode == "tome":
        r = _r_from_frac(T_patch, args.merge_frac)
        merge = SimilarityMergeCodec(
            D, n_prefix=args.n_prefix, r=r, grid=(16, 16), hidden=256,
            use_decoder=False, match_mode="tome").to(device).eval()
        print(f"  ToMe merge  frac={args.merge_frac}  r={r}  "
              f"(no merge ckpt, copy-mean decode)")
    else:
        if not args.merge_ckpt:
            args.merge_ckpt = str(
                Path(HERE) / "results" / "similarity_merge" / args.backbone /
                f"{args.layer}_r128_h256_lr0.0003_ep100_s42.pt")
        if not os.path.isfile(args.merge_ckpt):
            raise FileNotFoundError(
                f"merge checkpoint missing: {args.merge_ckpt}")
        merge, merge_meta = load_merge_codec(args.merge_ckpt, device=device)
        print(f"  loaded merge r={merge.r} hidden={merge.hidden}  "
              f"ckpt={args.merge_ckpt}")
    Tm = T_full - merge.r
    side_bits = merge_side_bits(
        merge.r, T_patch, getattr(merge, "match_mode", args.match_mode))
    print(f"  Tm={Tm}  match_side={side_bits:.0f} bits/img  "
          f"(BPFP {side_bits / (T_full * D):.5f})")

    if 0 < args.max_train_images < len(features_train):
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(features_train), args.max_train_images,
                         replace=False)
        features_train_sub = [features_train[i] for i in idx]
    else:
        features_train_sub = features_train
    n_val = min(args.n_val, len(features_train))
    rng_val = np.random.RandomState(args.seed + 1)
    val_idx = rng_val.choice(len(features_train), n_val, replace=False)
    val_features = [features_train[i] for i in val_idx]

    print(f"\nLoading DINOv2 ({args.backbone})...")
    wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone,
                            device=device)

    results = {
        'config': vars(args), 'D': D, 'T_full': T_full, 'T_patch': T_patch,
        'G': G, 'd': d, 'K': args.K, 'Tm': Tm, 'side_bits': side_bits,
        'merge_ckpt': args.merge_ckpt,
    }

    if not args.skip_ceiling:
        print(f"\n{'=' * 60}\n  [ceiling] frozen merge, no PQ\n{'=' * 60}")
        from similarity_merge import SimilarityMergePQCodec
        from soft_pq import SoftPQ, OrthogonalTransform
        dummy = SoftPQ(G, args.K, d, lmbda=0.0).to(device)
        ceil = SimilarityMergePQCodec(
            merge, dummy, OrthogonalTransform(D).to(device)).to(device)
        ceil.quantize = False
        wrapper.backbone.to(device)
        if wrapper.head is not None:
            wrapper.head.to(device)
        xhat = soft_pq_encode_decode(features_test, ceil, args.norm_mode,
                                     device, chunk_images=args.batch_size)
        acc_c = evaluate_accuracy(xhat, basenames_test, gt_test,
                                  wrapper, layer_idx, device)
        del xhat
        tail_c = FrozenTail(list(wrapper.backbone.blocks[layer_idx + 1:]),
                            wrapper.backbone.norm, device=device)
        dl_c = evaluate_delta_l_ref(features_test, tail_c, args.norm_mode,
                                    device, codec=ceil,
                                    batch_size=args.batch_size)
        tail_c.to('cpu')
        results['ceiling'] = {'acc': float(acc_c), 'delta_l': float(dl_c)}
        print(f"  Acc={acc_c:.4f}  ΔL_ref={dl_c:.1f}")
        del ceil, dummy
        torch.cuda.empty_cache()

    if args.orfc_ckpt and os.path.isfile(args.orfc_ckpt):
        print(f"\n{'=' * 60}\n  [ORFC ref] {os.path.basename(args.orfc_ckpt)}\n"
              f"{'=' * 60}")
        ref = load_orfc_codec(args.orfc_ckpt, device=device)
        m_ref = _eval_acc_dl_rate(
            ref, features_test, basenames_test, gt_test, wrapper, layer_idx,
            device, args.norm_mode, args.batch_size, features_train_sub,
            Tm=T_full, T_full=T_full, D=D, side_bits=0.0)
        results['orfc_ref'] = m_ref
        print(f"  Acc={m_ref['acc']:.4f}  ΔL={m_ref['delta_l']:.1f}  "
              f"rANS={m_ref['rans_bpt']:.2f}b/t  BPFP={m_ref['bpfp']:.4f}")
        del ref
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}\n  [train] R+PQ from scratch, freeze merge, K={args.K}\n"
          f"{'=' * 60}")
    tail_train = _rebuild_train_tail(wrapper, layer_idx, device)
    t0 = time.time()
    codec, hist = train_merge_pq_codec(
        features_train=features_train_sub, tail=tail_train, merge=merge,
        G=G, K=args.K, d=d, n_prefix=args.n_prefix, norm_mode=args.norm_mode,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        device=device, seed=args.seed, val_features=val_features,
        verbose=True, lmbda=args.lmbda, grad_clip=args.grad_clip,
        tau_start=args.tau_start, tau_end=args.tau_end)
    print(f"  training: {time.time() - t0:.1f}s")
    results['history'] = hist

    print(f"\n{'=' * 60}\n  [trained] merge+PQ K={args.K}\n{'=' * 60}")
    m = _eval_acc_dl_rate(
        codec, features_test, basenames_test, gt_test, wrapper, layer_idx,
        device, args.norm_mode, args.batch_size, features_train_sub,
        Tm=Tm, T_full=T_full, D=D, side_bits=side_bits)
    results['trained'] = m
    print(f"  Acc={m['acc']:.4f}  ΔL={m['delta_l']:.1f}  "
          f"rANS={m['rans_bpt']:.2f}b/t  "
          f"BPFP={m['bpfp']:.4f}  BPFP+side={m['bpfp_with_match_side']:.4f}")

    out_dir = Path(HERE) / 'results' / 'similarity_merge_pq' / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"{args.layer}_r{merge.r}_K{args.K}_emb{d}_lmbda{args.lmbda}"
           f"_lr{args.lr}_ep{args.epochs}_s{args.seed}")
    if args.match_mode == "tome":
        tag = (f"{args.layer}_tome_f{args.merge_frac}_r{merge.r}_K{args.K}"
               f"_emb{d}_lmbda{args.lmbda}_lr{args.lr}_ep{args.epochs}"
               f"_s{args.seed}")
    if args.save_codec:
        save_merge_pq_codec(
            codec, out_dir / f"{tag}.pt",
            meta_extra={'merge_ckpt': args.merge_ckpt, 'layer': args.layer,
                        'match_mode': args.match_mode,
                        'merge_frac': args.merge_frac})
    out_path = out_dir / f"{tag}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")
    return results


if __name__ == '__main__':
    main()
