#!/usr/bin/env python
"""
Comprehensive evaluation of ORFCv1/v1.1 checkpoints.

Evaluates both classification (ImageNet-500) and segmentation (VOC2012-100)
for each checkpoint, plus OPQ baseline.

Usage:
    python eval_v1_1_all.py --gpu 7

    python eval_v1_1_all.py --gpu 7 \
        --ckpt_dirs checkpoints/dinov2_vitl14/v1_1_20260726T011657Z \
                    checkpoints/dinov2_vitl14/formal_k8_20260726T175547Z \
        --pattern "*.pt"
"""

import os, sys, argparse, json, glob, time
import numpy as np
import torch
from pathlib import Path

V1_ROOT = os.path.dirname(os.path.abspath(__file__))
ORFC_ROOT = os.path.normpath(os.path.join(V1_ROOT, '..', 'orfc'))
PROJECT_ROOT = os.path.normpath(os.path.join(V1_ROOT, '..', '..', '..'))

if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)
if V1_ROOT not in sys.path:
    sys.path.insert(0, V1_ROOT)

from run_multilayer_calibrator import preload_features, load_gt, evaluate_accuracy
from opq import batch_normalize_gpu, batch_inv_normalize_gpu, batched_assign
from backbone.wrapper import Dinov2Wrapper, SegmentationEvaluator
from codec_v1 import load_codec_v1

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
warnings.filterwarnings("ignore", message="torchvision.datapoints")
warnings.filterwarnings("ignore", message="The given NumPy array")
import logging
try:
    from mmcv.utils import get_logger
    logger = get_logger('mmcv')
except ImportError:
    logger = logging.getLogger('mmcv')
logger.setLevel(logging.WARNING)


class CodecV1SegEval(SegmentationEvaluator):
    """Thin wrapper: quantize via FeatureCodecV1, then feed to seg head."""

    def __init__(self, codec, norm_mode, layer_idx,
                 voc_root, weights_root, device, model_name='dinov2_vitl14'):
        self.codec = codec
        self.norm_mode = norm_mode
        self.layer_idx = layer_idx
        self.voc_root = voc_root
        self.weights_root = weights_root
        self.device = device
        self.feat_dim = 1024
        self.model_name = model_name

    @torch.no_grad()
    def quantize_tokens(self, tokens_np):
        X = torch.from_numpy(tokens_np).float().unsqueeze(0).to(self.device)
        Y, Mu, Std = batch_normalize_gpu(X, mode=self.norm_mode)
        Y_hat, _ = self.codec(Y)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        return X_hat.squeeze(0)


class OPQSegEval(SegmentationEvaluator):
    """Quantize via standard OPQ (R + hard PQ assignment)."""

    def __init__(self, R, codebooks, embedding_dim, norm_mode, layer_idx,
                 voc_root, weights_root, device, model_name='dinov2_vitl14'):
        self.R_t = torch.from_numpy(R).float().to(device)
        self.cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
        self.G = len(codebooks)
        self.d = embedding_dim
        self.norm_mode = norm_mode
        self.layer_idx = layer_idx
        self.voc_root = voc_root
        self.weights_root = weights_root
        self.device = device
        self.feat_dim = 1024
        self.model_name = model_name

    @torch.no_grad()
    def quantize_tokens(self, tokens_np):
        C = tokens_np.shape[1]
        X = torch.from_numpy(tokens_np).float().unsqueeze(0).to(self.device)
        Y, Mu, Std = batch_normalize_gpu(X, mode=self.norm_mode)
        flat = Y.reshape(-1, C)
        Z = flat @ self.R_t
        z_3d = Z.reshape(-1, self.G, self.d).permute(1, 0, 2).contiguous()
        z_hat_3d, _ = batched_assign(z_3d, self.cb_t, device=self.device)
        flat_hat = z_hat_3d.permute(1, 0, 2).reshape(-1, C)
        Y_hat = (flat_hat @ self.R_t.T).reshape(Y.shape)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        return X_hat.squeeze(0)


def codec_encode_decode(features, codec, norm_mode, device, chunk=200):
    """Quantize a list of per-image features through a FeatureCodecV1."""
    codec.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(features), chunk):
            e = min(s + chunk, len(features))
            X = torch.from_numpy(np.stack(features[s:e])).float().to(device)
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            Y_hat, _ = codec(Y)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            for i in range(X_hat.shape[0]):
                out.append(X_hat[i].cpu().numpy())
            del X, Y, Mu, Std, Y_hat, X_hat
        torch.cuda.empty_cache()
    return out


def opq_encode_decode(features, codebooks, embedding_dim, norm_mode,
                      device, R, chunk=500):
    """Quantize via standard OPQ."""
    G = len(codebooks)
    C = features[0].shape[1]
    cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
    R_t = torch.from_numpy(R).float().to(device)
    out = []
    with torch.no_grad():
        for s in range(0, len(features), chunk):
            e = min(s + chunk, len(features))
            X = torch.from_numpy(np.stack(features[s:e])).float().to(device)
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            flat = Y.reshape(-1, C)
            Z = flat @ R_t
            z_3d = Z.reshape(-1, G, embedding_dim).permute(1, 0, 2).contiguous()
            z_hat_3d, _ = batched_assign(z_3d, cb_t, device=device)
            flat_hat = z_hat_3d.permute(1, 0, 2).reshape(-1, C)
            Y_hat = (flat_hat @ R_t.T).reshape(Y.shape)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            for i in range(X_hat.shape[0]):
                out.append(X_hat[i].cpu().numpy())
            del X, Y, Mu, Std, Y_hat, X_hat
        torch.cuda.empty_cache()
    return out


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--ckpt_dirs", nargs="+", default=[])
    parser.add_argument("--ckpts", nargs="+", default=[])
    parser.add_argument("--pattern", type=str, default="*.pt")
    parser.add_argument("--opq_artifact", type=str, default="")
    parser.add_argument("--v1_baselines", nargs="+", default=[],
                        help="v1 formal ckpts to include as baselines")
    parser.add_argument("--layer", type=str, default="blk20")
    parser.add_argument("--norm_mode", type=str, default="per_image")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--backbone", type=str, default="dinov2_vitl14")
    parser.add_argument("--skip_seg", action="store_true")
    parser.add_argument("--skip_cls", action="store_true")

    parser.add_argument("--feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features"))
    parser.add_argument("--test_subset", type=str, default="test")
    parser.add_argument("--gt_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "utils",
                                             "imagenet_selected_label500.txt"))
    parser.add_argument("--seg_feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features",
                                             "voc2012_100"))
    parser.add_argument("--voc_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "data",
                                             "VOCdevkit", "VOC2012"))
    parser.add_argument("--seg_image_list", type=str,
                        default=os.path.join(PROJECT_ROOT, "utils",
                                             "voc2012_val_100.txt"))
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device("cuda:0")

    layer_idx = int(args.layer[-2:])
    seg_feat_dir = str(Path(args.seg_feat_root) / args.backbone / args.layer)

    # ---- Collect checkpoints ----
    ckpt_list = list(args.ckpts)
    for d in args.ckpt_dirs:
        ckpt_list.extend(sorted(glob.glob(os.path.join(d, args.pattern))))
    for b in args.v1_baselines:
        if os.path.isfile(b):
            ckpt_list.append(b)

    print(f"{'#' * 70}")
    print(f"  ORFC v1/v1.1 Comprehensive Evaluation")
    print(f"  GPU {args.gpu}, {len(ckpt_list)} checkpoints"
          f"{' + OPQ' if args.opq_artifact else ''}")
    print(f"  Cls: {'ON' if not args.skip_cls else 'OFF'}  "
          f"Seg: {'ON' if not args.skip_seg else 'OFF'}")
    print(f"{'#' * 70}")

    for p in ckpt_list:
        print(f"  {os.path.basename(p)}")
    if args.opq_artifact:
        print(f"  [OPQ] {os.path.basename(args.opq_artifact)}")

    # ---- Load DINOv2 ----
    print(f"\nLoading DINOv2 ({args.backbone})...")
    wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone,
                            device=device)

    # ---- Load test features (cls) ----
    features_test, basenames_test, gt_test = None, None, None
    if not args.skip_cls:
        test_dir = Path(args.feat_root) / args.test_subset / args.backbone / args.layer
        test_files = sorted(test_dir.glob("*.npy"))
        print(f"\nLoading {len(test_files)} test features from {test_dir}...")
        features_test, basenames_test = preload_features(test_files, num_workers=4)
        gt_test = load_gt(args.gt_path)
        print(f"  GT labels: {len(gt_test)}")

    # ---- Results ----
    all_results = []

    # ---- OPQ baseline ----
    if args.opq_artifact:
        print(f"\n{'=' * 60}")
        print(f"  [OPQ Baseline]")
        print(f"{'=' * 60}")
        art = np.load(args.opq_artifact, allow_pickle=True)
        R_std = art['R']
        G = R_std.shape[0] // args.embedding_dim
        codebooks_std = [art[f'cb_{g}'] for g in range(G)]

        r = {'name': 'OPQ (baseline)', 'type': 'opq'}

        if not args.skip_cls:
            xhat_opq = opq_encode_decode(
                features_test, codebooks_std, args.embedding_dim,
                args.norm_mode, device, R_std)
            wrapper.backbone.to(device)
            if wrapper.head is not None:
                wrapper.head.to(device)
            torch.cuda.empty_cache()
            acc = evaluate_accuracy(
                xhat_opq, basenames_test, gt_test, wrapper, layer_idx, device)
            r['cls_acc'] = float(acc)
            print(f"  Cls Acc = {acc:.4f}")
            del xhat_opq

        if not args.skip_seg:
            wrapper.backbone.cpu()
            if wrapper.head is not None:
                wrapper.head.cpu()
            torch.cuda.empty_cache()
            evaluator = OPQSegEval(
                R_std, codebooks_std, args.embedding_dim,
                args.norm_mode, layer_idx,
                args.voc_root, wrapper.weights_root, device,
                model_name=args.backbone)
            seg = evaluator.evaluate(
                seg_feat_dir=seg_feat_dir,
                image_list=args.seg_image_list, verbose=False)
            r['seg_miou'] = float(seg['miou'])
            r['seg_acc'] = float(seg['acc'])
            print(f"  Seg mIoU = {seg['miou']:.4f}  aAcc = {seg['acc']:.4f}")
            del evaluator

        all_results.append(r)

    # ---- Checkpoints ----
    for ckpt_path in ckpt_list:
        ckpt_name = os.path.basename(ckpt_path).replace('.pt', '')
        short = _short_name(ckpt_name)
        print(f"\n{'=' * 60}")
        print(f"  {short}")
        print(f"{'=' * 60}")

        codec = load_codec_v1(ckpt_path, device=device)
        r = {'name': short, 'ckpt': ckpt_path, 'type': 'codec'}

        if not args.skip_cls:
            xhat = codec_encode_decode(
                features_test, codec, args.norm_mode, device)
            wrapper.backbone.to(device)
            if wrapper.head is not None:
                wrapper.head.to(device)
            torch.cuda.empty_cache()
            acc = evaluate_accuracy(
                xhat, basenames_test, gt_test, wrapper, layer_idx, device)
            r['cls_acc'] = float(acc)
            print(f"  Cls Acc = {acc:.4f}")
            del xhat

        if not args.skip_seg:
            wrapper.backbone.cpu()
            if wrapper.head is not None:
                wrapper.head.cpu()
            torch.cuda.empty_cache()
            evaluator = CodecV1SegEval(
                codec, args.norm_mode, layer_idx,
                args.voc_root, wrapper.weights_root, device,
                model_name=args.backbone)
            seg = evaluator.evaluate(
                seg_feat_dir=seg_feat_dir,
                image_list=args.seg_image_list, verbose=False)
            r['seg_miou'] = float(seg['miou'])
            r['seg_acc'] = float(seg['acc'])
            print(f"  Seg mIoU = {seg['miou']:.4f}  aAcc = {seg['acc']:.4f}")
            del evaluator

        del codec
        torch.cuda.empty_cache()
        all_results.append(r)

    # ---- Summary table ----
    print(f"\n{'#' * 70}")
    print(f"  COMPARISON TABLE")
    print(f"{'#' * 70}")
    header = f"{'Name':45s}"
    if not args.skip_cls:
        header += f"  {'ClsAcc':>7s}"
    if not args.skip_seg:
        header += f"  {'mIoU':>7s}  {'SegAcc':>7s}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        line = f"{r['name']:45s}"
        if not args.skip_cls:
            line += f"  {r.get('cls_acc', 0):7.4f}"
        if not args.skip_seg:
            line += f"  {r.get('seg_miou', 0):7.4f}  {r.get('seg_acc', 0):7.4f}"
        print(line)

    # ---- Save ----
    out_path = args.out
    if not out_path:
        out_path = os.path.join(V1_ROOT, 'results', 'eval_comparison.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


def _short_name(ckpt_name):
    """Produce a human-readable short name from checkpoint filename."""
    name = ckpt_name
    for prefix in ['blk20_K8_emb32_', 'blk20_K8_']:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    for suffix in ['_ep100_s42', '_ep100_s43', '_ep100_s44']:
        name = name.replace(suffix, '')
    name = name.replace('lr0.0003_', '')
    name = name.replace('tau0.0063977931_', '')
    name = name.replace('gpb4_', '')
    name = name.replace('joint_', '')
    name = name.replace('a0.1,0.5,1.0_', '')
    return name


if __name__ == '__main__':
    main()
