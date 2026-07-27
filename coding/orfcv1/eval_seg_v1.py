#!/usr/bin/env python
"""
ORFCv1 segmentation evaluation on VOC2012 test set.

Loads trained FeatureCodecV1 checkpoints and evaluates mIoU via
CodecSegmentationEvaluator (slide inference, matching run_soft_pq.py).

Usage:
    python eval_seg_v1.py --ckpt path/to/codec.pt --gpu 5

    # Batch mode (all final seeds):
    python eval_seg_v1.py \
        --ckpt_dir checkpoints/dinov2_vitl14/formal_k8_20260726T175547Z \
        --pattern "final_seed*" --gpu 5
"""

import os, sys, argparse, json, glob, time, math
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]
    except (ValueError, IndexError):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

from opq import batch_normalize_gpu, batch_inv_normalize_gpu
from backbone.wrapper import Dinov2Wrapper, SegmentationEvaluator
from codec_v1 import load_codec_v1

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
import logging
try:
    from mmcv.utils import get_logger
    logger = get_logger('mmcv')
except ImportError:
    logger = logging.getLogger('mmcv')
logger.setLevel(logging.WARNING)


class CodecV1SegmentationEvaluator(SegmentationEvaluator):
    """SegmentationEvaluator for FeatureCodecV1 (same interface as run_soft_pq.py)."""

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


def evaluate_one_ckpt(ckpt_path, args, wrapper, device):
    """Evaluate a single checkpoint, return result dict."""
    ckpt_name = os.path.basename(ckpt_path)
    print(f"\n{'=' * 60}")
    print(f"  Evaluating: {ckpt_name}")
    print(f"{'=' * 60}")

    codec = load_codec_v1(ckpt_path, device=device)
    layer_idx = int(args.layer[-2:])

    evaluator = CodecV1SegmentationEvaluator(
        codec=codec,
        norm_mode=args.norm_mode,
        layer_idx=layer_idx,
        voc_root=args.voc_root,
        weights_root=wrapper.weights_root,
        device=device,
        feat_dim=1024,
        model_name=args.backbone,
    )

    t0 = time.time()
    seg_result = evaluator.evaluate(
        seg_feat_dir=str(args.seg_feat_dir),
        image_list=args.seg_image_list,
        verbose=True,
    )
    elapsed = time.time() - t0

    miou = seg_result['miou']
    acc = seg_result['acc']
    class_iou = seg_result['class_iou']

    print(f"  mIoU = {miou:.4f}  aAcc = {acc:.4f}  ({elapsed:.1f}s)")
    for i, name in enumerate(SegmentationEvaluator.VOC_CLASSES):
        iou_val = class_iou[i]
        print(f"    {name:15s}: {iou_val:.4f}" if not np.isnan(iou_val)
              else f"    {name:15s}: NaN")

    result = {
        'ckpt': ckpt_path,
        'miou': float(miou),
        'acc': float(acc),
        'class_iou': {name: float(v) for name, v in
                      zip(SegmentationEvaluator.VOC_CLASSES, class_iou)
                      if not np.isnan(v)},
        'eval_time': float(elapsed),
    }
    del codec, evaluator
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="ORFCv1 segmentation evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", type=str, default="",
                        help="Single checkpoint path")
    parser.add_argument("--ckpt_dir", type=str, default="",
                        help="Directory of checkpoints (batch mode)")
    parser.add_argument("--pattern", type=str, default="*.pt",
                        help="Glob pattern within --ckpt_dir")
    parser.add_argument("--layer", type=str, default="blk20")
    parser.add_argument("--norm_mode", type=str, default="per_image")
    parser.add_argument("--backbone", type=str, default="dinov2_vitl14")
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--seg_feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features", "voc2012_100"))
    parser.add_argument("--voc_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "VOCdevkit", "VOC2012"))
    parser.add_argument("--seg_image_list", type=str,
                        default=os.path.join(PROJECT_ROOT, "utils", "voc2012_val_100.txt"))
    parser.add_argument("--out_dir", type=str, default="",
                        help="Output directory for results JSON (default: alongside ckpt)")

    args = parser.parse_args()

    args.seg_feat_dir = Path(args.seg_feat_root) / args.backbone / args.layer

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Collect checkpoints
    ckpt_list = []
    if args.ckpt:
        ckpt_list = [args.ckpt]
    elif args.ckpt_dir:
        pat = os.path.join(args.ckpt_dir, args.pattern)
        ckpt_list = sorted(glob.glob(pat))
        if not ckpt_list:
            raise FileNotFoundError(f"No checkpoints matching {pat}")
    else:
        parser.error("Provide --ckpt or --ckpt_dir")

    print(f"Checkpoints to evaluate: {len(ckpt_list)}")
    for p in ckpt_list:
        print(f"  {os.path.basename(p)}")

    print(f"\nLoading DINOv2 ({args.backbone}) on GPU {args.gpu}...")
    wrapper = Dinov2Wrapper(head_layers=1, model_name=args.backbone, device=device)
    wrapper.backbone.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()

    all_results = []
    for ckpt_path in ckpt_list:
        result = evaluate_one_ckpt(ckpt_path, args, wrapper, device)
        all_results.append(result)

    # Summary
    print(f"\n{'#' * 60}")
    print(f"  SEGMENTATION RESULTS SUMMARY")
    print(f"{'#' * 60}")
    for r in all_results:
        name = os.path.basename(r['ckpt']).replace('.pt', '')
        print(f"  {name}")
        print(f"    mIoU={r['miou']:.4f}  aAcc={r['acc']:.4f}")

    if len(all_results) > 1:
        mious = [r['miou'] for r in all_results]
        import statistics
        print(f"\n  Mean mIoU = {statistics.mean(mious):.4f} "
              f"± {statistics.stdev(mious):.4f}" if len(mious) > 1
              else f"\n  Mean mIoU = {statistics.mean(mious):.4f}")

    # Save
    out_dir = args.out_dir
    if not out_dir:
        if args.ckpt_dir:
            out_dir = args.ckpt_dir.replace('checkpoints', 'results')
        else:
            out_dir = os.path.dirname(args.ckpt) or '.'
    os.makedirs(out_dir, exist_ok=True)

    if len(all_results) == 1:
        tag = os.path.basename(all_results[0]['ckpt']).replace('.pt', '')
        out_path = os.path.join(out_dir, f'seg_eval_{tag}.json')
    else:
        out_path = os.path.join(out_dir, 'seg_eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
