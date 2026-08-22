"""Frozen downstream evaluation for the matched-rate patch/block VQ arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ORFC = Path("/data4/workspace/zlt/featcodec/ORFC/coding/orfc")
ORFCV1 = Path("/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1")
ROOT = Path("/data4/workspace/zlt/featcodec")
sys.path[:0] = [str(ORFC), str(ORFCV1)]

from opq import batch_inv_normalize_gpu, batch_normalize_gpu  # noqa: E402
from run_multilayer_calibrator import evaluate_accuracy, load_gt, preload_features  # noqa: E402
from run_soft_pq import CodecSegmentationEvaluator  # noqa: E402
from backbone.wrapper import Dinov2Wrapper  # noqa: E402
from soft_pq import OrthogonalTransform, SoftPQ  # noqa: E402
from phase1.v34.block_quantizer import (  # noqa: E402
    BlockVQ, PatchOnlyCodec, ProductBlockVQ, TokenPairVQ,
)


def load_codec(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu")
    arm = payload["arm"]
    variant = payload.get("variant", "block_k256" if arm == "block" else "orfc_k4")
    if arm == "orfc":
        pq = SoftPQ(32, 4, 32)
    elif variant == "block_pq_2xk16":
        pq = ProductBlockVQ()
    elif variant == "token_pair_k16":
        pq = TokenPairVQ()
    else:
        pq = BlockVQ()
    codec = PatchOnlyCodec(
        OrthogonalTransform(1024), pq, arm, pad_mode="replicate")
    codec.load_state_dict(payload["state_dict"])
    return codec.to(device).eval(), arm


@torch.no_grad()
def reconstruct(features, codec, device, batch):
    decoded = []
    for first in range(0, len(features), batch):
        x = torch.from_numpy(np.stack(features[first:first + batch])).float().to(device)
        y, mu, std = batch_normalize_gpu(x, mode="per_image")
        yhat, _ = codec(y)
        xhat = batch_inv_normalize_gpu(yhat, mu, std).cpu().numpy()
        decoded.extend(xhat)
    return decoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--block", default="blk20",
                        choices=("blk05", "blk10", "blk15", "blk20"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    codec, arm = load_codec(args.checkpoint, device)
    layer = int(args.block[-2:])

    feature_dir = ROOT / f"features/test/dinov2_vitl14/{args.block}"
    features, names = preload_features(sorted(feature_dir.glob("*.npy")), num_workers=4)
    decoded = reconstruct(features, codec, device, args.batch)
    wrapper = Dinov2Wrapper(head_layers=1, model_name="dinov2_vitl14", device=device)
    labels = load_gt(ROOT / "ORFC/utils/imagenet_selected_label500.txt")
    cls_acc = evaluate_accuracy(decoded, names, labels, wrapper, layer, device)
    del decoded, features, wrapper
    torch.cuda.empty_cache()
    report = {"block": args.block, "arm": arm, "checkpoint": str(args.checkpoint),
              "classification_images": len(names), "cls_acc": float(cls_acc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    segmenter = CodecSegmentationEvaluator(
        codec=codec, norm_mode="per_image", layer_idx=layer,
        voc_root=str(ROOT / "data/VOCdevkit/VOC2012"),
        weights_root="/data4/workspace/zlt/cache/torch/hub/checkpoints",
        device=device,
        feat_dim=1024, model_name="dinov2_vitl14")
    seg = segmenter.evaluate(
        str(ROOT / f"features/voc2012_100/dinov2_vitl14/{args.block}"),
        str(ROOT / "ORFC/utils/voc2012_val_100.txt"))
    report.update({"segmentation_images": 100,
                   "seg_miou": float(seg["miou"]),
                   "seg_acc": float(seg["acc"])})
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
