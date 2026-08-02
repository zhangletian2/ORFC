"""Matched ImageNet-500/VOC-100 evaluation for joint allocation and ORFC."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

V1 = Path(__file__).resolve().parents[2]
ORFC = V1.parent / "orfc"
PROJECT = V1.parents[2]
for path in (str(V1), str(ORFC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from backbone.wrapper import Dinov2Wrapper, SegmentationEvaluator
from opq import batch_inv_normalize_gpu, batch_normalize_gpu
from run_multilayer_calibrator import evaluate_accuracy, load_gt, preload_features
from soft_pq import load_codec as load_orfc
from codec_v1 import load_codec_v1


class CodecSegmentation(SegmentationEvaluator):
    def __init__(self, codec, modes, device, voc_root, weights_root, layer_idx=20):
        self.codec, self.modes, self.device = codec, modes, device
        self.norm_mode, self.layer_idx = "per_image", int(layer_idx)
        self.voc_root, self.weights_root = voc_root, weights_root
        self.feat_dim, self.model_name = 1024, "dinov2_vitl14"

    @torch.no_grad()
    def quantize_tokens(self, tokens_np):
        x = torch.from_numpy(tokens_np).float().unsqueeze(0).to(self.device)
        y, mu, std = batch_normalize_gpu(x, mode=self.norm_mode)
        yhat, _ = (self.codec(y, modes=self.modes) if self.modes is not None
                   else self.codec(y))
        return batch_inv_normalize_gpu(yhat, mu, std).squeeze(0)


@torch.no_grad()
def reconstruct(features, codec, modes, device, chunk=16):
    output = []
    for start in range(0, len(features), int(chunk)):
        x = torch.from_numpy(np.stack(features[start:start + chunk])).float().to(device)
        y, mu, std = batch_normalize_gpu(x, mode="per_image")
        yhat, _ = (codec(y, modes=modes) if modes is not None else codec(y))
        xhat = batch_inv_normalize_gpu(yhat, mu, std)
        output.extend(item.cpu().numpy() for item in xhat)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("joint", "orfc"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--allocation", default="")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--layer", type=int, default=20)
    args = parser.parse_args(argv)

    device = torch.device("cuda")
    codec = (load_codec_v1(args.checkpoint, device=device) if args.kind == "joint"
             else load_orfc(args.checkpoint, device=device))
    modes = np.load(args.allocation).astype(np.int64).tolist() if args.allocation else None
    if args.kind == "joint" and modes is None:
        parser.error("joint codec requires --allocation")
    if args.kind == "orfc" and modes is not None:
        parser.error("ORFC baseline must use its uniform checkpoint")
    if modes is None:
        nominal_rate = int(codec.pq.G * round(np.log2(codec.pq.K)))
    else:
        bits = np.log2(np.asarray(codec.pq.mode_sizes)).astype(np.int64)
        nominal_rate = int(bits[np.asarray(modes)].sum())

    wrapper = Dinov2Wrapper(head_layers=1, model_name="dinov2_vitl14", device=device)
    feature_root = PROJECT / "features"
    block = f"blk{args.layer:02d}"
    test_dir = feature_root / "test" / "dinov2_vitl14" / block
    files = sorted(test_dir.glob("*.npy"))
    features, basenames = preload_features(files, num_workers=4)
    labels = load_gt(PROJECT / "utils" / "imagenet_selected_label500.txt")
    reconstructed = reconstruct(features, codec, modes, device, args.chunk)
    wrapper.backbone.to(device)
    if wrapper.head is not None:
        wrapper.head.to(device)
    accuracy = evaluate_accuracy(
        reconstructed, basenames, labels, wrapper, args.layer, device)
    del reconstructed

    wrapper.backbone.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()
    evaluator = CodecSegmentation(
        codec, modes, device, PROJECT / "data" / "VOCdevkit" / "VOC2012",
        wrapper.weights_root, args.layer)
    segmentation = evaluator.evaluate(
        seg_feat_dir=str(feature_root / "voc2012_100" / "dinov2_vitl14" / block),
        image_list=str(PROJECT / "utils" / "voc2012_val_100.txt"),
        verbose=False)
    result = {"name": args.name, "kind": args.kind,
              "checkpoint": args.checkpoint, "allocation": args.allocation or None,
              "nominal_rate": nominal_rate, "layer": args.layer, "block": block,
              "classification_images": len(files),
              "segmentation_images": 100, "cls_acc": float(accuracy),
              "seg_miou": float(segmentation["miou"]),
              "seg_acc": float(segmentation["acc"])}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
