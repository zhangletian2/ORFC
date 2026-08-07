"""Frozen ImageNet-500/VOC-100 evaluation for a V31 nested codec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..v12.eval_downstream import CodecSegmentation, PROJECT, reconstruct
from ..v30 import nested
from backbone.wrapper import Dinov2Wrapper
from run_multilayer_calibrator import evaluate_accuracy, load_gt, preload_features


class NestedAdapter:
    def __init__(self, codec):
        self.codec, self.pq = codec, codec.pq

    def __call__(self, y, modes):
        return nested.reconstruct(self.codec, y, modes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--allocation", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--layer", type=int, default=20)
    args = parser.parse_args(argv)
    device = torch.device("cuda")
    codec, _ = nested.load_checkpoint(args.checkpoint, device)
    adapter = NestedAdapter(codec)
    modes = np.load(args.allocation).astype(np.int64).tolist()
    bits = np.asarray(codec.pq.mode_bits, dtype=np.int64)
    nominal_rate = int(bits[np.asarray(modes)].sum())

    wrapper = Dinov2Wrapper(
        head_layers=1, model_name="dinov2_vitl14", device=device)
    feature_root = PROJECT / "features"
    block = f"blk{args.layer:02d}"
    test_dir = feature_root / "test" / "dinov2_vitl14" / block
    files = sorted(test_dir.glob("*.npy"))
    features, basenames = preload_features(files, num_workers=4)
    labels = load_gt(PROJECT / "utils" / "imagenet_selected_label500.txt")
    reconstructed = reconstruct(
        features, adapter, modes, device, args.chunk)
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
        adapter, modes, device,
        PROJECT / "data" / "VOCdevkit" / "VOC2012",
        wrapper.weights_root, args.layer)
    segmentation = evaluator.evaluate(
        seg_feat_dir=str(
            feature_root / "voc2012_100" / "dinov2_vitl14" / block),
        image_list=str(PROJECT / "utils" / "voc2012_val_100.txt"),
        verbose=False)
    result = {
        "name": args.name, "kind": "v31_nested",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "allocation": str(Path(args.allocation).resolve()),
        "nominal_rate": nominal_rate, "layer": args.layer, "block": block,
        "classification_images": len(files), "segmentation_images": 100,
        "cls_acc": float(accuracy), "seg_miou": float(segmentation["miou"]),
        "seg_acc": float(segmentation["acc"])}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
