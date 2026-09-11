#!/usr/bin/env python3
"""Extract DINOv3 ViT-L/16 slide-inference features for VOC and NYU eval.

VOC: CROP=512, STRIDE=341, patch_size=16 → padded to multiple of 16
     Each image → (num_slides, 5+H*W, 1024) saved as .npy
NYU: center-pad to multiple of 16, single crop
     Each image → (1, 5+H*W, 1024) saved as .npy

Usage:
    python extract_dinov3_voc_nyu.py --dataset voc --gpu 0
    python extract_dinov3_voc_nyu.py --dataset nyu --gpu 0
    python extract_dinov3_voc_nyu.py --dataset all --gpu 0
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PATCH_SIZE = 16
N_PREFIX = 5  # 1 CLS + 4 reg
CROP_SIZE = (512, 512)
STRIDE = (341, 341)
LAYERS = [5, 10, 15, 20]

PROJECT = Path(__file__).resolve().parent.parent.parent  # ORFC/
FEATCODEC = PROJECT.parent  # featcodec/

VOC_ROOT = FEATCODEC / "ORFC" / "data" / "VOCdevkit" / "VOC2012"
VOC_LIST = FEATCODEC / "ORFC" / "utils" / "voc2012_val_100.txt"
VOC_OUT = FEATCODEC / "features" / "voc2012_100" / "dinov3_vitl16"

NYU_ROOT = FEATCODEC / "CoFAI" / "data" / "NYU"
NYU_LIST_FILE = NYU_ROOT / "nyu_test.txt"
NYU_OUT = FEATCODEC / "features" / "nyu_depth_80" / "dinov3_vitl16"

CKPT = "/data4/workspace/zlt/cache/torch/hub/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"


class BlockHook:
    def __init__(self, blocks, indices):
        self.buf = {}
        for idx in indices:
            key = f"blk{idx:02d}"
            def _hook(m, inp, out, k=key):
                t = out[0] if isinstance(out, tuple) else out
                self.buf[k] = t.detach().cpu().float().numpy()
            blocks[idx].register_forward_hook(_hook)

    def pop(self):
        out = dict(self.buf)
        self.buf.clear()
        return out


def load_model(device):
    import timm
    model = timm.create_model("vit_large_patch16_dinov3",
                               pretrained=False, img_size=224,
                               dynamic_img_size=True)
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    sd.pop("mask_token", None)
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    return model


def get_slide_crops(h_img, w_img, crop_size, stride):
    h_crop, w_crop = crop_size
    h_stride, w_stride = stride
    crops = []
    h_grids = max(1, math.ceil((h_img - h_crop) / h_stride) + 1)
    w_grids = max(1, math.ceil((w_img - w_crop) / w_stride) + 1)
    for i in range(h_grids):
        for j in range(w_grids):
            y1 = min(i * h_stride, h_img - h_crop)
            x1 = min(j * w_stride, w_img - w_crop)
            crops.append((y1, x1, y1 + h_crop, x1 + w_crop))
    return crops


def pad_to_multiple(img_tensor, multiple):
    _, h, w = img_tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h))
    return img_tensor


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def extract_voc(args, model, hook, device):
    if not VOC_LIST.is_file():
        print(f"VOC list not found: {VOC_LIST}")
        return
    with open(VOC_LIST) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    print(f"VOC: {len(names)} images, {len(LAYERS)} layers")

    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    for layer_idx in LAYERS:
        out_dir = VOC_OUT / f"blk{layer_idx:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

    for name in tqdm(names, desc="VOC extract"):
        # Check if all layers done
        all_exist = all(
            (VOC_OUT / f"blk{l:02d}" / f"{name}.npy").is_file() for l in LAYERS
        )
        if all_exist and not args.overwrite:
            continue

        img_path = VOC_ROOT / "JPEGImages" / f"{name}.jpg"
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        h_img, w_img = img_np.shape[:2]

        # Pad image to crop_size minimum
        pad_h = max(0, CROP_SIZE[0] - h_img)
        pad_w = max(0, CROP_SIZE[1] - w_img)
        if pad_h or pad_w:
            img_np = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        padded_h, padded_w = img_np.shape[:2]

        crops = get_slide_crops(padded_h, padded_w, CROP_SIZE, STRIDE)

        # Per-layer accumulator
        layer_slides = {l: [] for l in LAYERS}

        for y1, x1, y2, x2 in crops:
            crop_img = img_np[y1:y2, x1:x2]
            t = transforms.ToTensor()(Image.fromarray(crop_img))
            t = pad_to_multiple(t, PATCH_SIZE)
            t = normalize(t).unsqueeze(0).to(device)

            with torch.no_grad(), torch.amp.autocast("cuda"):
                model(t)

            feats = hook.pop()
            for l in LAYERS:
                key = f"blk{l:02d}"
                layer_slides[l].append(feats[key][0])  # (T, D)

        for l in LAYERS:
            out_path = VOC_OUT / f"blk{l:02d}" / f"{name}.npy"
            arr = np.stack(layer_slides[l], axis=0)  # (num_slides, T, D)
            np.save(out_path, arr.astype(np.float32))


def extract_nyu(args, model, hook, device):
    if not NYU_LIST_FILE.is_file():
        print(f"NYU list not found: {NYU_LIST_FILE}")
        return
    with open(NYU_LIST_FILE) as f:
        items = []
        for ln in f:
            parts = ln.strip().split()
            if len(parts) >= 2:
                img_rel = parts[0]
                stem = img_rel.replace("/", "__").replace(".jpg", "")
                rgb_path = NYU_ROOT / "test" / img_rel
                if not rgb_path.is_file():
                    rgb_path = NYU_ROOT / img_rel
                items.append((stem, rgb_path))
    print(f"NYU: {len(items)} images, {len(LAYERS)} layers")

    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    for layer_idx in LAYERS:
        out_dir = NYU_OUT / f"blk{layer_idx:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

    for stem, rgb_path in tqdm(items, desc="NYU extract"):
        all_exist = all(
            (NYU_OUT / f"blk{l:02d}" / f"{stem}.npy").is_file() for l in LAYERS
        )
        if all_exist and not args.overwrite:
            continue

        img = Image.open(rgb_path).convert("RGB")
        t = transforms.ToTensor()(img)
        t = pad_to_multiple(t, PATCH_SIZE)
        t = normalize(t).unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            model(t)

        feats = hook.pop()
        for l in LAYERS:
            key = f"blk{l:02d}"
            arr = feats[key]  # (1, T, D)
            out_path = NYU_OUT / f"blk{l:02d}" / f"{stem}.npy"
            np.save(out_path, arr.astype(np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("voc", "nyu", "all"), default="all")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    print("Loading DINOv3 ViT-L/16...", flush=True)
    model = load_model(device)
    hook = BlockHook(model.blocks, LAYERS)
    print("Model loaded.", flush=True)

    if args.dataset in ("voc", "all"):
        extract_voc(args, model, hook, device)
    if args.dataset in ("nyu", "all"):
        extract_nyu(args, model, hook, device)
    print("Done.")


if __name__ == "__main__":
    main()
