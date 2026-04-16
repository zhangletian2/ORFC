# -*- coding: utf-8 -*-
"""
DINOv2 语义分割推理脚本
基于 dinov2_seg_pipeline.py，对指定图片做端到端语义分割并保存可视化结果。
"""
from __future__ import annotations

import os
import sys
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import logging
import mmcv
from mmcv.parallel import collate
from mmseg.datasets.pipelines import Compose
from mmcv.utils import get_logger

logger = get_logger('mmcv')
logger.setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['USE_XFORMERS'] = '0'

ROOT = Path(__file__).resolve().parents[1]
DINOV2_PATH = str(ROOT / "backbone" / "dinov2")
sys.path.insert(0, DINOV2_PATH)

import dinov2.eval.segmentation.models  # noqa: F401

from dinov2_seg_pipeline import (
    MODEL_REGISTRY, CROP_SIZE, STRIDE, PATCH_SIZE, NUM_CLASSES,
    VOC_CLASSES, CenterPadding, LoadImage, SegmentationHead,
    get_slide_crops, _load_backbone, _get_config_path,
)

DEFAULT_VOC_ROOT = os.path.join(str(ROOT), "data", "VOCdevkit", "VOC2012")
DEFAULT_WEIGHTS_ROOT = os.path.join(str(ROOT), "pretrained")

VOC_COLORMAP = np.array([
    [0, 0, 0],       [128, 0, 0],     [0, 128, 0],     [128, 128, 0],
    [0, 0, 128],     [128, 0, 128],   [0, 128, 128],   [128, 128, 128],
    [64, 0, 0],      [192, 0, 0],     [64, 128, 0],    [192, 128, 0],
    [64, 0, 128],    [192, 0, 128],   [64, 128, 128],  [192, 128, 128],
    [0, 64, 0],      [128, 64, 0],    [0, 192, 0],     [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)


def load_seg_head(weights_path, in_channels=1024, num_classes=21, device='cuda'):
    head = SegmentationHead(in_channels, num_classes)
    ckpt = torch.load(weights_path, map_location='cpu')
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    head_state = {}
    for k, v in ckpt.items():
        if k.startswith('decode_head.'):
            head_state[k.replace('decode_head.', '')] = v
    head.load_state_dict(head_state, strict=True)
    return head.to(device).eval()


@torch.no_grad()
def slide_inference_full(backbone, head, img_tensor, device):
    """完整的滑窗推理：backbone 全部层 + head"""
    h_img, w_img = img_tensor.shape[2], img_tensor.shape[3]
    crops = get_slide_crops(h_img, w_img, CROP_SIZE, STRIDE)
    center_pad = CenterPadding(PATCH_SIZE)

    preds = torch.zeros((1, NUM_CLASSES, h_img, w_img), device=device)
    count_mat = torch.zeros((1, 1, h_img, w_img), device=device)

    crop_imgs = []
    for y1, x1, y2, x2 in crops:
        crop_img = img_tensor[:, :, y1:y2, x1:x2]
        crop_imgs.append(center_pad(crop_img))

    batch_size = 8
    all_patch_tokens = []
    for start in range(0, len(crop_imgs), batch_size):
        batch = torch.cat(crop_imgs[start:start+batch_size], dim=0).to(device)
        feats = backbone.forward_features(batch)
        patch_tokens = feats["x_norm_patchtokens"]
        all_patch_tokens.append(patch_tokens.cpu())

    all_patch_tokens = torch.cat(all_patch_tokens, dim=0)

    for i, (y1, x1, y2, x2) in enumerate(crops):
        tokens = all_patch_tokens[i:i+1].to(device)

        actual_h, actual_w = y2 - y1, x2 - x1
        padded_h = math.ceil(actual_h / PATCH_SIZE) * PATCH_SIZE
        padded_w = math.ceil(actual_w / PATCH_SIZE) * PATCH_SIZE
        feat_h, feat_w = padded_h // PATCH_SIZE, padded_w // PATCH_SIZE

        tokens_2d = tokens.reshape(1, feat_h, feat_w, -1).permute(0, 3, 1, 2)
        logits = head(tokens_2d)
        logits_up = F.interpolate(logits, size=CROP_SIZE, mode='bilinear', align_corners=False)
        logits_crop = logits_up[:, :, :actual_h, :actual_w]

        preds[:, :, y1:y2, x1:x2] += logits_crop
        count_mat[:, :, y1:y2, x1:x2] += 1

    preds = preds / count_mat
    return preds


def overlay_segmentation(img_pil, seg_pred, alpha=0.55):
    """将分割结果叠加到原图上"""
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]

    color_mask = VOC_COLORMAP[seg_pred]
    fg_mask = seg_pred > 0

    result = img_np.copy()
    result[fg_mask] = (
        img_np[fg_mask] * (1 - alpha) + color_mask[fg_mask] * alpha
    ).astype(np.uint8)

    return Image.fromarray(result)


def main():
    ap = argparse.ArgumentParser("DINOv2 语义分割推理 + 可视化")
    ap.add_argument('--model', default='vitl14', choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument('--voc_root', default=DEFAULT_VOC_ROOT)
    ap.add_argument('--weights_root', default=DEFAULT_WEIGHTS_ROOT)
    ap.add_argument('--config', default=None)
    ap.add_argument('--img_dir', required=True, help='输入图片目录')
    ap.add_argument('--out_dir', required=True, help='输出可视化目录')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    config_path = _get_config_path(args)
    cfg = mmcv.Config.fromfile(config_path)
    cfg.data_root = args.voc_root
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    print("[1/3] 加载 DINOv2 backbone...")
    backbone, reg = _load_backbone(args)

    print("[2/3] 加载分割头...")
    head_ckpt = os.path.join(args.weights_root, reg["seg_head"])
    head = load_seg_head(head_ckpt, in_channels=reg["embed_dim"],
                         num_classes=NUM_CLASSES, device=device)
    print(f"  Head loaded: {head_ckpt}")

    img_files = sorted([f for f in os.listdir(args.img_dir) if f.endswith(('.jpg', '.png'))])
    print(f"[3/3] 推理 {len(img_files)} 张图片...")

    for fname in tqdm(img_files, desc="Inference"):
        img_path = os.path.join(args.img_dir, fname)
        img_pil = Image.open(img_path).convert('RGB')
        ori_h, ori_w = img_pil.size[1], img_pil.size[0]

        img_np = np.array(img_pil)[:, :, ::-1]
        data = dict(img=img_np)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        img_tensor = data['img'][0]

        preds = slide_inference_full(backbone, head, img_tensor, device)

        if preds.shape[2] != ori_h or preds.shape[3] != ori_w:
            preds = F.interpolate(preds, size=(ori_h, ori_w),
                                  mode='bilinear', align_corners=False)

        seg_pred = preds.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        result_img = overlay_segmentation(img_pil, seg_pred)
        out_name = Path(fname).stem + '.png'
        result_img.save(os.path.join(args.out_dir, out_name), quality=95)

    print(f"\n完成! 结果保存在 {args.out_dir}")


if __name__ == '__main__':
    main()
