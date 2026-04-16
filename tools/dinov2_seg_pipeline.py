# -*- coding: utf-8 -*-
"""
DINOv2 语义分割特征管线（VOC2012）
支持 ViT-L/14 (24 blocks, dim=1024) 和 ViT-G/14 (40 blocks, dim=1536)。

功能：
  - extract：滑窗提取指定block的特征，保存 .npy [num_slides, 1+N, D]
  - replay ：从特征重放，滑窗融合计算mIoU，支持MSE计算

依赖：
  本地 dinov2 源码路径（backbone/dinov2）
  mmseg (conda activate featcodec2)
用法：
# 提取（ViT-G/14 滑窗模式）
export TORCH_HOME=$PROJECT_ROOT/pretrained
conda activate featcodec2
python tools/dinov2_seg_pipeline.py extract \
    --model vitg14 \
    --out_root features/voc2012_5000/dinov2_vitg14 \
    --blocks 9,19,29,39 \
    --image_list utils/voc2012_all_5000.txt

# 重放
python tools/dinov2_seg_pipeline.py replay \
    --model vitg14 \
    --feature_root features/voc2012_100/dinov2_vitg14 \
    --layer blk09 blk19 blk29 blk39 \
    --image_list utils/voc2012_val_100.txt
"""

from __future__ import annotations

import os
import sys
import math
import time
import glob
import argparse
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import logging
import mmcv
from mmcv.parallel import collate, scatter
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

import dinov2.eval.segmentation.models

# ========================= 配置 =========================
DEFAULT_VOC_ROOT = os.path.join(str(ROOT), "data", "VOCdevkit", "VOC2012")
DEFAULT_WEIGHTS_ROOT = os.path.join(str(ROOT), "pretrained")

MODEL_REGISTRY = {
    "vitl14": {
        "vit_fn": "vit_large",
        "vit_kwargs": dict(patch_size=14, img_size=518, init_values=1.0, block_chunks=0),
        "embed_dim": 1024,
        "pretrain": "dinov2_vitl14_pretrain.pth",
        "seg_head": "dinov2_vitl14_voc2012_linear_head.pth",
        "default_config": str(ROOT / "utils" / "dinov2_vitl14_voc2012_linear_config.py"),
    },
    "vitg14": {
        "vit_fn": "vit_giant2",
        "vit_kwargs": dict(patch_size=14, img_size=518, init_values=1.0, block_chunks=0, ffn_layer="swiglufused"),
        "embed_dim": 1536,
        "pretrain": "dinov2_vitg14_pretrain.pth",
        "seg_head": "dinov2_vitg14_voc2012_linear_head.pth",
        "default_config": str(ROOT / "utils" / "dinov2_vitg14_voc2012_linear_config.py"),
    },
}

CROP_SIZE = (512, 512)
STRIDE = (341, 341)
PATCH_SIZE = 14

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
NUM_CLASSES = 21
IGNORE_INDEX = 255


# ========================= 工具函数 =========================

class CenterPadding(torch.nn.Module):
    """将图像pad到patch_size的整数倍（中心对齐）"""
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        import itertools
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


class LoadImage:
    """mmseg兼容的图像加载pipeline"""
    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def load_val_list(voc_root):
    """读取验证集列表"""
    val_txt = os.path.join(voc_root, 'ImageSets/Segmentation/val.txt')
    with open(val_txt, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def fast_hist(label, prediction, n):
    """计算混淆矩阵"""
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(int) + prediction[k].astype(int), minlength=n * n).reshape(n, n)


def per_class_miou(hist):
    """计算每类IoU"""
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    return iou


def _load_backbone(args):
    """根据 args.model 加载 DINOv2 backbone，返回 (backbone, reg)"""
    from dinov2.models import vision_transformer as vits
    reg = MODEL_REGISTRY[args.model]
    builder = getattr(vits, reg["vit_fn"])
    backbone = builder(**reg["vit_kwargs"])
    ckpt_path = os.path.join(args.weights_root, reg["pretrain"])
    backbone.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)
    backbone = backbone.to(args.device).eval()
    print(f"  Backbone loaded: {ckpt_path}  ({reg['vit_fn']}, dim={reg['embed_dim']})")
    return backbone, reg


def _get_config_path(args):
    """返回 mmseg config 路径，优先使用 --config 显式指定的，否则从 registry 取默认值"""
    if args.config is not None:
        return args.config
    return MODEL_REGISTRY[args.model]["default_config"]


# ========================= Slide Inference =========================

def get_slide_crops(h_img, w_img, crop_size, stride):
    """
    计算滑窗裁剪区域
    返回: list of (y1, x1, y2, x2)
    """
    h_crop, w_crop = crop_size
    h_stride, w_stride = stride

    crops = []
    for h_idx in range(0, max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1):
        for w_idx in range(0, max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crops.append((y1, x1, y2, x2))
    return crops


def slide_inference_encode(backbone, img, layer_idx, crop_size, stride, patch_size, batch_size=8):
    """
    滑窗提取特征（不做norm）- 批量推理版本

    Returns:
        feature_list: list of [1, 1+N, D] tensors，每个crop一个
        crops: list of (y1, x1, y2, x2)
        img_shape: (h_img, w_img)
    """
    h_img, w_img = img.shape[2], img.shape[3]
    h_crop, w_crop = crop_size

    crops = get_slide_crops(h_img, w_img, crop_size, stride)
    center_pad = CenterPadding(patch_size)

    crop_imgs = []
    for y1, x1, y2, x2 in crops:
        crop_img = img[:, :, y1:y2, x1:x2]
        crop_padded = center_pad(crop_img)
        crop_imgs.append(crop_padded)

    feature_list = []
    num_crops = len(crop_imgs)

    for start_idx in range(0, num_crops, batch_size):
        end_idx = min(start_idx + batch_size, num_crops)
        batch_crops = crop_imgs[start_idx:end_idx]
        batch_tensor = torch.cat(batch_crops, dim=0)

        feats = backbone.get_intermediate_layers(
            batch_tensor, n=[layer_idx], reshape=False, norm=False, return_class_token=True
        )
        patch_tokens, cls_token = feats[0]
        full_seq = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)

        for i in range(full_seq.shape[0]):
            feature_list.append(full_seq[i:i+1])

    return feature_list, crops, (h_img, w_img)


def slide_inference_decode(backbone, head, feature_list, crops, img_shape, layer_idx, patch_size, device, ori_shape=None):
    """
    滑窗解码并融合

    Returns:
        seg_pred: [H, W] numpy array（原图尺寸）
    """
    h_img, w_img = img_shape
    h_crop, w_crop = CROP_SIZE
    num_blocks = len(backbone.blocks)

    preds = torch.zeros((1, NUM_CLASSES, h_img, w_img), device=device)
    count_mat = torch.zeros((1, 1, h_img, w_img), device=device)

    for i, (y1, x1, y2, x2) in enumerate(crops):
        feat = feature_list[i]
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat).to(device)
        else:
            feat = feat.to(device)

        x = feat
        for blk_idx in range(layer_idx + 1, num_blocks):
            x = backbone.blocks[blk_idx](x)
        x = backbone.norm(x)

        patch_tokens = x[:, 1:, :]

        actual_h = y2 - y1
        actual_w = x2 - x1
        padded_h = math.ceil(actual_h / patch_size) * patch_size
        padded_w = math.ceil(actual_w / patch_size) * patch_size
        feat_h = padded_h // patch_size
        feat_w = padded_w // patch_size

        patch_tokens = patch_tokens.reshape(1, feat_h, feat_w, -1).permute(0, 3, 1, 2)
        logits = head(patch_tokens)
        logits_up = F.interpolate(logits, size=(h_crop, w_crop), mode='bilinear', align_corners=False)
        logits_crop = logits_up[:, :, :actual_h, :actual_w]

        preds[:, :, y1:y2, x1:x2] += logits_crop
        count_mat[:, :, y1:y2, x1:x2] += 1

    assert (count_mat == 0).sum() == 0, "Zero count in count matrix"
    preds = preds / count_mat

    if ori_shape is not None and (ori_shape[0] != h_img or ori_shape[1] != w_img):
        preds = F.interpolate(preds, size=ori_shape, mode='bilinear', align_corners=False)

    seg_pred = preds.argmax(dim=1).squeeze(0).cpu().numpy()
    return seg_pred


# ========================= 分割头 =========================

class SegmentationHead(nn.Module):
    """BN + 1x1 Conv 分割头（与官方VOC2012 linear head一致）"""
    def __init__(self, in_channels=1024, num_classes=21):
        super().__init__()
        self.bn = nn.SyncBatchNorm(in_channels)
        self.conv_seg = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv_seg(x)
        return x


def load_seg_head(weights_path, in_channels=1024, num_classes=21, device='cuda'):
    """加载分割头权重"""
    head = SegmentationHead(in_channels, num_classes)
    ckpt = torch.load(weights_path, map_location='cpu')
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']

    head_state = {}
    for k, v in ckpt.items():
        if k.startswith('decode_head.'):
            new_k = k.replace('decode_head.', '')
            head_state[new_k] = v

    head.load_state_dict(head_state, strict=True)
    return head.to(device).eval()


# ========================= Extract命令 =========================

@torch.no_grad()
def cmd_extract(args):
    """
    滑窗提取中间层特征

    特征格式: [num_slides, 1+N, D]
    - 第0列是CLS token
    - 所有层都不做norm（与seg.py一致）
    """
    device = args.device
    voc_root = args.voc_root
    out_root = args.out_root
    layers = [int(x) for x in args.blocks.split(",")]
    crop_size = CROP_SIZE
    stride = STRIDE
    patch_size = PATCH_SIZE

    os.makedirs(out_root, exist_ok=True)
    for k in layers:
        os.makedirs(os.path.join(out_root, f"blk{k:02d}"), exist_ok=True)

    print("[1/4] 加载配置...")
    config_path = _get_config_path(args)
    cfg = mmcv.Config.fromfile(config_path)
    cfg.data_root = voc_root

    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    print("[2/4] 加载DINOv2 backbone...")
    backbone, reg = _load_backbone(args)

    if args.image_list:
        with open(args.image_list, 'r') as f:
            val_list = [line.strip() for line in f if line.strip()]
        print(f"[3/4] 从 {args.image_list} 读取 {len(val_list)} 个样本")
    else:
        val_list = load_val_list(voc_root)
        print(f"[3/4] 验证集样本数: {len(val_list)}")

    print(f"[4/4] 提取特征 (slide: crop={crop_size}, stride={stride}, batch_size={args.batch_size})...")
    print(f"  layers={layers}, norm=False")
    t0 = time.time()

    for name in tqdm(val_list, desc="Extracting"):
        img_path = os.path.join(voc_root, 'JPEGImages', f'{name}.jpg')
        if not os.path.isfile(img_path):
            print(f"[warn] missing image: {img_path}")
            continue

        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)[:, :, ::-1]

        data = dict(img=img_np)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)

        img_tensor = data['img'][0].to(device)

        for k in layers:
            key = f"blk{k:02d}"

            feature_list, crops, img_shape = slide_inference_encode(
                backbone, img_tensor, k, crop_size, stride, patch_size, args.batch_size
            )

            features = torch.cat(feature_list, dim=0).cpu().numpy().astype(np.float32)

            layer_dir = os.path.join(out_root, key)
            save_path = os.path.join(layer_dir, f"{name}.npy")
            np.save(save_path, features)

    print(f"\n[extract] Done. N={len(val_list)} layers={layers} ({time.time()-t0:.2f}s)")


# ========================= Replay命令 =========================

@torch.no_grad()
def cmd_replay(args):
    """
    从特征重放，计算mIoU

    特征格式: [num_slides, 1+N, D]
    replay时继续前向 + norm
    """
    device = args.device
    voc_root = args.voc_root
    feature_root = args.feature_root
    layers = args.layer
    org_feature_root = args.org_feature_root
    patch_size = PATCH_SIZE

    print("[1/5] 加载配置...")
    config_path = _get_config_path(args)
    cfg = mmcv.Config.fromfile(config_path)
    cfg.data_root = voc_root

    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    reg = MODEL_REGISTRY[args.model]

    print("[2/5] 加载分割头...")
    head_ckpt = os.path.join(args.weights_root, reg["seg_head"])
    head = load_seg_head(head_ckpt, in_channels=reg["embed_dim"], num_classes=NUM_CLASSES, device=device)
    print(f"  Head loaded: {head_ckpt}  (in_channels={reg['embed_dim']})")

    print("[3/5] 加载backbone...")
    backbone, _ = _load_backbone(args)

    all_meta = {layer: {} for layer in layers}
    for layer in layers:
        layer_dir = os.path.join(feature_root, layer)
        if os.path.isdir(layer_dir):
            npy_files = glob.glob(os.path.join(layer_dir, "*.npy"))
            for npy_file in npy_files:
                name = os.path.splitext(os.path.basename(npy_file))[0]
                all_meta[layer][name] = {"id": name, "path": npy_file}

    if args.image_list:
        with open(args.image_list, 'r') as f:
            val_list = [line.strip() for line in f if line.strip()]
        print(f"[4/5] 从 {args.image_list} 读取 {len(val_list)} 个样本")
    else:
        val_list = load_val_list(voc_root)
        print(f"[4/5] 验证集样本数: {len(val_list)}")

    for layer in layers:
        print(f"  {layer}: {len(all_meta[layer])} 个特征")

    def eval_single_layer(layer, meta_map):
        layer_idx = int(layer[-2:])
        hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        mse_list = []
        t0 = time.time()
        missing = 0

        for name in tqdm(val_list, desc=f"Replay {layer}", leave=False):
            if name not in meta_map:
                missing += 1
                continue

            img_path = os.path.join(voc_root, 'JPEGImages', f'{name}.jpg')
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)[:, :, ::-1]

            data = dict(img=img_np)
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)

            img_tensor = data['img'][0]
            h_img, w_img = img_tensor.shape[2], img_tensor.shape[3]

            crops = get_slide_crops(h_img, w_img, CROP_SIZE, STRIDE)

            gt_path = os.path.join(voc_root, 'SegmentationClass', f'{name}.png')
            gt = np.array(Image.open(gt_path))
            ori_h, ori_w = gt.shape[:2]

            feat_path = meta_map[name]['path']
            features = np.load(feat_path)

            feature_list = [features[i:i+1] for i in range(features.shape[0])]

            seg_pred = slide_inference_decode(
                backbone, head, feature_list, crops, (h_img, w_img),
                layer_idx, patch_size, device, ori_shape=(ori_h, ori_w)
            )

            mask = gt != IGNORE_INDEX
            hist += fast_hist(gt[mask], seg_pred[mask], NUM_CLASSES)

            if org_feature_root:
                org_layer_dir = os.path.join(org_feature_root, layer)
                org_feat_path = os.path.join(org_layer_dir, f"{name}.npy")
                if os.path.exists(org_feat_path):
                    org_feat = np.load(org_feat_path)
                    mse = np.mean((org_feat - features) ** 2)
                    mse_list.append(mse)

        all_iou = per_class_miou(hist)
        miou = np.nanmean(all_iou)
        acc = np.diag(hist).sum() / (hist.sum() + 1e-10)
        elapsed = time.time() - t0

        return miou, acc, all_iou, mse_list, missing, elapsed

    print("[5/5] 评估中...")
    results = {}
    for layer in layers:
        meta_map = all_meta[layer]
        if not meta_map:
            print(f"[warn] {layer}: 无特征，跳过")
            continue
        miou, acc, class_iou, mse_list, missing, elapsed = eval_single_layer(layer, meta_map)
        results[layer] = {
            'miou': miou,
            'acc': acc,
            'class_iou': class_iou,
            'mse': np.mean(mse_list) if mse_list else None,
            'missing': missing,
            'time': elapsed
        }

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    header = f"{'Layer':<10} {'mIoU':>10} {'Acc':>10}"
    if org_feature_root:
        header += f" {'MSE':>12}"
    header += f" {'Time':>10}"
    print(header)
    print("-" * 70)

    for layer in layers:
        if layer in results:
            res = results[layer]
            line = f"{layer:<10} {res['miou']*100:>9.2f}% {res['acc']*100:>9.2f}%"
            if org_feature_root and res['mse'] is not None:
                line += f" {res['mse']:>12.8f}"
            line += f" {res['time']:>9.2f}s"
            print(line)
    print("-" * 70)
    print("\n完成!")


# ========================= CLI =========================

def build_parser():
    ap = argparse.ArgumentParser("DINOv2 VOC2012分割：特征提取/重放 (slide inference)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="滑窗提取中间层特征到.npy")
    pe.add_argument('--model', default='vitl14', choices=list(MODEL_REGISTRY.keys()),
                    help='模型变体 (default: vitl14)')
    pe.add_argument('--voc_root', default=DEFAULT_VOC_ROOT, help='VOC2012根目录')
    pe.add_argument('--weights_root', default=DEFAULT_WEIGHTS_ROOT, help='权重目录')
    pe.add_argument('--config', default=None, help='mmseg配置文件（不指定则根据 --model 自动选择）')
    pe.add_argument('--out_root', required=True, help='特征输出目录')
    pe.add_argument('--blocks', default='5,11,17,23', help='0-based块索引，逗号分隔')
    pe.add_argument('--image_list', default=None, help='图片名列表文件')
    pe.add_argument('--batch_size', type=int, default=64, help='批量推理的batch大小（控制显存）')
    pe.add_argument('--device', default='cuda')

    pr = sub.add_parser("replay", help='从特征重放，计算mIoU')
    pr.add_argument('--model', default='vitl14', choices=list(MODEL_REGISTRY.keys()),
                    help='模型变体 (default: vitl14)')
    pr.add_argument('--voc_root', default=DEFAULT_VOC_ROOT, help='VOC2012根目录')
    pr.add_argument('--weights_root', default=DEFAULT_WEIGHTS_ROOT, help='权重目录')
    pr.add_argument('--config', default=None, help='mmseg配置文件（不指定则根据 --model 自动选择）')
    pr.add_argument('--feature_root', required=True, help='特征目录')
    pr.add_argument('--layer', required=True, nargs='+', help='层名列表')
    pr.add_argument('--org_feature_root', default=None, help='原始特征目录（用于MSE）')
    pr.add_argument('--image_list', default=None, help='图片名列表文件')
    pr.add_argument('--device', default='cuda')

    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    if args.cmd == 'extract':
        cmd_extract(args)
    elif args.cmd == 'replay':
        cmd_replay(args)


if __name__ == '__main__':
    main()
