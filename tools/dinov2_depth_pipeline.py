# -*- coding: utf-8 -*-
"""
DINOv2 NYU Depth V2 特征管线 (ViT-L/14)
支持 ViT-L/14 (24 blocks, dim=1024)。

功能：
  - extract：整图提取指定 block 的特征，保存 .npy [1, 1+N, D]
  - replay ：从特征重放，继续前向计算深度图，支持 MSE 及深度指标评估

依赖：
  本地 dinov2 源码路径（backbone/dinov2）
  conda activate featcodec2

用法：
# 提取
conda activate featcodec2
python tools/dinov2_depth_pipeline.py extract \
    --model vitl14 \
    --data_root /data4/workspace/zlt/LaMoFC/Data_example/dinov2/dpt/source/NYU_Test80 \
    --split /data4/workspace/zlt/featcodec/ORFC/utils/nyu_test_80.txt \
    --out_root features/nyu_depth_80/dinov2_vitl14 \
    --blocks 5,10,15,20

# 重放
python tools/dinov2_depth_pipeline.py replay \
    --model vitl14 \
    --data_root /data4/workspace/zlt/LaMoFC/Data_example/dinov2/dpt/source/NYU_Test80 \
    --split /data4/workspace/zlt/featcodec/ORFC/utils/nyu_test_80.txt \
    --feature_root features/nyu_depth_80/dinov2_vitl14 \
    --layer blk05 blk10 blk15 blk20
"""

from __future__ import annotations

import os
import sys
import math
import time
import glob
import argparse
import itertools
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['USE_XFORMERS'] = '0'

ROOT = Path(__file__).resolve().parents[1]
DINOV2_PATH = str(ROOT / "backbone" / "dinov2")
sys.path.insert(0, DINOV2_PATH)

# ========================= 配置 =========================
DEFAULT_WEIGHTS_ROOT = "/data4/workspace/zlt/cache/torch/hub/checkpoints"

MODEL_REGISTRY = {
    "vitl14": {
        "vit_fn": "vit_large",
        "vit_kwargs": dict(patch_size=14, img_size=518, init_values=1.0, block_chunks=0),
        "embed_dim": 1024,
        "num_blocks": 24,
        "pretrain": "dinov2_vitl14_pretrain.pth",
        "depth_head": "dinov2_vitl14_nyu_linear_head.pth",
    },
}

PATCH_SIZE = 14
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# NYU Depth V2 标准范围
MIN_DEPTH = 0.001
MAX_DEPTH = 10.0


# ========================= 工具函数 =========================

class CenterPadding(nn.Module):
    """将张量 pad 到 patch_size 整数倍（中心对齐）"""
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
        pads = list(itertools.chain.from_iterable(
            self._get_pad(m) for m in x.shape[:1:-1]
        ))
        return F.pad(x, pads)


def preprocess_image(img_path):
    """
    加载并预处理图像，返回 (img_tensor [1,3,H',W'], ori_shape (H,W), pad_shape (H',W'))
    H',W' 是 CenterPad 到 patch_size 整数倍后的尺寸
    """
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    ori_h, ori_w = img_np.shape[:2]

    img_norm = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()

    center_pad = CenterPadding(PATCH_SIZE)
    img_padded = center_pad(img_tensor)
    pad_h, pad_w = img_padded.shape[2], img_padded.shape[3]

    return img_padded, (ori_h, ori_w), (pad_h, pad_w)


def parse_split_file(split_path):
    """
    解析 NYU split 文件，每行格式: scene/rgb_XXXXX.jpg scene/sync_depth_XXXXX.png focal
    返回 list of (rgb_rel_path, depth_rel_path, focal)
    """
    samples = []
    with open(split_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rgb_rel = parts[0]
            depth_rel = parts[1] if len(parts) > 1 else None
            focal = float(parts[2]) if len(parts) > 2 else 518.8579
            samples.append((rgb_rel, depth_rel, focal))
    return samples


def load_depth_gt(depth_path):
    """
    加载 NYU 深度真值图 (16-bit PNG, 单位 mm → m)
    返回 [H, W] float32 深度图 (单位: 米)
    """
    depth_png = np.array(Image.open(depth_path), dtype=np.float32)
    depth_m = depth_png / 1000.0
    return depth_m


# ========================= 深度评估指标 =========================

def compute_depth_metrics(pred, gt, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH):
    """
    计算标准 NYU 深度评估指标
    pred, gt: [H, W] numpy array, 单位 m
    返回 dict: {a1, a2, a3, abs_rel, rmse, log10, rmse_log, silog, sq_rel}
    """
    valid = (gt > min_depth) & (gt < max_depth)
    pred_valid = pred[valid]
    gt_valid = gt[valid]

    if len(gt_valid) == 0:
        return {k: 0.0 for k in ['a1', 'a2', 'a3', 'abs_rel', 'rmse', 'log10', 'rmse_log', 'silog', 'sq_rel']}

    pred_valid = np.clip(pred_valid, min_depth, max_depth)

    thresh = np.maximum(gt_valid / pred_valid, pred_valid / gt_valid)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    sq_rel = np.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
    rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))

    log_pred = np.log(pred_valid)
    log_gt = np.log(gt_valid)
    rmse_log = np.sqrt(np.mean((log_gt - log_pred) ** 2))
    log10_err = np.mean(np.abs(np.log10(gt_valid) - np.log10(pred_valid)))

    diff_log = log_gt - log_pred
    silog = np.sqrt(np.mean(diff_log ** 2) - (np.mean(diff_log)) ** 2) * 100.0

    return {
        'a1': a1, 'a2': a2, 'a3': a3,
        'abs_rel': abs_rel, 'rmse': rmse, 'log10': log10_err,
        'rmse_log': rmse_log, 'silog': silog, 'sq_rel': sq_rel
    }


# ========================= 模型加载 =========================

def _load_backbone(args):
    """加载 DINOv2 backbone"""
    from dinov2.models import vision_transformer as vits
    reg = MODEL_REGISTRY[args.model]
    builder = getattr(vits, reg["vit_fn"])
    backbone = builder(**reg["vit_kwargs"])
    ckpt_path = os.path.join(args.weights_root, reg["pretrain"])
    backbone.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=False), strict=True)
    backbone = backbone.to(args.device).eval()
    print(f"  Backbone loaded: {ckpt_path}  ({reg['vit_fn']}, dim={reg['embed_dim']})")
    return backbone, reg


def _load_depth_head(args):
    """
    加载 BNHead 深度估计头
    layers=1: in_channels=[embed_dim], channels=embed_dim*2, in_index=[0]

    支持两种 checkpoint 格式:
    1. 官方格式: {'state_dict': {'decode_head.conv_depth.weight': [n_bins, C, 1, 1], ...}}
    2. 简化格式: {'weight': [n_bins, C], 'bias': [n_bins]} (Linear 格式，自动 reshape)
    """
    from dinov2.hub.depth import BNHead

    reg = MODEL_REGISTRY[args.model]
    embed_dim = reg["embed_dim"]
    channels = embed_dim * 2

    ckpt_path = os.path.join(args.weights_root, reg["depth_head"])
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']

    conv_w_key = next((k for k in ckpt if 'conv_depth.weight' in k), None)
    if conv_w_key:
        n_bins = ckpt[conv_w_key].shape[0]
    else:
        n_bins = ckpt['weight'].shape[0]

    head = BNHead(
        classify=True,
        n_bins=n_bins,
        bins_strategy="UD",
        norm_strategy="linear",
        upsample=4,
        in_channels=[embed_dim],
        in_index=[0],
        input_transform="resize_concat",
        channels=channels,
        align_corners=False,
        min_depth=MIN_DEPTH,
        max_depth=MAX_DEPTH,
        loss_decode=(),
    )

    if conv_w_key:
        head_state = {}
        for k, v in ckpt.items():
            new_k = k.replace('decode_head.', '') if k.startswith('decode_head.') else k
            head_state[new_k] = v
        head.load_state_dict(head_state, strict=True)
    else:
        w = ckpt['weight']
        if w.dim() == 2:
            w = w.reshape(n_bins, channels, 1, 1)
        head.conv_depth.weight.data = w
        head.conv_depth.bias.data = ckpt['bias']

    head = head.to(args.device).eval()
    print(f"  Depth head loaded: {ckpt_path}  (BNHead, n_bins={n_bins}, channels={channels})")
    return head


# ========================= 特征提取（整图） =========================

@torch.no_grad()
def encode_image(backbone, img_tensor, layer_idx, device):
    """
    整图提取指定 block 的特征（不做 norm）

    Returns:
        feature: [1, 1+N, D] tensor (CLS + patch tokens, unnormalized)
    """
    img_tensor = img_tensor.to(device)
    feats = backbone.get_intermediate_layers(
        img_tensor, n=[layer_idx], reshape=False, norm=False, return_class_token=True
    )
    patch_tokens, cls_token = feats[0]
    full_seq = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
    return full_seq


@torch.no_grad()
def decode_depth(backbone, head, feature, layer_idx, pad_shape, ori_shape, device, n_prefix=1, patch_size=None, rope=None):
    """
    从中间层特征重放，获取深度图

    DINOv2 深度估计的官方流程不对特征做 norm（与分割不同）：
      backbone.get_intermediate_layers(..., norm=False)
    因此 replay 时续传完剩余 blocks 后不施加 LayerNorm。

    Args:
        feature: [1, 1+N, D] (CLS + patches, unnormalized)
        layer_idx: 提取层索引
        pad_shape: (pad_h, pad_w) CenterPad 后尺寸
        ori_shape: (ori_h, ori_w) 原始图像尺寸

    Returns:
        depth_pred: [ori_h, ori_w] numpy array
    """
    num_blocks = len(backbone.blocks)
    feat = feature.to(device)

    x = feat
    for blk_idx in range(layer_idx + 1, num_blocks):
        if rope is not None:
            x = backbone.blocks[blk_idx](x, rope=rope)
            continue
        x = backbone.blocks[blk_idx](x)
    # 注意：深度估计不做 norm（官方 depthers.py 中 norm=False）

    cls_token = x[:, 0, :]
    patch_tokens = x[:, n_prefix:, :]

    pad_h, pad_w = pad_shape
    feat_h = pad_h // (patch_size or PATCH_SIZE)
    feat_w = pad_w // (patch_size or PATCH_SIZE)
    patch_map = patch_tokens.reshape(1, feat_h, feat_w, -1).permute(0, 3, 1, 2)

    head_input = [(patch_map, cls_token)]
    depth_out = head(head_input, img_metas=None)

    depth_out = torch.clamp(depth_out, min=MIN_DEPTH, max=MAX_DEPTH)

    ori_h, ori_w = ori_shape
    depth_out = F.interpolate(
        depth_out, size=(ori_h, ori_w), mode='bilinear', align_corners=False
    )

    return depth_out.squeeze().cpu().numpy()


# ========================= Extract 命令 =========================

@torch.no_grad()
def cmd_extract(args):
    """
    整图提取中间层特征

    特征格式: [1, 1+N, D]
    - 第0列是 CLS token，后续为 patch tokens
    - 不做 norm（replay 时续传后做 norm）
    """
    device = args.device
    data_root = args.data_root
    out_root = args.out_root
    layers = [int(x) for x in args.blocks.split(",")]

    os.makedirs(out_root, exist_ok=True)
    for k in layers:
        os.makedirs(os.path.join(out_root, f"blk{k:02d}"), exist_ok=True)

    print("[1/3] 加载 DINOv2 backbone...")
    backbone, reg = _load_backbone(args)

    print("[2/3] 解析数据列表...")
    samples = parse_split_file(args.split)
    print(f"  共 {len(samples)} 个样本, 数据目录: {data_root}")

    print(f"[3/3] 提取特征 (layers={layers}, norm=False, whole image)...")
    t0 = time.time()

    for rgb_rel, _, _ in tqdm(samples, desc="Extracting"):
        img_path = os.path.join(data_root, rgb_rel)
        if not os.path.isfile(img_path):
            print(f"[warn] missing image: {img_path}")
            continue

        img_tensor, ori_shape, pad_shape = preprocess_image(img_path)
        img_tensor = img_tensor.to(device)

        name = rgb_rel.replace('/', '_').rsplit('.', 1)[0]

        for k in layers:
            key = f"blk{k:02d}"
            feature = encode_image(backbone, img_tensor, k, device)
            feat_np = feature.cpu().numpy().astype(np.float32)

            save_path = os.path.join(out_root, key, f"{name}.npy")
            np.save(save_path, feat_np)

    elapsed = time.time() - t0
    print(f"\n[extract] Done. N={len(samples)}, layers={layers}, time={elapsed:.2f}s")


# ========================= Replay 命令 =========================

@torch.no_grad()
def cmd_replay(args):
    """
    从特征重放，计算深度图，评估指标

    特征格式: [1, 1+N, D]
    replay: 续传 blocks + norm + BNHead → depth
    """
    device = args.device
    data_root = args.data_root
    feature_root = args.feature_root
    layers = args.layer
    org_feature_root = args.org_feature_root

    print("[1/4] 加载 DINOv2 backbone...")
    backbone, reg = _load_backbone(args)

    print("[2/4] 加载深度估计头...")
    head = _load_depth_head(args)

    print("[3/4] 解析数据列表...")
    samples = parse_split_file(args.split)
    print(f"  共 {len(samples)} 个样本")

    all_meta = {layer: {} for layer in layers}
    for layer in layers:
        layer_dir = os.path.join(feature_root, layer)
        if os.path.isdir(layer_dir):
            npy_files = glob.glob(os.path.join(layer_dir, "*.npy"))
            for npy_file in npy_files:
                name = os.path.splitext(os.path.basename(npy_file))[0]
                all_meta[layer][name] = npy_file
        print(f"  {layer}: {len(all_meta[layer])} 个特征文件")

    def eval_single_layer(layer, meta_map):
        layer_idx = int(layer[-2:])
        all_metrics = []
        mse_list = []
        t0 = time.time()
        missing = 0

        for rgb_rel, depth_rel, _ in tqdm(samples, desc=f"Replay {layer}", leave=False):
            name = rgb_rel.replace('/', '_').rsplit('.', 1)[0]
            if name not in meta_map:
                missing += 1
                continue

            img_path = os.path.join(data_root, rgb_rel)
            _, ori_shape, pad_shape = preprocess_image(img_path)

            feat_path = meta_map[name]
            feature = np.load(feat_path)
            if feature.ndim == 2:
                feature = feature[np.newaxis, ...]
            feature_tensor = torch.from_numpy(feature).to(device)

            depth_pred = decode_depth(
                backbone, head, feature_tensor, layer_idx, pad_shape, ori_shape, device
            )

            if depth_rel is not None:
                gt_path = os.path.join(data_root, depth_rel)
                if os.path.isfile(gt_path):
                    depth_gt = load_depth_gt(gt_path)
                    metrics = compute_depth_metrics(depth_pred, depth_gt)
                    all_metrics.append(metrics)

            if org_feature_root:
                org_feat_path = os.path.join(org_feature_root, layer, f"{name}.npy")
                if os.path.exists(org_feat_path):
                    org_feat = np.load(org_feat_path)
                    mse = np.mean((org_feat - feature) ** 2)
                    mse_list.append(mse)

        elapsed = time.time() - t0
        return all_metrics, mse_list, missing, elapsed

    print("[4/4] 评估中...")
    results = {}
    for layer in layers:
        meta_map = all_meta[layer]
        if not meta_map:
            print(f"[warn] {layer}: 无特征文件，跳过")
            continue
        all_metrics, mse_list, missing, elapsed = eval_single_layer(layer, meta_map)
        results[layer] = {
            'metrics': all_metrics,
            'mse': np.mean(mse_list) if mse_list else None,
            'missing': missing,
            'time': elapsed
        }

    print("\n" + "=" * 90)
    print("NYU Depth V2 评估汇总")
    print("=" * 90)
    header = f"{'Layer':<8} {'RMSE':>8} {'AbsRel':>8} {'log10':>8} {'δ<1.25':>8} {'δ<1.25²':>8} {'δ<1.25³':>8}"
    if org_feature_root:
        header += f" {'FeatMSE':>12}"
    header += f" {'Time':>8}"
    print(header)
    print("-" * 90)

    for layer in layers:
        if layer not in results:
            continue
        res = results[layer]
        if not res['metrics']:
            print(f"{layer:<8} {'(no GT)':>8}")
            continue

        avg = {k: np.mean([m[k] for m in res['metrics']]) for k in res['metrics'][0]}
        line = (
            f"{layer:<8} "
            f"{avg['rmse']:>8.4f} "
            f"{avg['abs_rel']:>8.4f} "
            f"{avg['log10']:>8.4f} "
            f"{avg['a1']:>8.4f} "
            f"{avg['a2']:>8.4f} "
            f"{avg['a3']:>8.4f}"
        )
        if org_feature_root and res['mse'] is not None:
            line += f" {res['mse']:>12.8f}"
        line += f" {res['time']:>7.2f}s"
        print(line)

    print("-" * 90)
    print("完成!")


# ========================= Compare 命令 =========================

@torch.no_grad()
def direct_inference(backbone, head, img_tensor, pad_shape, ori_shape, device,
                     patch_size=None):
    """
    端到端直接推理深度图（无中间特征提取/保存环节）
    """
    img_tensor = img_tensor.to(device)
    num_blocks = len(backbone.blocks)
    feats = backbone.get_intermediate_layers(
        img_tensor, n=[num_blocks - 1], reshape=False, norm=False, return_class_token=True
    )
    patch_tokens, cls_token = feats[0]

    pad_h, pad_w = pad_shape
    feat_h = pad_h // (patch_size or PATCH_SIZE)
    feat_w = pad_w // (patch_size or PATCH_SIZE)
    patch_map = patch_tokens.reshape(1, feat_h, feat_w, -1).permute(0, 3, 1, 2)

    head_input = [(patch_map, cls_token)]
    depth_out = head(head_input, img_metas=None)
    depth_out = torch.clamp(depth_out, min=MIN_DEPTH, max=MAX_DEPTH)

    ori_h, ori_w = ori_shape
    depth_out = F.interpolate(
        depth_out, size=(ori_h, ori_w), mode='bilinear', align_corners=False
    )
    return depth_out.squeeze().cpu().numpy()


@torch.no_grad()
def cmd_compare(args):
    """
    对比 direct inference vs feature replay:
      - 两者都在同一批样本上跑
      - 对比深度图的 pixel-wise 差异 (MSE, MAE, MaxErr)
      - 对比评估指标是否一致
    """
    device = args.device
    data_root = args.data_root
    feature_root = args.feature_root
    layers = args.layer

    print("[1/4] 加载 DINOv2 backbone...")
    backbone, reg = _load_backbone(args)

    print("[2/4] 加载深度估计头...")
    head = _load_depth_head(args)

    print("[3/4] 解析数据列表...")
    samples = parse_split_file(args.split)
    print(f"  共 {len(samples)} 个样本")

    all_meta = {layer: {} for layer in layers}
    for layer in layers:
        layer_dir = os.path.join(feature_root, layer)
        if os.path.isdir(layer_dir):
            npy_files = glob.glob(os.path.join(layer_dir, "*.npy"))
            for npy_file in npy_files:
                name = os.path.splitext(os.path.basename(npy_file))[0]
                all_meta[layer][name] = npy_file
        print(f"  {layer}: {len(all_meta[layer])} 个特征文件")

    print("[4/4] 对比评估中...")
    print("  - Direct: image → backbone(all blocks) → head → depth")
    print("  - Replay: loaded feature → backbone(remaining blocks) → head → depth\n")

    results = {}
    for layer in layers:
        layer_idx = int(layer[-2:])
        meta_map = all_meta[layer]
        if not meta_map:
            print(f"[warn] {layer}: 无特征文件，跳过")
            continue

        direct_metrics_all = []
        replay_metrics_all = []
        diff_mse_list = []
        diff_mae_list = []
        diff_max_list = []

        for rgb_rel, depth_rel, _ in tqdm(samples, desc=f"Compare {layer}"):
            name = rgb_rel.replace('/', '_').rsplit('.', 1)[0]
            if name not in meta_map:
                continue

            img_path = os.path.join(data_root, rgb_rel)
            if not os.path.isfile(img_path):
                continue

            img_tensor, ori_shape, pad_shape = preprocess_image(img_path)

            depth_direct = direct_inference(
                backbone, head, img_tensor, pad_shape, ori_shape, device
            )

            feat_path = meta_map[name]
            feature = np.load(feat_path)
            if feature.ndim == 2:
                feature = feature[np.newaxis, ...]
            feature_tensor = torch.from_numpy(feature).to(device)
            depth_replay = decode_depth(
                backbone, head, feature_tensor, layer_idx, pad_shape, ori_shape, device
            )

            diff = depth_direct - depth_replay
            diff_mse_list.append(np.mean(diff ** 2))
            diff_mae_list.append(np.mean(np.abs(diff)))
            diff_max_list.append(np.max(np.abs(diff)))

            if depth_rel is not None:
                gt_path = os.path.join(data_root, depth_rel)
                if os.path.isfile(gt_path):
                    depth_gt = load_depth_gt(gt_path)
                    direct_metrics_all.append(compute_depth_metrics(depth_direct, depth_gt))
                    replay_metrics_all.append(compute_depth_metrics(depth_replay, depth_gt))

        results[layer] = {
            'direct_metrics': direct_metrics_all,
            'replay_metrics': replay_metrics_all,
            'diff_mse': diff_mse_list,
            'diff_mae': diff_mae_list,
            'diff_max': diff_max_list,
        }

    print("\n" + "=" * 100)
    print("Direct vs Replay 对比结果")
    print("=" * 100)

    for layer in layers:
        if layer not in results:
            continue
        res = results[layer]
        n = len(res['diff_mse'])
        if n == 0:
            continue

        print(f"\n{'─' * 100}")
        print(f"Layer: {layer}  ({n} samples)")
        print(f"{'─' * 100}")

        avg_mse = np.mean(res['diff_mse'])
        avg_mae = np.mean(res['diff_mae'])
        avg_max = np.mean(res['diff_max'])
        max_max = np.max(res['diff_max'])
        print(f"\n  深度图像素差异 (Direct - Replay):")
        print(f"    Mean MSE:    {avg_mse:.2e}")
        print(f"    Mean MAE:    {avg_mae:.2e}")
        print(f"    Mean MaxErr: {avg_max:.2e}")
        print(f"    Max  MaxErr: {max_max:.2e}")

        if res['direct_metrics'] and res['replay_metrics']:
            metric_keys = ['rmse', 'abs_rel', 'log10', 'a1', 'a2', 'a3']
            print(f"\n  {'Metric':<10} {'Direct':>10} {'Replay':>10} {'Diff':>12} {'Match?':>8}")
            print(f"  {'-' * 52}")
            for k in metric_keys:
                d_val = np.mean([m[k] for m in res['direct_metrics']])
                r_val = np.mean([m[k] for m in res['replay_metrics']])
                diff = d_val - r_val
                match = "✓" if abs(diff) < 1e-6 else ("≈" if abs(diff) < 1e-4 else "✗")
                print(f"  {k:<10} {d_val:>10.6f} {r_val:>10.6f} {diff:>12.2e} {match:>8}")

    print(f"\n{'=' * 100}")
    print("完成！如果 Diff 全为零/极小 (< 1e-6)，则特征重放与直接推理完全等价。")
    print("=" * 100)


# ========================= CLI =========================

def build_parser():
    ap = argparse.ArgumentParser("DINOv2 NYU Depth V2: 特征提取/重放 (whole image inference)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="整图提取中间层特征到 .npy")
    pe.add_argument('--model', default='vitl14', choices=list(MODEL_REGISTRY.keys()))
    pe.add_argument('--data_root', required=True, help='NYU 数据目录 (含 scene/rgb_*.jpg)')
    pe.add_argument('--split', required=True, help='数据列表文件 (nyu_test_80.txt)')
    pe.add_argument('--weights_root', default=DEFAULT_WEIGHTS_ROOT, help='权重目录')
    pe.add_argument('--out_root', required=True, help='特征输出目录')
    pe.add_argument('--blocks', default='5,10,15,20', help='0-based 块索引，逗号分隔')
    pe.add_argument('--device', default='cuda', help='设备 (cuda / cuda:0 / cpu)')

    pr = sub.add_parser("replay", help='从特征重放，计算深度指标')
    pr.add_argument('--model', default='vitl14', choices=list(MODEL_REGISTRY.keys()))
    pr.add_argument('--data_root', required=True, help='NYU 数据目录')
    pr.add_argument('--split', required=True, help='数据列表文件')
    pr.add_argument('--weights_root', default=DEFAULT_WEIGHTS_ROOT, help='权重目录')
    pr.add_argument('--feature_root', required=True, help='特征目录')
    pr.add_argument('--layer', required=True, nargs='+', help='层名列表 (如 blk05 blk10)')
    pr.add_argument('--org_feature_root', default=None, help='原始特征目录（用于 MSE 对比）')
    pr.add_argument('--device', default='cuda', help='设备 (cuda / cuda:0 / cpu)')

    pc = sub.add_parser("compare", help='对比直接推理与特征重放的深度估计结果')
    pc.add_argument('--model', default='vitl14', choices=list(MODEL_REGISTRY.keys()))
    pc.add_argument('--data_root', required=True, help='NYU 数据目录')
    pc.add_argument('--split', required=True, help='数据列表文件')
    pc.add_argument('--weights_root', default=DEFAULT_WEIGHTS_ROOT, help='权重目录')
    pc.add_argument('--feature_root', required=True, help='已提取的特征目录')
    pc.add_argument('--layer', required=True, nargs='+', help='层名列表 (如 blk05 blk20 blk23)')
    pc.add_argument('--device', default='cuda', help='设备 (cuda / cuda:0 / cpu)')

    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    if args.cmd == 'extract':
        cmd_extract(args)
    elif args.cmd == 'replay':
        cmd_replay(args)
    elif args.cmd == 'compare':
        cmd_compare(args)


if __name__ == '__main__':
    main()
