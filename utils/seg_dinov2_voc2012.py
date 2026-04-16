# -*- coding: utf-8 -*-
"""
DINOv2 VOC2012 语义分割评估脚本（mmseg官方方式）

用法: 
    conda activate featcodec2
    python utils/seg_dinov2_voc2012.py

参考: backbone/dinov2/notebooks/semantic_segmentation.ipynb
"""
import os
import sys
import math
import argparse
import itertools
from pathlib import Path
from functools import partial

import torch
import torch.nn.functional as F
import mmcv
from mmcv.parallel import MMDataParallel
from mmseg.apis import init_segmentor, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset

# 添加本地dinov2源码路径
ROOT = Path(__file__).resolve().parents[1]
DINOV2_PATH = str(ROOT / "backbone" / "dinov2")
sys.path.insert(0, DINOV2_PATH)

# 注册mmseg组件 (BNHead, DinoVisionTransformer)
import dinov2.eval.segmentation.models

# ========================= 默认配置 =========================
DEFAULT_VOC_ROOT = os.path.join(str(ROOT), "data", "VOCdevkit", "VOC2012")
DEFAULT_WEIGHTS_ROOT = os.path.join(str(ROOT), "pretrained")
CONFIG_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_voc2012_linear_config.py"


# ========================= 工具函数 =========================
class CenterPadding(torch.nn.Module):
    """将输入pad到patch_size的整数倍（官方实现）"""
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
            self._get_pad(m) for m in x.shape[:1:-1]))
        return F.pad(x, pads)


def load_config_from_url(url: str) -> str:
    """从URL下载配置文件内容"""
    import urllib.request
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


def load_config_from_file(path: str) -> str:
    """从本地文件加载配置"""
    with open(path, 'r') as f:
        return f.read()


def create_segmenter(cfg, backbone_model):
    """
    创建分割模型，替换backbone为真正的DINOv2
    
    这是官方notebook中的核心技巧：
    1. init_segmentor创建mmseg模型（backbone是空壳DinoVisionTransformer）
    2. 替换backbone.forward为真正的dinov2.get_intermediate_layers
    3. 注册pre_hook做center padding
    """
    model = init_segmentor(cfg)
    model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    if hasattr(backbone_model, "patch_size"):
        model.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
    model.init_weights()
    return model


# ========================= 主函数 =========================
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="DINOv2 VOC2012 Segmentation Evaluation")
    parser.add_argument('--voc_root', type=str, default=DEFAULT_VOC_ROOT,
                        help='VOC2012数据集根目录')
    parser.add_argument('--weights_root', type=str, default=DEFAULT_WEIGHTS_ROOT,
                        help='权重文件目录')
    parser.add_argument('--config', type=str, default=None,
                        help='本地配置文件路径（默认从URL下载）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--workers', type=int, default=4,
                        help='dataloader workers')
    args = parser.parse_args()

    backbone_ckpt = os.path.join(args.weights_root, "dinov2_vitl14_pretrain.pth")
    head_ckpt = os.path.join(args.weights_root, "dinov2_vitl14_voc2012_linear_head.pth")

    # 检查权重文件
    for ckpt in [backbone_ckpt, head_ckpt]:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"权重文件不存在: {ckpt}")

    print("=" * 60)
    print("DINOv2-L/14 VOC2012 语义分割评估")
    print("=" * 60)
    print(f"VOC Root: {args.voc_root}")
    print(f"Backbone: {backbone_ckpt}")
    print(f"Head: {head_ckpt}")
    print("=" * 60)

    # 1. 加载并修改配置
    print("\n[1/6] 加载配置...")
    if args.config and os.path.exists(args.config):
        cfg_str = load_config_from_file(args.config)
        print(f"  从本地加载: {args.config}")
    else:
        cfg_str = load_config_from_url(CONFIG_URL)
        print(f"  从URL下载: {CONFIG_URL}")
    
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
    
    # 修改数据路径
    cfg.data_root = args.voc_root
    cfg.data.test.data_root = args.voc_root
    cfg.data.val.data_root = args.voc_root
    if hasattr(cfg.data, 'train'):
        cfg.data.train.data_root = args.voc_root
    print(f"  数据路径已修改为: {args.voc_root}")

    # 2. 加载DINOv2 backbone
    print("\n[2/6] 加载DINOv2 backbone...")
    from dinov2.models import vision_transformer as vits
    backbone_model = vits.vit_large(
        patch_size=14, 
        img_size=518, 
        init_values=1.0, 
        block_chunks=0
    )
    state_dict = torch.load(backbone_ckpt, map_location="cpu")
    backbone_model.load_state_dict(state_dict, strict=True)
    backbone_model = backbone_model.to(args.device).eval()
    print(f"  Backbone加载完成: vit_large (patch_size=14, embed_dim=1024)")

    # 3. 创建分割模型
    print("\n[3/6] 创建分割模型...")
    model = create_segmenter(cfg, backbone_model)
    print(f"  模型结构: EncoderDecoder + BNHead")
    print(f"  测试模式: slide (crop_size=512, stride=341)")

    # 4. 加载分割头权重
    print("\n[4/6] 加载分割头权重...")
    head_state = torch.load(head_ckpt, map_location="cpu")
    # 官方权重格式: {'meta': ..., 'state_dict': ..., 'optimizer': ...}
    if 'state_dict' in head_state:
        head_state = head_state['state_dict']
    model.load_state_dict(head_state, strict=False)
    model = model.to(args.device).eval()
    print(f"  分割头加载完成: 21类 (VOC2012)")

    # 5. 构建测试数据集
    print("\n[5/6] 构建测试数据集...")
    dataset = build_dataset(cfg.data.test)
    dataloader = build_dataloader(
        dataset, 
        samples_per_gpu=1, 
        workers_per_gpu=args.workers, 
        dist=False, 
        shuffle=False
    )
    print(f"  验证集样本数: {len(dataset)}")

    # 6. 推理评估
    print("\n[6/6] 开始推理评估...")
    model = MMDataParallel(model, device_ids=[0])
    results = single_gpu_test(model, dataloader, pre_eval=True)

    # 7. 计算mIoU
    print("\n计算mIoU指标...")
    metrics = dataset.evaluate(results, metric="mIoU")
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    
    # 类别名称
    class_names = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    # 打印每类IoU
    if 'IoU' in metrics:
        print("\nPer-class IoU:")
        for i, (name, iou) in enumerate(zip(class_names, metrics['IoU'])):
            print(f"  {i:2d}. {name:15s}: {iou*100:.2f}%")
    
    # 打印mIoU
    print("\n" + "-" * 40)
    if 'mIoU' in metrics:
        print(f"mIoU: {metrics['mIoU']*100:.2f}%")
    if 'aAcc' in metrics:
        print(f"aAcc: {metrics['aAcc']*100:.2f}%")
    print("-" * 40)
    print("\n完成!")


if __name__ == "__main__":
    main()
