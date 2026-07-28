#!/data4/workspace/zlt/featcodec/CoFAI/.venv/bin/python
# -*- coding: utf-8 -*-
"""
DINOv3 ViT-L/16 特征提取脚本
从 ImageNet val 5k 子集提取指定 block 层特征，保存为 .npy (float32)

模型参数：
  patch_size=16, embed_dim=1024, depth=24, n_storage_tokens=4
  224x224 输入 → 14×14=196 patches → token 序列 [1 CLS + 4 reg + 196 patch] = 201

用法：
  python dinov3_feat_extract.py \
      --root /data4/workspace/zlt/CAPO_cls/dataset/imagenet/images/val \
      --list /data4/workspace/zlt/featcodec/ORFC/utils/imagenet_selected_pathname5000.txt \
      --out_root /data4/workspace/zlt/featcodec/features/train/dinov3_vitl16/blk05 \
      --layer 5 --device cuda

  # 最后一层 (blk23, ViT-L/16 depth=24)
  python dinov3_feat_extract.py \
      --root /data4/workspace/zlt/CAPO_cls/dataset/imagenet/images/val \
      --list /data4/workspace/zlt/featcodec/ORFC/utils/imagenet_selected_pathname5000.txt \
      --out_root /data4/workspace/zlt/featcodec/features/train/dinov3_vitl16/blk23 \
      --layer 23 --device cuda
"""

import os, sys, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", category=UserWarning)

# DINOv3 本地源码 & 权重
DINOV3_DIR = Path("/data4/workspace/zlt/fasterrcnn-pytorch-training-pipeline/dinov3")
BACKBONE_WEIGHTS = DINOV3_DIR / "weights" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_LAYER = 5


def build_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_list(list_txt):
    """读取 pathname 列表，每行 '<wnid> <basename>'"""
    with open(list_txt, "r") as f:
        return [line.strip().split() for line in f if line.strip()]


class BlockOutputCatcher:
    """注册 forward hook 到指定 block，收集输出 [B, N, D]"""

    def __init__(self, backbone: nn.Module, block_indices):
        self.indices = sorted(set(int(i) for i in block_indices))
        self._buf = {}
        self._handles = []
        assert hasattr(backbone, "blocks"), "backbone 缺少 .blocks"
        blocks = list(backbone.blocks)
        n_blocks = len(blocks)

        def _make_hook(idx):
            key = f"blk{idx:02d}"
            def hook(module, inp, out):
                # DINOv3 block returns List[Tensor] (batch-list); 单图时取 [0]
                if isinstance(out, list):
                    self._buf[key] = out[0].detach().cpu()
                else:
                    self._buf[key] = out.detach().cpu()
            return hook

        for idx in self.indices:
            if idx >= n_blocks:
                raise IndexError(f"block index {idx} >= n_blocks={n_blocks}")
            h = blocks[idx].register_forward_hook(_make_hook(idx))
            self._handles.append(h)

    def pop(self):
        outs = self._buf
        self._buf = {}
        return outs

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def build_backbone(device):
    sys.path.insert(0, str(DINOV3_DIR))
    backbone = torch.hub.load(
        str(DINOV3_DIR), "dinov3_vitl16",
        source="local",
        weights=str(BACKBONE_WEIGHTS),
    )
    backbone = backbone.to(device).eval()
    return backbone


@torch.no_grad()
def cmd_extract(args):
    device = args.device
    tfm = build_transform()
    pairs = load_list(args.list)
    os.makedirs(args.out_root, exist_ok=True)

    backbone = build_backbone(device)
    n_blocks = len(backbone.blocks)
    if not (0 <= args.layer < n_blocks):
        raise ValueError(f"--layer must be in [0, {n_blocks - 1}], got {args.layer}")
    layer_tag = f"blk{args.layer:02d}"
    print(f"  Extracting {layer_tag} (block {args.layer}/{n_blocks - 1})")

    catcher = BlockOutputCatcher(backbone, [args.layer])

    t0, n, skipped = time.time(), 0, 0

    try:
        for wnid, base in tqdm(pairs, desc=f"Extracting {layer_tag}"):
            img_path = os.path.join(args.root, wnid, base + ".JPEG")
            if not os.path.isfile(img_path):
                print(f"[warn] missing: {img_path}")
                skipped += 1
                continue

            save_path = os.path.join(args.out_root, f"{base}.npy")
            if args.skip_existing and os.path.isfile(save_path):
                n += 1
                continue

            img = Image.open(img_path).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)

            _ = backbone(x)

            outs = catcher.pop()
            key = f"blk{args.layer:02d}"
            arr = outs[key].squeeze(0).numpy().astype(np.float32)  # [201, 1024]
            np.save(save_path, arr)
            n += 1
    finally:
        catcher.close()

    elapsed = time.time() - t0
    print(f"[extract] Done. saved={n} skipped={skipped} "
          f"out_root={args.out_root} ({elapsed:.1f}s, {n / max(elapsed, 1):.1f} img/s)")


def main():
    ap = argparse.ArgumentParser("DINOv3 ViT-L/16 特征提取")
    ap.add_argument("--root", required=True, help="ImageNet val 根目录 (含 wnid 子目录)")
    ap.add_argument("--list", required=True, help="pathname 列表 txt: <wnid> <basename>")
    ap.add_argument("--out_root", required=True, help="特征输出目录")
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                    help="Block index to extract (0-based, default=5 for blk05; 23 for last layer)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_existing", action="store_true",
                    help="跳过已存在的 .npy 文件")
    args = ap.parse_args()
    cmd_extract(args)


if __name__ == "__main__":
    main()
