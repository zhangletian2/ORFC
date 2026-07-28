# -*- coding: utf-8 -*-
"""
SigLIP2 So400m-patch14-224 COCO Retrieval 特征提取：
- 从固定的 image list 加载图片，提取 blk07/15/23 中间层 token 特征
- 保存为 .npy（[N, D]，N=256 patch tokens，float32）

用法：
  python siglip2_ret_extract.py \
      --image_list  /path/to/coco_selected_pathname500.txt \
      --image_root  /path/to/coco2014 \
      --image_subdir val2014 \
      --out_root    /path/to/features/coco_ret/siglip2_so400m \
      --layers      7,15,23
"""

import os, sys, time, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from siglip2_feat_pipeline import load_siglip2, EncoderLayerCatcher


class COCOListDataset(Dataset):
    """从 pathname list 加载 COCO 图片，返回 PIL Image 和索引。"""

    def __init__(self, image_list, image_root, image_subdir="val2014"):
        self.image_ids = []
        with open(image_list, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.image_ids.append(line)

        self.image_root = image_root
        self.image_subdir = image_subdir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(
            self.image_root, self.image_subdir, f"{img_id}.jpg")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return img, idx


def collate_pil(batch):
    imgs, indices = zip(*batch)
    return list(imgs), list(indices)


@torch.no_grad()
def extract(args):
    device = args.device
    model, processor = load_siglip2(args.model_id, device, cache_dir=args.cache_dir)

    vision_model = model.vision_model
    layers = [int(x) for x in args.layers.split(",")]
    num_layers = len(vision_model.encoder.layers)
    for l in layers:
        assert 0 <= l < num_layers, f"layer {l} 越界，模型共 {num_layers} 层"

    catcher = EncoderLayerCatcher(vision_model, layers)

    dataset = COCOListDataset(args.image_list, args.image_root, args.image_subdir)
    print(f"从 {args.image_list} 加载 {len(dataset)} 张图片")

    os.makedirs(args.out_root, exist_ok=True)
    for l in layers:
        os.makedirs(os.path.join(args.out_root, f"blk{l:02d}"), exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_pil,
        pin_memory=False,
        shuffle=False,
    )

    t0, n = time.time(), 0
    for imgs, indices in tqdm(loader, desc="Extracting"):
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        _ = model.get_image_features(**inputs)

        outs = catcher.pop()
        for b_idx, global_idx in enumerate(indices):
            img_id = dataset.image_ids[global_idx]
            for l in layers:
                key = f"blk{l:02d}"
                arr = outs[key][b_idx].numpy().astype(np.float32)
                np.save(os.path.join(args.out_root, key, f"{img_id}.npy"), arr)
            n += 1

    catcher.close()
    elapsed = time.time() - t0
    print(f"[extract] 完成: {n} 张图, layers={layers}, "
          f"batch_size={args.batch_size}, 耗时 {elapsed:.1f}s")
    print(f"特征保存至: {args.out_root}")


def main():
    ap = argparse.ArgumentParser(
        "SigLIP2 COCO Retrieval 中间层特征提取")
    ap.add_argument('--model_id', default='google/siglip2-so400m-patch14-224')
    ap.add_argument('--cache_dir', default=None)
    ap.add_argument('--image_list', required=True,
                    help='图片 ID 列表 (coco_selected_pathname500.txt)')
    ap.add_argument('--image_root', required=True,
                    help='COCO 图片根目录 (包含 val2014/ 子目录)')
    ap.add_argument('--image_subdir', default='val2014',
                    help='图片子目录名')
    ap.add_argument('--out_root', required=True,
                    help='特征输出根目录')
    ap.add_argument('--layers', default='7,15,23',
                    help='0-based encoder layer 索引，逗号分隔')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    extract(args)


if __name__ == '__main__':
    main()
