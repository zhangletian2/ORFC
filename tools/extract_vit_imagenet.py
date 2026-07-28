#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL ViT-only ImageNet 特征提取。

从 ImageNet 图片直接提取视觉塔 block[layer] 的 hidden_states [N, 1024]，
保存为 .npy（float32）。只加载视觉塔，不走 LLM。

输入格式（与 siglip2_feat_pipeline.py 一致）：
  --list  每行: "<wnid> <basename>"   (pathname list)
  图片路径: {root}/{wnid}/{basename}.JPEG

多 GPU 并行：
    python extract_vit_imagenet.py ... --num_workers 4 --worker_id 0 --device cuda:0
    python extract_vit_imagenet.py ... --num_workers 4 --worker_id 1 --device cuda:1
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_visual_model(model_path, device="cuda", dtype="bf16"):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[dtype]
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map=device,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)

    visual = model.model.visual
    del model
    torch.cuda.empty_cache()
    return visual, processor


def load_list(path):
    """读取 pathname list，每行: <wnid> <basename>"""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            pairs.append((parts[0], parts[1]))
    return pairs


def prepare_pixel_values(processor, img):
    """通过 Qwen3-VL processor 获取 pixel_values 和 image_grid_thw。"""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "."},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True,
        return_tensors="pt", add_generation_prompt=False,
    )
    return inputs["pixel_values"], inputs["image_grid_thw"]


class BlockHook:
    def __init__(self, block):
        self.output = None
        self._handle = block.register_forward_hook(self._hook)

    def _hook(self, _mod, _inp, out):
        self.output = out.detach().cpu().float()

    def pop(self):
        out = self.output
        self.output = None
        return out

    def close(self):
        self._handle.remove()


def main():
    ap = argparse.ArgumentParser(
        description="Qwen3-VL ViT-only ImageNet feature extraction")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--root", required=True,
                    help="ImageNet images root (contains wnid subdirs)")
    ap.add_argument("--list", required=True,
                    help="pathname list: each line '<wnid> <basename>'")
    ap.add_argument("--layer", type=int, default=5,
                    help="ViT block index to extract (default: 5)")
    ap.add_argument("--out_dir", required=True,
                    help="Output .npy directory")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16",
                    choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--num_workers", type=int, default=1,
                    help="Total worker count (multi-GPU)")
    ap.add_argument("--worker_id", type=int, default=0,
                    help="Current worker id (0-based)")
    args = ap.parse_args()

    all_pairs = load_list(args.list)
    total = len(all_pairs)

    chunk_size = (total + args.num_workers - 1) // args.num_workers
    start = args.worker_id * chunk_size
    end = min(start + chunk_size, total)
    my_pairs = all_pairs[start:end]

    print(f"[Worker {args.worker_id}/{args.num_workers}] "
          f"{len(my_pairs)} images, layer={args.layer}")

    visual, processor = load_visual_model(
        args.model_path, args.device, args.dtype)
    vit_dtype = next(visual.parameters()).dtype
    vit_device = next(visual.parameters()).device

    hook = BlockHook(visual.blocks[args.layer])
    os.makedirs(args.out_dir, exist_ok=True)

    grid_thw_dict = {}
    skipped = 0
    t0 = time.time()
    for wnid, basename in tqdm(my_pairs, desc=f"vit[{args.worker_id}]"):
        out_path = os.path.join(args.out_dir, f"{basename}.npy")
        if os.path.exists(out_path):
            # Still collect grid_thw for already-extracted features
            meta_path = os.path.join(args.out_dir, f"{basename}_grid.json")
            if os.path.exists(meta_path):
                continue
            # Need to get grid_thw even for existing features
            img_path = os.path.join(args.root, wnid, basename + ".JPEG")
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    _, grid = prepare_pixel_values(processor, img)
                    grid_thw_dict[basename] = grid[0].tolist()
                except Exception:
                    pass
            continue

        img_path = os.path.join(args.root, wnid, basename + ".JPEG")
        if not os.path.exists(img_path):
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        pv, grid = prepare_pixel_values(processor, img)
        pv = pv.to(device=vit_device, dtype=vit_dtype)
        grid = grid.to(device=vit_device)

        with torch.no_grad():
            _ = visual(pv, grid_thw=grid)

        feat = hook.pop()
        np.save(out_path, feat.numpy())
        grid_thw_dict[basename] = grid[0].cpu().tolist()

    hook.close()

    # Save grid_thw metadata
    meta_out = os.path.join(args.out_dir, f"grid_thw_worker{args.worker_id}.json")
    with open(meta_out, 'w') as f:
        json.dump(grid_thw_dict, f)
    print(f"[Worker {args.worker_id}] Saved grid_thw for {len(grid_thw_dict)} images -> {meta_out}")

    elapsed = time.time() - t0
    n_done = len(my_pairs) - skipped
    vram_gb = (torch.cuda.max_memory_allocated() / 1024**3
               if torch.cuda.is_available() else 0)
    print(f"\n[Worker {args.worker_id}] Done: {n_done} extracted, "
          f"{skipped} skipped, {elapsed:.1f}s "
          f"({elapsed/max(n_done,1)*1000:.0f}ms/img), "
          f"peak_vram={vram_gb:.2f}GB")


if __name__ == "__main__":
    main()
