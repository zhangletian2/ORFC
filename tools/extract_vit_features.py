#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL ViT-only 特征提取（不走 LLM 生成，~50-100x 加速）。

只加载视觉塔，对每张图片运行 ViT 前向，通过 hook 抓取 block[layer]
的 hidden_states [N, 1024] 并保存为 .npy。

支持多 GPU 并行：
    # 单卡
    python extract_vit_features.py --sample_list list.txt --out_dir feats/

    # 4 卡并行
    python extract_vit_features.py ... --num_workers 4 --worker_id 0 --device cuda:0
    python extract_vit_features.py ... --num_workers 4 --worker_id 1 --device cuda:1
    ...
"""

import argparse
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


def load_sample_list(path):
    indices = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                indices.append(int(line.split("\t")[0]))
    return indices


def prepare_pixel_values(processor, img):
    if not isinstance(img, Image.Image):
        from io import BytesIO
        img = Image.open(BytesIO(img)).convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

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
        description="Qwen3-VL ViT-only block feature extraction")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_dir", required=True,
                    help="MMBench dataset directory")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--data_subdir", default="en")
    ap.add_argument("--sample_list", required=True,
                    help="采样 list 文件路径")
    ap.add_argument("--layer", type=int, default=5,
                    help="ViT block index to extract")
    ap.add_argument("--out_dir", required=True,
                    help="输出 .npy 目录")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16",
                    choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--num_workers", type=int, default=1,
                    help="总 worker 数（多卡并行）")
    ap.add_argument("--worker_id", type=int, default=0,
                    help="当前 worker 编号（0-based）")
    args = ap.parse_args()

    from datasets import load_dataset

    kwargs = {}
    if args.data_subdir:
        kwargs["data_dir"] = args.data_subdir
    ds = load_dataset(args.data_dir, split=args.split, **kwargs)

    all_indices = load_sample_list(args.sample_list)
    chunk_size = (len(all_indices) + args.num_workers - 1) // args.num_workers
    start = args.worker_id * chunk_size
    end = min(start + chunk_size, len(all_indices))
    my_indices = set(all_indices[start:end])

    keep = [i for i, s in enumerate(ds) if s["index"] in my_indices]
    ds = ds.select(keep)
    print(f"[Worker {args.worker_id}/{args.num_workers}] "
          f"{len(ds)} samples, layer={args.layer}")

    visual, processor = load_visual_model(
        args.model_path, args.device, args.dtype)
    vit_dtype = next(visual.parameters()).dtype
    vit_device = next(visual.parameters()).device

    hook = BlockHook(visual.blocks[args.layer])
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    for sample in tqdm(ds, desc=f"vit[{args.worker_id}]"):
        pv, grid = prepare_pixel_values(processor, sample["image"])
        pv = pv.to(device=vit_device, dtype=vit_dtype)
        grid = grid.to(device=vit_device)

        with torch.no_grad():
            _ = visual(pv, grid_thw=grid)

        feat = hook.pop()
        sid = str(sample["index"])
        np.save(os.path.join(args.out_dir, f"{sid}.npy"), feat.numpy())

    hook.close()
    elapsed = time.time() - t0
    vram_gb = (torch.cuda.max_memory_allocated() / 1024**3
               if torch.cuda.is_available() else 0)
    print(f"\n[Worker {args.worker_id}] Done: {len(ds)} samples, "
          f"{elapsed:.1f}s ({elapsed/max(len(ds),1)*1000:.0f}ms/sample), "
          f"peak_vram={vram_gb:.2f}GB")


if __name__ == "__main__":
    main()
