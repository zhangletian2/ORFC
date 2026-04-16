# -*- coding: utf-8 -*-
"""
make_text_emb.py
生成 CLIP ViT-L/14 的文本嵌入矩阵 text_emb.npy （形状 [num_classes, D]，已 L2 归一化）

输入:
  --classnames  类别文件，每行一个类别名（如 "tench, Tinca tinca"）
输出:
  text_emb.npy  （保存到 --out_path）

默认模板: "a photo of a {}"
"""

import os, argparse
import numpy as np
import torch
import clip  # pip install git+https://github.com/openai/CLIP.git

def main():
    ap = argparse.ArgumentParser("生成 CLIP 文本嵌入矩阵 text_emb.npy")
    ap.add_argument("--classnames", required=True, help="类别文件，每行一个类别名")
    ap.add_argument("--out_path", required=True, help="输出 .npy 路径")
    ap.add_argument("--template", default="a photo of a {}", help="提示模板")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # 载入 CLIP ViT-L/14
    device = args.device
    model, _ = clip.load("ViT-L/14", device=device)
    model.eval()

    # 读取类别名
    with open(args.classnames, "r") as f:
        classnames = [ln.strip() for ln in f if ln.strip()]
    print(f"Loaded {len(classnames)} class names.")

    # 构造文本 prompts
    prompts = [args.template.format(name) for name in classnames]
    tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)  # [C, D]
        text_features /= text_features.norm(dim=-1, keepdim=True)  # L2 归一化
    np.save(args.out_path, text_features.cpu().numpy().astype(np.float32))
    print(f"[done] text_emb saved to {args.out_path}  shape={tuple(text_features.shape)}")

if __name__ == "__main__":
    main()
