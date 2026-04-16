# -*- coding: utf-8 -*-
import os, time, argparse
from pathlib import Path

import torch
import timm
from PIL import Image
from timm.data import resolve_data_config, create_transform

def load_list(list_txt):
    with open(list_txt, 'r') as f:
        return [line.strip().split() for line in f if line.strip()]  # [(wnid, base)]

def load_labels(label_txt):
    m = {}
    with open(label_txt, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            base, idx = ln.split()
            m[base] = int(idx)
    return m

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--list', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--model', default='swin_large_patch4_window7_224.ms_in22k_ft_in1k')
    args = ap.parse_args()

    model = timm.create_model(args.model, pretrained=True).to(args.device).eval()
    cfg = resolve_data_config(model.default_cfg)
    tfm = create_transform(**cfg)

    pairs = load_list(args.list)
    labels = load_labels(args.labels)

    n = len(pairs)
    top1 = top5 = 0
    t0 = time.time()
    for wnid, base in pairs:
        img_path = os.path.join(args.root, wnid, base + ".JPEG")
        if not os.path.isfile(img_path):
            print(f"[warn] missing image: {img_path}")
            continue
        gt = labels.get(base, None)
        if gt is None:
            print(f"[warn] missing label for: {base}")
            continue

        img = Image.open(img_path).convert('RGB')
        x = tfm(img).unsqueeze(0).to(args.device)
        logits = model(x)  # [1,1000]
        top5_idx = torch.topk(logits, k=5, dim=-1).indices.squeeze(0).tolist()

        if top5_idx[0] == gt:
            top1 += 1
        if gt in top5_idx:
            top5 += 1

    dt = time.time() - t0
    print(f"[Swin-L] N={n}  Top-1={top1/n*100:.2f}%  Top-5={top5/n*100:.2f}%  ({dt:.2f}s)")

if __name__ == '__main__':
    main()
