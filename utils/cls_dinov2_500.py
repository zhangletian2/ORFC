# -*- coding: utf-8 -*-
import os, sys, time, argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# 让 python 能 import 你本地的 dinov2 源码（backbone/dinov2）
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "backbone" / "dinov2"))
from dinov2.hub.classifiers import dinov2_vitl14_lc

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def load_list(list_txt):
    with open(list_txt, 'r') as f:
        return [line.strip().split() for line in f if line.strip()]  # [(wnid, base)]

def load_labels(label_txt):
    # base -> int idx
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
    ap.add_argument('--root', required=True, help='ImageNet val 根目录，如 $PROJECT_ROOT/data/imagenet/images')
    ap.add_argument('--list', required=True, help='500张列表 txt：<wnid> <basename>')
    ap.add_argument('--labels', required=True, help='500张的 label txt：<basename> <idx>')
    ap.add_argument('--weights_root', default=os.path.join(str(ROOT), 'pretrained'),
                    help='dinov2 权重所在目录，含 dinov2_vitl14_pretrain.pth / *_linear_head.pth')
    ap.add_argument('--head_layers', type=int, default=1, choices=[1,4],
                    help='1或4层 head（与下载的 head 文件匹配）')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    # 构建模型：一次性加载 backbone + head
    back = str(Path(args.weights_root) / "dinov2_vitl14_pretrain.pth")
    head = str(Path(args.weights_root) / ("dinov2_vitl14_linear_head.pth"
                                          if args.head_layers==1 else
                                          "dinov2_vitl14_linear4_head.pth"))
    model = dinov2_vitl14_lc(layers=args.head_layers, pretrained=True, weights=[back, head])
    model = model.to(args.device).eval()

    tfm = build_transform()
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
    print(f"[DINOv2-L/14] N={n}  Top-1={top1/n*100:.2f}%  Top-5={top5/n*100:.2f}%  ({dt:.2f}s)")

if __name__ == '__main__':
    main()
