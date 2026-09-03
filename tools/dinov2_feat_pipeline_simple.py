# -*- coding: utf-8 -*-
"""
DINOv2 极简特征管线（使用模型自带 forward + forward hook 抽取）
支持 ViT-L/14 (24 blocks) 和 ViT-G/14 (40 blocks)。

功能：
  - extract：注册 hook 到指定 block，正常 forward(images)，在 hook 中拿到 block 输出（含 CLS），保存 .npy(float32)
  - replay ：从某块特征 .npy 重载，接着从该块的下一层继续前向到 head，统计 Top-1/Top-5

输入格式沿用你的分类脚本：
  --list   每行: "<wnid> <basename>"
  --labels 每行: "<basename> <idx>"

依赖：
  本地 dinov2 源码路径（backbone/dinov2）
  权重目录含对应模型的 pretrain / linear_head 权重
"""

import os, sys, time, argparse, glob, json, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", category=UserWarning) # Disable xFormers UserWarning

# === 按你环境：把本地 dinov2 源码加到 PYTHONPATH（参考 cls_dinov2_500.py） ===
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "backbone" / "dinov2"))
from dinov2.hub.classifiers import dinov2_vitl14_lc, dinov2_vitg14_lc

MODEL_REGISTRY = {
    "vitl14": {
        "builder": dinov2_vitl14_lc,
        "pretrain": "dinov2_vitl14_pretrain.pth",
        "head1": "dinov2_vitl14_linear_head.pth",
        "head4": "dinov2_vitl14_linear4_head.pth",
        "tag": "dinov2_vitl14",
    },
    "vitg14": {
        "builder": dinov2_vitg14_lc,
        "pretrain": "dinov2_vitg14_pretrain.pth",
        "head1": "dinov2_vitg14_linear_head.pth",
        "head4": "dinov2_vitg14_linear4_head.pth",
        "tag": "dinov2_vitg14",
    },
}

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
    m = {}
    with open(label_txt, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            base, idx = ln.split()
            m[base] = int(idx)
    return m  # base -> int

def sha1_file(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        while True:
            b = f.read(1<<20)
            if not b: break
            h.update(b)
    return h.hexdigest()

# ====== 抽取：用 forward hook 抓 block 输出（完全复用模型内部 forward） ======

class BlockOutputCatcher:
    """
    注册到指定 blocks 的 forward hook，收集输出（含 CLS 的 token 序列 [B,N,D]）
    用法：
        catcher = BlockOutputCatcher(backbone, [5,11,17,23])
        logits = model(images)  # 正常前向
        outs = catcher.pop()    # {'blk05': [B,N,D], ...} 全是 CPU tensor
    """
    def __init__(self, backbone: nn.Module, block_indices):
        self.backbone = backbone
        self.indices = sorted(set(int(i) for i in block_indices))
        self._buf = {}
        self._handles = []
        assert hasattr(backbone, "blocks"), "backbone 缺少 .blocks"
        blocks = list(backbone.blocks)
        n_blocks = len(blocks)

        def _make_hook(idx):
            key = f"blk{idx:02d}"
            def hook(module, inp, out):
                self._buf[key] = out.detach().cpu()
            return hook

        for idx in self.indices:
            if idx >= n_blocks:
                raise IndexError(
                    f"block index {idx} out of range, backbone only has {n_blocks} blocks (0-{n_blocks-1})"
                )
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

@torch.no_grad()
def forward_from_tokens(backbone, head, tokens, start_block_idx):
    """
    从中间层 tokens (含 CLS，[B,N,D]) 继续前向，
    构造官方 LinearClassifierWrapper(layers=1) 的输入：
        linear_input = cat([x_norm_clstoken, mean(x_norm_patchtokens)], dim=1)
    返回 logits: [B, num_classes]
    """
    x = tokens
    for i in range(start_block_idx + 1, len(backbone.blocks)):
        x = backbone.blocks[i](x)
    x = backbone.norm(x)
    cls_token = x[:, 0]
    patch_tokens = x[:, 1:]
    mean_patch = patch_tokens.mean(dim=1)
    linear_input = torch.cat([cls_token, mean_patch], dim=1)
    if hasattr(head, "in_features") and head.in_features != linear_input.shape[1]:
        raise RuntimeError(
            f"Head expects {head.in_features}D, but got {linear_input.shape[1]}D. "
            f"请确认权重与模型匹配。"
        )
    return head(linear_input)


# ====== 子命令：extract / replay ======

def _load_classifier(args):
    """根据 args.model 加载分类器，返回 (clf, backbone, reg)"""
    reg = MODEL_REGISTRY[args.model]
    back = str(Path(args.weights_root) / reg["pretrain"])
    head = str(Path(args.weights_root) / (reg["head1"] if args.head_layers==1 else reg["head4"]))
    clf = reg["builder"](layers=args.head_layers, pretrained=True, weights=[back, head]).to(args.device).eval()
    backbone = getattr(clf, "backbone", clf)
    return clf, backbone, reg

class _ImgListDataset(torch.utils.data.Dataset):
    """每项返回 (tensor, basename, img_path)；缺失图像返回 None 占位由 collate 过滤。"""
    def __init__(self, root, pairs, transform):
        self.root = root
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        wnid, base = self.pairs[idx]
        img_path = os.path.join(self.root, wnid, base + ".JPEG")
        if not os.path.isfile(img_path):
            return None
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), base, img_path


def _collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, bases, paths = zip(*batch)
    return torch.stack(xs, 0), list(bases), list(paths)


def cmd_extract(args):
    device = args.device
    tfm = build_transform()
    pairs = load_list(args.list)
    os.makedirs(args.out_root, exist_ok=True)
    layers = [int(x) for x in args.blocks.split(",")]
    for k in layers:
        os.makedirs(os.path.join(args.out_root, f"blk{k:02d}"), exist_ok=True)

    clf, backbone, reg = _load_classifier(args)
    catcher = BlockOutputCatcher(backbone, layers)

    loader = torch.utils.data.DataLoader(
        _ImgListDataset(args.root, pairs, tfm),
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=(device.startswith("cuda")),
        persistent_workers=(args.num_workers > 0),
        collate_fn=_collate_skip_none,
    )

    mf = open(os.path.join(args.out_root, "manifest.jsonl"), "a", encoding="utf-8") if args.write_manifest else None
    t0, n = time.time(), 0

    try:
        for batch in tqdm(loader, desc="Extracting"):
            if batch is None:
                continue
            x, bases, paths = batch
            x = x.to(device, non_blocking=True)
            _ = clf(x)
            outs = catcher.pop()
            B = len(bases)
            for i in range(B):
                for k in layers:
                    key = f"blk{k:02d}"
                    arr = outs[key][i].numpy().astype(np.float32)
                    save_path = os.path.join(args.out_root, key, f"{bases[i]}.npy")
                    np.save(save_path, arr)
                    if mf:
                        rec = {
                            "id": bases[i],
                            "model": reg["tag"],
                            "layer": key,
                            "path": save_path,
                            "shape": list(arr.shape),
                            "dtype": "float32",
                            "img_path": paths[i],
                            "sha1": sha1_file(save_path),
                        }
                        mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    finally:
        catcher.close()
        if mf: mf.close()

    print(f"[extract] Done. N={n} layers={layers} bs={args.batch_size} "
          f"workers={args.num_workers} out_root={args.out_root} ({time.time()-t0:.2f}s)")

def cmd_replay(args):
    device = args.device
    labels = load_labels(args.labels)
    layers = args.layer

    clf, backbone, reg = _load_classifier(args)
    head_mod = getattr(clf, "linear_head", None)
    assert head_mod is not None, "未找到分类头（head）"

    print("BLK\t\tAcc@1\t\tAcc@5\t\tTime")
    for layer in layers:
        feat_dir = os.path.join(args.feature_root, layer)
        files = sorted(glob.glob(os.path.join(feat_dir, "*.npy")))
        if not files:
            raise RuntimeError(f"未找到特征：{feat_dir}/*.npy")
        start_idx = int(layer[-2:])
        top1 = top5 = 0
        n = 0
        t0 = time.time()
        for p in files:
            base = os.path.splitext(os.path.basename(p))[0]
            gt = labels.get(base, None)
            if gt is None:
                print(f"[warn] missing label for: {base}")
                continue
            arr = np.load(p)
            tok = torch.from_numpy(arr).unsqueeze(0).to(device)
            logits = forward_from_tokens(backbone, head_mod, tok, start_idx)
            top5_idx = torch.topk(logits, k=5, dim=-1).indices.squeeze(0).tolist()
            if top5_idx[0] == gt: top1 += 1
            if gt in top5_idx:    top5 += 1
            n += 1
        print(f"{layer}\t\t{top1/n*100:.2f}%\t\t{top5/n*100:.2f}%\t\t{time.time()-t0:.2f}s")

# ====== CLI ======

def build_parser():
    ap = argparse.ArgumentParser("DINOv2：hook 抽取 / 分层回放分类（极简）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 提取
    pe = sub.add_parser("extract", help="forward hook 抽取 tokens（含CLS）到 .npy")
    pe.add_argument('--root',   required=True, help='ImageNet val 根目录')
    pe.add_argument('--list',   required=True, help='列表 txt：<wnid> <basename>')
    pe.add_argument('--weights_root', required=True, help='权重目录')
    pe.add_argument('--model', default='vitl14', choices=list(MODEL_REGISTRY.keys()),
                    help='模型变体 (default: vitl14)')
    pe.add_argument('--head_layers', type=int, default=1, choices=[1,4])
    pe.add_argument('--out_root', required=True, help='特征输出根目录')
    pe.add_argument('--blocks', default='5,11,17,23', help='0-based 块索引，逗号分隔')
    pe.add_argument('--batch_size', type=int, default=32, help='提取 batch size')
    pe.add_argument('--num_workers', type=int, default=8, help='DataLoader workers')
    pe.add_argument('--device', default='cuda')
    pe.add_argument('--write_manifest', action='store_true')

    # 回放
    pr = sub.add_parser("replay", help='从某块 .npy 重载，继续前向到 head，统计 Top-1/Top-5')
    pr.add_argument('--feature_root', required=True, help='extract 阶段的 out_root')
    pr.add_argument('--layer', type=str, required=True, nargs="*", help='如 blk09 blk19')
    pr.add_argument('--labels', required=True, help='label txt：<basename> <idx>')
    pr.add_argument('--weights_root', required=True)
    pr.add_argument('--model', default='vitl14', choices=list(MODEL_REGISTRY.keys()),
                    help='模型变体 (default: vitl14)')
    pr.add_argument('--head_layers', type=int, default=1, choices=[1,4])
    pr.add_argument('--device', default='cuda')

    return ap

def main():
    ap = build_parser()
    args = ap.parse_args()

    if args.cmd == 'extract':
        layers = [int(x) for x in args.blocks.split(",")]
        for k in layers:
            os.makedirs(os.path.join(args.out_root, f"blk{k:02d}"), exist_ok=True)
        cmd_extract(args)
    elif args.cmd == 'replay':
        cmd_replay(args)

if __name__ == '__main__':
    main()
