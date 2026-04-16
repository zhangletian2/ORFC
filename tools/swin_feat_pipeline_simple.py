# -*- coding: utf-8 -*-
"""
Swin-L 极简特征管线（参照 cls_swin_500.py）
- extract：按 --stages 指定的 stage（1..4）抽取，forward hook 抓取 (x,H,W)，保存 x 为 .npy(float32)；
          H、W 写入 out_root/manifest.jsonl（轻量索引）
- replay ：指定 --layer（stage1..4）重载 .npy，H、W 从 manifest 查回，从下一 stage 继续前向到 head，统计 Top-1/Top-5

输入格式与 cls_swin_500.py 一致：
  --list   每行: "<wnid> <basename>"
  --labels 每行: "<basename> <idx>"
"""

import os, time, argparse, glob, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import timm
from timm.data import resolve_data_config, create_transform
from tqdm import tqdm

# -------- I/O --------
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

def open_manifest(path):
    idx = {}
    if not os.path.isfile(path):
        return idx
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["layer"], rec["id"])  # (stageX, basename)
            idx[key] = rec
    return idx

def append_manifest(path, rec):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# -------- Hook 抽取 --------
class StageCatcher:
    """
    注册到 timm Swin 的 model.layers[i]（BasicLayer），捕获 (x,H,W)
    存储到 self.buf: {"stage1": (x.cpu(), H, W), ...}
    """
    def __init__(self, model, stages):
        self.model = model
        self.indices0 = [s-1 for s in sorted(set(stages))]  # 1..4 -> 0..3
        self.buf = {}
        self.handles = []
        layers = list(getattr(model, "layers"))
        assert len(layers) >= 4, "Swin 模型应包含 4 个 stages"

        def _make_hook(idx0, tag):
            def hook(module, inp, out):
                if isinstance(out, (list, tuple)) and len(out) == 3:
                    x, H, W = out
                else:
                    x = out
                    # 兜底：若返回不是三元组，尝试从模型推断 H,W（常见 timm Swin 会返回三元组）
                    grid = getattr(self.model.patch_embed, "grid_size", (56, 56))
                    H = grid[0] // (2 ** idx0)
                    W = grid[1] // (2 ** idx0)
                self.buf[tag] = (x.detach().cpu(), int(H), int(W))
            return hook

        tag_map = {0:"stage1", 1:"stage2", 2:"stage3", 3:"stage4"}
        for i0 in self.indices0:
            h = layers[i0].register_forward_hook(_make_hook(i0, tag_map[i0]))
            self.handles.append(h)

    def pop(self):
        out = self.buf
        self.buf = {}
        return out

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

# -------- 回放：从下一 stage 继续到 head --------
@torch.no_grad()
def forward_from_stage(model, x, start_stage_idx0: int):
    """
    兼容 timm>=0.9 的 Swin。
    BasicLayer.forward 需要 [B,H,W,C]。
    Head 期望 [B,C]。
    """
    # 若输入为 [B,N,C]，还原为 [B,H,W,C]
    if x.ndim == 3:
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.view(B, H, W, C)

    # 逐层前向
    for i in range(start_stage_idx0 + 1, len(model.layers)):
        x = model.layers[i](x)

    # 展平 + norm
    B, H, W, C = x.shape
    x = x.view(B, H * W, C)
    if model.norm is not None:
        x = model.norm(x)  # [B, N, C]

    # 正确的 global pool：对 token 维做平均
    x = x.mean(dim=1)  # [B, C]

    # 直接进入线性层
    logits = model.head.fc(x)  # 避开 head 内的 global_pool，再走 fc
    return logits



# -------- 子命令 --------
@torch.no_grad()
def cmd_extract(args):
    device = args.device
    model = timm.create_model(args.model, pretrained=True).to(device).eval()
    cfg = resolve_data_config(model.default_cfg)
    tfm = create_transform(**cfg)

    stages = [int(x) for x in args.stages.split(",")]  # e.g., "1,2,3,4"
    catcher = StageCatcher(model, stages)

    pairs = load_list(args.list)
    os.makedirs(args.out_root, exist_ok=True)
    # 预建各 stage 目录
    for s in stages:
        os.makedirs(os.path.join(args.out_root, f"stage{s}"), exist_ok=True)
    mani_path = os.path.join(args.out_root, "manifest.jsonl")

    t0, n = time.time(), 0
    for wnid, base in tqdm(pairs, desc="Extracting"):
        img_path = os.path.join(args.root, wnid, base + ".JPEG")
        if not os.path.isfile(img_path):
            print(f"[warn] missing image: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        _ = model(x)   # 正常前向；hook 捕获

        outs = catcher.pop()     # {'stageK': (x,H,W), ...}
        for s in stages:
            tag = f"stage{s}"
            x_s, H, W = outs[tag]
            arr = x_s.squeeze(0).numpy().astype(np.float32)  # [H*W, C]
            save_dir = os.path.join(args.out_root, tag)
            save_path = os.path.join(save_dir, f"{base}.npy")
            np.save(save_path, arr)

            rec = {
                "id": base,
                "model": args.model,
                "layer": tag,
                "path": save_path,
                "shape": list(arr.shape),  # [H*W, C]
                "dtype": "float32",
                "img_path": img_path,
                "hw": [H, W],
            }
            append_manifest(mani_path, rec)
        n += 1

    catcher.close()
    print(f"[extract][{args.model}] N={n} stages={stages} out_root={args.out_root} ({time.time()-t0:.2f}s)")

@torch.no_grad()
def cmd_replay(args):
    device = args.device
    labels = load_labels(args.labels)
    model = timm.create_model(args.model, pretrained=True).to(device).eval()

    layer = args.layer  # 'stage1'..'stage4'
    stage_idx0 = {"stage1":0,"stage2":1,"stage3":2,"stage4":3}[layer]

    mani_path = os.path.join(args.feature_root, "manifest.jsonl")
    mani = open_manifest(mani_path)
    # 收集该层的样本列表（按 .npy 存在性过滤）
    layer_dir = os.path.join(args.feature_root, layer)
    files = sorted(glob.glob(os.path.join(layer_dir, "*.npy")))
    if not files:
        raise RuntimeError(f"未找到特征：{layer_dir}/*.npy")

    top1 = top5 = 0
    n = 0
    t0 = time.time()
    for p in tqdm(files, desc="Reloading"):
        base = os.path.splitext(os.path.basename(p))[0]
        gt = labels.get(base, None)
        if gt is None:
            continue
        rec = mani.get((layer, base), None)
        if rec is None:
            # manifest 缺失该条目的话无法得知 H,W（为保持极简，不做更多分支）
            print(f"[warn] missing manifest for {layer}/{base}, skip")
            continue
        H, W = rec["hw"]
        # 载回 .npy
        arr = np.load(p)  # (56, 56, 192)
        x = torch.from_numpy(arr).float().unsqueeze(0).to(device)  # [1, 56, 56, 192]
        logits = forward_from_stage(model, x, start_stage_idx0=stage_idx0)

        top5_idx = torch.topk(logits, k=5, dim=-1).indices.squeeze(0).tolist()
        if top5_idx[0] == gt: top1 += 1
        if gt in top5_idx:    top5 += 1
        n += 1

    print(f"[replay@{layer}][{args.model}] N={n} Top-1={top1/n*100:.2f}% Top-5={top5/n*100:.2f}% ({time.time()-t0:.2f}s)")

# -------- CLI --------
def build_parser():
    ap = argparse.ArgumentParser("Swin-L：分层抽取 / 分层回放（极简 .npy + manifest）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="hook 抽取 stage 输出并保存 .npy（H、W 记入 manifest.jsonl）")
    pe.add_argument('--root', required=True, help='ImageNet val 根目录')
    pe.add_argument('--list', required=True, help='500张列表 txt：<wnid> <basename>')
    pe.add_argument('--out_root', required=True, help='特征输出根目录，如 features/swin_large')
    pe.add_argument('--stages', default='1,2,3,4', help='要抽取的 stage，逗号分隔，如 1,3,4')
    pe.add_argument('--model', default='swin_large_patch4_window7_224.ms_in22k_ft_in1k')
    pe.add_argument('--device', default='cuda')

    pr = sub.add_parser("replay", help='从某 stage 的 .npy 重载（H、W 从 manifest 读取），继续前向到 head')
    pr.add_argument('--feature_root', required=True, help='extract 的 out_root')
    pr.add_argument('--layer', required=True, choices=['stage1','stage2','stage3','stage4'])
    pr.add_argument('--labels', required=True, help='500张 label txt：<basename> <idx>')
    pr.add_argument('--model', default='swin_large_patch4_window7_224.ms_in22k_ft_in1k')
    pr.add_argument('--device', default='cuda')

    return ap

def main():
    ap = build_parser()
    args = ap.parse_args()
    if args.cmd == 'extract':
        # 预建目录（仅所需 stage）
        for s in [int(x) for x in args.stages.split(",")]:
            os.makedirs(os.path.join(args.out_root, f"stage{s}"), exist_ok=True)
        cmd_extract(args)
    elif args.cmd == 'replay':
        cmd_replay(args)

if __name__ == '__main__':
    main()
