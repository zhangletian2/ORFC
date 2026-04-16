# -*- coding: utf-8 -*-
"""
EVA02-L/14 (timm: eva02_large_patch14_224.mim_m38m) 极简特征管线
- extract：注册 forward hook 到指定 blocks（0-based），正常前向；在 hook 中取 block 输出（含 CLS），保存 .npy(float32, [N,D])
- replay ：从某 block 的 .npy 重载 tokens（[1,N,D]，含 CLS），从该 block 的下一层继续前向到 head，统计 Top-1/Top-5

输入文件格式（与 clip/dinov2/swin 保持一致）：
  --list   每行: "<wnid> <basename>"
  --labels 每行: "<basename> <idx>"

依赖：timm>=0.9
模型名：eva02_large_patch14_224.mim_m38m（Hugging Face 权重别名由 timm 自动拉取）
"""

import os, time, argparse, glob, re
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import timm
from timm.data import resolve_data_config, create_transform

# ---------- I/O ----------
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

# ---------- Hook 抽取：抓 blocks[k] 输出（[B,N,D]，含 CLS） ----------
class BlockCatcher:
    """
    注册到 VisionTransformer 风格的 model.blocks[k]，收集输出（含 CLS 的 token 序列 [B,N,D]）
    """
    def __init__(self, model, block_indices):
        self.model = model
        self.indices = sorted(set(int(i) for i in block_indices))
        self.buf = {}
        self.handles = []
        assert hasattr(model, "blocks"), "EVA02 模型缺少 .blocks"
        blocks = list(model.blocks)

        def _make_hook(idx):
            key = f"blk{idx:02d}"
            def hook(module, inp, out):
                # out: [B, N, D]（VisionTransformer 的 Block 输出）
                self.buf[key] = out.detach().cpu().to(torch.float32)
            return hook

        for idx in self.indices:
            h = blocks[idx].register_forward_hook(_make_hook(idx))
            self.handles.append(h)

    def pop(self):
        out = self.buf
        self.buf = {}
        return out

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

# ---------- 从中间 tokens 继续前向到 head ----------
@torch.no_grad()
def forward_from_tokens_to_logits(model: nn.Module, tokens_bnd: torch.Tensor, start_block_idx: int):
    """
    tokens_bnd: [B, N, D]（含 CLS；N = 1 + H*W/patch^2）
    从 start_block_idx 的下一层继续跑到模型 head，返回 logits: [B, num_classes]
    兼容 timm 的 VisionTransformer 实现（EVA02 复用了 ViT 主干）
    """
    x = tokens_bnd  # [B, N, D]

    # 继续后续 blocks
    for i in range(start_block_idx + 1, len(model.blocks)):
        x = model.blocks[i](x)  # [B, N, D]

    # 最终层 norm（与 timm ViT 对齐）
    if getattr(model, "norm", None) is not None:
        x = model.norm(x)  # [B, N, D]

    # ✅ 优先走官方实现：自动处理 global_pool / fc_norm / head
    if hasattr(model, "forward_head"):

        logits = model.forward_head(x, pre_logits=False)  # [B, num_classes]
        return logits

    # ⬇️ 兜底：手动与 timm 一致
    # pool（avg 或 cls）
    if getattr(model, "global_pool", None) == "avg":
        x = x[:, 1:, :].mean(dim=1)   # [B, D]
    else:
        x = x[:, 0, :]                # [B, D]

    # fc_norm（很多 EVA/ViT 变体使用它）
    if getattr(model, "fc_norm", None) is not None:
        x = model.fc_norm(x)          # [B, D]

    # head（Linear 或模块）
    head = getattr(model, "head", None)
    if head is None:
        raise RuntimeError("未找到分类头 model.head")
    logits = head(x) if isinstance(head, nn.Module) else head(x)
    return logits

# ---------- 子命令 ----------
@torch.no_grad()
def cmd_extract(args):
    device = args.device
    model = timm.create_model('eva02_large_patch14_224.mim_in22k', pretrained=True).to(device).eval()
    cfg = resolve_data_config(model.default_cfg)
    tfm = create_transform(**cfg)

    blocks = [int(x) for x in args.blocks.split(",")]  # e.g., 5,11,17,23
    catcher = BlockCatcher(model, blocks)

    pairs = load_list(args.list)
    os.makedirs(args.out_root, exist_ok=True)
    for b in blocks:
        os.makedirs(os.path.join(args.out_root, f"blk{b:02d}"), exist_ok=True)

    t0, n = time.time(), 0
    for wnid, base in pairs:
        img_path = os.path.join(args.root, wnid, base + ".JPEG")
        if not os.path.isfile(img_path):
            print(f"[warn] missing image: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        _ = model(x)  # 正常前向；hook 捕获中间层

        outs = catcher.pop()  # {'blk05': [1,N,D], ...}
        for b in blocks:
            key = f"blk{b:02d}"
            arr = outs[key].squeeze(0).numpy().astype(np.float32)  # [N, D]
            save_path = os.path.join(args.out_root, key, f"{base}.npy")
            np.save(save_path, arr)
        n += 1

    catcher.close()
    print(f"[extract][EVA02-L/14] N={n} blocks={blocks} out_root={args.out_root} ({time.time()-t0:.2f}s)")

@torch.no_grad()
def cmd_replay(args):
    import copy
    import torch.nn as nn

    device = args.device
    labels = load_labels(args.labels)

    # 1) 创建 224 骨干（dst）
    model = timm.create_model('eva02_large_patch14_224.mim_m38m',
                              pretrained=True).to(device).eval().float()

    # 2) 从 448 分类模型（src）迁移 fc_norm + head 到 224 骨干（dst）
    src = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
                            pretrained=True).to(device).eval().float()

    # 2.1 迁移 fc_norm（若 224 没有或是 Identity，则直接替换；否则加载权重）
    if hasattr(src, 'fc_norm'):
        if (not hasattr(model, 'fc_norm')) or isinstance(model.fc_norm, nn.Identity):
            model.fc_norm = copy.deepcopy(src.fc_norm).to(device)
        else:
            try:
                model.fc_norm.load_state_dict(src.fc_norm.state_dict(), strict=True)
            except Exception:
                # 避免个别实现上属性不完全一致
                model.fc_norm = copy.deepcopy(src.fc_norm).to(device)

    # 2.2 迁移 head（常见为 Linear；若是模块则直接深拷贝）
    if isinstance(src.head, nn.Linear):
        new_head = nn.Linear(src.head.in_features, src.head.out_features,
                             bias=(src.head.bias is not None)).to(device)
        new_head.load_state_dict(src.head.state_dict(), strict=True)
        model.head = new_head
    else:
        model.head = copy.deepcopy(src.head).to(device)

    # 标注 num_classes，避免后续分支/断言误判
    if getattr(model, 'num_classes', 0) == 0:
        model.num_classes = getattr(src, 'num_classes', 1000)

    # 简要健诊：现在 head 应该可用
    _head = getattr(model, 'head', None)
    assert _head is not None and (not hasattr(_head, 'out_features') or _head.out_features > 0), \
        "迁移后的分类头无效，请检查 head/fc_norm 复制是否成功。"

    # 释放 src 减少显存占用
    del src
    torch.cuda.empty_cache()

    # 3) 解析起始层索引（兼容 blk5 / blk05 / blk11）
    m = re.search(r"(\d+)$", args.layer)
    assert m, f"layer 名称格式应类似 blk05/blk11：收到 {args.layer}"
    start_idx = int(m.group(1))

    # 4) 参考序列长度（1 + num_patches）
    try:
        num_patches = int(getattr(model.patch_embed, "num_patches"))
    except Exception:
        # 兜底：224/14=16 -> 16*16=256
        num_patches = 256
    seq_len_ref = 1 + num_patches

    # 5) 读取特征并重载
    feat_dir = os.path.join(args.feature_root, args.layer)
    files = sorted(glob.glob(os.path.join(feat_dir, "*.npy")))
    if not files:
        raise RuntimeError(f"未找到特征：{feat_dir}/*.npy")

    top1 = top5 = 0
    n = 0
    t0 = time.time()
    for p in files:
        base = os.path.splitext(os.path.basename(p))[0]
        gt = labels.get(base, None)
        if gt is None:
            continue

        arr = np.load(p)  # [N, D]（含 CLS）
        tok = torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=torch.float32)  # [1,N,D]

        # 形状校验（不做自动分支）
        assert tok.ndim == 3 and tok.shape[1] == seq_len_ref, \
            f"token 形状应为 [B,{seq_len_ref},D]（含 CLS）；实际 {tok.shape}，文件：{p}"

        # 从中间层 token 继续跑到 head（内部优先使用 model.forward_head）
        logits = forward_from_tokens_to_logits(model, tok, start_block_idx=start_idx)  # [1,1000]
        top5_idx = torch.topk(logits, k=5, dim=-1).indices.squeeze(0).tolist()

        if top5_idx[0] == gt:
            top1 += 1
        if gt in top5_idx:
            top5 += 1
        n += 1

    print(f"[replay@{args.layer}][EVA02-L/14] N={n} Top-1={top1/n*100:.2f}% Top-5={top5/n*100:.2f}% ({time.time()-t0:.2f}s)")


# ---------- CLI ----------
def build_parser():
    ap = argparse.ArgumentParser("EVA02-L/14：分层抽取 / 分层回放（极简 .npy，含 CLS）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="hook 抽取 blocks 输出（含 CLS）并保存 .npy")
    pe.add_argument('--root', required=True, help='ImageNet val 根目录')
    pe.add_argument('--list', required=True, help='500张列表 txt：<wnid> <basename>')
    pe.add_argument('--out_root', required=True, help='特征输出根目录，如 features/eva02_large')
    pe.add_argument('--blocks', default='5,11,17,23', help='0-based block 索引，逗号分隔，如 5,11,17,23')
    pe.add_argument('--device', default='cuda')

    pr = sub.add_parser("replay", help='从某 block 的 .npy 重载，继续前向到 head，统计 Top-1/Top-5')
    pr.add_argument('--feature_root', required=True, help='extract 的 out_root')
    pr.add_argument('--layer', required=True, help='如 blk05/blk11/blk17/blk23（支持 blk5/blk23 等）')
    pr.add_argument('--labels', required=True, help='500张 label txt：<basename> <idx>')
    pr.add_argument('--device', default='cuda')

    return ap

def main():
    ap = build_parser()
    args = ap.parse_args()
    if args.cmd == 'extract':
        for b in [int(x) for x in args.blocks.split(",")]:
            os.makedirs(os.path.join(args.out_root, f"blk{b:02d}"), exist_ok=True)
        cmd_extract(args)
    elif args.cmd == 'replay':
        cmd_replay(args)

if __name__ == '__main__':
    main()
