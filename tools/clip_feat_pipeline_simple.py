# -*- coding: utf-8 -*-
"""
CLIP ViT-L/14 极简特征管线（带“中间层重载分类”修复）：
- extract：按 --blocks 指定的 resblocks（0-based）注册 forward hook，正常前向；在 hook 中取 block 输出（含 CLS），保存 .npy(float32)
- replay ：从某 block 的 .npy 重载 tokens（[N_tokens, D]，含 CLS），从该 block 的下一层继续前向到 visual.proj，得到 image_feat；
           文本嵌入 text_emb 可在回放阶段“就地生成”（推荐，保证类别顺序一致），或从 .npy 载入（会强制 L2 归一化）。

输入文件格式：
  --list   每行: "<wnid> <basename>"
  --labels 每行: "<basename> <idx>"

依赖：openai/clip
"""

import os, time, argparse, glob, re
import numpy as np
import torch
from PIL import Image
import clip  # pip install git+https://github.com/openai/CLIP.git
from tqdm import tqdm

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

# ---------- classnames & text_emb ----------
def load_classnames_wnid_format(path):
    """解析 classnames.txt，格式：<wnid> <class name...>；行序即类别索引（0..999）。"""
    names, wnids = [], []
    with open(path, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            wnid = parts[0]
            cls_name = " ".join(parts[1:])
            wnids.append(wnid)
            names.append(cls_name)
    assert len(names) > 0, "classnames 文件为空"
    return names, wnids

@torch.no_grad()
def build_text_emb_from_classnames(model, classnames_path, device, template="a photo of a {}"):
    names, _ = load_classnames_wnid_format(classnames_path)
    prompts = [template.format(n) for n in names]
    text_tokens = clip.tokenize(prompts).to(device)
    text_feat = model.encode_text(text_tokens)                # [C, D]
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return text_feat.to(torch.float32)

# ---------- 抽取：forward hook 抓 resblocks[k] 输出 ----------
class ResblockCatcher:
    """
    注册到 visual.transformer.resblocks[k] 的 forward hook，收集输出（[B, N, D]，含 CLS）
    用法：
        catcher = ResblockCatcher(model.visual, [5,11,17,23])
        _ = model.encode_image(images)  # 正常前向；在 hook 中收集中间层输出
        outs = catcher.pop()  # {'blk05': Tensor[B,N,D], ...}（均在 CPU）
    """
    def __init__(self, visual, block_indices):
        self.visual = visual
        self.indices = sorted(set(int(i) for i in block_indices))
        self.buf = {}
        self.handles = []

        resblocks = list(self.visual.transformer.resblocks)

        def _make_hook(idx):
            key = f"blk{idx:02d}"
            def hook(module, inp, out):
                # resblock 输出是 [L, B, D]；保存为 [B, L, D]（含 CLS）
                out_bld = out.permute(1, 0, 2).contiguous()  # -> [B, L, D]
                self.buf[key] = out_bld.detach().cpu().to(torch.float32)

            return hook

        for idx in self.indices:
            h = resblocks[idx].register_forward_hook(_make_hook(idx))
            self.handles.append(h)

    def pop(self):
        out = self.buf
        self.buf = {}
        return out

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

# ---------- 回放：从下一 block 继续到 visual.proj ----------
@torch.no_grad()
def continue_from_tokens_to_imgfeat(model, tokens_bld, start_block_idx):
    """
    tokens_bld: [B, L, D]（含 CLS, L = 1 + H*W/patch^2）
    从 start_block_idx 的下一层继续跑到末层，然后取 CLS → ln_post → proj
    """
    visual = model.visual
    # BLD -> LBD（与 CLIP 内部 transformer 一致）
    x = tokens_bld.permute(1, 0, 2).contiguous()  # [L, B, D]

    # 继续后续 resblocks（start_block_idx + 1 ... end）
    for i in range(start_block_idx + 1, len(visual.transformer.resblocks)):
        x = visual.transformer.resblocks[i](x)   # x: [L, B, D]

    # LBD -> BLD，读出 CLS → ln_post → proj
    x = x.permute(1, 0, 2).contiguous()          # [B, L, D]
    cls = x[:, 0, :]                             # [B, D]
    if hasattr(visual, "ln_post") and visual.ln_post is not None:
        cls = visual.ln_post(cls)                # 先取 CLS，再 ln_post（与官方一致）
    elif hasattr(visual, "ln_final") and visual.ln_final is not None:
        # 少数实现会有 ln_final 作用在全序列；无 ln_post 时可不处理
        pass

    if hasattr(visual, "proj") and visual.proj is not None:
        if isinstance(visual.proj, torch.Tensor):
            img_feat = cls @ visual.proj         # [B, D_out]
        else:
            img_feat = visual.proj(cls)
    else:
        img_feat = cls
    return img_feat


@torch.no_grad()
def logits_from_imgfeat_and_text(model, img_feat, text_emb):
    """
    img_feat: [B, D]（未归一化）；text_emb: [C, D]（建议已 L2 归一化）
    返回 logits: [B, C]
    """
    # 归一化图像特征
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    if text_emb is None:
        raise ValueError("text_emb is None")
    # 若 text_emb 未归一化，可在载入时手动归一化；此处默认已归一化
    logit_scale = model.logit_scale.exp()
    return logit_scale * img_feat @ text_emb.t()

# ---------- 子命令 ----------
@torch.no_grad()
def cmd_extract(args):
    device = args.device
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    blocks = [int(x) for x in args.blocks.split(",")]  # e.g., 5,11,17,23
    catcher = ResblockCatcher(model.visual, blocks)

    pairs = load_list(args.list)
    os.makedirs(args.out_root, exist_ok=True)
    for b in blocks:
        os.makedirs(os.path.join(args.out_root, f"blk{b:02d}"), exist_ok=True)

    t0, n = time.time(), 0
    for wnid, base in tqdm(pairs, desc="Extracting"):
        img_path = os.path.join(args.root, wnid, base + ".JPEG")
        if not os.path.isfile(img_path):
            print(f"[warn] missing image: {img_path}")
            continue
        img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

        # 正常前向；在 hook 中取中间层输出
        _ = model.encode_image(img)

        outs = catcher.pop()  # {'blk05': [1,N,D], ...}
        for b in blocks:
            key = f"blk{b:02d}"
            arr = outs[key].squeeze(0).numpy().astype(np.float32)  # [N, D]

            save_path = os.path.join(args.out_root, key, f"{base}.npy")
            np.save(save_path, arr)
        n += 1

    catcher.close()
    print(f"[extract][CLIP ViT-L/14] N={n} blocks={blocks} out_root={args.out_root} ({time.time()-t0:.2f}s)")

@torch.no_grad()
def cmd_replay(args):
    device = args.device
    labels = load_labels(args.labels)
    layers = args.layer            # 如 ["blk05" "blk11"]

    # 模型（统一 float32，避免半精冲突）
    model, _ = clip.load("ViT-L/14", device=device)
    model.eval().float()

    # 文本嵌入：优先根据 classnames 现场生成（推荐），否则从 .npy 载入后强制归一化
    if args.classnames:
        text_emb = build_text_emb_from_classnames(model, args.classnames, device, template=args.template)
    else:
        assert args.text_emb is not None, "请提供 --classnames 或 --text_emb 之一"
        text_emb = torch.from_numpy(np.load(args.text_emb)).to(device)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        text_emb = text_emb.to(torch.float32)
    print("BLK\t\tAcc@1\t\tAcc@5\t\tTime")
    for layer in layers:

        feat_dir = os.path.join(args.feature_root, layer)
        files = sorted(glob.glob(os.path.join(feat_dir, "*.npy")))
        if not files:
            raise RuntimeError(f"未找到特征：{feat_dir}/*.npy")
        start_idx = int(layer[-2:])  # 23
        top1 = top5 = 0
        n = 0
        t0 = time.time()
        for p in files:
            base = os.path.splitext(os.path.basename(p))[0]
            gt = labels.get(base, None)
            if gt is None:
                continue
            arr = np.load(p)  # 现在约定保存为 [B, L, D]（含 CLS）
            tok = torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=torch.float32)

            # 形状校验（不做自动分支）
            seq_len_ref = model.visual.positional_embedding.shape[0]  # L = 1 + H*W/patch^2
            assert tok.ndim == 3 and tok.shape[1] == seq_len_ref, \
                f"token 形状应为 [B,{seq_len_ref},D]（含 CLS）；实际 {tok.shape}，文件：{p}"

            img_feat = continue_from_tokens_to_imgfeat(model, tok, start_idx)

            logits = logits_from_imgfeat_and_text(model, img_feat, text_emb)   # [1, C]
            top5_idx = torch.topk(logits, k=5, dim=-1).indices.squeeze(0).tolist()

            if top5_idx[0] == gt: top1 += 1
            if gt in top5_idx:    top5 += 1
            n += 1
        print(f"{layer}\t\t{top1/n*100:.2f}%\t\t{top5/n*100:.2f}%\t\t{time.time()-t0:.2f}s")

# ---------- CLI ----------
def build_parser():
    ap = argparse.ArgumentParser("CLIP ViT-L/14：分层抽取 / 分层回放（极简 .npy，类别就地对齐版）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="hook 抽取 resblocks 输出（含 CLS）并保存 .npy")
    pe.add_argument('--root', required=True, help='ImageNet val 根目录')
    pe.add_argument('--list', required=True, help='500张列表 txt：<wnid> <basename>')
    pe.add_argument('--out_root', required=True, help='特征输出根目录，如 features/clip_vitl14')
    pe.add_argument('--blocks', default='5,11,17,23', help='0-based resblock 索引，逗号分隔，如 5,11,17,23')
    pe.add_argument('--device', default='cuda')

    pr = sub.add_parser("replay", help='从某 block 的 .npy 重载，继续前向到 visual.proj，再与 text_emb 做零样本评测')
    pr.add_argument('--feature_root', required=True, help='extract 的 out_root')
    pr.add_argument('--layer', type=str, required=True, nargs="*", help='["blk23", "blk11", "blk05"]')
    pr.add_argument('--labels', required=True, help='500张 label txt：<basename> <idx>')
    # 二选一：classnames 优先；若无，则提供 text_emb.npy
    pr.add_argument('--classnames', default=None, help='1000 类文件：每行 <wnid> <class name...>，行序=官方索引')
    pr.add_argument('--template', default='a photo of a {}', help='生成 prompt 的模板')
    pr.add_argument('--text_emb', default=None, help='预先保存的 text_emb.npy，形状 [C, D]；若提供则会强制归一化')
    pr.add_argument('--device', default='cuda')

    return ap

def main():
    ap = build_parser()
    args = ap.parse_args()
    if args.cmd == 'extract':
        # 预建目录
        if hasattr(args, 'blocks'):
            for b in [int(x) for x in args.blocks.split(",")]:
                os.makedirs(os.path.join(args.out_root, f"blk{b:02d}"), exist_ok=True)
        cmd_extract(args)
    elif args.cmd == 'replay':
        cmd_replay(args)

if __name__ == '__main__':
    main()
