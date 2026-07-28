# -*- coding: utf-8 -*-
"""
SigLIP2 So400m-patch14-224 特征管线（分层抽取 / 分层回放零样本分类）：
- extract：按 --layers 指定的 encoder layer（0-based）注册 forward hook，
           保存每层输出 [N, D] 为 .npy（float32）。SigLIP2 无 CLS token，
           N = (224/14)^2 = 256 patch tokens。
- replay ：从某 layer 的 .npy 重载 tokens（[N, D]），从该层的下一层继续
           前向到末层 → post_layernorm → MultiheadAttentionPoolingHead，
           得到 image_emb；文本嵌入 text_emb 由 processor + model 在回放
           阶段就地生成。logits = scale * img@text.T + bias。

输入文件格式：
  --list   每行: "<wnid> <basename>"
  --labels 每行: "<basename> <idx>"

依赖：transformers>=4.49, torch>=2.1, sentencepiece
模型：google/siglip2-so400m-patch14-224
"""

import os, sys, time, argparse, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ---------- I/O ----------
def load_list(list_txt):
    with open(list_txt, 'r') as f:
        return [line.strip().split() for line in f if line.strip()]

def load_labels(label_txt):
    m = {}
    with open(label_txt, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            base, idx = ln.split()
            m[base] = int(idx)
    return m

def load_classnames_wnid_format(path):
    """解析 classnames.txt，格式：<wnid> <class name...>；行序即类别索引（0..999）。"""
    names, wnids = [], []
    with open(path, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            wnids.append(parts[0])
            names.append(" ".join(parts[1:]))
    assert len(names) > 0, "classnames 文件为空"
    return names, wnids

# ---------- 模型加载 ----------
def load_siglip2(model_id, device, cache_dir=None):
    from transformers import AutoModel, AutoProcessor
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    processor = AutoProcessor.from_pretrained(model_id, **kwargs)
    model = AutoModel.from_pretrained(model_id, **kwargs).eval().to(device)
    return model, processor

# ---------- 文本嵌入 ----------
@torch.no_grad()
def build_text_emb(model, processor, classnames_path, device,
                   template="This is a photo of {}."):
    """
    SigLIP2 文本嵌入：processor 自动 lowercase + padding=max_length, max_length=64。
    返回 [C, D]，已 L2 归一化。
    """
    names, _ = load_classnames_wnid_format(classnames_path)
    prompts = [template.format(n) for n in names]

    text_inputs = processor(
        text=prompts,
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    text_out = model.get_text_features(**text_inputs)
    text_emb = text_out.pooler_output if hasattr(text_out, 'pooler_output') else text_out  # [C, D]
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return text_emb.float()

# ---------- 抽取：forward hook ----------
class EncoderLayerCatcher:
    """
    注册到 model.vision_model.encoder.layers[k] 的 forward hook。
    SigLIP2 encoder layer 输出是 tuple，hidden_states 在 [0]，形状 [B, N, D]。
    """
    def __init__(self, vision_model, layer_indices):
        self.indices = sorted(set(int(i) for i in layer_indices))
        self.buf = {}
        self.handles = []

        layers = vision_model.encoder.layers
        for idx in self.indices:
            key = f"blk{idx:02d}"
            def _hook(module, inp, out, _key=key):
                if isinstance(out, tuple):
                    hs = out[0]
                else:
                    hs = out
                self.buf[_key] = hs.detach().cpu().float()   # [B, N, D]
            self.handles.append(layers[idx].register_forward_hook(_hook))

    def pop(self):
        out = self.buf
        self.buf = {}
        return out

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

# ---------- 回放：从中间层继续前向 ----------
@torch.no_grad()
def continue_from_tokens(vision_model, tokens_bnd, start_layer_idx):
    """
    tokens_bnd: [B, N, D]（N = 256 patch tokens，无 CLS）
    从 start_layer_idx+1 继续到末层 → post_layernorm → head (MAP)
    返回 pooled image embedding [B, D_out]
    """
    x = tokens_bnd
    layers = vision_model.encoder.layers
    for i in range(start_layer_idx + 1, len(layers)):
        layer_out = layers[i](x, attention_mask=None)
        x = layer_out[0] if isinstance(layer_out, tuple) else layer_out

    x = vision_model.post_layernorm(x)                       # [B, N, D]

    if hasattr(vision_model, 'head') and vision_model.head is not None:
        pooled = vision_model.head(x)                         # [B, D]
    else:
        pooled = x.mean(dim=1)
    return pooled

@torch.no_grad()
def compute_logits(model, img_emb, text_emb):
    """
    SigLIP2: logits = scale * cosine + bias
    """
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    logit_scale = model.logit_scale.exp()
    logit_bias = model.logit_bias
    return logit_scale * img_emb @ text_emb.t() + logit_bias

# ---------- 图片数据集 ----------
class ImageListDataset(Dataset):
    """从 list_txt 加载 (wnid, basename) 对，返回 PIL Image 和 basename。"""
    def __init__(self, list_txt, root):
        self.root = root
        self.pairs = load_list(list_txt)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        wnid, base = self.pairs[idx]
        img_path = os.path.join(self.root, wnid, base + ".JPEG")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return img, base


def collate_pil(batch):
    imgs, bases = zip(*batch)
    return list(imgs), list(bases)


# ---------- 子命令：extract ----------
@torch.no_grad()
def cmd_extract(args):
    device = args.device
    model, processor = load_siglip2(args.model_id, device, cache_dir=args.cache_dir)
    vision_model = model.vision_model

    layers = [int(x) for x in args.layers.split(",")]
    num_layers = len(vision_model.encoder.layers)
    for l in layers:
        assert 0 <= l < num_layers, f"layer {l} 越界，模型共 {num_layers} 层"

    catcher = EncoderLayerCatcher(vision_model, layers)

    os.makedirs(args.out_root, exist_ok=True)
    for l in layers:
        os.makedirs(os.path.join(args.out_root, f"blk{l:02d}"), exist_ok=True)

    dataset = ImageListDataset(args.list, args.root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_pil,
        pin_memory=False,
        shuffle=False,
    )

    t0, n, skipped = time.time(), 0, 0
    for imgs, bases in tqdm(loader, desc="Extracting", total=len(loader)):
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        _ = model.get_image_features(**inputs)

        outs = catcher.pop()
        for b_idx, base in enumerate(bases):
            for l in layers:
                key = f"blk{l:02d}"
                arr = outs[key][b_idx].numpy().astype(np.float32)  # [N, D]
                np.save(os.path.join(args.out_root, key, f"{base}.npy"), arr)
            n += 1

    catcher.close()
    print(f"[extract][SigLIP2-So400m] N={n} layers={layers} "
          f"batch_size={args.batch_size} workers={args.num_workers} "
          f"out_root={args.out_root} ({time.time()-t0:.2f}s)")

# ---------- 子命令：replay ----------
@torch.no_grad()
def cmd_replay(args):
    device = args.device
    labels = load_labels(args.labels)
    blk_names = args.layer

    model, processor = load_siglip2(args.model_id, device, cache_dir=args.cache_dir)
    model.float()
    vision_model = model.vision_model

    if args.classnames:
        text_emb = build_text_emb(model, processor, args.classnames, device,
                                  template=args.template)
    else:
        assert args.text_emb is not None, "请提供 --classnames 或 --text_emb"
        text_emb = torch.from_numpy(np.load(args.text_emb)).to(device).float()
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    num_layers = len(vision_model.encoder.layers)
    seq_len_ref = (224 // 14) ** 2     # 256

    print("BLK\t\tAcc@1\t\tAcc@5\t\tTime")
    for blk in blk_names:
        feat_dir = os.path.join(args.feature_root, blk)
        files = sorted(glob.glob(os.path.join(feat_dir, "*.npy")))
        if not files:
            raise RuntimeError(f"未找到特征：{feat_dir}/*.npy")
        start_idx = int(blk[-2:])
        assert 0 <= start_idx < num_layers, f"layer {start_idx} 越界"

        top1 = top5 = n = 0
        t0 = time.time()
        for p in files:
            base = os.path.splitext(os.path.basename(p))[0]
            gt = labels.get(base)
            if gt is None:
                continue

            arr = np.load(p)
            tok = torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=torch.float32)

            assert tok.ndim == 3 and tok.shape[1] == seq_len_ref, \
                f"token 形状应为 [1,{seq_len_ref},D]；实际 {tok.shape}，文件：{p}"

            img_emb = continue_from_tokens(vision_model, tok, start_idx)
            logits = compute_logits(model, img_emb, text_emb)
            top5_idx = torch.topk(logits, k=5, dim=-1).indices.squeeze(0).tolist()

            if top5_idx[0] == gt:
                top1 += 1
            if gt in top5_idx:
                top5 += 1
            n += 1

        print(f"{blk}\t\t{top1/n*100:.2f}%\t\t{top5/n*100:.2f}%\t\t{time.time()-t0:.2f}s")

# ---------- CLI ----------
def build_parser():
    ap = argparse.ArgumentParser(
        "SigLIP2-So400m-patch14-224：分层抽取 / 分层回放零样本分类")
    ap.add_argument('--model_id', default='google/siglip2-so400m-patch14-224',
                    help='HuggingFace 模型 ID')
    ap.add_argument('--cache_dir', default=None,
                    help='模型缓存目录（默认 ~/.cache/huggingface）')
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- extract ---
    pe = sub.add_parser("extract",
        help="hook 抽取 encoder layer 输出并保存 .npy（[N,D]，N=256）")
    pe.add_argument('--root', required=True, help='ImageNet val 根目录')
    pe.add_argument('--list', required=True,
                    help='图片列表 txt：每行 <wnid> <basename>')
    pe.add_argument('--out_root', required=True,
                    help='特征输出根目录')
    pe.add_argument('--layers', default='7,15,23',
                    help='0-based encoder layer 索引，逗号分隔（共27层）')
    pe.add_argument('--batch_size', type=int, default=32,
                    help='推理 batch size')
    pe.add_argument('--num_workers', type=int, default=4,
                    help='DataLoader worker 数')
    pe.add_argument('--device', default='cuda')

    # --- replay ---
    pr = sub.add_parser("replay",
        help='从 .npy 重载中间层 tokens，继续前向 → MAP head → 零样本分类')
    pr.add_argument('--feature_root', required=True,
                    help='extract 的 out_root')
    pr.add_argument('--layer', type=str, required=True, nargs="*",
                    help='回放层名，如 blk26 blk20 blk13')
    pr.add_argument('--labels', required=True,
                    help='label txt：每行 <basename> <idx>')
    pr.add_argument('--classnames', default=None,
                    help='1000 类文件：每行 <wnid> <class name...>')
    pr.add_argument('--template', default='This is a photo of {}.',
                    help='文本 prompt 模板（SigLIP2 推荐 "This is a photo of {label}."）')
    pr.add_argument('--text_emb', default=None,
                    help='预存 text_emb.npy [C,D]')
    pr.add_argument('--device', default='cuda')

    return ap

def main():
    args = build_parser().parse_args()
    if args.cmd == 'extract':
        cmd_extract(args)
    elif args.cmd == 'replay':
        cmd_replay(args)

if __name__ == '__main__':
    main()
