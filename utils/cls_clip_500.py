# featcodec/tools/cls_clip_500.py
# -*- coding: utf-8 -*-
import os, time, argparse
import torch
import clip
from PIL import Image

TEMPLATE = "a photo of a {}"

def load_list(list_txt):
    # [(wnid, basename)]
    with open(list_txt, 'r') as f:
        return [line.strip().split() for line in f if line.strip()]

def load_labels(label_txt):
    # base -> int idx（idx 必须与 classnames 的行序一致）
    m = {}
    with open(label_txt, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            base, idx = ln.split()
            m[base] = int(idx)
    return m

def load_classnames_wnid_format(path):
    """
    解析 classnames.txt，格式：<wnid> <class name...>
    行序即类别索引（0..999）。
    返回：
      names: [class_name_str_0, class_name_str_1, ...]  长度=1000
      wnids: [wnid_0, wnid_1, ...]（可选用来核对）
    """
    names, wnids = [], []
    with open(path, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            wnid = parts[0]
            cls_name = " ".join(parts[1:])  # 保留空格
            wnids.append(wnid)
            names.append(cls_name)
    assert len(names) > 0, "classnames 文件为空"
    return names, wnids

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='ImageNet val 根目录（含 wnid 子目录）')
    ap.add_argument('--list', required=True, help='500张列表 txt：<wnid> <basename>')
    ap.add_argument('--labels', required=True, help='500 张的 label txt：<basename> <idx>')
    ap.add_argument('--classnames', required=True, help='1000 类文件：每行 <wnid> <class name...>，行序=官方索引')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--download_root', default=None, help='clip 本地缓存目录（可选）')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu'
    model, preprocess = clip.load("ViT-L/14", device=device, download_root=args.download_root)
    model.eval().to(device)

    names, wnids = load_classnames_wnid_format(args.classnames)
    prompts = [TEMPLATE.format(n) for n in names]

    pairs = load_list(args.list)
    labels = load_labels(args.labels)

    # 安全检查：label 索引不能越界
    max_label = max(labels.values())
    assert max_label < len(names), f"label 索引越界：{max_label} >= {len(names)}"

    n = len(pairs)
    top1 = top5 = 0
    t0 = time.time()
    text = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
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
        x = preprocess(img).unsqueeze(0).to(device)
        img_feat = model.encode_image(x)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        logits_per_image = 100.0 * img_feat @ text_feat.t()  # (1,1000)
        # logits_per_image, _ = model(x, text)  # [1, 1000]
        top5_idx = torch.topk(logits_per_image, k=5, dim=-1).indices.squeeze(0).tolist()

        if top5_idx[0] == gt:
            top1 += 1
        if gt in top5_idx:
            top5 += 1

    dt = time.time() - t0
    print(f"[CLIP ViT-L/14 zero-shot] N={n}  Top-1={top1/n*100:.2f}%  Top-5={top5/n*100:.2f}%  ({dt:.2f}s)")

if __name__ == '__main__':
    main()
