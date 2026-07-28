# -*- coding: utf-8 -*-
"""
验证 SigLIP2 So400m 特征提取正确性：
1) 直接推理 (end-to-end) 零样本分类精度（anchor）
2) 各层 (blk07/blk15/blk23) 回放精度
3) 逐样本对比回放 image_emb 与直接推理 image_emb 的 CosSim
"""

import os, sys, time, argparse, glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from siglip2_feat_pipeline import (
    load_siglip2, build_text_emb, continue_from_tokens,
    compute_logits, load_list, load_labels,
    load_classnames_wnid_format, EncoderLayerCatcher,
)


@torch.no_grad()
def direct_inference(model, processor, img_path, device):
    """直接推理，返回 image_emb [1, D]"""
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    out = model.get_image_features(**inputs)
    img_emb = out.pooler_output if hasattr(out, 'pooler_output') else out
    return img_emb.float()


@torch.no_grad()
def replay_inference(vision_model, feat_path, start_layer_idx, device):
    """从保存特征回放，返回 image_emb [1, D]"""
    arr = np.load(feat_path)
    tok = torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=torch.float32)
    img_emb = continue_from_tokens(vision_model, tok, start_layer_idx)
    return img_emb.float()


def cos_sim(a, b):
    a = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    b = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return (a * b).sum(dim=-1).item()


def main():
    ap = argparse.ArgumentParser("验证 SigLIP2 特征提取正确性")
    ap.add_argument('--model_id', default='google/siglip2-so400m-patch14-224')
    ap.add_argument('--cache_dir', default=None)
    ap.add_argument('--img_root', required=True, help='ImageNet val 根目录')
    ap.add_argument('--list', required=True, help='pathname txt: <wnid> <basename>')
    ap.add_argument('--labels', required=True, help='label txt: <basename> <idx>')
    ap.add_argument('--classnames', required=True, help='classnames.txt')
    ap.add_argument('--feature_root', required=True, help='提取的特征根目录')
    ap.add_argument('--layers', default='7,15,23', help='要验证的层 (0-based)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max_samples', type=int, default=0,
                    help='最大验证样本数，0 = 全部')
    args = ap.parse_args()

    device = args.device
    layers = [int(x) for x in args.layers.split(",")]
    blk_names = [f"blk{l:02d}" for l in layers]

    print("Loading model ...")
    model, processor = load_siglip2(args.model_id, device, cache_dir=args.cache_dir)
    model.float()
    vision_model = model.vision_model

    print("Building text embeddings ...")
    text_emb = build_text_emb(model, processor, args.classnames, device)

    pairs = load_list(args.list)
    labels = load_labels(args.labels)

    if args.max_samples > 0:
        pairs = pairs[:args.max_samples]

    n_total = len(pairs)
    print(f"Verifying {n_total} samples, layers={layers}")

    direct_top1 = 0
    direct_top5 = 0
    replay_top1 = {b: 0 for b in blk_names}
    replay_top5 = {b: 0 for b in blk_names}
    cossim_sums = {b: 0.0 for b in blk_names}
    cossim_mins = {b: 1.0 for b in blk_names}
    n = 0

    for wnid, base in tqdm(pairs, desc="Verifying"):
        gt = labels.get(base)
        if gt is None:
            continue

        img_path = os.path.join(args.img_root, wnid, base + ".JPEG")
        if not os.path.isfile(img_path):
            continue

        direct_emb = direct_inference(model, processor, img_path, device)
        direct_emb_norm = direct_emb / direct_emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        logits_direct = compute_logits(model, direct_emb, text_emb)
        top5_direct = torch.topk(logits_direct, k=5, dim=-1).indices.squeeze(0).tolist()
        if top5_direct[0] == gt:
            direct_top1 += 1
        if gt in top5_direct:
            direct_top5 += 1

        for blk, layer_idx in zip(blk_names, layers):
            feat_path = os.path.join(args.feature_root, blk, f"{base}.npy")
            if not os.path.isfile(feat_path):
                continue

            rep_emb = replay_inference(vision_model, feat_path, layer_idx, device)
            rep_emb_norm = rep_emb / rep_emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)

            cs = cos_sim(direct_emb_norm, rep_emb_norm)
            cossim_sums[blk] += cs
            cossim_mins[blk] = min(cossim_mins[blk], cs)

            logits_rep = compute_logits(model, rep_emb, text_emb)
            top5_rep = torch.topk(logits_rep, k=5, dim=-1).indices.squeeze(0).tolist()
            if top5_rep[0] == gt:
                replay_top1[blk] += 1
            if gt in top5_rep:
                replay_top5[blk] += 1

        n += 1

    print(f"\n{'='*60}")
    print(f"验证结果 ({n} 样本)")
    print(f"{'='*60}")
    print(f"{'方法':<12} {'Acc@1':>8} {'Acc@5':>8} {'AvgCosSim':>10} {'MinCosSim':>10}")
    print(f"{'-'*60}")
    print(f"{'直接推理':<12} {direct_top1/n*100:>7.2f}% {direct_top5/n*100:>7.2f}% {'---':>10} {'---':>10}")
    for blk in blk_names:
        avg_cs = cossim_sums[blk] / n
        min_cs = cossim_mins[blk]
        print(f"{blk+'回放':<12} {replay_top1[blk]/n*100:>7.2f}% {replay_top5[blk]/n*100:>7.2f}% {avg_cs:>10.6f} {min_cs:>10.6f}")

    print(f"\n回放精度 vs 直接推理差异:")
    for blk in blk_names:
        diff1 = replay_top1[blk]/n*100 - direct_top1/n*100
        diff5 = replay_top5[blk]/n*100 - direct_top5/n*100
        status = "PASS" if abs(diff1) < 0.01 and abs(diff5) < 0.01 else "MISMATCH"
        print(f"  {blk}: dAcc@1={diff1:+.2f}%  dAcc@5={diff5:+.2f}%  [{status}]")

    all_pass = all(
        abs(replay_top1[b]/n - direct_top1/n) < 1e-4 and
        abs(replay_top5[b]/n - direct_top5/n) < 1e-4
        for b in blk_names
    )
    print(f"\n总体结论: {'ALL PASS - 回放精度与直接推理完全一致' if all_pass else 'MISMATCH - 存在精度差异，需排查'}")


if __name__ == '__main__':
    main()
