# -*- coding: utf-8 -*-
"""
SigLIP2 So400m-patch14-224 COCO Image-to-Text Retrieval 评估：
- direct : 完整模型直接推理 → image_emb, text_emb → 检索评估
- replay : 从中间层 .npy 回放 → image_emb, 文本直接编码 → 检索评估
- 支持同时对比 direct 与多层 replay 的 R@1/5/10

用法：
  # 直接推理
  python siglip2_retrieval.py direct \
      --meta_json /path/to/subset_meta.json \
      --image_root /path/to/coco2014

  # 回放评估（可同时评估多层）
  python siglip2_retrieval.py replay \
      --meta_json /path/to/subset_meta.json \
      --feature_root /path/to/features \
      --layers blk07 blk15 blk23

  # 一次性对比 direct + replay
  python siglip2_retrieval.py compare \
      --meta_json /path/to/subset_meta.json \
      --image_root /path/to/coco2014 \
      --feature_root /path/to/features \
      --layers blk07 blk15 blk23
"""

import os, sys, json, time, argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from siglip2_feat_pipeline import load_siglip2, continue_from_tokens


# ──────────────────────── 数据加载 ────────────────────────

def load_meta(meta_json):
    """加载由 siglip2_ret_extract.py 生成的 subset_meta.json。
    返回 entries, img2txt, txt2img。
    """
    with open(meta_json, 'r') as f:
        entries = json.load(f)

    img2txt = {}
    txt2img = {}
    caption_list = []
    cap_idx = 0
    for i, e in enumerate(entries):
        gt_indices = []
        for cap in e['caption']:
            caption_list.append(cap)
            txt2img[cap_idx] = i
            gt_indices.append(cap_idx)
            cap_idx += 1
        img2txt[i] = gt_indices

    return entries, caption_list, img2txt, txt2img


# ──────────────────────── 文本编码 ────────────────────────

@torch.no_grad()
def encode_texts(model, processor, captions, device, batch_size=128):
    """编码所有 caption → [N_txt, D]，已 L2 归一化。"""
    all_emb = []
    for i in tqdm(range(0, len(captions), batch_size), desc="Encoding texts"):
        batch = captions[i:i + batch_size]
        inputs = processor(
            text=batch,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        out = model.get_text_features(**inputs)
        emb = out.pooler_output if hasattr(out, 'pooler_output') else out
        emb = emb.float()
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        all_emb.append(emb.cpu())
    return torch.cat(all_emb, dim=0)


# ──────────────────────── 图片编码（直接推理） ────────────────────────

@torch.no_grad()
def encode_images_direct(model, processor, entries, image_root, device,
                         batch_size=32):
    """直接推理编码图片 → [N_img, D]，已 L2 归一化。"""
    all_emb = []
    for i in tqdm(range(0, len(entries), batch_size), desc="Encoding images"):
        batch_entries = entries[i:i + batch_size]
        imgs = []
        for e in batch_entries:
            path = os.path.join(image_root, e['image'])
            try:
                imgs.append(Image.open(path).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (224, 224)))

        inputs = processor(images=imgs, return_tensors="pt").to(device)
        out = model.get_image_features(**inputs)
        emb = out.pooler_output if hasattr(out, 'pooler_output') else out
        emb = emb.float()
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        all_emb.append(emb.cpu())
    return torch.cat(all_emb, dim=0)


# ──────────────────────── 图片编码（中间层回放） ────────────────────────

@torch.no_grad()
def encode_images_replay(model, entries, feature_root, blk_name, device):
    """从中间层特征回放 → [N_img, D]，已 L2 归一化。"""
    vision_model = model.vision_model
    start_idx = int(blk_name.replace("blk", ""))
    feat_dir = os.path.join(feature_root, blk_name)

    all_emb = []
    for e in tqdm(entries, desc=f"Replay ({blk_name})"):
        img_id = e['image_id']
        feat_path = os.path.join(feat_dir, f"{img_id}.npy")
        arr = np.load(feat_path)
        tok = torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=torch.float32)
        emb = continue_from_tokens(vision_model, tok, start_idx)
        emb = emb.float()
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        all_emb.append(emb.cpu())
    return torch.cat(all_emb, dim=0)


# ──────────────────────── 检索评估 ────────────────────────

def compute_retrieval_scores(img_emb, txt_emb, model, device):
    """计算相似度矩阵 [N_img, N_txt]。
    使用 SigLIP2 的 logit_scale（bias 为标量，不影响排序）。
    """
    logit_scale = model.logit_scale.exp().item()
    score = logit_scale * (img_emb @ txt_emb.t())
    return score.numpy()


def eval_retrieval(score_matrix, img2txt, txt2img):
    """计算 I2T / T2I 的 R@1, R@5, R@10。"""
    n_img, n_txt = score_matrix.shape

    # Image-to-Text
    i2t_r1 = i2t_r5 = i2t_r10 = 0
    for i in range(n_img):
        ranking = np.argsort(-score_matrix[i])
        gt_set = set(img2txt[i])
        for rank, idx in enumerate(ranking):
            if idx in gt_set:
                if rank < 1:
                    i2t_r1 += 1
                if rank < 5:
                    i2t_r5 += 1
                if rank < 10:
                    i2t_r10 += 1
                break

    # Text-to-Image
    t2i_r1 = t2i_r5 = t2i_r10 = 0
    for j in range(n_txt):
        ranking = np.argsort(-score_matrix[:, j])
        gt = txt2img[j]
        rank = np.where(ranking == gt)[0][0]
        if rank < 1:
            t2i_r1 += 1
        if rank < 5:
            t2i_r5 += 1
        if rank < 10:
            t2i_r10 += 1

    results = {
        "i2t_R@1":  i2t_r1  / n_img * 100,
        "i2t_R@5":  i2t_r5  / n_img * 100,
        "i2t_R@10": i2t_r10 / n_img * 100,
        "t2i_R@1":  t2i_r1  / n_txt * 100,
        "t2i_R@5":  t2i_r5  / n_txt * 100,
        "t2i_R@10": t2i_r10 / n_txt * 100,
    }
    return results


def print_results(label, results):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Image→Text   R@1={results['i2t_R@1']:.2f}%  "
          f"R@5={results['i2t_R@5']:.2f}%  R@10={results['i2t_R@10']:.2f}%")
    print(f"  Text→Image   R@1={results['t2i_R@1']:.2f}%  "
          f"R@5={results['t2i_R@5']:.2f}%  R@10={results['t2i_R@10']:.2f}%")


# ──────────────────────── 子命令 ────────────────────────

def cmd_direct(args):
    device = args.device
    entries, captions, img2txt, txt2img = load_meta(args.meta_json)
    print(f"数据: {len(entries)} 张图片, {len(captions)} 条 caption")

    model, processor = load_siglip2(args.model_id, device, cache_dir=args.cache_dir)
    model.float()

    t0 = time.time()
    txt_emb = encode_texts(model, processor, captions, device,
                           batch_size=args.text_batch_size)
    img_emb = encode_images_direct(model, processor, entries, args.image_root,
                                   device, batch_size=args.image_batch_size)

    score_matrix = compute_retrieval_scores(img_emb, txt_emb, model, device)
    results = eval_retrieval(score_matrix, img2txt, txt2img)
    elapsed = time.time() - t0

    print_results(f"Direct Inference ({elapsed:.1f}s)", results)
    _save_results(args.output, "direct", results)
    return results


def cmd_replay(args):
    device = args.device
    entries, captions, img2txt, txt2img = load_meta(args.meta_json)
    print(f"数据: {len(entries)} 张图片, {len(captions)} 条 caption")

    model, processor = load_siglip2(args.model_id, device, cache_dir=args.cache_dir)
    model.float()

    txt_emb = encode_texts(model, processor, captions, device,
                           batch_size=args.text_batch_size)

    all_results = {}
    for blk in args.layers:
        t0 = time.time()
        img_emb = encode_images_replay(model, entries, args.feature_root,
                                       blk, device)
        score_matrix = compute_retrieval_scores(img_emb, txt_emb, model, device)
        results = eval_retrieval(score_matrix, img2txt, txt2img)
        elapsed = time.time() - t0

        print_results(f"Replay {blk} ({elapsed:.1f}s)", results)
        all_results[blk] = results

    _save_results(args.output, "replay", all_results)
    return all_results


def cmd_compare(args):
    """一次性对比 direct + 多层 replay。"""
    device = args.device
    entries, captions, img2txt, txt2img = load_meta(args.meta_json)
    print(f"数据: {len(entries)} 张图片, {len(captions)} 条 caption")

    model, processor = load_siglip2(args.model_id, device, cache_dir=args.cache_dir)
    model.float()

    txt_emb = encode_texts(model, processor, captions, device,
                           batch_size=args.text_batch_size)

    all_results = {}

    # Direct
    t0 = time.time()
    img_emb_direct = encode_images_direct(
        model, processor, entries, args.image_root,
        device, batch_size=args.image_batch_size)
    score_direct = compute_retrieval_scores(img_emb_direct, txt_emb, model, device)
    res_direct = eval_retrieval(score_direct, img2txt, txt2img)
    print_results(f"Direct Inference ({time.time()-t0:.1f}s)", res_direct)
    all_results["direct"] = res_direct

    # Replay per layer
    for blk in args.layers:
        t0 = time.time()
        img_emb_rep = encode_images_replay(model, entries, args.feature_root,
                                           blk, device)
        score_rep = compute_retrieval_scores(img_emb_rep, txt_emb, model, device)
        res_rep = eval_retrieval(score_rep, img2txt, txt2img)
        print_results(f"Replay {blk} ({time.time()-t0:.1f}s)", res_rep)
        all_results[blk] = res_rep

        # cosine similarity between direct and replay embeddings
        cs = torch.nn.functional.cosine_similarity(
            img_emb_direct, img_emb_rep, dim=-1)
        print(f"  CosSim vs Direct: mean={cs.mean():.6f}  min={cs.min():.6f}")

    # Summary table
    print(f"\n{'='*72}")
    print(f"  {'方法':<16} {'i2t R@1':>8} {'i2t R@5':>8} {'i2t R@10':>9} "
          f"{'t2i R@1':>8} {'t2i R@5':>8} {'t2i R@10':>9}")
    print(f"{'─'*72}")
    for key, res in all_results.items():
        print(f"  {key:<16} {res['i2t_R@1']:>7.2f}% {res['i2t_R@5']:>7.2f}% "
              f"{res['i2t_R@10']:>8.2f}% {res['t2i_R@1']:>7.2f}% "
              f"{res['t2i_R@5']:>7.2f}% {res['t2i_R@10']:>8.2f}%")
    print(f"{'='*72}")

    _save_results(args.output, "compare", all_results)
    return all_results


def _save_results(output_path, prefix, results):
    if output_path is None:
        return
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    path = output_path if output_path.endswith('.json') \
        else f"{output_path}_{prefix}.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"结果已保存: {path}")


# ──────────────────────── CLI ────────────────────────

def _add_common_args(parser):
    parser.add_argument('--model_id', default='google/siglip2-so400m-patch14-224')
    parser.add_argument('--cache_dir', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--text_batch_size', type=int, default=128)
    parser.add_argument('--output', default=None, help='结果输出 JSON 路径')


def build_parser():
    ap = argparse.ArgumentParser(
        "SigLIP2 COCO Image-to-Text Retrieval 评估")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ── direct ──
    pd = sub.add_parser("direct", help="完整模型直接推理检索评估")
    _add_common_args(pd)
    pd.add_argument('--meta_json', required=True,
                    help='subset_meta.json (由 siglip2_ret_extract.py 生成)')
    pd.add_argument('--image_root', required=True,
                    help='COCO 图片根目录')
    pd.add_argument('--image_batch_size', type=int, default=32)

    # ── replay ──
    pr = sub.add_parser("replay", help="从中间层特征回放检索评估")
    _add_common_args(pr)
    pr.add_argument('--meta_json', required=True)
    pr.add_argument('--feature_root', required=True,
                    help='特征根目录 (含 blkXX/ 子目录)')
    pr.add_argument('--layers', nargs='+', default=['blk07', 'blk15', 'blk23'],
                    help='回放层名')

    # ── compare ──
    pc = sub.add_parser("compare", help="对比 direct + 多层 replay")
    _add_common_args(pc)
    pc.add_argument('--meta_json', required=True)
    pc.add_argument('--image_root', required=True)
    pc.add_argument('--feature_root', required=True)
    pc.add_argument('--layers', nargs='+', default=['blk07', 'blk15', 'blk23'])
    pc.add_argument('--image_batch_size', type=int, default=32)

    return ap


def main():
    args = build_parser().parse_args()
    if args.cmd == 'direct':
        cmd_direct(args)
    elif args.cmd == 'replay':
        cmd_replay(args)
    elif args.cmd == 'compare':
        cmd_compare(args)


if __name__ == '__main__':
    main()
