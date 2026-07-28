#!/usr/bin/env python
"""
Hyperprior COCO Image-to-Text Retrieval evaluation for SigLIP2.

Evaluation pipeline (same metrics as eval_ret_soft_pq.py):
  - Anchor: raw feature replay (no compression)
  - Hyperprior: compress/decompress features with trained model, then replay

Usage:
    python eval_ret_hyperprior.py \
        --layer blk23 \
        --feat_root /path/to/coco_ret/siglip2_so400m \
        --meta_json /path/to/subset_meta.json \
        --ckpt_dir  /path/to/checkpoints/lamofc_v2
"""

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from compressai.ops import compute_padding
from compressai.zoo.image import model_architectures
from siglip2_feat_pipeline import load_siglip2, continue_from_tokens

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ---- Truncation ranges (must match training) ----------------------------

TRUN_RANGES = {
    "blk07": (-1, 1),
    "blk15": (-5, 5),
    "blk23": (-10, 10),
}

# ---- Data loading --------------------------------------------------------

def load_meta(meta_json):
    with open(meta_json, 'r') as f:
        entries = json.load(f)
    img2txt, txt2img, captions = {}, {}, []
    cap_idx = 0
    for i, e in enumerate(entries):
        gt = []
        for cap in e['caption']:
            captions.append(cap)
            txt2img[cap_idx] = i
            gt.append(cap_idx)
            cap_idx += 1
        img2txt[i] = gt
    return entries, captions, img2txt, txt2img


def load_features(feat_dir, entries):
    feats = []
    for e in entries:
        path = os.path.join(feat_dir, f"{e['image_id']}.npy")
        feats.append(np.load(path))
    return feats


# ---- Text encoding -------------------------------------------------------

@torch.no_grad()
def encode_texts(model, processor, captions, device, batch_size=128):
    all_emb = []
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i + batch_size]
        inputs = processor(
            text=batch, padding="max_length",
            max_length=64, truncation=True, return_tensors="pt",
        ).to(device)
        out = model.get_text_features(**inputs)
        emb = out.pooler_output if hasattr(out, 'pooler_output') else out
        emb = emb.float()
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        all_emb.append(emb.cpu())
    return torch.cat(all_emb, dim=0)


# ---- Image replay --------------------------------------------------------

@torch.no_grad()
def replay_to_embeddings(vision_model, features, start_idx, device,
                         batch_size=16):
    all_emb = []
    for i in range(0, len(features), batch_size):
        batch = features[i:i + batch_size]
        tok = torch.from_numpy(np.stack(batch)).to(
            device=device, dtype=torch.float32)
        emb = continue_from_tokens(vision_model, tok, start_idx)
        emb = emb.float()
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        all_emb.append(emb.cpu())
    return torch.cat(all_emb, dim=0)


# ---- Retrieval metrics ---------------------------------------------------

def eval_retrieval(score_matrix, img2txt, txt2img):
    n_img, n_txt = score_matrix.shape
    i2t_r1 = i2t_r5 = i2t_r10 = 0
    for i in range(n_img):
        ranking = np.argsort(-score_matrix[i])
        gt_set = set(img2txt[i])
        for rank, idx in enumerate(ranking):
            if idx in gt_set:
                if rank < 1:  i2t_r1 += 1
                if rank < 5:  i2t_r5 += 1
                if rank < 10: i2t_r10 += 1
                break
    t2i_r1 = t2i_r5 = t2i_r10 = 0
    for j in range(n_txt):
        ranking = np.argsort(-score_matrix[:, j])
        gt = txt2img[j]
        rank = int(np.where(ranking == gt)[0][0])
        if rank < 1:  t2i_r1 += 1
        if rank < 5:  t2i_r5 += 1
        if rank < 10: t2i_r10 += 1
    return {
        "i2t_R@1":  i2t_r1  / n_img * 100,
        "i2t_R@5":  i2t_r5  / n_img * 100,
        "i2t_R@10": i2t_r10 / n_img * 100,
        "t2i_R@1":  t2i_r1  / n_txt * 100,
        "t2i_R@5":  t2i_r5  / n_txt * 100,
        "t2i_R@10": t2i_r10 / n_txt * 100,
    }


# ---- Hyperprior compress / decompress ------------------------------------

def truncation(feat, trun_low, trun_high):
    return np.clip(feat, trun_low, trun_high).astype(np.float32)


def uniform_quantization(feat, min_v, max_v, bit_depth=1):
    scale = ((2 ** bit_depth) - 1) / (max_v - min_v)
    return ((feat - min_v) * scale).astype(np.float32)


def uniform_dequantization(feat, min_v, max_v, bit_depth=1):
    scale = ((2 ** bit_depth) - 1) / (max_v - min_v)
    return (feat / scale + min_v).astype(np.float32)


def load_hyperprior_model(ckpt_path, device, model_name="bmshj2018-hyperprior"):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    model_cls = model_architectures[model_name]
    net = model_cls.from_state_dict(state_dict)
    net = net.to(device).eval()
    net.update(force=True)
    return net


@torch.no_grad()
def hyperprior_encode_decode(features, net, trun_low, trun_high,
                             bit_depth, device):
    """Compress + decompress features with hyperprior; return decoded features and avg BPFP."""
    decoded = []
    total_bits = 0
    total_points = 0
    nan_count = 0

    for feat in features:
        feat_t = truncation(feat, trun_low, trun_high)
        feat_q = uniform_quantization(feat_t, trun_low, trun_high, bit_depth)

        x = torch.from_numpy(feat_q).unsqueeze(0).unsqueeze(0).to(device)
        h, w = x.size(2), x.size(3)
        pad, unpad = compute_padding(h, w, min_div=2 ** 6)
        x_padded = F.pad(x, pad, mode="constant", value=0)

        out_enc = net.compress(x_padded)
        out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
        x_hat = F.pad(out_dec["x_hat"], unpad)

        if not torch.isfinite(x_hat).all():
            nan_count += 1
            decoded.append(feat.copy())
            continue

        num_points = h * w
        bits = sum(len(s[0]) for s in out_enc["strings"]) * 8.0
        total_bits += bits
        total_points += num_points

        x_hat_np = x_hat.squeeze(0).squeeze(0).cpu().numpy()
        feat_dec = uniform_dequantization(x_hat_np, trun_low, trun_high, bit_depth)
        decoded.append(feat_dec)

    bpfp = total_bits / total_points if total_points > 0 else 0.0
    if nan_count > 0:
        print(f"  WARNING: {nan_count} samples with NaN in decompress output")
    return decoded, bpfp


# ---- Checkpoint discovery ------------------------------------------------

def discover_checkpoints(ckpt_root, backbone, layer):
    """Find best checkpoints: .../lamofc_v2/{backbone}_{layer}_lmb{lambda}/checkpoint_best.pth.tar"""
    pattern = f"{backbone}_{layer}_lmb*"
    dirs = sorted(Path(ckpt_root).glob(pattern))
    configs = []
    for d in dirs:
        best = d / "checkpoint_best.pth.tar"
        if not best.exists():
            continue
        name = d.name
        lmb_str = name.split("_lmb")[-1]
        tag = f"lmb{lmb_str}"
        configs.append((tag, float(lmb_str), str(best)))
    configs.sort(key=lambda x: x[1])
    return configs


def _print_ret(res, ref=None):
    di = dt = ""
    if ref:
        di = f"  (dR@1={res['i2t_R@1'] - ref['i2t_R@1']:+.2f}%)"
        dt = f"  (dR@1={res['t2i_R@1'] - ref['t2i_R@1']:+.2f}%)"
    print(f"  I2T  R@1={res['i2t_R@1']:.2f}%  "
          f"R@5={res['i2t_R@5']:.2f}%  "
          f"R@10={res['i2t_R@10']:.2f}%{di}")
    print(f"  T2I  R@1={res['t2i_R@1']:.2f}%  "
          f"R@5={res['t2i_R@5']:.2f}%  "
          f"R@10={res['t2i_R@10']:.2f}%{dt}")


# ---- Main ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("Hyperprior COCO Retrieval Evaluation")
    ap.add_argument("--layer", type=str, required=True)
    ap.add_argument("--feat_root", type=str, required=True,
                    help="COCO ret feature root (contains blkXX/)")
    ap.add_argument("--meta_json", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, required=True,
                    help="Root of hyperprior checkpoints (e.g. .../lamofc_v2)")
    ap.add_argument("--backbone", type=str, default="siglip2_so400m")
    ap.add_argument("--model_id", type=str,
                    default="google/siglip2-so400m-patch14-224")
    ap.add_argument("--bit_depth", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    layer = args.layer
    start_idx = int(layer.replace("blk", ""))

    if layer not in TRUN_RANGES:
        ap.error(f"Unknown layer {layer}, supported: {list(TRUN_RANGES.keys())}")
    trun_low, trun_high = TRUN_RANGES[layer]

    entries, captions, img2txt, txt2img = load_meta(args.meta_json)
    print(f"Data: {len(entries)} images, {len(captions)} captions")

    configs = discover_checkpoints(args.ckpt_dir, args.backbone, layer)
    print(f"Layer {layer}: {len(configs)} hyperprior checkpoints")
    for tag, lmb, path in configs:
        print(f"  {tag}: {path}")

    feat_dir = os.path.join(args.feat_root, layer)
    print(f"\nLoading COCO ret features from {feat_dir} ...")
    features = load_features(feat_dir, entries)
    print(f"  {len(features)} x {features[0].shape}")

    print(f"\nLoading SigLIP2 model ...")
    model, processor = load_siglip2(args.model_id, device)
    model.float()
    vision_model = model.vision_model
    logit_scale = model.logit_scale.exp().item()

    print("Encoding texts ...")
    txt_emb = encode_texts(model, processor, captions, device)

    model.text_model.cpu()
    torch.cuda.empty_cache()

    # -- Anchor --
    print(f"\n{'=' * 70}")
    print(f"  Anchor: {layer} (raw feature replay)")
    print(f"{'=' * 70}")
    t0 = time.time()
    img_emb_anchor = replay_to_embeddings(
        vision_model, features, start_idx, device)
    score_anchor = logit_scale * (img_emb_anchor @ txt_emb.t())
    res_anchor = eval_retrieval(score_anchor.numpy(), img2txt, txt2img)
    _print_ret(res_anchor)
    print(f"  ({time.time() - t0:.1f}s)")

    all_results = {"anchor": res_anchor}

    # -- Hyperprior checkpoints --
    for tag, lmb, ckpt_path in configs:
        print(f"\n{'~' * 70}")
        print(f"  {layer} / HP_{tag}  (lambda={lmb})")
        print(f"{'~' * 70}")

        t0 = time.time()
        net = load_hyperprior_model(ckpt_path, device)

        decoded, bpfp = hyperprior_encode_decode(
            features, net, trun_low, trun_high, args.bit_depth, device)
        print(f"  BPFP = {bpfp:.4f}")

        img_emb_hp = replay_to_embeddings(
            vision_model, decoded, start_idx, device)
        score_hp = logit_scale * (img_emb_hp @ txt_emb.t())
        res_hp = eval_retrieval(score_hp.numpy(), img2txt, txt2img)

        cos = torch.nn.functional.cosine_similarity(
            img_emb_anchor, img_emb_hp, dim=-1)
        _print_ret(res_hp, res_anchor)
        print(f"  CosSim: mean={cos.mean():.6f}  min={cos.min():.6f}")
        print(f"  ({time.time() - t0:.1f}s)")

        all_results[f"HP_{tag}"] = {
            **res_hp, "bpfp": bpfp, "lambda": lmb,
            "cos_mean": cos.mean().item()}

        del net, decoded, img_emb_hp
        torch.cuda.empty_cache()

    # -- Summary table --
    print(f"\n{'=' * 95}")
    print(f"  {layer} Summary  (trun=[{trun_low}, {trun_high}])")
    print(f"{'=' * 95}")
    print(f"  {'Config':<18} {'BPFP':>6} {'i2t R@1':>8} {'i2t R@5':>8} "
          f"{'i2t R@10':>9} {'t2i R@1':>8} {'t2i R@5':>8} {'t2i R@10':>9}")
    print(f"  {'-' * 89}")

    print(f"  {'Anchor':<18} {'':>6} {res_anchor['i2t_R@1']:>7.2f}% "
          f"{res_anchor['i2t_R@5']:>7.2f}% {res_anchor['i2t_R@10']:>8.2f}% "
          f"{res_anchor['t2i_R@1']:>7.2f}% {res_anchor['t2i_R@5']:>7.2f}% "
          f"{res_anchor['t2i_R@10']:>8.2f}%")

    for key in sorted(k for k in all_results if k != "anchor"):
        r = all_results[key]
        if "i2t_R@1" not in r:
            continue
        print(f"  {key:<18} {r['bpfp']:>6.4f} "
              f"{r['i2t_R@1']:>7.2f}% "
              f"{r['i2t_R@5']:>7.2f}% "
              f"{r['i2t_R@10']:>8.2f}% "
              f"{r['t2i_R@1']:>7.2f}% "
              f"{r['t2i_R@5']:>7.2f}% "
              f"{r['t2i_R@10']:>8.2f}%")
    print(f"{'=' * 95}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({layer: all_results}, f, indent=2)
        print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
