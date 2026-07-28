#!/usr/bin/env python
"""
Soft-PQ + OPQ COCO Image-to-Text Retrieval evaluation.

For each layer, evaluates:
  - Anchor (raw feature replay, no quantization)
  - OPQ baseline (train R+codebooks on ImageNet, encode/decode coco_ret)
  - Soft-PQ codec (load checkpoint, encode/decode coco_ret)

Usage:
    python eval_ret_soft_pq.py \
        --layer blk07 \
        --feat_root /path/to/coco_ret/siglip2_so400m \
        --train_feat_root /path/to/features/train/siglip2_so400m \
        --meta_json /path/to/subset_meta.json \
        --ckpt_dir  /path/to/checkpoints/siglip2_so400m
"""

import os, sys, json, math, time, argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from soft_pq import load_codec, soft_pq_encode_decode
from opq import (batch_normalize_gpu, batch_inv_normalize_gpu,
                 learn_opq_rotation, batched_assign)
from siglip2_feat_pipeline import load_siglip2, continue_from_tokens

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ---- Data loading ----------------------------------------------------

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


def load_train_features(feat_dir, max_images=5000):
    files = sorted(Path(feat_dir).glob("*.npy"))[:max_images]
    feats = [np.load(str(f)) for f in tqdm(files, desc="Loading train")]
    return feats


# ---- Text encoding ---------------------------------------------------

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


# ---- Image replay ----------------------------------------------------

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


# ---- Retrieval metrics -----------------------------------------------

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


# ---- rANS BPFP -------------------------------------------------------

def _rans_bpfp_from_labels(labels_np, G, K, dim):
    pmfs = []
    for g in range(G):
        counts = np.zeros(K, dtype=np.float64)
        np.add.at(counts, labels_np[g], 1)
        counts += 1.0
        pmfs.append(counts / counts.sum())
    try:
        from compressai._CXX import pmf_to_quantized_cdf
        from compressai import ans
        encoder = ans.RansEncoder()
        N = labels_np.shape[1]
        cdfs, cdf_sizes = [], []
        for g in range(G):
            p = torch.from_numpy(pmfs[g]).float()
            p = torch.cat([p, (1.0 - p.sum()).clamp_min(0).unsqueeze(0)])
            cdfs.append(pmf_to_quantized_cdf(p.tolist(), 16))
            cdf_sizes.append(K + 2)
        symbols, cdf_indices = [], []
        for n in range(N):
            for g in range(G):
                symbols.append(int(labels_np[g, n]))
                cdf_indices.append(g)
        bs = encoder.encode_with_indexes(
            symbols, cdf_indices, cdfs, cdf_sizes, [0] * G)
        return len(bs) * 8 / N / dim
    except (ImportError, ModuleNotFoundError):
        return G * math.log2(K) / dim


def compute_bpfp_codec(features, codec, norm_mode, dim, device):
    codec.eval()
    pq = codec.pq
    G, K = pq.G, pq.K
    all_labels = []
    with torch.no_grad():
        for s in range(0, len(features), 4):
            e = min(s + 4, len(features))
            X = torch.from_numpy(np.stack(features[s:e])).float().to(device)
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
            _ = codec(Y)
            all_labels.append(pq._last_labels.cpu())
            del X, Y
    labels = torch.cat(all_labels, dim=1).numpy()
    return _rans_bpfp_from_labels(labels, G, K, dim)


def compute_bpfp_opq(features, R, codebooks, emb_dim, norm_mode, dim, device):
    G = len(codebooks)
    K = codebooks[0].shape[0]
    R_t = torch.from_numpy(R).float().to(device)
    cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
    all_labels = []
    with torch.no_grad():
        for s in range(0, len(features), 4):
            e = min(s + 4, len(features))
            X = torch.from_numpy(np.stack(features[s:e])).float().to(device)
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
            flat = Y.reshape(-1, dim) @ R_t
            sub = flat.reshape(-1, G, emb_dim).permute(1, 0, 2).contiguous()
            dists = torch.cdist(sub, cb_t)
            lbl = dists.argmin(dim=-1)
            all_labels.append(lbl.cpu())
            del X, Y, flat, sub, dists, lbl
    labels_np = torch.cat(all_labels, dim=1).numpy()
    del R_t, cb_t
    torch.cuda.empty_cache()
    return _rans_bpfp_from_labels(labels_np, G, K, dim)


# ---- OPQ encode/decode -----------------------------------------------

def opq_encode_decode(features, codebooks, emb_dim, norm_mode,
                      R, dim, device, chunk=200):
    G = len(codebooks)
    cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
    R_t = torch.from_numpy(R).float().to(device)
    all_xhat = []
    with torch.no_grad():
        for s in range(0, len(features), chunk):
            e = min(s + chunk, len(features))
            X = torch.from_numpy(np.stack(features[s:e])).float().to(device)
            B, T, C = X.shape
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            flat = Y.reshape(-1, C) @ R_t
            sub = flat.reshape(-1, G, emb_dim).permute(1, 0, 2).contiguous()
            sub_hat, _ = batched_assign(sub, cb_t, device=device)
            flat_hat = sub_hat.permute(1, 0, 2).reshape(-1, C)
            Y_hat = (flat_hat @ R_t.T).reshape(B, T, C)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            for i in range(B):
                all_xhat.append(X_hat[i].cpu().numpy())
            del X, Y, Mu, Std, flat, sub, sub_hat, flat_hat, Y_hat, X_hat
        torch.cuda.empty_cache()
    del cb_t, R_t
    return all_xhat


# ---- OPQ training ----------------------------------------------------

def train_opq(train_features, K, emb_dim, norm_mode, dim, device,
              max_samples=2_000_000, seed=42):
    G = dim // emb_dim
    all_vecs = []
    for s in range(0, len(train_features), 200):
        e = min(s + 200, len(train_features))
        X = torch.from_numpy(
            np.stack(train_features[s:e])).float().to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
        all_vecs.append(Y.reshape(-1, dim).cpu().numpy())
        del X, Y
    flat = np.concatenate(all_vecs, axis=0)
    del all_vecs

    max_flat = max_samples // G
    if flat.shape[0] > max_flat:
        rng = np.random.RandomState(seed)
        idx = rng.choice(flat.shape[0], max_flat, replace=False)
        flat = flat[idx]

    R, codebooks, hist = learn_opq_rotation(
        flat, G, emb_dim, K,
        max_iter_opq=20, max_iter_kmeans=100,
        device=device, verbose=False)
    mse = hist[-1][0] if hist else float('nan')
    del flat
    torch.cuda.empty_cache()
    return R, codebooks, mse


# ---- Checkpoint discovery --------------------------------------------

def discover_checkpoints(ckpt_dir, layer):
    pattern = f"{layer}_K*_emb*_bt1152_ws_lmbda0.0_tau0.5_lr0.0003_ep100_n5000_s42.pt"
    found = sorted(Path(ckpt_dir).glob(pattern))
    configs = []
    for p in found:
        name = p.stem
        parts = name.split('_')
        K = int([x for x in parts if x.startswith('K')][0][1:])
        emb = int([x for x in parts if x.startswith('emb')][0][3:])
        tag = f"K{K}_e{emb}"
        configs.append((tag, K, emb, str(p)))
    configs.sort(key=lambda x: (x[2], x[1]))
    return configs


def _print_ret(res, ref=None):
    di = ""
    dt = ""
    if ref:
        di = f"  (dR@1={res['i2t_R@1']-ref['i2t_R@1']:+.2f}%)"
        dt = f"  (dR@1={res['t2i_R@1']-ref['t2i_R@1']:+.2f}%)"
    print(f"  I2T  R@1={res['i2t_R@1']:.2f}%  "
          f"R@5={res['i2t_R@5']:.2f}%  "
          f"R@10={res['i2t_R@10']:.2f}%{di}")
    print(f"  T2I  R@1={res['t2i_R@1']:.2f}%  "
          f"R@5={res['t2i_R@5']:.2f}%  "
          f"R@10={res['t2i_R@10']:.2f}%{dt}")


# ---- Main ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        "Soft-PQ + OPQ COCO Retrieval Evaluation")
    ap.add_argument("--layer", type=str, required=True)
    ap.add_argument("--feat_root", type=str, required=True,
                    help="COCO ret feature root (contains blkXX/)")
    ap.add_argument("--train_feat_root", type=str, default=None,
                    help="ImageNet training feature root (required for OPQ)")
    ap.add_argument("--meta_json", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--model_id", type=str,
                    default="google/siglip2-so400m-patch14-224")
    ap.add_argument("--norm_mode", type=str, default="per_image")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--method", type=str, default="both",
                    choices=["opq", "softpq", "both"],
                    help="Which method(s) to evaluate")
    args = ap.parse_args()

    run_opq = args.method in ("opq", "both")
    run_softpq = args.method in ("softpq", "both")

    if run_opq and not args.train_feat_root:
        ap.error("--train_feat_root is required when --method includes OPQ")

    device = torch.device(args.device)
    layer = args.layer
    start_idx = int(layer.replace("blk", ""))
    dim = 1152

    entries, captions, img2txt, txt2img = load_meta(args.meta_json)
    print(f"Data: {len(entries)} images, {len(captions)} captions")

    configs = discover_checkpoints(args.ckpt_dir, layer)
    print(f"Layer {layer}: {len(configs)} Soft-PQ checkpoints")

    feat_dir = os.path.join(args.feat_root, layer)
    print(f"\nLoading COCO ret features from {feat_dir} ...")
    features = load_features(feat_dir, entries)
    print(f"  {len(features)} x {features[0].shape}")

    train_features = None
    if run_opq:
        train_feat_dir = os.path.join(args.train_feat_root, layer)
        print(f"Loading training features from {train_feat_dir} ...")
        train_features = load_train_features(train_feat_dir)
        print(f"  {len(train_features)} x {train_features[0].shape}")

    print(f"\nLoading SigLIP2 model ...")
    model, processor = load_siglip2(args.model_id, device)
    model.float()
    vision_model = model.vision_model
    logit_scale = model.logit_scale.exp().item()

    print("Encoding texts ...")
    txt_emb = encode_texts(model, processor, captions, device)

    # -- Anchor --
    print(f"\n{'='*70}")
    print(f"  Anchor: {layer} (raw feature replay)")
    print(f"{'='*70}")
    t0 = time.time()
    img_emb_anchor = replay_to_embeddings(
        vision_model, features, start_idx, device)
    score_anchor = logit_scale * (img_emb_anchor @ txt_emb.t())
    res_anchor = eval_retrieval(score_anchor.numpy(), img2txt, txt2img)
    _print_ret(res_anchor)
    print(f"  ({time.time()-t0:.1f}s)")

    # Load existing results for merging (when running partial method)
    existing_results = {}
    if args.output and os.path.exists(args.output):
        with open(args.output, 'r') as f:
            saved = json.load(f)
            if layer in saved:
                existing_results = saved[layer]

    all_results = {**existing_results, "anchor": res_anchor}

    # -- OPQ baselines --
    if not run_opq:
        print(f"\n  [Skipping OPQ — method={args.method}]")
    for tag, K, emb, _ in (configs if run_opq else []):
        opq_tag = f"OPQ_{tag}"
        print(f"\n{'~'*70}")
        print(f"  {layer} / {opq_tag}  (training OPQ ...)")
        print(f"{'~'*70}")

        t0 = time.time()
        R, codebooks, mse = train_opq(
            train_features, K, emb, args.norm_mode, dim, device)
        print(f"  OPQ trained: MSE={mse:.6f}  ({time.time()-t0:.1f}s)")

        bpfp = compute_bpfp_opq(
            features, R, codebooks, emb, args.norm_mode, dim, device)
        print(f"  BPFP = {bpfp:.4f}")

        recs = opq_encode_decode(
            features, codebooks, emb, args.norm_mode, R, dim, device)
        img_emb_opq = replay_to_embeddings(
            vision_model, recs, start_idx, device)
        score_opq = logit_scale * (img_emb_opq @ txt_emb.t())
        res_opq = eval_retrieval(score_opq.numpy(), img2txt, txt2img)

        cos = torch.nn.functional.cosine_similarity(
            img_emb_anchor, img_emb_opq, dim=-1)
        _print_ret(res_opq, res_anchor)
        print(f"  CosSim: mean={cos.mean():.6f}  min={cos.min():.6f}")
        print(f"  ({time.time()-t0:.1f}s total)")

        all_results[opq_tag] = {
            **res_opq, "bpfp": bpfp, "cos_mean": cos.mean().item()}
        del R, codebooks, recs, img_emb_opq
        torch.cuda.empty_cache()

    if train_features is not None:
        del train_features

    # -- Soft-PQ Codec --
    if not run_softpq:
        print(f"\n  [Skipping Soft-PQ — method={args.method}]")
    for tag, K, emb, ckpt_path in (configs if run_softpq else []):
        codec_tag = f"SoftPQ_{tag}"
        print(f"\n{'~'*70}")
        print(f"  {layer} / {codec_tag}")
        print(f"{'~'*70}")

        t0 = time.time()
        codec = load_codec(ckpt_path, device=device)

        bpfp = compute_bpfp_codec(
            features, codec, args.norm_mode, dim, device)
        print(f"  BPFP = {bpfp:.4f}")

        recs = soft_pq_encode_decode(features, codec, args.norm_mode, device)
        img_emb_codec = replay_to_embeddings(
            vision_model, recs, start_idx, device)
        score_codec = logit_scale * (img_emb_codec @ txt_emb.t())
        res_codec = eval_retrieval(score_codec.numpy(), img2txt, txt2img)

        cos = torch.nn.functional.cosine_similarity(
            img_emb_anchor, img_emb_codec, dim=-1)
        _print_ret(res_codec, res_anchor)
        print(f"  CosSim: mean={cos.mean():.6f}  min={cos.min():.6f}")
        print(f"  ({time.time()-t0:.1f}s)")

        all_results[codec_tag] = {
            **res_codec, "bpfp": bpfp, "cos_mean": cos.mean().item()}
        del codec, recs, img_emb_codec
        torch.cuda.empty_cache()

    # -- Summary table --
    print(f"\n{'='*95}")
    print(f"  {layer} Summary")
    print(f"{'='*95}")
    print(f"  {'Config':<18} {'BPFP':>6} {'i2t R@1':>8} {'i2t R@5':>8} "
          f"{'i2t R@10':>9} {'t2i R@1':>8} {'t2i R@5':>8} {'t2i R@10':>9}")
    print(f"  {'-'*89}")

    print(f"  {'Anchor':<18} {'':>6} {res_anchor['i2t_R@1']:>7.2f}% "
          f"{res_anchor['i2t_R@5']:>7.2f}% {res_anchor['i2t_R@10']:>8.2f}% "
          f"{res_anchor['t2i_R@1']:>7.2f}% {res_anchor['t2i_R@5']:>7.2f}% "
          f"{res_anchor['t2i_R@10']:>8.2f}%")

    all_tags = sorted(set(k for k in all_results if k != "anchor"),
                       key=lambda k: (0 if k.startswith("OPQ") else 1, k))
    for key in all_tags:
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
    print(f"{'='*95}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({layer: all_results}, f, indent=2)
        print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
