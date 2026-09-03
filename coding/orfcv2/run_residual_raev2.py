#!/usr/bin/env python3
"""orfcv2: K2 residual SoftPQ, share R, pixel L1 KD through RAEv2.

Reuses ResidualFeatureCodec / share_base_transform from residual_pq.py.
Only residual codebooks (+ prior if λ>0) are trained; base K1 and R stay frozen.

Protocol (rae_tail/train.py Mode B, RAEv2 k1 instead of DINOv2 RAE):
  stage2 ImageNet 5k, 256×256 crop → DINOv3 encode to slot → freeze K1+R
  residual = Y − Ŷ_base → PQ_K2 in shared-R coordinates
  remaining blocks + affine-free LN → ViTXL → L1 vs bypass teacher pixels

Features: features/train/dinov3_vitl16_stage2 (RAEv2 256px, T=261).

    python run_residual_raev2.py \\
        --base_ckpt ../orfc/checkpoints/dinov3_vitl16/blk20_K16_....pt \\
        --K 2 --epochs 100 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

ORFCV2_ROOT = Path(__file__).resolve().parent
ORFC_ROOT = ORFCV2_ROOT.parent / "orfc"
for p in (ORFCV2_ROOT, ORFC_ROOT, ORFC_ROOT / "eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dinov3_eval_common import (  # noqa: E402
    BACKBONE_CKPT,
    COFAI_ROOT,
    IMAGENET_ROOT,
    IMAGENET_TRAIN_FEAT_STAGE2,
    IMAGENET_TRAIN_LIST_STAGE2,
    N_PREFIX,
    NORM_MODE,
    PATCH,
    RAEV2_IMG_SIZE,
    RAEV2_TOKEN_HW,
    decode_slot,
    layer_idx,
    parse_ckpt_name,
)
from opq import batch_inv_normalize_gpu, batch_normalize_gpu  # noqa: E402
from raev2_head import load_raev2_head, patches_to_pixels, raev2_preprocess  # noqa: E402
from residual_pq import (  # noqa: E402
    ResidualFeatureCodec,
    _cuda_mem,
    build_residual_codec,
    collect_residuals,
    save_residual_codec,
    share_base_transform,
)
from soft_pq import compute_perplexity, load_codec  # noqa: E402
from backbone.dinov3_tail import build_dinov3_tail  # noqa: E402

os.environ.setdefault("PROJECT_ROOT", str(COFAI_ROOT))
from cofai.backbone.timm import Dinov3TimmBackbone  # noqa: E402


def load_image_pairs(list_txt: Path, root: Path):
    pairs = []
    with open(list_txt) as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) < 2:
                continue
            wnid, base = parts[0], parts[1]
            pairs.append((base, root / wnid / f"{base}.JPEG"))
    return pairs


def extract_features(backbone, pairs, img_size, device, batch_size=4):
    feats, names = [], []
    print(f"  Extracting {len(pairs)} images at {img_size}px...")
    with torch.inference_mode():
        for i in tqdm(range(0, len(pairs), batch_size), desc="Extract"):
            chunk = pairs[i : i + batch_size]
            imgs, keep = [], []
            for base, path in chunk:
                if not path.is_file():
                    print(f"  [warn] missing {path}")
                    continue
                imgs.append(raev2_preprocess(Image.open(path).convert("RGB"), img_size))
                keep.append(base)
            if not imgs:
                continue
            x = torch.stack(imgs).to(device)
            h = backbone.encode(x)
            for j, name in enumerate(keep):
                feats.append(h[j].detach().cpu().numpy().astype(np.float32))
                names.append(name)
            del x, h
    print(f"  extracted {len(feats)}  shape={feats[0].shape}")
    return feats, names


def _load_single_npy(npy_path):
    return np.load(npy_path).astype(np.float32)


def load_cached_features(feat_cache_dir, img_names, num_workers=8):
    """Parallel .npy load — same pattern as rae_tail/train.py."""
    cache_dir = Path(feat_cache_dir)
    npy_paths = []
    for name in img_names:
        npy_path = cache_dir / f"{name}.npy"
        if not npy_path.is_file():
            raise FileNotFoundError(f"Feature cache not found: {npy_path}")
        npy_paths.append(npy_path)

    feats = [None] * len(npy_paths)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i, arr in enumerate(executor.map(_load_single_npy, npy_paths)):
            feats[i] = arr
    print(
        f"  Loaded {len(feats)} cached features from {cache_dir} "
        f"({num_workers} workers)"
    )
    print(f"    token shape: {feats[0].shape}")
    return feats


def save_cached_features(cache_dir: Path, names, feats):
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, feat in zip(names, feats):
        np.save(cache_dir / f"{name}.npy", feat)


def rae_patches(h_full, tail, n_prefix):
    """Remaining blocks + last LN → patch tokens (RAEv2 k1)."""
    return tail(h_full)[:, n_prefix:, :]


def ckpt_stem(args, base_stem: str, n_img: int) -> str:
    return (
        f"{base_stem}_resK{args.K}_shareR_raev2l1"
        f"_lmbda{args.lmbda}_tau{args.tau_start}"
        f"_lr{args.lr}_ep{args.epochs}_n{n_img}_s{args.seed}"
    )


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    info = parse_ckpt_name(args.base_ckpt)
    layer = args.layer or info.get("layer")
    if not layer:
        raise SystemExit("need --layer or a base ckpt named blkXX_...")
    slot = decode_slot(layer)
    lyr_i = layer_idx(layer)
    img_size = RAEV2_IMG_SIZE
    token_hw = RAEV2_TOKEN_HW
    n_prefix = N_PREFIX
    k2 = args.K

    print(f"\n{'=' * 70}")
    print("  orfcv2 residual  share R  |  RAEv2 pixel L1 KD")
    print(f"  layer={layer} slot={slot}  K2={k2}  λ={args.lmbda}")
    print(f"  {img_size}px  token_hw={token_hw}  n_prefix={n_prefix}")
    print(f"  base={Path(args.base_ckpt).name}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  device={device}")
    print(f"{'=' * 70}")

    print("\n[1/5] backbone + frozen tail")
    backbone = (
        Dinov3TimmBackbone(
            model_size="large",
            img_size=img_size,
            patch_size=PATCH,
            dynamic_size=False,
            slot=slot,
            n_last_blocks=1,
            cast_dtype="float32",
            pretrained=False,
            ckpt_path=BACKBONE_CKPT,
            device=str(device),
        )
        .eval()
        .to(device)
    )
    dim = int(backbone.model.embed_dim)
    backbone.model.norm = nn.LayerNorm(dim, elementwise_affine=False).to(device)
    print(f"  last LN → affine-free  dim={dim}")

    pairs = load_image_pairs(Path(args.pathname_list), Path(args.imagenet_root))
    print(f"  Loaded {len(pairs)} images from pathname list")
    if args.max_images > 0 and args.max_images < len(pairs):
        pairs = pairs[: args.max_images]
        print(f"  Truncated to {len(pairs)} images")
    names = [b for b, _ in pairs]
    cache_dir = Path(args.feat_cache_dir) / layer if args.feat_cache_dir else None
    if cache_dir:
        print(f"[2/5] Loading cached features from: {cache_dir}")
        feats = load_cached_features(cache_dir, names)
    else:
        print("[2/5] Extracting features (no cache)...")
        feats, names = extract_features(
            backbone, pairs, img_size, device, batch_size=args.extract_batch_size
        )

    n_img = len(feats)
    t_tokens, d = feats[0].shape
    expect_t = n_prefix + token_hw[0] * token_hw[1]
    print(f"  D={d}, T={t_tokens}")
    if t_tokens != expect_t:
        raise SystemExit(
            f"feature T={t_tokens} != {expect_t} for {img_size}px "
            f"(need RAEv2 256 / T=261, not 224 / T=201)"
        )

    features_array = np.stack(feats)
    del feats
    print(f"  features_array: {features_array.shape} "
          f"({features_array.nbytes / 1e9:.2f} GB)")

    print("[3/5] Building FrozenTail...")
    tail = build_dinov3_tail(backbone, lyr_i, token_hw, device)
    n_tail = len(getattr(tail, "blocks", []))
    print(f"  FrozenTail: {n_tail} blocks + affine-free LN")
    backbone.model.cpu()
    torch.cuda.empty_cache()
    _cuda_mem(device, " after backbone.cpu()")

    print("[4/5] freeze base, share R, k-means residual in R-space")
    base = load_codec(args.base_ckpt, device=str(device))
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    g, d_sub = base.pq.G, base.pq.d
    if g * d_sub != d:
        raise SystemExit(f"base G*d={g * d_sub} != D={d}")
    print(f"  base K={base.pq.K} G={g} d={d_sub}  bits/token={g * math.log2(base.pq.K):.0f}")

    res = build_residual_codec(
        d, g, k2, d_sub, ecvq_lmbda=args.lmbda, transform_kind="orthogonal"
    ).to(device)
    share_base_transform(res, base)
    residual_vecs = collect_residuals(
        features_array,
        base,
        NORM_MODE,
        device,
        max_vectors=args.kmeans_max_samples,
        n_prefix=n_prefix,
        seed=args.seed,
    )
    print(f"  residual vectors: {residual_vecs.shape}  "
          f"mean square={float((residual_vecs ** 2).mean()):.6f}")
    z = torch.from_numpy(np.ascontiguousarray(residual_vecs)).float()
    with torch.no_grad():
        z = res.transform.to("cpu").encode(z)
    res.transform.to(device)
    print(f"  Residual init: k-means on {z.shape[0]:,} vectors in shared-R space")
    res.pq.init_from_kmeans(z, device=str(device))
    if res.pq.use_rate:
        g_pq, k_pq, d_pq = res.pq.G, res.pq.K, res.pq.d
        counts = np.zeros((g_pq, k_pq), dtype=np.float64)
        c = res.pq.codebooks.detach().to(device)
        chunk = 200_000
        with torch.no_grad():
            for s in range(0, z.shape[0], chunk):
                e = min(s + chunk, z.shape[0])
                zg = (z[s:e].to(device)
                      .reshape(-1, g_pq, d_pq).permute(1, 0, 2).contiguous())
                labels = torch.cdist(zg, c).argmin(dim=-1).cpu().numpy()
                for gi in range(g_pq):
                    np.add.at(counts[gi], labels[gi], 1)
                del zg
        res.pq.init_prior_from_freq(counts)
        p = counts / counts.sum(-1, keepdims=True)
        ppl = float(np.exp(-(p * np.log(p + 1e-30)).sum(-1)).mean())
        print(f"    residual prior init from k-means frequency (ppl={ppl:.1f})")
    del residual_vecs, z
    for p in res.transform.parameters():
        p.requires_grad_(False)
    print("  Frozen: residual transform (share R); training codebooks only")
    torch.cuda.empty_cache()

    codec = ResidualFeatureCodec(base, res).to(device)
    trainable = [p for p in codec.res.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    # FrozenTail back to GPU (backbone.model.cpu() moved shared modules)
    tail.to(str(device))

    print("[5/5] RAEv2 ViTXL (deferred after k-means) + teacher pixels")
    decoder, _, _, _ = load_raev2_head(device)
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"  RAEv2 Decoder loaded ({n_dec:,} params, all frozen)")
    print("  Loss mode: pixel-level L1 (RAE tail)")
    _cuda_mem(device, " after decoder")

    teacher_pixel_cache = np.empty((n_img, 3, img_size, img_size), dtype=np.float16)
    teacher_bs = min(args.batch_size, 4)
    print("  Pre-computing teacher (bypass pixel reconstruction via RAE tail)...")
    with torch.no_grad():
        for start in tqdm(range(0, n_img, teacher_bs), desc="Teacher pixels"):
            end = min(start + teacher_bs, n_img)
            x = torch.from_numpy(features_array[start:end]).float().to(device)
            patches = rae_patches(x, tail, n_prefix)
            pix = patches_to_pixels(decoder, patches, clamp=True)
            teacher_pixel_cache[start:end] = pix.cpu().half().numpy()
            del x, patches, pix
    torch.cuda.empty_cache()
    print(f"  Teacher pixel cache: {teacher_pixel_cache.nbytes / 1e9:.2f} GB")
    _cuda_mem(device, " after teacher")

    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    print("  Training...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    indices = np.arange(n_img)
    for epoch in range(args.epochs):
        t0 = time.time()
        if args.epochs > 1:
            progress = epoch / (args.epochs - 1)
            tau = args.tau_start * (args.tau_end / args.tau_start) ** progress
        else:
            tau = args.tau_start
        codec.res.pq.temperature = tau
        np.random.shuffle(indices)
        tot_d = tot_r = 0.0
        usage_acc = torch.zeros(g, k2, device=device)
        codec.train()
        for start in range(0, n_img, args.batch_size):
            end = min(start + args.batch_size, n_img)
            idx = indices[start:end]
            b = len(idx)
            x = torch.from_numpy(features_array[idx]).float().to(device)
            with torch.no_grad():
                y, mu, std = batch_normalize_gpu(x, mode=NORM_MODE, n_prefix=n_prefix)
            y_hat, usage = codec(y)
            x_hat = batch_inv_normalize_gpu(y_hat, mu, std)
            patches = rae_patches(x_hat, tail, n_prefix)
            rec = patches_to_pixels(decoder, patches, clamp=False)
            tgt = torch.from_numpy(
                teacher_pixel_cache[idx].astype(np.float32)
            ).to(device)
            distortion = F.l1_loss(rec, tgt, reduction="sum") / b
            if codec.use_rate:
                rate_bits = codec._last_rate * t_tokens
                loss = rate_bits + args.lmbda * distortion
            else:
                loss = distortion
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            tot_d += distortion.item() * b
            tot_r += (
                codec._last_rate.item() * b
                if codec.use_rate
                else math.log2(k2) * g * b
            )
            usage_acc += usage.detach()
            del x, y, mu, std, y_hat, x_hat, patches, rec, tgt, loss, distortion
        scheduler.step()
        avg_d = tot_d / n_img
        avg_r = tot_r / n_img
        ppl = compute_perplexity(usage_acc)
        dead = int((usage_acc == 0).sum().item())
        info_ep = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "l1": avg_d,
            "rate_bpt": avg_r,
            "perplexity": ppl,
            "dead_entries": dead,
            "temperature": float(codec.res.pq.temperature),
            "time": time.time() - t0,
        }
        history.append(info_ep)
        if epoch % max(1, args.log_every) == 0 or epoch == args.epochs - 1:
            print(
                f"  ep {epoch:3d}/{args.epochs}  L1={avg_d:.4f}  "
                f"R={avg_r:.2f}b/t  ppl={ppl:.1f}  dead={dead}  "
                f"τ={codec.res.pq.temperature:.4f}  ({info_ep['time']:.1f}s)"
            )
            _cuda_mem(device, f" ep{epoch}")

    if args.no_save:
        print("\n  --no_save: skip checkpoint")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = ckpt_stem(args, Path(args.base_ckpt).stem, n_img)
    ckpt_path = out_dir / f"{stem}.pt"
    extra = {
        "layer": layer,
        "slot": slot,
        "norm_mode": NORM_MODE,
        "n_prefix": n_prefix,
        "img_size": img_size,
        "token_hw": list(token_hw),
        "K2": k2,
        "share_r": True,
        "loss": "raev2_pixel_l1",
        "history": history,
    }
    save_residual_codec(codec, str(ckpt_path), args.base_ckpt, extra=extra)
    json_path = out_dir / f"{stem}.json"
    payload = {k: v for k, v in extra.items() if k != "history"}
    payload.update({
        "base_ckpt": str(Path(args.base_ckpt).resolve()),
        "ckpt": str(ckpt_path),
        "n_img": n_img,
        "history": history,
    })
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  saved {ckpt_path}")
    print(f"  json  {json_path}\n")


def parse_args():
    p = argparse.ArgumentParser(
        "orfcv2 residual K2 + share R, RAEv2 pixel L1 KD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_ckpt", required=True)
    p.add_argument("--layer", default="", help="blk05/10/15/20; parsed from ckpt if empty")
    p.add_argument("--K", "--K2", dest="K", type=int, default=2,
                   help="Residual codebook size (K2)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lmbda", type=float, default=0.0,
                   help="Residual-training rate λ (R+λD; 0 = distortion only, no prior)")
    p.add_argument("--tau_start", type=float, default=0.5)
    p.add_argument("--tau_end", type=float, default=0.005)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=16,
                   help="Unified batch size (4090, no grad-checkpoint)")
    p.add_argument("--extract_batch_size", type=int, default=4)
    p.add_argument("--max_images", type=int, default=0, help="0 = all in pathname_list")
    p.add_argument("--kmeans_max_samples", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no_save", action="store_true",
                   help="Skip writing ckpt (batch-size probe)")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--pathname_list", default=str(IMAGENET_TRAIN_LIST_STAGE2))
    p.add_argument("--imagenet_root", default=str(IMAGENET_ROOT))
    p.add_argument(
        "--feat_cache_dir",
        default=str(IMAGENET_TRAIN_FEAT_STAGE2),
    )
    p.add_argument(
        "--out_dir",
        default=str(ORFCV2_ROOT / "checkpoints" / "dinov3_vitl16_raev2_res"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
