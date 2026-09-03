#!/usr/bin/env python3
"""ORFC + RAEv2 (DINOv3-L k1) image reconstruction on ImageNet-500.

Protocol (aligned with CoFAI rae_tail/eval_rae.py and RAEv2 stage1):
  256×256 crop → DINOv3-L/16 encode to slot → optional Soft-PQ
  → remaining blocks + affine-free LN (RAEv2 k1) → patch tokens
  → stats round-trip → ViTXL decoder → PSNR / MS-SSIM / LPIPS

Same 500 images as classification (`imagenet_selected_pathname500.txt`).
RAEv2 weights: third_party/RAEv2/pretrained_models/stage1/imagenet/dinov3l-k1/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

_EVAL_DIR = Path(__file__).resolve().parent
_ORFC = _EVAL_DIR.parent
_ORFCV2 = _ORFC.parent / "orfcv2"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
if str(_ORFC) not in sys.path:
    sys.path.insert(0, str(_ORFC))
if str(_ORFCV2) not in sys.path:
    sys.path.insert(0, str(_ORFCV2))

from dinov3_eval_common import (  # noqa: E402
    BACKBONE_CKPT,
    CKPT_DIR,
    COFAI_ROOT,
    IMAGENET_ROOT,
    IMAGENET_TEST_LIST,
    N_PREFIX,
    NORM_MODE,
    PATCH,
    RAEV2_DECODER_CFG,
    RAEV2_DECODER_PT,
    RAEV2_IMG_SIZE,
    RAEV2_LOG_DIR,
    RAEV2_RESULTS_DIR,
    RAEV2_SRC,
    RAEV2_STATS_PT,
    RAEV2_TOKEN_HW,
    decode_slot,
    layer_idx,
    parse_ckpt_name,
)

os.environ.setdefault("PROJECT_ROOT", str(COFAI_ROOT))

from cofai.backbone.timm import Dinov3TimmBackbone  # noqa: E402
from opq import batch_inv_normalize_gpu, batch_normalize_gpu  # noqa: E402
from residual_pq import ResidualFeatureCodec, load_any_codec  # noqa: E402

if str(RAEV2_SRC) not in sys.path:
    sys.path.insert(0, str(RAEV2_SRC))
from stage1.decoders.decoder import GeneralDecoder  # noqa: E402
from stage1.decoders.utils import ViTMAEConfig  # noqa: E402
from stage1.rae import _load_normalization_stats  # noqa: E402


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


def raev2_preprocess(img: Image.Image, size: int) -> torch.Tensor:
    """Center-crop to square then resize — RAEv2 `scripts/stage1/sample.py`."""
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((size, size), Image.BICUBIC)
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32)) / 255.0
    return arr.permute(2, 0, 1)


def compute_psnr(img1, img2):
    mse = F.mse_loss(img1, img2, reduction="none").mean(dim=[1, 2, 3])
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


def compute_ms_ssim(img1, img2):
    try:
        from pytorch_msssim import ms_ssim

        return ms_ssim(img1, img2, data_range=1.0, size_average=False)
    except ImportError:
        from torchmetrics.functional.image import (
            multiscale_structural_similarity_index_measure as ms_ssim_fn,
        )

        return ms_ssim_fn(img1, img2, data_range=1.0)


def build_backbone(device, slot: int, affine_free: bool):
    backbone = (
        Dinov3TimmBackbone(
            model_size="large",
            img_size=RAEV2_IMG_SIZE,
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
    if affine_free:
        dim = int(backbone.model.embed_dim)
        backbone.model.norm = nn.LayerNorm(dim, elementwise_affine=False).to(device)
        print(f"  last LN replaced with affine-free LayerNorm(dim={dim})  [RAEv2 k1]")
    return backbone


def _load_raev2_vitxl(num_patches: int) -> nn.Module:
    """Load ViTXL without `AutoConfig` (config.json has patch_size='SHOULD BE RELOADED')."""
    cfg_path = Path(RAEV2_DECODER_CFG) / "config.json"
    if not cfg_path.is_file():
        cfg_path = Path(RAEV2_DECODER_CFG)
    raw = json.loads(cfg_path.read_text())
    config = ViTMAEConfig(
        hidden_size=1024,
        patch_size=16,
        image_size=int(16 * math.sqrt(num_patches)),
        decoder_hidden_size=int(raw["decoder_hidden_size"]),
        decoder_intermediate_size=int(raw["decoder_intermediate_size"]),
        decoder_num_attention_heads=int(raw["decoder_num_attention_heads"]),
        decoder_num_hidden_layers=int(raw["decoder_num_hidden_layers"]),
        hidden_act=raw.get("hidden_act", "gelu"),
        hidden_dropout_prob=float(raw.get("hidden_dropout_prob", 0.0)),
        attention_probs_dropout_prob=float(raw.get("attention_probs_dropout_prob", 0.0)),
        layer_norm_eps=float(raw.get("layer_norm_eps", 1e-12)),
        initializer_range=float(raw.get("initializer_range", 0.02)),
        num_channels=int(raw.get("num_channels", 3)),
        qkv_bias=bool(raw.get("qkv_bias", True)),
        num_attention_heads=int(raw.get("num_attention_heads", 12)),
        num_hidden_layers=int(raw.get("num_hidden_layers", 12)),
        intermediate_size=int(raw.get("intermediate_size", 3072)),
    )
    decoder = GeneralDecoder(config, num_patches=num_patches)
    print(f"Loading pretrained decoder from {RAEV2_DECODER_PT}")
    state = torch.load(RAEV2_DECODER_PT, map_location="cpu", weights_only=False)
    keys = decoder.load_state_dict(state, strict=False)
    if keys.missing_keys:
        print(f"  missing keys: {keys.missing_keys}")
    if keys.unexpected_keys:
        print(f"  unexpected keys: {len(keys.unexpected_keys)}")
    return decoder


def load_raev2_head(device):
    if not RAEV2_DECODER_PT.is_file() or not RAEV2_STATS_PT.is_file():
        raise FileNotFoundError(
            f"RAEv2 weights missing:\n  {RAEV2_DECODER_PT}\n  {RAEV2_STATS_PT}\n"
            "Download with hf-mirror, e.g.\n"
            "  curl -L --noproxy '*' -o decoder.pt "
            "'https://hf-mirror.com/nyu-visionx/RAEv2-models/resolve/main/"
            "stage1/imagenet/dinov3l-k1/decoder.pt?download=true'"
        )
    n_patches = RAEV2_TOKEN_HW[0] * RAEV2_TOKEN_HW[1]
    decoder = _load_raev2_vitxl(n_patches).to(device).eval()
    mean, var, do_norm = _load_normalization_stats(str(RAEV2_STATS_PT))
    if mean is not None:
        mean = mean.to(device)
        var = var.to(device)
    print(f"  RAEv2 decoder={RAEV2_DECODER_PT.name}  stats={tuple(mean.shape) if mean is not None else None}")
    return decoder, mean, var, do_norm


@torch.no_grad()
def orfc_recon(h: torch.Tensor, codec, device) -> torch.Tensor:
    y, mu, std = batch_normalize_gpu(h, mode=NORM_MODE, n_prefix=N_PREFIX)
    y_hat, _ = codec(y)
    return batch_inv_normalize_gpu(y_hat, mu, std)


@torch.no_grad()
def patches_after_tail(backbone, h: torch.Tensor, token_hw):
    outs = backbone.decode_whole(h, token_res=token_hw)
    full = outs[-1]
    n_pfx = int(getattr(backbone.model, "num_prefix_tokens", N_PREFIX))
    return full[:, n_pfx:, :]


@torch.no_grad()
def decode_patches(patches, decoder, mean, var, do_norm, eps=1e-5):
    """RAEv2 `RAE.encode` stats + `RAE.decode` (denorm → ViTXL → unpatchify)."""
    b, n, c = patches.shape
    h = w = int(math.sqrt(n))
    z = patches.transpose(1, 2).contiguous().view(b, c, h, w)
    if do_norm and mean is not None:
        z = (z - mean) / torch.sqrt(var + eps)
        z = z * torch.sqrt(var + eps) + mean
    z = z.view(b, c, h * w).transpose(1, 2)
    logits = decoder(z, drop_cls_token=False).logits
    rec = decoder.unpatchify(logits)
    return rec.clamp(0.0, 1.0)


def dump_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)
    print(f"[ok] {path}")


@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    info = parse_ckpt_name(args.ckpt_path) if args.ckpt_path else {}
    layer = args.layer or info.get("layer")
    if not layer:
        raise SystemExit("need --layer (or a ckpt whose name starts with blkXX)")
    slot = decode_slot(layer)
    token_hw = RAEV2_TOKEN_HW
    img_size = RAEV2_IMG_SIZE
    n_patches = token_hw[0] * token_hw[1]
    n_tokens = N_PREFIX + n_patches

    tag = "bypass_" + layer if args.bypass or not args.ckpt_path else Path(args.ckpt_path).stem
    out_path = RAEV2_RESULTS_DIR / f"recon_{tag}.json"
    if out_path.is_file() and not args.force:
        try:
            with open(out_path) as f:
                prev = json.load(f)
            need = args.max_images if args.max_images > 0 else 400
            if int(prev.get("n_samples", 0)) >= need:
                print(f"[skip] {out_path}")
                return
        except Exception:
            pass

    print(f"\n{'=' * 70}")
    print(f"  RAEv2 recon  layer={layer} slot={slot}  {img_size}px  token_hw={token_hw}")
    print(f"  device={device}  affine_free={not args.keep_affine_ln}")
    print(f"{'=' * 70}")

    print("[1/4] backbone")
    backbone = build_backbone(device, slot, affine_free=not args.keep_affine_ln)
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)
    backbone.encode(dummy)
    print(f"  prefix={backbone.model.num_prefix_tokens}  blocks={len(backbone.model.blocks)}")

    print("[2/4] RAEv2 ViTXL decoder")
    decoder, mean, var, do_norm = load_raev2_head(device)

    codec = None
    rate = {}
    if args.ckpt_path and not args.bypass:
        print("[3/4] ORFC codec")
        codec = load_any_codec(args.ckpt_path, device=str(device))
        codec.eval()
        if isinstance(codec, ResidualFeatureCodec):
            kb, gb = codec.base.pq.K, codec.base.pq.G
            kr, gr = codec.res.pq.K, codec.res.pq.G
            bpfp = math.log2(kb) * gb + math.log2(kr) * gr
            rate = {
                "K": int(kb),
                "K2": int(kr),
                "G": int(gb),
                "emb": int(codec.base.pq.d),
                "bpfp": float(bpfp),
                "bpfp_base": float(math.log2(kb) * gb),
                "bpfp_res": float(math.log2(kr) * gr),
                "bits_per_image": float(bpfp * n_tokens),
                "bpp": float(bpfp * n_tokens / (img_size * img_size)),
                "n_tokens": n_tokens,
                "residual": True,
            }
            print(
                f"  residual  K1={kb} K2={kr} G={gb}  "
                f"{bpfp:.2f} bits/token  {rate['bpp']:.4f} bpp"
            )
        else:
            k, g = codec.pq.K, codec.pq.G
            bpfp = math.log2(k) * g
            rate = {
                "K": int(k),
                "G": int(g),
                "emb": int(codec.pq.d),
                "bpfp": float(bpfp),
                "bits_per_image": float(bpfp * n_tokens),
                "bpp": float(bpfp * n_tokens / (img_size * img_size)),
                "n_tokens": n_tokens,
            }
            print(f"  K={k} G={g}  {bpfp:.2f} bits/token  {rate['bpp']:.4f} bpp")
    else:
        print("[3/4] bypass (no codec)")

    lpips_fn = None
    try:
        import lpips

        lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as e:
        print(f"  [warn] LPIPS unavailable: {e}")

    pairs = load_image_pairs(Path(args.pathname_list), Path(args.imagenet_root))
    if args.max_images > 0:
        pairs = pairs[: args.max_images]
    print(f"[4/4] {len(pairs)} images  bs={args.batch_size}")

    acc = {
        "psnr_bp": [],
        "psnr_cd": [],
        "msssim_bp": [],
        "msssim_cd": [],
        "lpips_bp": [],
        "lpips_cd": [],
    }
    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for start in tqdm(range(0, len(pairs), args.batch_size), desc="recon"):
        chunk = pairs[start : start + args.batch_size]
        gts, names, keep = [], [], []
        for base, path in chunk:
            if not path.is_file():
                print(f"[warn] missing {path}")
                continue
            img = Image.open(path).convert("RGB")
            gts.append(raev2_preprocess(img, img_size))
            names.append(base)
            keep.append(path)
        if not gts:
            continue
        gt = torch.stack(gts).to(device)
        h = backbone.encode(gt)

        feat_bp = patches_after_tail(backbone, h, token_hw)
        rec_bp = decode_patches(feat_bp, decoder, mean, var, do_norm)

        rec_cd = None
        if codec is not None:
            h_hat = orfc_recon(h, codec, device)
            feat_cd = patches_after_tail(backbone, h_hat, token_hw)
            rec_cd = decode_patches(feat_cd, decoder, mean, var, do_norm)

        acc["psnr_bp"].extend(compute_psnr(rec_bp, gt).tolist())
        acc["msssim_bp"].extend(compute_ms_ssim(rec_bp, gt).tolist())
        if rec_cd is not None:
            acc["psnr_cd"].extend(compute_psnr(rec_cd, gt).tolist())
            acc["msssim_cd"].extend(compute_ms_ssim(rec_cd, gt).tolist())
        if lpips_fn is not None:
            acc["lpips_bp"].extend(
                lpips_fn(rec_bp * 2 - 1, gt * 2 - 1).reshape(-1).tolist()
            )
            if rec_cd is not None:
                acc["lpips_cd"].extend(
                    lpips_fn(rec_cd * 2 - 1, gt * 2 - 1).reshape(-1).tolist()
                )

        if vis_dir is not None and start == 0:
            n_vis = min(args.n_vis, rec_bp.size(0))
            for i in range(n_vis):
                def _save(t, name):
                    arr = (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(
                        np.uint8
                    )
                    Image.fromarray(arr).save(vis_dir / name)

                _save(gt[i], f"{names[i]}_gt.png")
                _save(rec_bp[i], f"{names[i]}_bypass.png")
                if rec_cd is not None:
                    _save(rec_cd[i], f"{names[i]}_orfc.png")

        del h, feat_bp, rec_bp, gt
        if rec_cd is not None:
            del rec_cd

    elapsed = time.time() - t0
    n = len(acc["psnr_bp"])
    metrics_bp = {
        "psnr": float(np.mean(acc["psnr_bp"])) if n else None,
        "ms_ssim": float(np.mean(acc["msssim_bp"])) if n else None,
        "lpips": float(np.mean(acc["lpips_bp"])) if acc["lpips_bp"] else None,
    }
    metrics_cd = None
    if acc["psnr_cd"]:
        metrics_cd = {
            "psnr": float(np.mean(acc["psnr_cd"])),
            "ms_ssim": float(np.mean(acc["msssim_cd"])),
            "lpips": float(np.mean(acc["lpips_cd"])) if acc["lpips_cd"] else None,
        }

    print(f"\n{'=' * 70}")
    print(f"  {n} images  {elapsed:.1f}s  layer={layer}")
    print(f"  {'Metric':<12} {'Bypass':<14} {'ORFC':<14} {'Delta':<10}")
    print(f"  {'-' * 50}")
    for key, label in (("psnr", "PSNR"), ("ms_ssim", "MS-SSIM"), ("lpips", "LPIPS")):
        bp = metrics_bp.get(key)
        cd = metrics_cd.get(key) if metrics_cd else None
        bp_s = f"{bp:.4f}" if bp is not None else "-"
        cd_s = f"{cd:.4f}" if cd is not None else "-"
        d_s = f"{cd - bp:+.4f}" if (bp is not None and cd is not None) else "-"
        print(f"  {label:<12} {bp_s:<14} {cd_s:<14} {d_s:<10}")
    if rate:
        print(f"  bpp={rate['bpp']:.4f}  bpfp={rate['bpfp']:.3f}")
    print(f"{'=' * 70}\n")

    payload = {
        "task": "recon_raev2",
        "tag": tag,
        "layer": layer,
        "slot": slot,
        "mode": "bypass" if codec is None else "orfc",
        "ckpt": args.ckpt_path or "",
        "n_samples": n,
        "img_size": img_size,
        "token_hw": list(token_hw),
        "affine_free_ln": not args.keep_affine_ln,
        "norm_mode": NORM_MODE,
        "metrics_bypass": metrics_bp,
        "metrics": metrics_cd if metrics_cd is not None else metrics_bp,
        "rate": rate,
        "elapsed_s": elapsed,
        "decoder": str(RAEV2_DECODER_PT),
    }
    dump_json(out_path, payload)


def parse_args():
    p = argparse.ArgumentParser("ORFC + RAEv2 k1 reconstruction")
    p.add_argument("--ckpt_path", default="", help="Soft-PQ .pt; empty with --bypass")
    p.add_argument("--bypass", action="store_true")
    p.add_argument("--layer", default="", help="blk05/10/15/20/23; parsed from ckpt if omitted")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep_affine_ln", action="store_true",
                   help="Keep DINOv3 affine last LN (default: RAEv2 affine-free)")
    p.add_argument("--pathname_list", default=str(IMAGENET_TEST_LIST))
    p.add_argument("--imagenet_root", default=str(IMAGENET_ROOT))
    p.add_argument("--vis_dir", default="")
    p.add_argument("--n_vis", type=int, default=8)
    p.add_argument("--ckpt_dir", default=str(CKPT_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    RAEV2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAEV2_LOG_DIR.mkdir(parents=True, exist_ok=True)
    evaluate(args)


if __name__ == "__main__":
    main()
