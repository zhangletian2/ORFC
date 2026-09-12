#!/usr/bin/env python
"""Merge the scattered DINOv3 results into dinov2-shaped *_tasks.json files.

The dinov2 pipeline gets one self-contained ``*_tasks.json`` per job from
eval_bilinear_orfc_joint_tasks.py, with rANS bits measured on the task images
themselves.  DINOv3 was evaluated by a different script (eval_dinov3_tasks.py,
ADE20K-2000 / NYU-654) that records quality only, and the rate lives elsewhere:

  plain ORFC  orfc/checkpoints/dinov3_vitl16/<stem>.json -> history[-1].rate_per_image
  stage-2     orfcv1/results/bilinear_orfc_jointopq/dinov3_vitl16/<stem>.json -> rate.bits_per_image

Both rates are per ImageNet-224 image (201 tokens), so they are directly
comparable to each other even though they are not the task images' own bits.
BPFP is recomputed here as bits / (201 * 1024); the ``bpfp`` field already in
the stage-2 JSON divides by 257 tokens, a dinov2 leftover that understates
DINOv3 rate by 257/201.

    python -u collect_dinov3_rd.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVAL_RES = ROOT / "orfc" / "eval" / "results" / "dinov3_vitl16"
ORFC_CKPT = ROOT / "orfc" / "checkpoints" / "dinov3_vitl16"
STAGE2_RES = HERE / "results" / "bilinear_orfc_jointopq" / "dinov3_vitl16"

SRC_TOKENS = 201       # 1 CLS + 4 register + 14*14 patches at 224px
FEAT_DIM = 1024

ORFC_SUF = ("_emb32_bt1024_ws_lmbda0.5_split_reg_cls_patch"
            "_tau0.5_lr0.0003_ep100_n5000_s42")
STAGE2_SUF = ("_emb32_bt1024_ws_lmbda0.5_tau2.0_te2.0_tscon"
              "_hlr0.1_bv_lr0.0003_ep100_n5000_nval200_s42")

# Uncompressed feature passed straight to the heads.
ANCHOR_MIOU = 0.5307
ANCHOR_RMSE = 0.3474

# ADE20K val is 2000 images; a few early plain-ORFC runs only did 100 and are
# not comparable, so they are dropped rather than silently plotted.
SEG_MIN_SAMPLES = 2000


def _quality(task_prefix, tag):
    """(value, n_samples) for one eval JSON, or (None, None) if absent."""
    fp = EVAL_RES / f"{task_prefix}_{tag}.json"
    if not fp.is_file():
        return None, None
    d = json.loads(fp.read_text())
    metric = "mIoU" if task_prefix == "semseg" else "rmse"
    return (d.get("metrics") or {}).get(metric), d.get("n_samples")


def _block(bits, miou, rmse, n_seg, n_depth):
    """One dinov2-style result block: quality + a timing stub carrying bpfp."""
    bpfp = bits / (SRC_TOKENS * FEAT_DIM)
    timing = {
        task: {
            "n_images": n,
            "avg_src_tokens_per_image": SRC_TOKENS,
            "bits_per_image": bits,
            "bpfp": bpfp,
        }
        for task, n in (("seg", n_seg), ("depth", n_depth))
    }
    return {
        "seg": None if miou is None else {"miou": miou},
        "depth": None if rmse is None else {"rmse": rmse},
        "timing": timing,
        "rans_prior": "train_bpt",
    }


def collect_orfc(out_dir: Path):
    n = 0
    for fp in sorted(ORFC_CKPT.glob(f"blk*_K*{ORFC_SUF}.json")):
        m = re.match(r"(blk\d+)_K(\d+)_", fp.name)
        if not m:
            continue
        layer, K = m.group(1), int(m.group(2))
        history = json.loads(fp.read_text()).get("history") or []
        if not history:
            print(f"skip {fp.name}: no history -> no rate")
            continue
        bits = history[-1].get("rate_per_image")
        tag = fp.stem
        miou, n_seg = _quality("semseg", tag)
        rmse, n_depth = _quality("depth", tag)
        if miou is not None and (n_seg or 0) < SEG_MIN_SAMPLES:
            print(f"drop seg {layer} K{K}: n_samples={n_seg} < {SEG_MIN_SAMPLES}")
            miou, n_seg = None, None
        if miou is None and rmse is None:
            continue
        doc = {
            "layer": layer,
            "K": K,
            "protocol": "ade2000_nyu654_imagenet_rate",
            "norm_mode": "split_reg_cls_patch",
            "tasks": ["seg", "depth"],
            "orfc_ckpt": str(ORFC_CKPT / f"{tag}.pt"),
            "orfc_only": True,
            "residual_ablation": None,
            "anchor_rmse": ANCHOR_RMSE,
            "anchor_miou": ANCHOR_MIOU,
            "orfc": _block(bits, miou, rmse, n_seg, n_depth),
        }
        (out_dir / f"{tag}_tasks.json").write_text(json.dumps(doc, indent=1) + "\n")
        n += 1
    return n


def collect_stage2(out_dir: Path):
    n = 0
    for fp in sorted(STAGE2_RES.glob(f"blk*_conv2_main_jointopq_K*{STAGE2_SUF}.json")):
        if "clsconv2" in fp.name:
            continue
        d = json.loads(fp.read_text())
        cfg, rate = d.get("config") or {}, d.get("rate") or {}
        layer, K = cfg.get("layer"), int(cfg.get("K"))
        bits = rate.get("bits_per_image")
        if bits is None:
            print(f"skip {fp.name}: no rate.bits_per_image")
            continue
        tag = f"stage2_main_{fp.stem}"
        miou, n_seg = _quality("semseg", tag)
        rmse, n_depth = _quality("depth", tag)
        if miou is None and rmse is None:
            print(f"skip {fp.name}: no eval results")
            continue
        doc = {
            "layer": layer,
            "K": K,
            "protocol": "ade2000_nyu654_imagenet_rate",
            "norm_mode": cfg.get("norm_mode"),
            "tasks": ["seg", "depth"],
            "orfc_ckpt": d.get("ckpt"),
            "spatial_ckpt": d.get("spatial_ckpt"),
            "orfc_only": False,
            "residual_ablation": "main",
            "n_coded": rate.get("n_coded_tokens"),
            "anchor_rmse": ANCHOR_RMSE,
            "anchor_miou": ANCHOR_MIOU,
            "bilinear": _block(bits, miou, rmse, n_seg, n_depth),
        }
        (out_dir / f"{fp.stem}_tasks.json").write_text(
            json.dumps(doc, indent=1) + "\n")
        n += 1
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out_dir",
        default=str(HERE / "results" / "bilinear_orfc_jointopq" / "tasks_dinov3"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_orfc = collect_orfc(out_dir)
    n_stage2 = collect_stage2(out_dir)
    print(f"wrote {n_orfc} ORFC + {n_stage2} stage-2 -> {out_dir}")


if __name__ == "__main__":
    main()
