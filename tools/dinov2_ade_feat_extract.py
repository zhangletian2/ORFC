#!/data4/workspace/zlt/featcodec/CoFAI/.venv/bin/python
# -*- coding: utf-8 -*-
"""
DINOv2 ViT-L/14 特征提取 — ADE train 5k（原生分辨率 + pad 到 14 的倍数）

与 DINOv3 ADE 提取共用同一份图像清单（ade_train_683x512_5k.txt）：
  原图 683×512，底/右 pad 到 patch=14 的倍数 → 686×518
  → 49×37 patches + 1 CLS = [1814, 1024]（无 register）

一次前向同时抽取 blk05 / blk10 / blk15 / blk20（0-based block 索引）。

用法：
  python tools/dinov2_ade_feat_extract.py --device cuda:4

  python tools/dinov2_ade_feat_extract.py \\
      --list utils/ade_train_683x512_5k.txt \\
      --ade_root /data4/workspace/zlt/featcodec/CoFAI/data/ADEChallengeData2016/images/training \\
      --out_root /data4/workspace/zlt/featcodec/ORFC/features/train/dinov2_vitl14_ade \\
      --blocks 5,10,15,20 --device cuda:4 --skip_existing
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backbone" / "dinov2"))

from dinov2.models import vision_transformer as vits  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 14
DEFAULT_BLOCKS = (5, 10, 15, 20)

DEFAULT_LIST = ROOT / "utils" / "ade_train_683x512_5k.txt"
DEFAULT_ADE_ROOT = Path(
    "/data4/workspace/zlt/featcodec/CoFAI/data/"
    "ADEChallengeData2016/images/training"
)
DEFAULT_OUT_ROOT = ROOT / "features" / "train" / "dinov2_vitl14_ade"
DEFAULT_WEIGHTS = ROOT / "pretrained" / "dinov2_vitl14_pretrain.pth"


def pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """Pad NCHW tensor on bottom/right so H,W are multiples of ``multiple``."""
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)


def load_ade_paths(pathname_list: str, ade_root: str):
    """One filename (or stem) per line → (path, stem) for .npy naming."""
    root = Path(ade_root)
    paths, names = [], []
    with open(pathname_list, "r") as f:
        for line in f:
            tok = line.strip().split()[0] if line.strip() else ""
            if not tok:
                continue
            name = Path(tok).name
            stem = Path(name).stem
            img_path = root / name
            if not img_path.exists():
                for ext in (".jpg", ".JPG", ".png", ".JPEG"):
                    cand = root / f"{stem}{ext}"
                    if cand.exists():
                        img_path = cand
                        break
            if img_path.exists():
                paths.append(img_path)
                names.append(stem)
            else:
                print(f"[warn] missing: {root / name}")
    return paths, names


class BlockOutputCatcher:
    """Hook specified backbone.blocks; collect [B, T, D] on CPU."""

    def __init__(self, backbone: nn.Module, block_indices):
        self.indices = sorted(set(int(i) for i in block_indices))
        self._buf = {}
        self._handles = []
        assert hasattr(backbone, "blocks"), "backbone 缺少 .blocks"
        blocks = list(backbone.blocks)
        n_blocks = len(blocks)

        def _make_hook(idx):
            key = f"blk{idx:02d}"

            def hook(module, inp, out):
                if isinstance(out, list):
                    self._buf[key] = out[0].detach().cpu()
                else:
                    self._buf[key] = out.detach().cpu()

            return hook

        for idx in self.indices:
            if idx >= n_blocks:
                raise IndexError(f"block index {idx} >= n_blocks={n_blocks}")
            h = blocks[idx].register_forward_hook(_make_hook(idx))
            self._handles.append(h)

    def pop(self):
        outs = self._buf
        self._buf = {}
        return outs

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def build_backbone(weights_path: str, device: str):
    backbone = vits.vit_large(
        patch_size=PATCH_SIZE, img_size=518, init_values=1.0, block_chunks=0
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    backbone.load_state_dict(state, strict=True)
    backbone = backbone.to(device).eval()
    print(f"  Backbone: {weights_path}")
    print(f"  blocks={len(backbone.blocks)}  embed={backbone.embed_dim}  "
          f"patch={backbone.patch_size}  registers={backbone.num_register_tokens}")
    return backbone


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("DINOv2 ViT-L/14 ADE 5k feature extract")
    ap.add_argument("--list", default=str(DEFAULT_LIST),
                    help="ADE filename list (one image per line)")
    ap.add_argument("--ade_root", default=str(DEFAULT_ADE_ROOT))
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT),
                    help="Output root; writes blkXX/*.npy underneath")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--blocks", default="5,10,15,20",
                    help="0-based block indices, comma-separated")
    ap.add_argument("--pad_multiple", type=int, default=PATCH_SIZE)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_images", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    layers = [int(x) for x in args.blocks.split(",") if x.strip()]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for k in layers:
        (out_root / f"blk{k:02d}").mkdir(parents=True, exist_ok=True)

    list_src = Path(args.list)
    list_dst = out_root / list_src.name
    if list_src.is_file() and not list_dst.exists():
        shutil.copy2(list_src, list_dst)

    print(f"\n{'=' * 70}")
    print("  DINOv2 ViT-L/14 ADE feature extract")
    print(f"  native size + pad_multiple={args.pad_multiple}")
    print(f"  blocks={['blk%02d' % k for k in layers]}")
    print(f"  list={args.list}")
    print(f"  images={args.ade_root}")
    print(f"  out={out_root}")
    print(f"{'=' * 70}")

    print("\n[1/3] Loading backbone...")
    backbone = build_backbone(args.weights, args.device)
    catcher = BlockOutputCatcher(backbone, layers)

    print("[2/3] Loading image list...")
    img_files, img_names = load_ade_paths(args.list, args.ade_root)
    if args.max_images > 0:
        img_files = img_files[: args.max_images]
        img_names = img_names[: args.max_images]
    print(f"  {len(img_files)} images")

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    saved = skipped = missing_skip = 0
    sample_hw = None
    t0 = time.time()

    print("[3/3] Extracting...")
    try:
        for i in tqdm(range(0, len(img_files), args.batch_size), desc="Extract"):
            batch_files = img_files[i:i + args.batch_size]
            batch_names = img_names[i:i + args.batch_size]
            todo_idx = []
            for j, name in enumerate(batch_names):
                if args.skip_existing and all(
                    (out_root / f"blk{k:02d}" / f"{name}.npy").is_file()
                    for k in layers
                ):
                    skipped += 1
                else:
                    todo_idx.append(j)
            if not todo_idx:
                continue

            imgs = []
            for j in todo_idx:
                imgs.append(to_tensor(Image.open(batch_files[j]).convert("RGB")))
            x = torch.stack(imgs).to(args.device)
            x = pad_to_multiple(x, args.pad_multiple)
            if sample_hw is None:
                sample_hw = (int(x.shape[2]), int(x.shape[3]))
            x = normalize(x)

            _ = backbone(x)
            outs = catcher.pop()
            for bi, j in enumerate(todo_idx):
                name = batch_names[j]
                for k in layers:
                    key = f"blk{k:02d}"
                    arr = outs[key][bi].numpy().astype(np.float32)
                    np.save(out_root / key / f"{name}.npy", arr)
                saved += 1
            del x, outs
    finally:
        catcher.close()

    elapsed = time.time() - t0
    print(f"\n  saved={saved}  skipped_existing={skipped}  "
          f"({elapsed:.1f}s, {saved / max(elapsed, 1):.2f} img/s)")

    sample = next((out_root / f"blk{layers[0]:02d}").glob("ADE_train_*.npy"), None)
    if sample is None:
        sample = next((out_root / f"blk{layers[0]:02d}").glob("*.npy"), None)
    if sample is None:
        raise SystemExit("No .npy written")
    arr = np.load(sample)
    if sample_hw is None:
        im0 = Image.open(img_files[0])
        w0, h0 = im0.size
        ph = (args.pad_multiple - h0 % args.pad_multiple) % args.pad_multiple
        pw = (args.pad_multiple - w0 % args.pad_multiple) % args.pad_multiple
        sample_hw = (h0 + ph, w0 + pw)
    th, tw = sample_hw[0] // PATCH_SIZE, sample_hw[1] // PATCH_SIZE
    expect_t = 1 + th * tw  # CLS + patches, no registers
    print(f"  Padded input HxW={sample_hw} → token_hw=({th},{tw})")
    print(f"  Sample {sample.name}: shape={arr.shape} (expect T={expect_t}, D=1024)")
    npy_counts = {
        f"blk{k:02d}": len(list((out_root / f"blk{k:02d}").glob("*.npy")))
        for k in layers
    }
    print(f"  npy counts: {npy_counts}")
    if arr.shape[0] != expect_t or arr.shape[1] != 1024:
        raise SystemExit(f"Unexpected shape {arr.shape}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
